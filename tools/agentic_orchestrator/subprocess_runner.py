"""Safe subprocess discovery, execution, classification, and redaction."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MAX_CAPTURE_CHARS = 2_000_000
SENSITIVE_ENV_RE = re.compile(
    r"(TOKEN|SECRET|PASSWORD|CREDENTIAL|API[_-]?KEY|PRIVATE[_-]?KEY)", re.I
)
REDACTIONS = (
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)((?:api[_-]?key|token|password|secret)\s*[:=]\s*)\S+"),
    re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", re.S),
)

USAGE_PATTERNS = (
    "rate limit",
    "rate-limit",
    "quota exceeded",
    "usage limit",
    "credits exhausted",
    "insufficient credits",
    "too many requests",
    "resource_exhausted",
)
AUTH_PATTERNS = (
    "not authenticated",
    "authentication required",
    "unauthorized",
    "please log in",
    "login required",
)


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    classification: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class ToolProbe:
    tool: str
    executable: str | None
    version: str | None
    authenticated: bool | None
    models: tuple[str, ...]
    configured_model: str | None
    status: str
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "TOOL": self.tool,
            "EXECUTABLE": self.executable,
            "VERSION": self.version,
            "AUTHENTICATED": self.authenticated,
            "MODELS": list(self.models),
            "CONFIGURED_MODEL": self.configured_model,
            "STATUS": self.status,
            "DETAIL": self.detail,
        }


def sanitize_text(text: str) -> str:
    sanitized = text
    for pattern in REDACTIONS:
        sanitized = pattern.sub(
            lambda match: f"{match.group(1) if match.lastindex else ''}[REDACTED]", sanitized
        )
    if len(sanitized) > MAX_CAPTURE_CHARS:
        return sanitized[:MAX_CAPTURE_CHARS] + "\n[TRUNCATED]"
    return sanitized


def safe_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not SENSITIVE_ENV_RE.search(key)}
    if extra:
        for key, value in extra.items():
            if SENSITIVE_ENV_RE.search(key):
                raise ValueError(f"refusing sensitive environment variable: {key}")
            env[key] = value
    return env


def classify_failure(returncode: int, stdout: str, stderr: str) -> str:
    if returncode == 0:
        return "OK"
    combined = f"{stdout}\n{stderr}".lower()
    if any(pattern in combined for pattern in USAGE_PATTERNS):
        return "USAGE_EXHAUSTED"
    if any(pattern in combined for pattern in AUTH_PATTERNS):
        return "AUTHENTICATION_REQUIRED"
    if returncode == 124 or "timed out" in combined:
        return "TIMEOUT"
    if returncode in {126, 127} or "not found" in combined or "access is denied" in combined:
        return "UNAVAILABLE"
    return "PROCESS_FAILURE"


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    stdin_text: str | None = None,
    timeout_seconds: float = 300.0,
    extra_env: Mapping[str, str] | None = None,
) -> CommandResult:
    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise ValueError("argv must contain non-empty strings")
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            shell=False,
            env=safe_environment(extra_env),
        )
        stdout = sanitize_text(completed.stdout)
        stderr = sanitize_text(completed.stderr)
        return CommandResult(
            argv=tuple(argv),
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            classification=classify_failure(completed.returncode, stdout, stderr),
        )
    except FileNotFoundError as exc:
        return CommandResult(tuple(argv), 127, "", sanitize_text(str(exc)), "UNAVAILABLE")
    except PermissionError as exc:
        return CommandResult(tuple(argv), 126, "", sanitize_text(str(exc)), "UNAVAILABLE")
    except subprocess.TimeoutExpired as exc:
        stdout = sanitize_text(_coerce_timeout_text(exc.stdout))
        stderr = sanitize_text(_coerce_timeout_text(exc.stderr) + "\nprocess timed out")
        return CommandResult(tuple(argv), 124, stdout, stderr, "TIMEOUT")


def _coerce_timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def discover_codex() -> Path | None:
    override = os.environ.get("CODEX_CLI_PATH")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    found = shutil.which("codex")
    if found:
        candidates.append(Path(found))
    if os.name == "nt":
        local = Path(os.environ.get("LOCALAPPDATA", "")) / "OpenAI" / "Codex" / "bin"
        if local.is_dir():
            candidates.extend(sorted(local.glob("*/codex.exe"), reverse=True))
    return _first_working(candidates, ("--version",))


def discover_cursor_agent() -> Path | None:
    override = os.environ.get("CURSOR_AGENT_CLI_PATH")
    candidates: list[Path] = [Path(override)] if override else []
    for name in ("agent", "cursor-agent"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    return _first_working(candidates, ("--version",))


def _first_working(candidates: Sequence[Path], probe_args: Sequence[str]) -> Path | None:
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen or not candidate.is_file():
            continue
        seen.add(key)
        result = run_command([str(candidate), *probe_args], cwd=Path.cwd(), timeout_seconds=15)
        if result.ok:
            return candidate.resolve()
    return None


def codex_configured_model() -> str | None:
    config = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
    try:
        data = tomllib.loads(config.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    model = data.get("model")
    return model if isinstance(model, str) and model else None


def probe_codex(repo: Path) -> ToolProbe:
    executable = discover_codex()
    configured = codex_configured_model()
    if executable is None:
        return ToolProbe("CODEX", None, None, None, (), configured, "UNAVAILABLE", "no executable")
    version_result = run_command([str(executable), "--version"], cwd=repo, timeout_seconds=15)
    auth_result = run_command([str(executable), "login", "status"], cwd=repo, timeout_seconds=20)
    auth_text = f"{auth_result.stdout}\n{auth_result.stderr}".lower()
    authenticated = auth_result.ok and "logged in" in auth_text
    status = "READY" if version_result.ok and authenticated else auth_result.classification
    detail = (
        auth_result.stdout.strip()
        or auth_result.stderr.strip()
        or "authentication status unavailable"
    )
    return ToolProbe(
        "CODEX",
        str(executable),
        version_result.stdout.strip() or None,
        authenticated,
        (),
        configured,
        status,
        detail,
    )


def probe_cursor(repo: Path) -> ToolProbe:
    executable = discover_cursor_agent()
    if executable is None:
        return ToolProbe(
            "CURSOR_AGENT",
            None,
            None,
            None,
            (),
            None,
            "UNAVAILABLE",
            "agent/cursor-agent not found; the Cursor editor CLI is not a substitute",
        )
    version_result = run_command([str(executable), "--version"], cwd=repo, timeout_seconds=15)
    auth_result = run_command([str(executable), "status"], cwd=repo, timeout_seconds=20)
    model_result = run_command([str(executable), "--list-models"], cwd=repo, timeout_seconds=30)
    models = tuple(line.strip() for line in model_result.stdout.splitlines() if line.strip())
    authenticated = auth_result.ok and "not authenticated" not in auth_result.stdout.lower()
    status = (
        "READY"
        if version_result.ok and authenticated and model_result.ok
        else (auth_result.classification if not authenticated else model_result.classification)
    )
    detail = auth_result.stdout.strip() or auth_result.stderr.strip() or "status unavailable"
    return ToolProbe(
        "CURSOR_AGENT",
        str(executable),
        version_result.stdout.strip() or None,
        authenticated,
        models,
        None,
        status,
        detail,
    )
