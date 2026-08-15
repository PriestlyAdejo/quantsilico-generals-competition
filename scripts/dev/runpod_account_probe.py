"""Read-only RunPod connectivity probe (auth, balance, pods, GPUs, volumes).

Queries mirror the official runpodctl (commit 6fc6f8b5) API surface:
GraphQL api.runpod.io/graphql for user/gpuTypes, REST rest.runpod.io/v1 for
pods/networkvolumes. Loads the key from ~/.runpod/config.toml (never from the
repo) and prints operational facts only; the key itself is never printed.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CONFIG = Path.home() / ".runpod" / "config.toml"
GQL_API = "https://api.runpod.io/graphql"
REST_API = "https://rest.runpod.io/v1"
USER_AGENT = "runpod-cli/1.0.0 (quantisilico-marathon)"


def load_key() -> str:
    text = CONFIG.read_text(encoding="utf-8")
    match = re.search(r'api_key\s*=\s*"([^"]+)"', text)
    if not match:
        raise SystemExit("api_key not found in ~/.runpod/config.toml")
    return match.group(1)


def _http(url: str, key: str, payload: dict | None = None, method: str | None = None) -> dict:
    last_error: Exception | None = None
    for attempt in range(4):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
            method=method or ("POST" if data else "GET"),
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:400]
            if exc.code in (400, 401, 403):
                raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
            last_error = RuntimeError(f"HTTP {exc.code}: {body}")
            print(f"HTTP {exc.code} attempt {attempt + 1}: {body}", file=sys.stderr)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_error = exc
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"unreachable after retries: {last_error}")


def gql(key: str, query: str) -> dict:
    result = _http(GQL_API, key, {"query": query, "variables": None})
    if result.get("errors"):
        raise RuntimeError(f"GraphQL errors: {result['errors']}")
    return result["data"]


def rest(key: str, endpoint: str, method: str = "GET", payload: dict | None = None) -> object:
    return _http(f"{REST_API}{endpoint}", key, payload=payload, method=method)


def main() -> int:
    key = load_key()
    out: dict = {"auth": "UNKNOWN", "endpoints": {"graphql": GQL_API, "rest": REST_API}}

    me = gql(
        key,
        """query { myself {
            id email clientBalance currentSpendPerHr spendLimit notifyLowBalance
        } }""",
    )
    out["auth"] = "OK"
    out["account"] = me["myself"]

    pods = rest(key, "/pods")
    out["pods"] = pods

    vols = rest(key, "/networkvolumes")
    out["network_volumes"] = vols

    gpus = gql(
        key,
        """query { gpuTypes {
            id displayName memoryInGb secureCloud communityCloud
            securePrice communityPrice
        } }""",
    )
    out["gpu_types"] = sorted(gpus["gpuTypes"], key=lambda t: t["displayName"])

    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
