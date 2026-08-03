#!/usr/bin/env bash
# Bootstrap the private Generals competition development environment (Linux/WSL/Git Bash).
set -euo pipefail

SCAFFOLD_ONLY=0
if [[ "${1:-}" == "--scaffold-only" ]]; then
  SCAFFOLD_ONLY=1
fi

assert_repo_root() {
  local markers=("pyproject.toml" "src/generals_bot" "third_party/generals-bots")
  for marker in "${markers[@]}"; do
    if [[ ! -e "$marker" ]]; then
      echo "Run bootstrap.sh from the repository root. Missing: $marker" >&2
      exit 1
    fi
  done
}

resolve_python() {
  if command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "Python 3.12 not found on PATH." >&2
  exit 1
}

assert_repo_root
PYTHON="$(resolve_python)"
VERSION="$("$PYTHON" --version 2>&1)"
echo "Interpreter: $PYTHON"
echo "Reported:    $VERSION"
if [[ "$VERSION" != "Python 3.12.10" ]]; then
  echo "Expected Python 3.12.10, got: $VERSION" >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating .venv with $PYTHON"
  "$PYTHON" -m venv .venv
fi

VENV_PY=".venv/bin/python"
"$VENV_PY" -m pip install --upgrade pip setuptools wheel

if [[ "$SCAFFOLD_ONLY" -eq 1 ]]; then
  echo "=== Scaffold-only mode: skipping official competition requirements and engine ==="
  echo "environment_parity: false"
  echo "status: bootstrap-only"
  "$VENV_PY" -m pip install -e ".[dev]"
  "$VENV_PY" -m pip check
else
  REQ="third_party/generals-bots/competition/requirements.txt"
  if [[ ! -f "$REQ" ]]; then
    echo "Official requirements missing: $REQ" >&2
    exit 1
  fi
  echo "Installing official competition requirements (authoritative)..."
  if ! "$VENV_PY" -m pip install -r "$REQ"; then
    cat >&2 <<'EOF'
Official competition dependency installation failed.
Record this failure, then re-run with --scaffold-only if you only need
repository scaffolding / lint / package tests.
Do not run the official matchup until dependencies install successfully.
EOF
    exit 1
  fi
  echo "Installing official engine editable with --no-deps..."
  "$VENV_PY" -m pip install --no-deps -e third_party/generals-bots
  echo "Installing private package with development extras..."
  "$VENV_PY" -m pip install -e ".[dev]"
  "$VENV_PY" -m pip check
fi

echo
echo "=== Environment versions ==="
"$VENV_PY" --version
"$VENV_PY" -c "import sys; print('executable:', sys.executable)"
"$VENV_PY" -m pip --version
"$VENV_PY" -c "import generals_bot; print('generals_bot', generals_bot.__version__)" || true

echo
echo "Bootstrap completed."
