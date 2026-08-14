#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
TRAINING_VENV="${TRAINING_VENV:-/home/liuyanlin.lyl/.venvs/expectation-step-rl}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

git -C "$REPO_ROOT" submodule update --init --recursive \
  project/expectation_step_rl/third_party/verl

if [[ ! -x "$TRAINING_VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$TRAINING_VENV"
fi

"$TRAINING_VENV/bin/python" -m pip install --upgrade pip wheel
"$TRAINING_VENV/bin/python" -m pip install "vllm==0.11.0"
"$TRAINING_VENV/bin/python" -m pip install -e "$PROJECT_DIR/third_party/verl"
"$TRAINING_VENV/bin/python" -m pip install -e "$REPO_ROOT"
"$TRAINING_VENV/bin/python" -m pip install -e "$PROJECT_DIR"

echo "Training environment ready: $TRAINING_VENV"
