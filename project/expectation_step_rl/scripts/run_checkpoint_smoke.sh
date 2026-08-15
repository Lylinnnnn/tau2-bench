#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_VENV="${TRAINING_VENV:-/home/liuyanlin.lyl/.venvs/expectation-step-rl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}"

"$SCRIPT_DIR/prepare_dataset.sh"
"$SCRIPT_DIR/prepare_training_calibration.sh"
"$SCRIPT_DIR/run_grpo.sh" checkpoint_smoke
"$TRAINING_VENV/bin/python" -m expectation_step_rl.verl_adapter.checkpoint \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --expected-step 1

echo "OSS checkpoint smoke passed: $CHECKPOINT_DIR"
