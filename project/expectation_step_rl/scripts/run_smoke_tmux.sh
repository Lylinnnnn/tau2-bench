#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${SESSION:-expectation-step-rl-smoke}"
LOG_DIR="$PROJECT_DIR/outputs/run_logs"
LOG_PATH="$LOG_DIR/smoke.log"
TRAINING_CUDA_VISIBLE_DEVICES="${TRAINING_CUDA_VISIBLE_DEVICES:-1,2,3,4}"

mkdir -p "$LOG_DIR"
tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT_DIR' && TRAINING_CUDA_VISIBLE_DEVICES='$TRAINING_CUDA_VISIBLE_DEVICES' bash scripts/run_grpo.sh smoke 2>&1 | tee '$LOG_PATH'"

echo "Started tmux session: $SESSION"
echo "Training GPUs: $TRAINING_CUDA_VISIBLE_DEVICES"
echo "Log: $LOG_PATH"
echo "Attach: tmux attach -t $SESSION"
