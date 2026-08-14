#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${SESSION:-expectation-step-rl-smoke}"
LOG_DIR="$PROJECT_DIR/outputs/run_logs"
LOG_PATH="$LOG_DIR/smoke.log"

mkdir -p "$LOG_DIR"
tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT_DIR' && bash scripts/run_grpo.sh smoke 2>&1 | tee '$LOG_PATH'"

echo "Started tmux session: $SESSION"
echo "Log: $LOG_PATH"
echo "Attach: tmux attach -t $SESSION"
