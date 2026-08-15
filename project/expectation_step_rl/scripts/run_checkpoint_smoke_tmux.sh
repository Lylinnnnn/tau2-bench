#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="${SESSION:-expectation-step-rl-ckpt-smoke}"
LOG_DIR="$PROJECT_DIR/outputs/run_logs"
LOG_PATH="${LOG_PATH:-$LOG_DIR/checkpoint_smoke_${TIMESTAMP}.log}"
TRAINING_CUDA_VISIBLE_DEVICES="${TRAINING_CUDA_VISIBLE_DEVICES:-1,2,3,4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/oss_bucket_0/yanlin/tau2/expectation_step_rl/checkpoints/checkpoint_smoke_${TIMESTAMP}}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-checkpoint_smoke_${TIMESTAMP}}"

mkdir -p "$LOG_DIR"
tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT_DIR' && CHECKPOINT_DIR='$CHECKPOINT_DIR' SCORER_GPU_IDS=0 TRAINER_EXPERIMENT_NAME='$TRAINER_EXPERIMENT_NAME' TRAINING_CUDA_VISIBLE_DEVICES='$TRAINING_CUDA_VISIBLE_DEVICES' bash scripts/with_managed_scorer_pool.sh bash scripts/run_checkpoint_smoke.sh 2>&1 | tee '$LOG_PATH'"

echo "Started tmux session: $SESSION"
echo "Log: $LOG_PATH"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Attach: tmux attach -t $SESSION"
