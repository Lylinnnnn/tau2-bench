#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${SESSION:-expectation-step-rl-full}"
LOG_DIR="$PROJECT_DIR/outputs/run_logs"
LOG_PATH="$LOG_DIR/full.log"
SCORER_GPU_IDS="${SCORER_GPU_IDS:-0,1}"
TRAINING_CUDA_VISIBLE_DEVICES="${TRAINING_CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/oss_bucket_0/yanlin/tau2/expectation_step_rl/checkpoints/qwen3_32b_full}"

mkdir -p "$LOG_DIR"
tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT_DIR' && CHECKPOINT_DIR='$CHECKPOINT_DIR' SCORER_GPU_IDS='$SCORER_GPU_IDS' TRAINING_CUDA_VISIBLE_DEVICES='$TRAINING_CUDA_VISIBLE_DEVICES' bash scripts/run_full_8gpu.sh 2>&1 | tee '$LOG_PATH'"

echo "Started tmux session: $SESSION"
echo "Log: $LOG_PATH"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Frozen scorer GPUs: $SCORER_GPU_IDS"
echo "Training/rollout GPUs: $TRAINING_CUDA_VISIBLE_DEVICES"
echo "Attach: tmux attach -t $SESSION"
