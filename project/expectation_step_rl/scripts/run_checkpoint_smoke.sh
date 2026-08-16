#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_VENV="${TRAINING_VENV:-/home/liuyanlin.lyl/.venvs/expectation-step-rl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}"
MODEL_PATH="${MODEL_PATH:-/data/oss_bucket_0/yanlin/tau2/models/Qwen3-32B}"
ADAPTER_INFERENCE_GPU="${ADAPTER_INFERENCE_GPU:-5}"

"$SCRIPT_DIR/prepare_dataset.sh"
"$SCRIPT_DIR/prepare_training_calibration.sh"
"$SCRIPT_DIR/run_grpo.sh" checkpoint_smoke
"$TRAINING_VENV/bin/python" -m expectation_step_rl.verl_adapter.checkpoint \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --expected-step 1
CUDA_VISIBLE_DEVICES="$ADAPTER_INFERENCE_GPU" \
  "$TRAINING_VENV/bin/python" -m expectation_step_rl.verl_adapter.inference_smoke \
  --model-path "$MODEL_PATH" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --expected-step 1 \
  --output "$CHECKPOINT_DIR/global_step_1/inference_smoke.json"

echo "OSS inference-adapter checkpoint smoke passed: $CHECKPOINT_DIR"
