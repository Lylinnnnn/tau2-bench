#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${EXPECTATION_SCORER_MODEL_PATH:-/data/oss_bucket_0/yanlin/tau2/models/Qwen3-32B}"
SERVED_NAME="${EXPECTATION_SCORER_MODEL:-qwen3-32b}"
HTTP_PORT="${SCORER_HTTP_PORT:?SCORER_HTTP_PORT is required}"
MAX_MODEL_LEN="${SCORER_MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${SCORER_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_BIN="${SCORER_VLLM_BIN:-/home/liuyanlin.lyl/.venvs/tau2-vllm/bin/vllm}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory does not exist: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vLLM executable does not exist: $VLLM_BIN" >&2
  exit 1
fi

exec "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 \
  --port "$HTTP_PORT" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
