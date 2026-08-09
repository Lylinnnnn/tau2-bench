#!/usr/bin/env bash
set -euo pipefail

model_path="${QWEN3_32B_MODEL_PATH:-/data/oss_bucket_0/yanlin/tau2/models/Qwen3-32B}"
served_model_name="${QWEN3_32B_SERVED_NAME:-qwen3-32b}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-4}"
max_model_len="${MAX_MODEL_LEN:-32768}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"
vllm_bin="${VLLM_BIN:-vllm}"

if [[ ! -d "${model_path}" ]]; then
  echo "Model directory does not exist: ${model_path}" >&2
  exit 1
fi

exec "${vllm_bin}" serve "${model_path}" \
  --served-model-name "${served_model_name}" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size "${tensor_parallel_size}" \
  --max-model-len "${max_model_len}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
