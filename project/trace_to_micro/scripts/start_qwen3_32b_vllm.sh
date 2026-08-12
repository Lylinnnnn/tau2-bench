#!/usr/bin/env bash
set -euo pipefail

model_path="${QWEN3_32B_MODEL_PATH:-/data/oss_bucket_0/yanlin/tau2/models/Qwen3-32B}"
served_model_name="${QWEN3_32B_SERVED_NAME:-qwen3-32b}"
http_port="${QWEN3_32B_HTTP_PORT:-8000}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-4}"
max_model_len="${MAX_MODEL_LEN:-32768}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"
vllm_bin="${VLLM_BIN:-/home/liuyanlin.lyl/.venvs/tau2-vllm/bin/vllm}"

if [[ ! -d "${model_path}" ]]; then
  echo "Model directory does not exist: ${model_path}" >&2
  exit 1
fi
if [[ ! -x "${vllm_bin}" ]]; then
  echo "vLLM executable does not exist or is not executable: ${vllm_bin}" >&2
  exit 1
fi

"${vllm_bin}" --version

exec "${vllm_bin}" serve "${model_path}" \
  --served-model-name "${served_model_name}" \
  --host 127.0.0.1 \
  --port "${http_port}" \
  --tensor-parallel-size "${tensor_parallel_size}" \
  --max-model-len "${max_model_len}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
