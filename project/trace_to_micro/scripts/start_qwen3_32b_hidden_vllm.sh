#!/usr/bin/env bash
set -euo pipefail

model_path="${QWEN3_32B_MODEL_PATH:-/data/oss_bucket_0/yanlin/tau2/models/Qwen3-32B}"
served_model_name="${QWEN3_32B_SERVED_NAME:-qwen3-32b}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-4}"
max_model_len="${MAX_MODEL_LEN:-32768}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"
vllm_bin="${VLLM_BIN:-/home/liuyanlin.lyl/.venvs/tau2-vllm/bin/vllm}"
if [[ -d /dev/shm && -w /dev/shm ]]; then
  default_hidden_dir="/dev/shm/trace_to_micro_hidden_states"
else
  default_hidden_dir="/tmp/trace_to_micro_hidden_states"
fi
hidden_dir="${TRACE_TO_MICRO_HIDDEN_TMP_DIR:-${default_hidden_dir}}"

if [[ ! -d "${model_path}" ]]; then
  echo "Model directory does not exist: ${model_path}" >&2
  exit 1
fi
if [[ ! -x "${vllm_bin}" ]]; then
  echo "vLLM executable does not exist or is not executable: ${vllm_bin}" >&2
  exit 1
fi

mkdir -p "${hidden_dir}"
export VLLM_USE_V2_MODEL_RUNNER=0
"${vllm_bin}" --version

exec "${vllm_bin}" serve "${model_path}" \
  --served-model-name "${served_model_name}" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size "${tensor_parallel_size}" \
  --max-model-len "${max_model_len}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --enforce-eager \
  --no-enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --speculative-config '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[31,47]}}}' \
  --kv-transfer-config "{\"kv_connector\":\"ExampleHiddenStatesConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"${hidden_dir}\"}}"
