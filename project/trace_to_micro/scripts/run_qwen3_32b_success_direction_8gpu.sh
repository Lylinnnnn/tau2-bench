#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <trajectories|trajectory-shards|merge-trajectories|activations|evaluate|all>" >&2
  exit 2
fi

stage="$1"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "${project_dir}/../.." && pwd)"
config="${project_dir}/configs/qwen3_32b_success_direction.toml"
num_shards="${NUM_SHARDS:-8}"
first_gpu="${FIRST_GPU:-0}"
base_port="${BASE_PORT:-8100}"
internal_base_port="${INTERNAL_BASE_PORT:-20000}"
internal_port_stride="${INTERNAL_PORT_STRIDE:-1000}"
master_base_port="${MASTER_BASE_PORT:-40000}"
force_overwrite="${FORCE_OVERWRITE:-1}"
shard_ids_raw="${SHARD_IDS:-}"
run_log_dir="${project_dir}/outputs/run_logs/success_direction_8gpu"
vllm_log_dir="${project_dir}/outputs/vllm_logs/success_direction_8gpu"

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-vllm}"
export PYTHONUNBUFFERED=1
export TAU2_NL_ASSERTIONS_LLM="${TAU2_NL_ASSERTIONS_LLM:-openai/qwen3-32b}"

mkdir -p "${run_log_dir}" "${vllm_log_dir}"
cd "${repo_dir}"

selected_shards=()

initialize_selected_shards() {
  local candidates=()
  if [[ -z "${shard_ids_raw}" ]]; then
    local shard
    for ((shard = 0; shard < num_shards; shard++)); do
      candidates+=("${shard}")
    done
  else
    read -r -a candidates <<<"${shard_ids_raw//,/ }"
  fi
  local -A seen=()
  local shard
  for shard in "${candidates[@]}"; do
    if [[ ! "${shard}" =~ ^[0-9]+$ ]] || ((shard >= num_shards)); then
      echo "Invalid shard ${shard@Q}; expected an integer in [0, ${num_shards})." >&2
      return 1
    fi
    if [[ -n "${seen[${shard}]:-}" ]]; then
      echo "Duplicate shard ${shard} in SHARD_IDS." >&2
      return 1
    fi
    seen["${shard}"]=1
    selected_shards+=("${shard}")
  done
}

require_explicit_shards() {
  if [[ -z "${shard_ids_raw}" ]]; then
    echo "trajectory-shards requires SHARD_IDS, for example SHARD_IDS='0 3 5'." >&2
    return 1
  fi
}

validate_port_layout() {
  if ((num_shards <= 0 || internal_port_stride < 2)); then
    echo "NUM_SHARDS must be positive and INTERNAL_PORT_STRIDE must be at least 2." >&2
    return 1
  fi
  local last_http_port=$((base_port + num_shards - 1))
  local last_internal_port=$((internal_base_port + num_shards * internal_port_stride - 1))
  local last_master_port=$((master_base_port + num_shards - 1))
  if ((base_port <= 0 || last_http_port > 65535)); then
    echo "HTTP port range is outside 1..65535: ${base_port}..${last_http_port}" >&2
    return 1
  fi
  if ((internal_base_port <= 0 || last_internal_port > 65535)); then
    echo "Internal port range is outside 1..65535: ${internal_base_port}..${last_internal_port}" >&2
    return 1
  fi
  if ((master_base_port <= 0 || last_master_port > 65535)); then
    echo "Master port range is outside 1..65535: ${master_base_port}..${last_master_port}" >&2
    return 1
  fi
}

run_checks() {
  uv run --no-sync pytest -c project/trace_to_micro/pyproject.toml \
    project/trace_to_micro/tests
  uv run --no-sync ruff check --config project/trace_to_micro/pyproject.toml \
    project/trace_to_micro/src project/trace_to_micro/tests \
    project/trace_to_micro/scripts
  uv run --no-sync ruff format --check \
    --config project/trace_to_micro/pyproject.toml \
    project/trace_to_micro/src project/trace_to_micro/tests \
    project/trace_to_micro/scripts
}

run_workers() {
  local mode="$1"
  local manager
  local command
  if [[ "${mode}" == "trajectories" ]]; then
    manager="${project_dir}/scripts/with_managed_qwen3_32b_vllm.sh"
    command="success-trajectory-shard"
  else
    manager="${project_dir}/scripts/with_managed_qwen3_32b_hidden_vllm.sh"
    command="success-activation-shard"
  fi

  local -A pids=()
  local shard
  for shard in "${selected_shards[@]}"; do
    local gpu=$((first_gpu + shard))
    local port=$((base_port + shard))
    local internal_port=$((internal_base_port + shard * internal_port_stride))
    local master_port=$((master_base_port + shard))
    local base_url="http://127.0.0.1:${port}/v1"
    local worker_log="${run_log_dir}/${mode}_shard_${shard}.log"
    local server_log="${vllm_log_dir}/${mode}_gpu_${gpu}_port_${port}.log"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export QWEN3_32B_HTTP_PORT="${port}"
      export VLLM_HOST_IP="127.0.0.1"
      export VLLM_PORT="${internal_port}"
      export VLLM_RPC_BASE_PATH="/tmp/trace_to_micro_vllm_rpc_${mode}_${shard}_${BASHPID}"
      mkdir -p "${VLLM_RPC_BASE_PATH}"
      export MASTER_ADDR="127.0.0.1"
      export MASTER_PORT="${master_port}"
      export TENSOR_PARALLEL_SIZE=1
      export OPENAI_API_BASE="${base_url}"
      export OPENAI_BASE_URL="${base_url}"
      export VLLM_LOG_PATH="${server_log}"
      export TRACE_TO_MICRO_HIDDEN_TMP_DIR="/dev/shm/trace_to_micro_hidden_states_gpu_${gpu}"
      export TAU2_NL_ASSERTIONS_LLM_ARGS="{\"temperature\":0,\"max_tokens\":1024,\"api_base\":\"${base_url}\",\"api_key\":\"EMPTY\",\"timeout\":180,\"num_retries\":0,\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
      local force_flag="--force"
      if [[ "${force_overwrite}" == "0" ]]; then
        force_flag="--no-force"
      fi
      local args=(
        uv run --no-sync python -m trace_to_micro.cli "${command}"
        --config "${config}"
        --shard-index "${shard}"
        --num-shards "${num_shards}"
        --base-url "${base_url}"
      )
      if [[ "${mode}" == "trajectories" ]]; then
        args+=("${force_flag}")
      fi
      "${manager}" "${args[@]}"
    ) >"${worker_log}" 2>&1 &
    pids["${shard}"]="$!"
    echo "Started ${mode} shard ${shard}: GPU ${gpu}, HTTP ${port}, internal ${internal_port}, master ${master_port}, log ${worker_log}"
  done

  local failed=0
  for shard in "${selected_shards[@]}"; do
    if ! wait "${pids[${shard}]}"; then
      echo "${mode} shard ${shard} failed; inspect ${run_log_dir}/${mode}_shard_${shard}.log" >&2
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    return 1
  fi
}

ensure_target_ports_are_free() {
  local occupied=0
  local shard
  for shard in "${selected_shards[@]}"; do
    local port=$((base_port + shard))
    if curl --fail --silent --show-error --max-time 2 \
      "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "Port ${port} is already serving a model; stop it or choose another BASE_PORT." >&2
      occupied=1
    fi
  done
  if [[ "${occupied}" -ne 0 ]]; then
    return 1
  fi
}

run_trajectories() {
  run_workers trajectories
  uv run --no-sync python -m trace_to_micro.cli success-merge-trajectories \
    --config "${config}" --num-shards "${num_shards}"
}

run_activations() {
  uv run --no-sync python -m trace_to_micro.cli success-activation-requests \
    --config "${config}"
  run_workers activations
  uv run --no-sync python -m trace_to_micro.cli success-merge-activations \
    --config "${config}" --num-shards "${num_shards}"
}

run_evaluate() {
  uv run --no-sync python -m trace_to_micro.cli success-evaluate \
    --config "${config}"
}

validate_port_layout
initialize_selected_shards
run_checks
case "${stage}" in
  trajectories)
    ensure_target_ports_are_free
    run_trajectories
    ;;
  trajectory-shards)
    require_explicit_shards
    ensure_target_ports_are_free
    run_workers trajectories
    ;;
  merge-trajectories)
    uv run --no-sync python -m trace_to_micro.cli success-merge-trajectories \
      --config "${config}" --num-shards "${num_shards}"
    ;;
  activations)
    ensure_target_ports_are_free
    run_activations
    ;;
  evaluate)
    run_evaluate
    ;;
  all)
    ensure_target_ports_are_free
    run_trajectories
    run_activations
    run_evaluate
    ;;
  *)
    echo "Unknown stage: ${stage}" >&2
    exit 2
    ;;
esac
