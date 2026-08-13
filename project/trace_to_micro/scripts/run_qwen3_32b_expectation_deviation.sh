#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <prepare|smoke|score|merge|evaluate|full>" >&2
  exit 2
fi

stage="$1"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "${project_dir}/../.." && pwd)"
config="${project_dir}/configs/qwen3_32b_expectation_deviation.toml"
num_shards="${NUM_SHARDS:-8}"
first_gpu="${FIRST_GPU:-0}"
smoke_gpu="${SMOKE_GPU:-0}"
base_port="${BASE_PORT:-8300}"
shard_zero_port="${SHARD_ZERO_PORT:-8000}"
internal_base_port="${INTERNAL_BASE_PORT:-24000}"
internal_port_stride="${INTERNAL_PORT_STRIDE:-1000}"
master_base_port="${MASTER_BASE_PORT:-44000}"
run_log_dir="${project_dir}/outputs/run_logs/expectation_deviation"
vllm_log_dir="${project_dir}/outputs/vllm_logs/expectation_deviation"

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-vllm}"
export PYTHONUNBUFFERED=1

mkdir -p "${run_log_dir}" "${vllm_log_dir}"
cd "${repo_dir}"

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

validate_port_layout() {
  local last_http_port=$((base_port + num_shards - 1))
  local last_internal_port=$((internal_base_port + num_shards * internal_port_stride - 1))
  local last_master_port=$((master_base_port + num_shards - 1))
  if ((num_shards <= 0 || internal_port_stride < 2)); then
    echo "NUM_SHARDS must be positive and INTERNAL_PORT_STRIDE at least 2." >&2
    return 1
  fi
  if ((base_port <= 0 || last_http_port > 65535)); then
    echo "Invalid HTTP port range ${base_port}..${last_http_port}." >&2
    return 1
  fi
  if ((shard_zero_port <= 0 || shard_zero_port > 65535)); then
    echo "Invalid SHARD_ZERO_PORT ${shard_zero_port}." >&2
    return 1
  fi
  if ((num_shards > 1 && shard_zero_port >= base_port + 1 && shard_zero_port <= last_http_port)); then
    echo "SHARD_ZERO_PORT overlaps another shard HTTP port." >&2
    return 1
  fi
  if ((internal_base_port <= 0 || last_internal_port > 65535)); then
    echo "Invalid internal port range." >&2
    return 1
  fi
  if ((master_base_port <= 0 || last_master_port > 65535)); then
    echo "Invalid master port range." >&2
    return 1
  fi
}

prepare() {
  uv run --no-sync python -m trace_to_micro.cli \
    expectation-deviation-prepare --config "${config}"
}

run_score_workers() {
  local -A pids=()
  local shard
  for ((shard = 0; shard < num_shards; shard++)); do
    local gpu=$((first_gpu + shard))
    local port=$((base_port + shard))
    if ((shard == 0)); then
      port="${shard_zero_port}"
    fi
    local internal_port=$((internal_base_port + shard * internal_port_stride))
    local master_port=$((master_base_port + shard))
    local base_url="http://127.0.0.1:${port}/v1"
    local worker_log="${run_log_dir}/score_shard_${shard}.log"
    local server_log="${vllm_log_dir}/gpu_${gpu}_port_${port}.log"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export QWEN3_32B_HTTP_PORT="${port}"
      export VLLM_HOST_IP="127.0.0.1"
      export VLLM_PORT="${internal_port}"
      export VLLM_RPC_BASE_PATH="/tmp/trace_to_micro_deviation_${shard}_${BASHPID}"
      mkdir -p "${VLLM_RPC_BASE_PATH}"
      export MASTER_ADDR="127.0.0.1"
      export MASTER_PORT="${master_port}"
      export TENSOR_PARALLEL_SIZE=1
      export OPENAI_API_BASE="${base_url}"
      export OPENAI_BASE_URL="${base_url}"
      export VLLM_LOG_PATH="${server_log}"
      "${project_dir}/scripts/with_managed_qwen3_32b_vllm.sh" \
        uv run --no-sync python -m trace_to_micro.cli \
          expectation-deviation-score-shard --config "${config}" \
          --shard-index "${shard}" --num-shards "${num_shards}" \
          --base-url "${base_url}"
    ) >"${worker_log}" 2>&1 &
    pids["${shard}"]="$!"
    echo "Started shard ${shard}: GPU ${gpu}, HTTP ${port}, log ${worker_log}"
  done
  local failed=0
  for ((shard = 0; shard < num_shards; shard++)); do
    if ! wait "${pids[${shard}]}"; then
      echo "Shard ${shard} failed; inspect ${run_log_dir}/score_shard_${shard}.log" >&2
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    return 1
  fi
}

merge() {
  uv run --no-sync python -m trace_to_micro.cli \
    expectation-deviation-merge --config "${config}" \
    --num-shards "${num_shards}"
}

evaluate() {
  uv run --no-sync python -m trace_to_micro.cli \
    expectation-deviation-evaluate --config "${config}"
}

case "${stage}" in
  prepare)
    run_checks
    prepare
    ;;
  smoke)
    run_checks
    prepare
    export CUDA_VISIBLE_DEVICES="${smoke_gpu}"
    export TENSOR_PARALLEL_SIZE=1
    exec "${project_dir}/scripts/with_managed_qwen3_32b_vllm.sh" \
      uv run --no-sync python -m trace_to_micro.cli \
        expectation-deviation-smoke --config "${config}" \
        --base-url "${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
    ;;
  score)
    validate_port_layout
    run_checks
    prepare
    run_score_workers
    ;;
  merge)
    merge
    ;;
  evaluate)
    evaluate
    ;;
  full)
    validate_port_layout
    run_checks
    prepare
    run_score_workers
    merge
    evaluate
    ;;
  *)
    echo "Unknown stage: ${stage}" >&2
    exit 2
    ;;
esac
