#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 <command> [args ...]" >&2
  exit 2
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_url="${OPENAI_API_BASE:-${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}}"
api_key="${OPENAI_API_KEY:-EMPTY}"
served_model_name="${QWEN3_32B_SERVED_NAME:-qwen3-32b}"
wait_seconds="${TRACE_TO_MICRO_SERVER_WAIT_SECONDS:-1800}"
poll_seconds="${TRACE_TO_MICRO_SERVER_POLL_SECONDS:-5}"
log_dir="${VLLM_LOG_DIR:-${project_dir}/outputs/vllm_logs}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="${VLLM_LOG_PATH:-${log_dir}/qwen3_32b_${timestamp}.log}"
server_pid=""

server_responds() {
  curl --fail --silent --show-error \
    --max-time 3 \
    --header "Authorization: Bearer ${api_key}" \
    "${base_url%/}/models" >/dev/null 2>&1
}

cleanup() {
  local status="$?"
  trap - EXIT
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    echo "Stopping managed Qwen3-32B vLLM process ${server_pid}"
    kill "${server_pid}"
    wait "${server_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if server_responds; then
  echo "Reusing the existing model server at ${base_url}"
else
  mkdir -p "${log_dir}"
  echo "Starting managed Qwen3-32B vLLM; log: ${log_path}"
  "${project_dir}/scripts/start_qwen3_32b_vllm.sh" >"${log_path}" 2>&1 &
  server_pid="$!"
  deadline=$((SECONDS + wait_seconds))
  while ! server_responds; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      wait "${server_pid}" || server_status="$?"
      echo "vLLM exited before becoming ready (status ${server_status:-0})." >&2
      tail -n 120 "${log_path}" >&2
      exit 1
    fi
    if ((SECONDS >= deadline)); then
      echo "vLLM did not become ready within ${wait_seconds}s." >&2
      tail -n 120 "${log_path}" >&2
      exit 1
    fi
    sleep "${poll_seconds}"
  done
  echo "Managed Qwen3-32B vLLM is ready at ${base_url}"
fi

OPENAI_API_KEY="${api_key}" \
  PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  uv run --no-sync python -m trace_to_micro.runtime.server_preflight \
    --base-url "${base_url}" --model-id "${served_model_name}" \
    --wait-seconds 3 --poll-seconds 1

"$@"
