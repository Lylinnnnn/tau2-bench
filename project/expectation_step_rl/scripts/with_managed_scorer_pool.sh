#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 <command> [args ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NUM_SCORERS=4
HTTP_BASE_PORT="${SCORER_HTTP_BASE_PORT:-8000}"
INTERNAL_BASE_PORT="${SCORER_INTERNAL_BASE_PORT:-24000}"
INTERNAL_PORT_STRIDE="${SCORER_INTERNAL_PORT_STRIDE:-1000}"
MASTER_BASE_PORT="${SCORER_MASTER_BASE_PORT:-44000}"
WAIT_SECONDS="${SCORER_WAIT_SECONDS:-1800}"
MODEL="${EXPECTATION_SCORER_MODEL:-qwen3-32b}"
API_KEY="${EXPECTATION_SCORER_API_KEY:-EMPTY}"
LOG_DIR="$PROJECT_DIR/outputs/run_logs/scorer_pool"
PIDS=()
URLS=()

mkdir -p "$LOG_DIR"

server_responds() {
  local base_url="$1"
  curl --fail --silent --show-error --max-time 3 \
    --header "Authorization: Bearer $API_KEY" \
    "${base_url}/models" | grep -Eq "\"id\"[[:space:]]*:[[:space:]]*\"$MODEL\""
}

cleanup() {
  local status="$?"
  trap - EXIT
  local pid
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      wait "$pid" 2>/dev/null || true
    fi
  done
  exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((index = 0; index < NUM_SCORERS; index++)); do
  gpu="$index"
  http_port=$((HTTP_BASE_PORT + index))
  internal_port=$((INTERNAL_BASE_PORT + index * INTERNAL_PORT_STRIDE))
  master_port=$((MASTER_BASE_PORT + index))
  base_url="http://127.0.0.1:${http_port}/v1"
  URLS+=("$base_url")
  if server_responds "$base_url"; then
    echo "Reusing frozen scorer on GPU $gpu at $base_url"
    continue
  fi
  log_path="$LOG_DIR/gpu_${gpu}_port_${http_port}.log"
  rpc_path="/tmp/expectation_step_rl_scorer_${index}_${BASHPID}"
  mkdir -p "$rpc_path"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export SCORER_HTTP_PORT="$http_port"
    export VLLM_HOST_IP="127.0.0.1"
    export VLLM_PORT="$internal_port"
    export VLLM_RPC_BASE_PATH="$rpc_path"
    export MASTER_ADDR="127.0.0.1"
    export MASTER_PORT="$master_port"
    exec "$SCRIPT_DIR/start_frozen_scorer.sh"
  ) >"$log_path" 2>&1 &
  PIDS+=("$!")
  echo "Starting scorer: GPU $gpu, HTTP $http_port, internal $internal_port, master $master_port"
done

deadline=$((SECONDS + WAIT_SECONDS))
for base_url in "${URLS[@]}"; do
  while ! server_responds "$base_url"; do
    for pid in "${PIDS[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "A scorer process exited before the pool became ready; inspect $LOG_DIR" >&2
        exit 1
      fi
    done
    if ((SECONDS >= deadline)); then
      echo "Scorer pool did not become ready within ${WAIT_SECONDS}s" >&2
      exit 1
    fi
    sleep 5
  done
  echo "Scorer ready: $base_url"
done

EXPECTATION_SCORER_BASE_URLS="$(IFS=,; echo "${URLS[*]}")" "$@"
