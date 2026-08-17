#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$EVALUATION_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/oss_bucket_0/yanlin/tau2/expectation_step_rl/checkpoints/qwen3_32b_full_ce56a20}"
SOURCE_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-}"
TASK_SPLIT="${TASK_SPLIT:-test}"
DOMAINS="${DOMAINS:-airline,retail}"
NUM_TRIALS="${NUM_TRIALS:-1}"
SEED="${SEED:-300}"
MAX_TASKS_PER_DOMAIN="${MAX_TASKS_PER_DOMAIN:-0}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"
RUN_TAG="${RUN_TAG:-$(basename "$CHECKPOINT_ROOT")_${TASK_SPLIT}_t${NUM_TRIALS}_s${SEED}}"
EVALUATION_ROOT="${EVALUATION_ROOT:-$EVALUATION_DIR/outputs}"
BASELINE_ROOT="${BASELINE_ROOT:-/data/oss_bucket_0/yanlin/tau2/expectation_step_rl/evaluation/baselines}"
RUN_ROOT="$EVALUATION_ROOT/$RUN_TAG"
SESSION="${SESSION:-expectation-step-rl-infer-${RUN_TAG}}"
LOG_PATH="${LOG_PATH:-$RUN_ROOT/logs/inference_${TIMESTAMP}.log}"

MAX_EVAL_GPUS="${MAX_EVAL_GPUS:-8}"
FIRST_GPU="${FIRST_GPU:-0}"
BASE_PORT="${BASE_PORT:-8200}"
INTERNAL_BASE_PORT="${INTERNAL_BASE_PORT:-28000}"
INTERNAL_PORT_STRIDE="${INTERNAL_PORT_STRIDE:-1000}"
MASTER_BASE_PORT="${MASTER_BASE_PORT:-46000}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-1800}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-2}"
MODEL_PATH="${MODEL_PATH:-/data/oss_bucket_0/yanlin/tau2/models/Qwen3-32B}"
SOURCE_MODEL_PATH="$MODEL_PATH"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-qwen3-32b}"
VLLM_BIN="${VLLM_BIN:-/home/liuyanlin.lyl/.venvs/tau2-vllm/bin/vllm}"
UV_BIN="${UV_BIN:-uv}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
STAGE_EVAL_INPUTS="${STAGE_EVAL_INPUTS:-0}"
LOCAL_STAGE_PARENT="${LOCAL_STAGE_PARENT:-/tmp}"
ALLOW_NETWORK_STAGE="${ALLOW_NETWORK_STAGE:-0}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/vllm_logs"

if [[ "${EXPECTATION_EVAL_IN_TMUX:-0}" != "1" ]]; then
  tmux new-session -d -s "$SESSION" \
    "exec bash -c 'cd \"$REPO_ROOT\" && EXPECTATION_EVAL_IN_TMUX=1 CHECKPOINT_ROOT=\"$CHECKPOINT_ROOT\" CHECKPOINT_STEPS=\"$CHECKPOINT_STEPS\" TASK_SPLIT=\"$TASK_SPLIT\" DOMAINS=\"$DOMAINS\" NUM_TRIALS=\"$NUM_TRIALS\" SEED=\"$SEED\" MAX_TASKS_PER_DOMAIN=\"$MAX_TASKS_PER_DOMAIN\" INCLUDE_BASELINE=\"$INCLUDE_BASELINE\" RUN_TAG=\"$RUN_TAG\" EVALUATION_ROOT=\"$EVALUATION_ROOT\" BASELINE_ROOT=\"$BASELINE_ROOT\" MAX_EVAL_GPUS=\"$MAX_EVAL_GPUS\" FIRST_GPU=\"$FIRST_GPU\" BASE_PORT=\"$BASE_PORT\" INTERNAL_BASE_PORT=\"$INTERNAL_BASE_PORT\" INTERNAL_PORT_STRIDE=\"$INTERNAL_PORT_STRIDE\" MASTER_BASE_PORT=\"$MASTER_BASE_PORT\" SERVER_WAIT_SECONDS=\"$SERVER_WAIT_SECONDS\" MAX_CONCURRENCY=\"$MAX_CONCURRENCY\" MODEL_PATH=\"$MODEL_PATH\" BASE_MODEL_NAME=\"$BASE_MODEL_NAME\" VLLM_BIN=\"$VLLM_BIN\" UV_BIN=\"$UV_BIN\" GPU_MEMORY_UTILIZATION=\"$GPU_MEMORY_UTILIZATION\" MAX_MODEL_LEN=\"$MAX_MODEL_LEN\" STAGE_EVAL_INPUTS=\"$STAGE_EVAL_INPUTS\" LOCAL_STAGE_PARENT=\"$LOCAL_STAGE_PARENT\" ALLOW_NETWORK_STAGE=\"$ALLOW_NETWORK_STAGE\" RSYNC_BIN=\"$RSYNC_BIN\" bash \"$0\" > >(tee \"$LOG_PATH\") 2>&1; status=\$?; echo inference-exit-code=\$status | tee -a \"$LOG_PATH\"; exit \$status'"
  echo "Started tmux session: $SESSION"
  echo "Log: $LOG_PATH"
  echo "Evaluation output: $RUN_ROOT"
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export PYTHONUNBUFFERED=1

if [[ -z "$CHECKPOINT_STEPS" && -f "$RUN_ROOT/experiment_manifest.json" ]]; then
  CHECKPOINT_STEPS="$(
    "$UV_BIN" run --frozen python -m expectation_step_rl.evaluation.tracking_cli \
      manifest-steps --run-root "$RUN_ROOT"
  )"
  echo "Resuming the checkpoint snapshot fixed by the existing manifest: $CHECKPOINT_STEPS"
fi

list_args=(
  "$UV_BIN" run --frozen python -m expectation_step_rl.evaluation.inference_cli
  list-checkpoints --checkpoint-root "$SOURCE_CHECKPOINT_ROOT" --format tsv
)
if [[ -n "$CHECKPOINT_STEPS" ]]; then
  list_args+=(--steps "$CHECKPOINT_STEPS")
fi
mapfile -t checkpoint_rows < <("${list_args[@]}")
if [[ "${#checkpoint_rows[@]}" -eq 0 ]]; then
  echo "No complete checkpoints discovered under $SOURCE_CHECKPOINT_ROOT" >&2
  exit 1
fi
discovered_steps=()
for row in "${checkpoint_rows[@]}"; do
  IFS=$'\t' read -r step _ <<<"$row"
  discovered_steps+=("$step")
done
MANIFEST_CHECKPOINT_STEPS="$(IFS=,; echo "${discovered_steps[*]}")"

baseline_id="$(
  "$UV_BIN" run --frozen python -m expectation_step_rl.evaluation.tracking_cli \
    prepare-run \
    --repo-root "$REPO_ROOT" \
    --evaluation-root "$EVALUATION_ROOT" \
    --run-tag "$RUN_TAG" \
    --checkpoint-root "$SOURCE_CHECKPOINT_ROOT" \
    --checkpoint-steps "$MANIFEST_CHECKPOINT_STEPS" \
    --source-model-path "$SOURCE_MODEL_PATH" \
    --base-model-name "$BASE_MODEL_NAME" \
    --domains "$DOMAINS" \
    --split "$TASK_SPLIT" \
    --num-trials "$NUM_TRIALS" \
    --seed "$SEED" \
    --max-tasks-per-domain "$MAX_TASKS_PER_DOMAIN" \
    --include-baseline "$INCLUDE_BASELINE" \
    --local-staging-enabled "$STAGE_EVAL_INPUTS"
)"
echo "Evaluation manifest: $RUN_ROOT/experiment_manifest.json"
echo "Canonical baseline ID: $baseline_id"

reuse_baseline=0
if [[ "$INCLUDE_BASELINE" == "1" ]] && \
  "$UV_BIN" run --frozen python -m expectation_step_rl.evaluation.tracking_cli \
    baseline-ready --run-root "$RUN_ROOT" --baseline-root "$BASELINE_ROOT" \
    >/dev/null; then
  reuse_baseline=1
  echo "Reusing canonical baseline: $BASELINE_ROOT/$baseline_id"
fi

IFS=',' read -r -a domain_list <<<"$DOMAINS"
expected_model_keys=()
if [[ "$INCLUDE_BASELINE" == "1" ]]; then
  expected_model_keys+=("base")
fi
model_keys=()
agent_models=()
adapter_paths=()
source_adapter_paths=()
if [[ "$INCLUDE_BASELINE" == "1" && "$reuse_baseline" == "0" ]]; then
  model_keys+=("base")
  agent_models+=("$BASE_MODEL_NAME")
  adapter_paths+=("")
  source_adapter_paths+=("")
fi
for row in "${checkpoint_rows[@]}"; do
  IFS=$'\t' read -r step key served_name adapter_path <<<"$row"
  expected_model_keys+=("$key")
  model_keys+=("$key")
  agent_models+=("$served_name")
  adapter_paths+=("$adapter_path")
  source_adapter_paths+=("$adapter_path")
done
EXPECTED_MODEL_KEYS="$(IFS=,; echo "${expected_model_keys[*]}")"
MODEL_COUNT="${#model_keys[@]}"
if [[ "$MAX_EVAL_GPUS" -le 0 ]]; then
  echo "MAX_EVAL_GPUS must be positive" >&2
  exit 2
fi
ACTIVE_SERVER_COUNT="$MAX_EVAL_GPUS"
if [[ "$MODEL_COUNT" -lt "$ACTIVE_SERVER_COUNT" ]]; then
  ACTIVE_SERVER_COUNT="$MODEL_COUNT"
fi
echo "Evaluation model queues: $MODEL_COUNT models on $ACTIVE_SERVER_COUNT GPUs"

server_pids=()
stage_dir=""
manifest_ready=1
cleanup() {
  local status="$?"
  trap - EXIT INT TERM
  local pid
  for pid in "${server_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      wait "$pid" 2>/dev/null || true
    fi
  done
  if [[ -n "$stage_dir" ]]; then
    case "$(basename "$stage_dir")" in
      expectation_step_rl_eval.*)
        rm -rf -- "$stage_dir"
        echo "Removed temporary evaluation inputs: $stage_dir"
        ;;
      *)
        echo "Refusing to remove unexpected staging path: $stage_dir" >&2
        status=1
        ;;
    esac
  fi
  if [[ "$status" -ne 0 && "$manifest_ready" == "1" ]]; then
    "$UV_BIN" run --frozen python -m expectation_step_rl.evaluation.tracking_cli \
      set-status --run-root "$RUN_ROOT" --status inference_failed \
      --error "inference shell exit code $status" >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((server = 0; server < ACTIVE_SERVER_COUNT; server++)); do
  port=$((BASE_PORT + server))
  if curl --fail --silent --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    echo "Evaluation port $port is already occupied; stop it or change BASE_PORT" >&2
    exit 1
  fi
done

if [[ "$STAGE_EVAL_INPUTS" == "1" ]]; then
  mkdir -p "$LOCAL_STAGE_PARENT"
  filesystem_type="$(stat -f -c %T "$LOCAL_STAGE_PARENT")"
  if [[ "$ALLOW_NETWORK_STAGE" != "1" && "$filesystem_type" == *fuse* ]]; then
    echo "LOCAL_STAGE_PARENT is still a FUSE filesystem ($filesystem_type): $LOCAL_STAGE_PARENT" >&2
    echo "Choose a node-local path such as a verified /tmp, or set ALLOW_NETWORK_STAGE=1 explicitly." >&2
    exit 1
  fi
  required_bytes="$(du -sb "$SOURCE_MODEL_PATH" | awk '{print $1}')"
  for source_adapter_path in "${source_adapter_paths[@]}"; do
    if [[ -n "$source_adapter_path" ]]; then
      adapter_bytes="$(du -sb "$source_adapter_path" | awk '{print $1}')"
      required_bytes=$((required_bytes + adapter_bytes))
    fi
  done
  available_bytes="$(df -PB1 --output=avail "$LOCAL_STAGE_PARENT" | tail -n 1 | tr -d ' ')"
  safety_bytes=$((5 * 1024 * 1024 * 1024))
  if ((available_bytes < required_bytes + safety_bytes)); then
    echo "Insufficient staging space under $LOCAL_STAGE_PARENT: required inputs plus 5 GiB safety margin" >&2
    exit 1
  fi
  stage_dir="$(mktemp -d "$LOCAL_STAGE_PARENT/expectation_step_rl_eval.XXXXXX")"
  local_model_path="$stage_dir/model/Qwen3-32B"
  mkdir -p "$local_model_path"
  echo "Staging base model once: $SOURCE_MODEL_PATH -> $local_model_path"
  "$RSYNC_BIN" -a --info=progress2 "$SOURCE_MODEL_PATH/" "$local_model_path/"
  MODEL_PATH="$local_model_path"
  for ((model_index = 0; model_index < MODEL_COUNT; model_index++)); do
    if [[ -z "${adapter_paths[$model_index]}" ]]; then
      continue
    fi
    local_adapter_path="$stage_dir/adapters/${model_keys[$model_index]}"
    mkdir -p "$local_adapter_path"
    echo "Staging LoRA ${model_keys[$model_index]}: ${adapter_paths[$model_index]}"
    "$RSYNC_BIN" -a --info=progress2 \
      "${adapter_paths[$model_index]}/" "$local_adapter_path/"
    adapter_paths[$model_index]="$local_adapter_path"
  done
  echo "Temporary inputs ready under $stage_dir"
fi

"$UV_BIN" run --frozen python -m expectation_step_rl.evaluation.tracking_cli \
  set-status --run-root "$RUN_ROOT" --status inference_running >/dev/null

if [[ "$reuse_baseline" == "1" ]]; then
  for domain in "${domain_list[@]}"; do
    "$UV_BIN" run --frozen python -m expectation_step_rl.evaluation.tracking_cli \
      materialize-baseline \
      --run-root "$RUN_ROOT" \
      --baseline-root "$BASELINE_ROOT" \
      --domain "$domain"
  done
fi

for ((server = 0; server < ACTIVE_SERVER_COUNT; server++)); do
  gpu=$((FIRST_GPU + server))
  port=$((BASE_PORT + server))
  internal_port=$((INTERNAL_BASE_PORT + server * INTERNAL_PORT_STRIDE))
  master_port=$((MASTER_BASE_PORT + server))
  # Use md5sum to shorten the path and avoid ZMQ IPC path length limit (108 chars)
  short_id=$(echo -n "${RUN_TAG}_${server}_${BASHPID}" | md5sum | cut -c1-16)
  rpc_path="/tmp/esrl_eval_${short_id}"
  server_log="$RUN_ROOT/vllm_logs/gpu_${gpu}_port_${port}.log"
  server_loras=()
  for ((model_index = server; model_index < MODEL_COUNT; model_index += ACTIVE_SERVER_COUNT)); do
    if [[ -n "${adapter_paths[$model_index]}" ]]; then
      server_loras+=("${agent_models[$model_index]}=${adapter_paths[$model_index]}")
    fi
  done
  server_cmd=(
    "$VLLM_BIN" serve "$MODEL_PATH"
    --served-model-name "$BASE_MODEL_NAME"
    --host 127.0.0.1
    --port "$port"
    --tensor-parallel-size 1
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser hermes
  )
  if [[ "${#server_loras[@]}" -gt 0 ]]; then
    server_cmd+=(
      --enable-lora
      --max-lora-rank 32
      --max-loras 1
      --max-cpu-loras "${#server_loras[@]}"
      --lora-modules "${server_loras[@]}"
    )
  fi
  mkdir -p "$rpc_path"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export VLLM_HOST_IP="127.0.0.1"
    export VLLM_PORT="$internal_port"
    export VLLM_RPC_BASE_PATH="$rpc_path"
    export MASTER_ADDR="127.0.0.1"
    export MASTER_PORT="$master_port"
    exec "${server_cmd[@]}"
  ) >"$server_log" 2>&1 &
  server_pids+=("$!")
  echo "Starting evaluation server: GPU $gpu, HTTP $port, log $server_log"
done

deadline=$((SECONDS + SERVER_WAIT_SECONDS))
for ((server = 0; server < ACTIVE_SERVER_COUNT; server++)); do
  port=$((BASE_PORT + server))
  while true; do
    models_json="$(curl --fail --silent --max-time 5 "http://127.0.0.1:${port}/v1/models" || true)"
    ready=1
    expected_server_models=("$BASE_MODEL_NAME")
    for ((model_index = server; model_index < MODEL_COUNT; model_index += ACTIVE_SERVER_COUNT)); do
      expected_server_models+=("${agent_models[$model_index]}")
    done
    for model in "${expected_server_models[@]}"; do
      if [[ "$models_json" != *"\"id\":\"$model\""* && "$models_json" != *"\"id\": \"$model\""* ]]; then
        ready=0
        break
      fi
    done
    if [[ "$ready" == "1" ]]; then
      break
    fi
    for pid in "${server_pids[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "An evaluation vLLM process exited during startup; inspect $RUN_ROOT/vllm_logs" >&2
        exit 1
      fi
    done
    if ((SECONDS >= deadline)); then
      echo "Evaluation vLLM pool did not become ready within ${SERVER_WAIT_SECONDS}s" >&2
      exit 1
    fi
    sleep 5
  done
  echo "Evaluation server ready: http://127.0.0.1:${port}/v1"
done

queue_pids=()
for ((server = 0; server < ACTIVE_SERVER_COUNT; server++)); do
  (
    for ((model_index = server; model_index < MODEL_COUNT; model_index += ACTIVE_SERVER_COUNT)); do
      model_key="${model_keys[$model_index]}"
      agent_model="${agent_models[$model_index]}"
      adapter_path="${adapter_paths[$model_index]}"
      source_adapter_path="${source_adapter_paths[$model_index]}"
      port=$((BASE_PORT + server))
      base_url="http://127.0.0.1:${port}/v1"
      for domain in "${domain_list[@]}"; do
        worker_log="$RUN_ROOT/logs/${model_key}_${domain}.log"
        {
          export TAU2_NL_ASSERTIONS_LLM="openai/$BASE_MODEL_NAME"
          export TAU2_NL_ASSERTIONS_LLM_ARGS="{\"temperature\":0,\"max_tokens\":1024,\"api_base\":\"$base_url\",\"api_key\":\"EMPTY\",\"timeout\":180,\"num_retries\":0,\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
          "$UV_BIN" run --frozen python -m expectation_step_rl.evaluation.inference_cli \
            run-shard \
            --repo-root "$REPO_ROOT" \
            --evaluation-root "$EVALUATION_ROOT" \
            --run-tag "$RUN_TAG" \
            --model-key "$model_key" \
            --agent-model "$agent_model" \
            --base-model "$BASE_MODEL_NAME" \
            --domain "$domain" \
            --split "$TASK_SPLIT" \
            --num-trials "$NUM_TRIALS" \
            --seed "$SEED" \
            --num-shards 1 \
            --max-tasks "$MAX_TASKS_PER_DOMAIN" \
            --shard-index 0 \
            --api-base "$base_url" \
            --max-concurrency "$MAX_CONCURRENCY"
          merge_args=(
            "$UV_BIN" run --frozen python -m expectation_step_rl.evaluation.inference_cli
            merge
            --repo-root "$REPO_ROOT"
            --evaluation-root "$EVALUATION_ROOT"
            --run-tag "$RUN_TAG"
            --model-key "$model_key"
            --agent-model "$agent_model"
            --base-model "$BASE_MODEL_NAME"
            --domain "$domain"
            --split "$TASK_SPLIT"
            --num-trials "$NUM_TRIALS"
            --seed "$SEED"
            --num-shards 1
            --max-tasks "$MAX_TASKS_PER_DOMAIN"
            --expected-model-keys "$EXPECTED_MODEL_KEYS"
            --expected-domains "$DOMAINS"
          )
          if [[ -n "$source_adapter_path" ]]; then
            merge_args+=(--adapter-path "$source_adapter_path")
          fi
          "${merge_args[@]}"
        } >"$worker_log" 2>&1
      done
    done
  ) &
  queue_pids+=("$!")
done

failed=0
for pid in "${queue_pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != "0" ]]; then
  echo "At least one model evaluation failed; inspect $RUN_ROOT/logs" >&2
  exit 1
fi

if [[ "$INCLUDE_BASELINE" == "1" && "$reuse_baseline" == "0" ]]; then
  for domain in "${domain_list[@]}"; do
    "$UV_BIN" run --frozen python -m expectation_step_rl.evaluation.tracking_cli \
      publish-baseline \
      --run-root "$RUN_ROOT" \
      --baseline-root "$BASELINE_ROOT" \
      --domain "$domain"
  done
fi

"$UV_BIN" run --frozen python -m expectation_step_rl.evaluation.tracking_cli \
  set-status --run-root "$RUN_ROOT" --status inference_complete >/dev/null

echo "Inference completed: $RUN_ROOT"
echo "Experiment registry: $EVALUATION_ROOT/experiment_registry.json"
echo "Next: RUN_TAG=$RUN_TAG bash $EVALUATION_DIR/scripts/compute_official_metrics.sh"
