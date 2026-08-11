#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "${project_dir}/../.." && pwd)"
config="${project_dir}/configs/qwen3_32b_thinking_model_preexperiment.toml"
model="openai/qwen3-32b"

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_API_BASE}}"
export TAU2_NL_ASSERTIONS_LLM="${TAU2_NL_ASSERTIONS_LLM:-${model}}"
if [[ -z "${TAU2_NL_ASSERTIONS_LLM_ARGS:-}" ]]; then
  export TAU2_NL_ASSERTIONS_LLM_ARGS='{"temperature":0,"max_tokens":1024,"response_format":{"type":"json_object"},"api_base":"http://127.0.0.1:8000/v1","api_key":"EMPTY","timeout":180,"num_retries":0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
fi

if [[ "${TRACE_TO_MICRO_MANAGED_VLLM:-0}" != "1" ]]; then
  export TRACE_TO_MICRO_MANAGED_VLLM=1
  exec "${project_dir}/scripts/with_managed_qwen3_32b_vllm.sh" "$0" "$@"
fi

cd "${repo_dir}"

uv run --no-sync python -m trace_to_micro.runtime.server_preflight \
  --config "${config}" \
  --base-url "${OPENAI_API_BASE}" \
  --wait-seconds "${TRACE_TO_MICRO_SERVER_WAIT_SECONDS:-600}" \
  --poll-seconds "${TRACE_TO_MICRO_SERVER_POLL_SECONDS:-5}"

run_case() {
  local save_to="$1"
  local llm_args="$2"

  uv run --frozen tau2 run \
    --domain retail \
    --task-ids 2 \
    --agent llm_agent \
    --agent-llm "${model}" \
    --agent-llm-args "${llm_args}" \
    --user user_simulator \
    --user-llm "${model}" \
    --user-llm-args "${llm_args}" \
    --num-trials 1 \
    --max-concurrency 1 \
    --max-retries 0 \
    --timeout 600 \
    --verbose-logs \
    --log-level DEBUG \
    --save-to "${save_to}"

  uv run --no-sync python -m trace_to_micro.cli audit-results \
    --results "data/simulations/${save_to}/results.json" \
    --domain retail \
    --expected-task-ids 2 \
    --expected-agent-model "${model}" \
    --expected-user-model "${model}"
}

nonthinking_args='{"temperature":0,"max_tokens":4096,"api_base":"http://127.0.0.1:8000/v1","api_key":"EMPTY","timeout":180,"num_retries":0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
thinking_args='{"temperature":0.6,"top_p":0.95,"max_tokens":4096,"api_base":"http://127.0.0.1:8000/v1","api_key":"EMPTY","timeout":180,"num_retries":0,"extra_body":{"top_k":20,"min_p":0.0,"chat_template_kwargs":{"enable_thinking":true}}}'

run_case "qwen3_32b_retail_task2_nonthinking_t0_smoke" "${nonthinking_args}"
run_case "qwen3_32b_retail_task2_thinking_t06_smoke" "${thinking_args}"
