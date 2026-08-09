#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG="${project_dir}/configs/qwen3_32b_thinking_model_preexperiment.toml"
export TAU2_NL_ASSERTIONS_LLM="${TAU2_NL_ASSERTIONS_LLM:-openai/qwen3-32b}"
if [[ -z "${TAU2_NL_ASSERTIONS_LLM_ARGS:-}" ]]; then
  export TAU2_NL_ASSERTIONS_LLM_ARGS='{"temperature":0,"max_tokens":1024,"response_format":{"type":"json_object"},"api_base":"http://127.0.0.1:8000/v1","api_key":"EMPTY","timeout":180,"num_retries":0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
fi

if [[ "${TRACE_TO_MICRO_MANAGED_VLLM:-0}" != "1" ]]; then
  export TRACE_TO_MICRO_MANAGED_VLLM=1
  exec "${project_dir}/scripts/with_managed_qwen3_32b_vllm.sh" "$0" "$@"
fi

exec "${project_dir}/scripts/run_server_model_preexperiment.sh" "$@"
