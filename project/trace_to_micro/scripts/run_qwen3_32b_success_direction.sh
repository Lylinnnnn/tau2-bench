#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <trajectories|activations|evaluate>" >&2
  exit 2
fi

stage="$1"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "${project_dir}/../.." && pwd)"
config="${project_dir}/configs/qwen3_32b_success_direction.toml"

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-vllm}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_API_BASE}}"
export PYTHONUNBUFFERED=1
export TAU2_NL_ASSERTIONS_LLM="${TAU2_NL_ASSERTIONS_LLM:-openai/qwen3-32b}"
if [[ -z "${TAU2_NL_ASSERTIONS_LLM_ARGS:-}" ]]; then
  export TAU2_NL_ASSERTIONS_LLM_ARGS='{"temperature":0,"max_tokens":1024,"api_base":"http://127.0.0.1:8000/v1","api_key":"EMPTY","timeout":180,"num_retries":0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
fi

cd "${repo_dir}"

uv run --no-sync pytest -c project/trace_to_micro/pyproject.toml \
  project/trace_to_micro/tests
uv run --no-sync ruff check --config project/trace_to_micro/pyproject.toml \
  project/trace_to_micro/src project/trace_to_micro/tests \
  project/trace_to_micro/scripts
uv run --no-sync ruff format --check \
  --config project/trace_to_micro/pyproject.toml \
  project/trace_to_micro/src project/trace_to_micro/tests \
  project/trace_to_micro/scripts

case "${stage}" in
  trajectories)
    exec "${project_dir}/scripts/with_managed_qwen3_32b_vllm.sh" \
      uv run --no-sync python -m trace_to_micro.cli success-trajectories \
        --config "${config}" --force
    ;;
  activations)
    uv run --no-sync python -m trace_to_micro.cli success-activation-requests \
      --config "${config}"
    exec "${project_dir}/scripts/with_managed_qwen3_32b_hidden_vllm.sh" \
      uv run --no-sync python -m trace_to_micro.cli success-activations \
        --config "${config}"
    ;;
  evaluate)
    exec uv run --no-sync python -m trace_to_micro.cli success-evaluate \
      --config "${config}"
    ;;
  *)
    echo "Unknown stage: ${stage}" >&2
    exit 2
    ;;
esac
