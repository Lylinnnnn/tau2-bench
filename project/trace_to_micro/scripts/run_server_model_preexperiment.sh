#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "${project_dir}/../.." && pwd)"
config="${project_dir}/configs/qwen3_30b_model_preexperiment.toml"

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-vllm}"
openai_base_url="${OPENAI_API_BASE:-${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}}"
export OPENAI_API_BASE="${openai_base_url}"
export OPENAI_BASE_URL="${openai_base_url}"

server_wait_seconds="${TRACE_TO_MICRO_SERVER_WAIT_SECONDS:-600}"
server_poll_seconds="${TRACE_TO_MICRO_SERVER_POLL_SECONDS:-5}"

cd "${repo_dir}"

uv run --no-sync pytest -c "${project_dir}/pyproject.toml" \
  "${project_dir}/tests"
uv run --no-sync ruff check --config "${project_dir}/pyproject.toml" \
  "${project_dir}/src" "${project_dir}/tests" "${project_dir}/scripts"
uv run --no-sync ruff format --check \
  --config "${project_dir}/pyproject.toml" \
  "${project_dir}/src" "${project_dir}/tests" "${project_dir}/scripts"

echo "Waiting for OpenAI-compatible model server at ${OPENAI_API_BASE}"
uv run --no-sync python -m trace_to_micro.server_preflight \
  --config "${config}" \
  --base-url "${OPENAI_API_BASE}" \
  --wait-seconds "${server_wait_seconds}" \
  --poll-seconds "${server_poll_seconds}"

uv run --no-sync python -m trace_to_micro.cli generate-trajectories \
  --config "${config}"
uv run --no-sync python -m trace_to_micro.cli analyze-trajectories \
  --config "${config}"
uv run --no-sync python -m trace_to_micro.cli paired-probe \
  --config "${config}"
uv run --no-sync python -m trace_to_micro.cli summarize-probe \
  --config "${config}"
