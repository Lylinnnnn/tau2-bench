#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "${project_dir}/../.." && pwd)"
export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${repo_dir}"

uv run --no-sync pytest -c "${project_dir}/pyproject.toml" \
  "${project_dir}/tests"
uv run --no-sync ruff check --config "${project_dir}/pyproject.toml" \
  "${project_dir}/src" "${project_dir}/tests" "${project_dir}/scripts"
uv run --no-sync ruff format --check --config "${project_dir}/pyproject.toml" \
  "${project_dir}/src" "${project_dir}/tests" "${project_dir}/scripts"
uv run --no-sync python -m trace_to_micro.cli oracle-preflight \
  --config "${project_dir}/configs/oracle_loco.toml"
