#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f pyproject.toml || ! -d project/trace_to_micro ]]; then
  echo "Run this script from the tau2-bench repository root." >&2
  exit 2
fi

export PYTHONPATH="$PWD/project/trace_to_micro/src${PYTHONPATH:+:$PYTHONPATH}"

uv run --no-sync pytest -c project/trace_to_micro/pyproject.toml \
  project/trace_to_micro/tests
uv run --no-sync ruff check --config project/trace_to_micro/pyproject.toml \
  project/trace_to_micro/src project/trace_to_micro/tests \
  project/trace_to_micro/scripts
uv run --no-sync ruff format --check \
  --config project/trace_to_micro/pyproject.toml \
  project/trace_to_micro/src project/trace_to_micro/tests \
  project/trace_to_micro/scripts

uv run --no-sync python -m trace_to_micro.cli model-preexperiment \
  --config project/trace_to_micro/configs/qwen3_30b_model_preexperiment.toml
