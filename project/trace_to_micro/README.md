# Trace-to-Micro

`trace_to_micro` contains training-free pre-experiments for testing whether local,
state-grounded transitions can be recovered across disjoint τ² Telecom task
compositions.

The first implemented experiment is an oracle structural preflight. It replays
the reference trajectory of each task, extracts exact assistant/user database
changes, and measures how well train transitions cover unseen test
compositions. Reference actions are evaluation-only upper bounds; they are not
treated as off-policy evidence.

The focused experiment plan is in
[`research/trace_to_micro_preexperiment_plan_zh.md`](../../research/trace_to_micro_preexperiment_plan_zh.md).

## Layout

```text
configs/                 Reproducible experiment configurations
scripts/                 Thin executable wrappers
src/trace_to_micro/      Tested extraction/orchestration code
src/trace_to_micro/evaluation/
                         Evaluation metrics kept separate from extraction
tests/                   Unit and integration tests
outputs/                 Generated reports (gitignored)
```

## Environment

The preflight adds no runtime dependency beyond the existing `tau2`
installation. Reuse the repository's server environment instead of creating a
second virtual environment inside this subproject.

From the repository root:

```bash
export PYTHONPATH="$PWD/project/trace_to_micro/src${PYTHONPATH:+:$PYTHONPATH}"
```

The nested `pyproject.toml` records package metadata and test/lint settings, but
running `uv sync` inside this directory is not required.

## Run the preflight

```bash
uv run python -m trace_to_micro.cli task-audit \
  --config project/trace_to_micro/configs/oracle_loco.toml

uv run python -m trace_to_micro.cli oracle-preflight \
  --config project/trace_to_micro/configs/oracle_loco.toml
```

Equivalent thin wrappers are available under `scripts/`.

The full preflight writes:

```text
project/trace_to_micro/outputs/oracle_loco/task_inventory.json
project/trace_to_micro/outputs/oracle_loco/oracle_transitions.jsonl
project/trace_to_micro/outputs/oracle_loco/oracle_support_report.json
```

## Tests

```bash
uv run pytest -c project/trace_to_micro/pyproject.toml \
  project/trace_to_micro/tests
uv run ruff check --config project/trace_to_micro/pyproject.toml \
  project/trace_to_micro/src project/trace_to_micro/tests \
  project/trace_to_micro/scripts
uv run ruff format --check --config project/trace_to_micro/pyproject.toml \
  project/trace_to_micro/src project/trace_to_micro/tests \
  project/trace_to_micro/scripts
```

Tests deliberately let replay and data errors surface. There are no live API
calls and no model training in this project stage.

On the configured server, the complete no-install check can be run with:

```bash
project/trace_to_micro/scripts/run_server_preflight.sh
```

It uses `uv run --no-sync`, so it reuses the existing server environment and
does not resolve or install packages.
