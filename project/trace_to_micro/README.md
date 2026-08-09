# Trace-to-Micro

`trace_to_micro` contains training-free pre-experiments for testing whether local,
state-grounded transitions can be recovered across disjoint τ² Telecom task
compositions, and whether clean micro-contexts improve a frozen model's local
decisions inside long trajectories.

The oracle structural preflight replays
the reference trajectory of each task, extracts exact assistant/user database
changes, and measures how well train transitions cover unseen test
compositions. Reference actions are evaluation-only upper bounds; they are not
treated as off-policy evidence.

The model-based pre-experiment is a separate pipeline. It generates one real
agent/user rollout per concrete task, refuses to analyze incomplete results,
extracts early/middle/late decision snapshots, and runs paired model inference
from the exact same replayed state under three context conditions:

- `long_raw`: the complete agent-visible prefix;
- `structured_state`: a label-free structured summary of that prefix;
- `clean_subtask`: the same summary plus an automatically mined local subgoal
  and observable success condition.

The context builder can see only the logged prefix. It cannot see the target
assistant message, future turns, the hidden user scenario, or reference actions.
For predicted assistant text, the real user simulator is run for one additional
turn. Assistant/user tool calls are executed against cloned factual states, so
the report measures actual state deltas, tool errors, and mutating no-ops rather
than counting text-only pseudo-results.

The focused experiment plan is in
[`research/trace_to_micro_preexperiment_plan_zh.md`](../../research/trace_to_micro_preexperiment_plan_zh.md).

## Layout

```text
configs/                 Oracle and model experiment configurations
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

## Run the oracle preflight

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

Tests deliberately let replay and data errors surface. Unit tests have no live
API calls. The model pre-experiment does call the configured agent, user, and
context-builder models, but does not train any model.

On the configured server, the complete no-install check can be run with:

```bash
project/trace_to_micro/scripts/run_server_preflight.sh
```

It uses `uv run --no-sync`, so it reuses the existing server environment and
does not resolve or install packages.

## Run the model-based pre-experiment

The checked-in configuration uses the server's OpenAI-compatible local
Qwen3-30B endpoint through LiteLLM:

```text
configs/qwen3_30b_model_preexperiment.toml
```

It runs all 114 Telecom `base` tasks once. The paired probe is evaluation-only
on the 40-task compositional-OOD `test` split and selects at most three
early/middle/late decisions per task. `num_trials = 1` and hallucination retry is
disabled, so the source data remains one heterogeneous behavior trajectory per
task rather than repeated rollouts of the same query.

One resumable command runs checks and all five stages:

```bash
project/trace_to_micro/scripts/run_server_model_preexperiment.sh
```

The equivalent stages are:

```bash
export PYTHONPATH="$PWD/project/trace_to_micro/src${PYTHONPATH:+:$PYTHONPATH}"
CONFIG=project/trace_to_micro/configs/qwen3_30b_model_preexperiment.toml

uv run --no-sync python -m trace_to_micro.cli generate-trajectories --config "$CONFIG"
uv run --no-sync python -m trace_to_micro.cli extract-snapshots --config "$CONFIG"
uv run --no-sync python -m trace_to_micro.cli logged-audit --config "$CONFIG"
uv run --no-sync python -m trace_to_micro.cli paired-probe --config "$CONFIG"
uv run --no-sync python -m trace_to_micro.cli summarize-probe --config "$CONFIG"
```

Both the τ² trajectory runner and the probe JSONL writer are resumable. The
generation stage saves to:

```text
data/simulations/trace_to_micro_qwen3_30b_telecom_base/results.json
```

The analysis stage writes compact artifacts under the gitignored directory:

```text
outputs/qwen3_30b_model_preexperiment/
├── trajectory_completeness.json
├── decision_snapshots.jsonl
├── snapshot_report.json
├── logged_transitions.jsonl
├── logged_support_report.json
├── structured_contexts.jsonl
├── paired_predictions.jsonl
└── paired_probe_report.json
```

`decision_snapshots.jsonl` stores pointers into the source results rather than
duplicating every long prefix. Verbose LLM logs are disabled to limit disk use.

Before the full run, an optional two-task endpoint smoke test is:

```bash
uv run --no-sync python -m trace_to_micro.cli generate-trajectories \
  --config "$CONFIG" \
  --num-tasks 2 \
  --save-to trace_to_micro_qwen3_30b_endpoint_smoke
```

The smoke output is intentionally separate and is never accepted by the
completeness-gated full analysis.
