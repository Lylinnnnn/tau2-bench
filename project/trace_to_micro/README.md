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

The model-based pre-experiment is a separate pipeline with two reports over the
same one-rollout-per-task corpus:

1. `experiment_1_trajectory_analysis` analyzes already-generated complete model
   trajectories. It reports outcomes by horizon, observed tool errors/no-ops and
   recovery, train leave-one-task-out transition support, and train-to-test
   transition support.
2. `experiment_2_context_probe` replays selected decisions to the exact same
   environment state and runs paired single-step model inference under three
   context conditions:

- `long_raw`: the complete agent-visible prefix;
- `structured_state`: a label-free structured summary of that prefix;
- `clean_subtask`: the same summary plus an automatically mined local subgoal
  and observable success condition.

Both experiments report `train` and `test` separately. Leave-one-task-out is
needed only for Experiment 1's within-train support calculation; Experiment 2
does not retrieve cross-task support as model input. The context builder can see
only the logged prefix. It cannot see the target
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
                         `model_experiment.py` keeps model experiment
                         orchestration out of the CLI
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

### Qwen3-32B thinking run

The Qwen3-32B configuration is isolated from the earlier 30B-A3B run:

```text
configs/qwen3_32b_thinking_model_preexperiment.toml
```

It expects the API model ID `qwen3-32b`. The model path is a server concern and
is deliberately not written into result metadata. On the GPU server, start the
downloaded checkpoint in its own tmux window:

```bash
tmux new -s qwen3-32b
cd /path/to/tau2-bench
TENSOR_PARALLEL_SIZE=4 \
project/trace_to_micro/scripts/start_qwen3_32b_vllm.sh
```

The launcher defaults to:

```text
model path: /data/oss_bucket_0/yanlin/tau2/models/Qwen3-32B
served ID:  qwen3-32b
endpoint:   http://127.0.0.1:8000/v1
context:    32768 tokens
```

Override `TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`, or
`GPU_MEMORY_UTILIZATION` when required by the server. The launcher uses the
checkpoint's Hermes-style JSON tool-call template, vLLM automatic tool choice,
and the `deepseek_r1` reasoning parser. It assumes `vllm` is available in the
active server environment; set `VLLM_BIN` to an absolute executable path when
needed.

Wait until the server prints that it is listening, then open a second tmux
window and run the two-case Retail task 2 smoke test:

```bash
tmux new-window -t qwen3-32b -n smoke
cd /path/to/tau2-bench
project/trace_to_micro/scripts/run_qwen3_32b_retail_smoke.sh
```

The first case is the closest controlled comparison to the old command:
non-thinking Qwen3-32B with `temperature=0`. The second is the intended
Qwen3-32B thinking protocol with the model-card sampling settings
(`temperature=0.6`, `top_p=0.95`, `top_k=20`). They write to separate result
directories:

```text
data/simulations/qwen3_32b_retail_task2_nonthinking_t0_smoke/
data/simulations/qwen3_32b_retail_task2_thinking_t06_smoke/
```

Each directory receives an `integrity_report.json` after generation. The audit
requires the requested task/trial and exact model IDs, checks tool-call/response
linkage, extracts reward and termination reason, and reports cross-role or
unknown tool calls separately. Such invalid actions remain valid observed
model behavior; missing tool responses or a model mismatch fail the audit.
The wrapper also points τ²'s NL-assertion evaluator at `openai/qwen3-32b` in
non-thinking JSON mode. This matters because the repository default still
names the earlier 30B-A3B endpoint; otherwise the dialogue can finish normally
but reward computation requests a model ID that the new server does not expose.
The selected evaluator model and arguments are copied into the integrity audit.

After both smoke outputs are structurally valid, run the complete Telecom
pre-experiment:

```bash
tmux new-window -t qwen3-32b -n telecom
cd /path/to/tau2-bench
project/trace_to_micro/scripts/run_qwen3_32b_model_preexperiment.sh
```

This generates all 114 official Telecom base tasks once, reports the official
train and test splits separately, and then runs the same-state context probe.
Its artifacts are isolated at:

```text
data/simulations/trace_to_micro_qwen3_32b_thinking_t06_telecom_base/results.json
project/trace_to_micro/outputs/qwen3_32b_thinking_t06_model_preexperiment/
```

The context builder runs the same checkpoint in non-thinking JSON mode with a
1,024-token output cap. Invalid JSON gets exactly one shorter retry. If that
also fails, the script keeps the already valid JSONL rows, stores bounded
head/tail diagnostics plus `finish_reason`, token usage, and output length in
`context_failures.jsonl`, and exits non-zero instead of silently accepting or
mixing malformed data.

### Earlier Qwen3-30B-A3B run

The checked-in configuration uses the server's OpenAI-compatible local
Qwen3-30B endpoint through LiteLLM:

```text
configs/qwen3_30b_model_preexperiment.toml
```

It runs all 114 official Telecom `base` tasks once: 74 `train` tasks and 40
held-out `test` tasks. Both experiments report the splits separately and select
at most three early/middle/late decisions per task. `num_trials = 1` and
hallucination retry is disabled, so the source data remains one heterogeneous
behavior trajectory per task rather than repeated rollouts of the same query.

Experiment 1 is observational: it measures what is already present in the
logged trajectories and makes no LLM calls. For its train support result, the query task is excluded
from support; for test transfer, support comes only from train. Experiment 2 is
the controlled mechanism probe: every `long_raw`, `structured_state`, and
`clean_subtask` branch is rebuilt from the same prefix and must have the same
pre-state fingerprint or the run fails. Experiment 2 calls the configured
context-builder, frozen agent, and—when the agent responds with text—the real
user simulator for one continuation.

One resumable command runs checks and all five stages:

```bash
project/trace_to_micro/scripts/run_server_model_preexperiment.sh
```

The script now treats model-server readiness as a hard gate. For an
unauthenticated local vLLM server it supplies a non-empty placeholder key and
the usual local endpoint automatically:

```text
OPENAI_API_KEY=local-vllm
OPENAI_API_BASE=http://127.0.0.1:8000/v1
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

Existing environment variables take precedence. If vLLM was started with
`--api-key`, export that value as `OPENAI_API_KEY` before invoking the script.
The preflight waits up to ten minutes by default, reads the exact served model
IDs required by the TOML, and verifies both tool calling and JSON response
mode. No trajectory file is opened until all checks pass. The wait can be
configured without editing code:

```bash
TRACE_TO_MICRO_SERVER_WAIT_SECONDS=1200 \
TRACE_TO_MICRO_SERVER_POLL_SECONDS=10 \
project/trace_to_micro/scripts/run_server_model_preexperiment.sh
```

The equivalent stages are:

```bash
export PYTHONPATH="$PWD/project/trace_to_micro/src${PYTHONPATH:+:$PYTHONPATH}"
CONFIG=project/trace_to_micro/configs/qwen3_30b_model_preexperiment.toml

uv run --no-sync python -m trace_to_micro.cli generate-trajectories --config "$CONFIG"
uv run --no-sync python -m trace_to_micro.cli analyze-trajectories --config "$CONFIG"
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
├── experiment_1_trajectory_analysis/
│   ├── trajectory_completeness.json
│   ├── logged_transitions.jsonl
│   ├── cross_split_support_report.json
│   ├── cross_split_report.json
│   ├── train/
│   │   ├── decision_snapshots.jsonl
│   │   ├── snapshot_report.json
│   │   └── trajectory_report.json
│   └── test/
│       ├── decision_snapshots.jsonl
│       ├── snapshot_report.json
│       └── trajectory_report.json
└── experiment_2_context_probe/
    ├── cross_split_report.json
    ├── train/
    │   ├── structured_contexts.jsonl
    │   ├── paired_predictions.jsonl
    │   └── paired_probe_report.json
    └── test/
        ├── structured_contexts.jsonl
        ├── paired_predictions.jsonl
        └── paired_probe_report.json
```

The decision manifests store pointers into the source results rather than
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
