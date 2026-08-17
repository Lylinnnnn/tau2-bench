#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$EVALUATION_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

RUN_TAG="${RUN_TAG:?Set RUN_TAG to the completed inference run}"
MODEL_KEYS="${MODEL_KEYS:-base,step_10,step_30}"
DOMAINS="${DOMAINS:-airline,retail}"
TASK_SPLIT="${TASK_SPLIT:-test}"
NUM_TRIALS="${NUM_TRIALS:-1}"
SEED="${SEED:-300}"
EVALUATION_ROOT="${EVALUATION_ROOT:-$EVALUATION_DIR/outputs}"
RUN_ROOT="$EVALUATION_ROOT/$RUN_TAG"
OUTPUT="${OUTPUT:-$RUN_ROOT/provisional_available_case_metrics.json}"
LOG_PATH="${LOG_PATH:-$RUN_ROOT/logs/provisional_available_case_metrics.log}"
SESSION="${SESSION:-expectation-step-rl-available-case-${RUN_TAG}}"
UV_BIN="${UV_BIN:-uv}"

mkdir -p "$RUN_ROOT/logs"
if [[ "${EXPECTATION_AVAILABLE_CASE_IN_TMUX:-0}" != "1" ]]; then
  tmux new-session -d -s "$SESSION" \
    "exec bash -c 'cd \"$REPO_ROOT\" && EXPECTATION_AVAILABLE_CASE_IN_TMUX=1 RUN_TAG=\"$RUN_TAG\" MODEL_KEYS=\"$MODEL_KEYS\" DOMAINS=\"$DOMAINS\" TASK_SPLIT=\"$TASK_SPLIT\" NUM_TRIALS=\"$NUM_TRIALS\" SEED=\"$SEED\" EVALUATION_ROOT=\"$EVALUATION_ROOT\" OUTPUT=\"$OUTPUT\" LOG_PATH=\"$LOG_PATH\" UV_BIN=\"$UV_BIN\" bash \"$0\" > >(tee \"$LOG_PATH\") 2>&1; status=\$?; echo available-case-metrics-exit-code=\$status | tee -a \"$LOG_PATH\"; exit \$status'"
  echo "Started tmux session: $SESSION"
  echo "Log: $LOG_PATH"
  echo "Output: $OUTPUT"
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi

export PYTHONPATH="$PROJECT_DIR/src:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$UV_BIN" run --frozen python -m expectation_step_rl.evaluation.available_case_metrics \
  --repo-root "$REPO_ROOT" \
  --evaluation-root "$EVALUATION_ROOT" \
  --run-tag "$RUN_TAG" \
  --model-keys "$MODEL_KEYS" \
  --domains "$DOMAINS" \
  --split "$TASK_SPLIT" \
  --num-trials "$NUM_TRIALS" \
  --seed "$SEED" \
  --output "$OUTPUT"
