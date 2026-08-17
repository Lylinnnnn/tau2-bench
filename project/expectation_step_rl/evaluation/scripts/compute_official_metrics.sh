#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$EVALUATION_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/oss_bucket_0/yanlin/tau2/expectation_step_rl/checkpoints/qwen3_32b_full_ce56a20}"
TASK_SPLIT="${TASK_SPLIT:-test}"
NUM_TRIALS="${NUM_TRIALS:-1}"
SEED="${SEED:-300}"
RUN_TAG="${RUN_TAG:-$(basename "$CHECKPOINT_ROOT")_${TASK_SPLIT}_t${NUM_TRIALS}_s${SEED}}"
EVALUATION_ROOT="${EVALUATION_ROOT:-$EVALUATION_DIR/outputs}"
RUN_ROOT="$EVALUATION_ROOT/$RUN_TAG"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="${SESSION:-expectation-step-rl-metrics-${RUN_TAG}}"
LOG_PATH="${LOG_PATH:-$RUN_ROOT/logs/official_metrics_${TIMESTAMP}.log}"
UV_BIN="${UV_BIN:-uv}"

mkdir -p "$RUN_ROOT/logs"
if [[ "${EXPECTATION_METRICS_IN_TMUX:-0}" != "1" ]]; then
  tmux new-session -d -s "$SESSION" \
    "exec bash -c 'cd \"$REPO_ROOT\" && EXPECTATION_METRICS_IN_TMUX=1 RUN_TAG=\"$RUN_TAG\" EVALUATION_ROOT=\"$EVALUATION_ROOT\" UV_BIN=\"$UV_BIN\" bash \"$0\" > >(tee \"$LOG_PATH\") 2>&1; status=\$?; echo official-metrics-exit-code=\$status | tee -a \"$LOG_PATH\"; exit \$status'"
  echo "Started tmux session: $SESSION"
  echo "Log: $LOG_PATH"
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
"$UV_BIN" run --frozen python -m expectation_step_rl.evaluation.official_metrics \
  --evaluation-run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/official_metrics.json"
echo "Official metrics: $RUN_ROOT/official_metrics.json"
