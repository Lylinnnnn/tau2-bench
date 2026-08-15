#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
TRAINING_VENV="${TRAINING_VENV:-/home/liuyanlin.lyl/.venvs/expectation-step-rl}"
DATA_DIR="${EXPECTATION_DATA_DIR:-$PROJECT_DIR/data/decisions_qwen3_32b_t06}"
RECORDS_PATH="$REPO_ROOT/project/trace_to_micro/outputs/qwen3_32b_thinking_t06_expectation_deviation/expectation_deviation_records.jsonl"
SCORES_PATH="$DATA_DIR/training_calibration_scores.jsonl"
CALIBRATION_PATH="$DATA_DIR/training_calibration.json"
BASE_URLS="${EXPECTATION_SCORER_BASE_URLS:?EXPECTATION_SCORER_BASE_URLS is required}"
CALIBRATION_ARGS=(
  --records "$RECORDS_PATH"
  --output "$SCORES_PATH"
  --base-urls "$BASE_URLS"
  --api-key "${EXPECTATION_SCORER_API_KEY:-EMPTY}"
  --model "${EXPECTATION_SCORER_MODEL:-qwen3-32b}"
  --workers-per-server 2
)
if [[ "${FORCE_CALIBRATION:-0}" == "1" ]]; then
  CALIBRATION_ARGS+=(--overwrite)
fi

"$TRAINING_VENV/bin/python" -m expectation_step_rl.expectation.calibration_data \
  "${CALIBRATION_ARGS[@]}"

"$TRAINING_VENV/bin/python" -m expectation_step_rl.expectation.calibration \
  --scores "$SCORES_PATH" \
  --output "$CALIBRATION_PATH" \
  --minimum-tool-count 5
