#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
TRAINING_VENV="${TRAINING_VENV:-/home/liuyanlin.lyl/.venvs/expectation-step-rl}"
OUTPUT_DIR="${EXPECTATION_DATA_DIR:-$PROJECT_DIR/data/decisions_qwen3_32b_t06}"
SCORES_PATH="$REPO_ROOT/project/trace_to_micro/outputs/qwen3_32b_thinking_t06_expectation_deviation/contextual_min_k_scores.jsonl"

"$TRAINING_VENV/bin/python" -m expectation_step_rl.data.compiler \
  --input "airline=$REPO_ROOT/data/simulations/trace_to_micro_qwen3_32b_thinking_t06_success_direction_airline_base/results.json" \
  --input "retail=$REPO_ROOT/data/simulations/trace_to_micro_qwen3_32b_thinking_t06_success_direction_retail_base/results.json" \
  --output-dir "$OUTPUT_DIR" \
  --overwrite

"$TRAINING_VENV/bin/python" -m expectation_step_rl.expectation.calibration \
  --scores "$SCORES_PATH" \
  --output "$OUTPUT_DIR/training_calibration.json" \
  --minimum-tool-count 5
