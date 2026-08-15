#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAINING_CUDA_VISIBLE_DEVICES="${TRAINING_CUDA_VISIBLE_DEVICES:-4,5,6,7}"

exec "$SCRIPT_DIR/with_managed_scorer_pool.sh" \
  bash -c "'$SCRIPT_DIR/prepare_dataset.sh' && '$SCRIPT_DIR/prepare_training_calibration.sh' && TRAINING_CUDA_VISIBLE_DEVICES='$TRAINING_CUDA_VISIBLE_DEVICES' '$SCRIPT_DIR/run_grpo.sh' full"
