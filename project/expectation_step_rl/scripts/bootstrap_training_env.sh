#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
TRAINING_VENV="${TRAINING_VENV:-/home/liuyanlin.lyl/.venvs/expectation-step-rl}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
FRAMEWORK_LOCK="$PROJECT_DIR/FRAMEWORK.lock"

lock_value() {
  sed -n "s/^$1=//p" "$FRAMEWORK_LOCK"
}

VLLM_VERSION="$(lock_value training_vllm)"
TORCH_VERSION="$(lock_value training_torch)"
FLASH_ATTN_VERSION="$(lock_value training_flash_attn)"
FLASHINFER_VERSION="$(lock_value training_flashinfer)"
TRANSFORMERS_VERSION="$(lock_value training_transformers)"
HUGGINGFACE_HUB_VERSION="$(lock_value training_huggingface_hub)"
TOKENIZERS_VERSION="$(lock_value training_tokenizers)"
SWANLAB_VERSION="$(lock_value training_swanlab)"
FLASH_ATTN_WHEEL="$(lock_value flash_attn_wheel)"

git -C "$REPO_ROOT" submodule update --init --recursive \
  project/expectation_step_rl/third_party/verl

if [[ ! -x "$TRAINING_VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$TRAINING_VENV"
fi

"$TRAINING_VENV/bin/python" -m pip install --upgrade pip wheel
"$TRAINING_VENV/bin/python" -m pip install "vllm==$VLLM_VERSION"
"$TRAINING_VENV/bin/python" -c \
  "import torch; assert torch.__version__.split('+', 1)[0] == '$TORCH_VERSION', torch.__version__"
"$TRAINING_VENV/bin/python" -m pip install --no-cache-dir "$FLASH_ATTN_WHEEL"
"$TRAINING_VENV/bin/python" -m pip install --no-cache-dir \
  "flashinfer-python==$FLASHINFER_VERSION"
"$TRAINING_VENV/bin/python" -m pip install -e "$PROJECT_DIR/third_party/verl"
"$TRAINING_VENV/bin/python" -m pip install -e "$REPO_ROOT"
"$TRAINING_VENV/bin/python" -m pip install -e "$PROJECT_DIR"
"$TRAINING_VENV/bin/python" -m pip install "swanlab==$SWANLAB_VERSION"
"$TRAINING_VENV/bin/python" -m pip install --no-deps \
  "transformers==$TRANSFORMERS_VERSION" \
  "huggingface-hub==$HUGGINGFACE_HUB_VERSION" \
  "tokenizers==$TOKENIZERS_VERSION"

"$TRAINING_VENV/bin/python" -c \
  "import flash_attn; assert flash_attn.__version__ == '$FLASH_ATTN_VERSION', flash_attn.__version__"
"$TRAINING_VENV/bin/python" -c \
  "import transformers; assert transformers.__version__ == '$TRANSFORMERS_VERSION', transformers.__version__"
"$TRAINING_VENV/bin/python" -c \
  "import swanlab; from importlib.metadata import version; assert version('swanlab') == '$SWANLAB_VERSION', version('swanlab')"

echo "Training environment ready: $TRAINING_VENV"
