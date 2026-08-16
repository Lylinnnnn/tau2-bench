#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
CONFIG="$PROJECT_DIR/configs/$MODE/grpo.env"

if [[ ! -f "$CONFIG" ]]; then
  echo "Unknown mode '$MODE'; expected smoke, checkpoint_smoke, pilot, or full" >&2
  exit 2
fi

set -a
source "$CONFIG"
set +a

TRAINING_VENV="${TRAINING_VENV:-/home/liuyanlin.lyl/.venvs/expectation-step-rl}"
MODEL_PATH="${MODEL_PATH:-/data/oss_bucket_0/yanlin/tau2/models/Qwen3-32B}"
EXPECTATION_DATA_DIR="${EXPECTATION_DATA_DIR:-$PROJECT_DIR/data/decisions_qwen3_32b_t06}"
EXPECTATION_CALIBRATION_PATH="${EXPECTATION_CALIBRATION_PATH:-$EXPECTATION_DATA_DIR/training_calibration.json}"
EXPECTATION_SCORER_BASE_URLS="${EXPECTATION_SCORER_BASE_URLS:-http://127.0.0.1:8000/v1}"
EXPECTATION_SCORER_API_KEY="${EXPECTATION_SCORER_API_KEY:-EMPTY}"
EXPECTATION_SCORER_MODEL="${EXPECTATION_SCORER_MODEL:-qwen3-32b}"
EXPECTATION_SCORER_TIMEOUT_SECONDS="${EXPECTATION_SCORER_TIMEOUT_SECONDS:-180}"
if [[ "$MODE" == "smoke" || "$MODE" == "checkpoint_smoke" ]]; then
  DEFAULT_TRAINING_GPUS="1,2,3,4"
else
  DEFAULT_TRAINING_GPUS="4,5,6,7"
fi
TRAINING_CUDA_VISIBLE_DEVICES="${TRAINING_CUDA_VISIBLE_DEVICES:-$DEFAULT_TRAINING_GPUS}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/$MODE}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/oss_bucket_0/yanlin/tau2/expectation_step_rl/checkpoints/qwen3_32b_${MODE}}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-[\"console\"]}"
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-tau2_expectation_step_rl}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-qwen3_32b_${MODE}}"
SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-/data/oss_bucket_0/yanlin/tau2/expectation_step_rl/swanlog}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-8}"
TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-disable}"
CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[]}"
CHECKPOINT_LOAD_CONTENTS="${CHECKPOINT_LOAD_CONTENTS:-[]}"

IFS=',' read -r -a TRAINING_GPU_IDS <<< "$TRAINING_CUDA_VISIBLE_DEVICES"
if [[ "${#TRAINING_GPU_IDS[@]}" -ne "$TRAINING_N_GPUS" ]]; then
  echo "TRAINING_N_GPUS=$TRAINING_N_GPUS but CUDA_VISIBLE_DEVICES exposes ${#TRAINING_GPU_IDS[@]} GPUs" >&2
  exit 2
fi

export EXPECTATION_CALIBRATION_PATH
export EXPECTATION_SCORER_API_KEY
export EXPECTATION_SCORER_BASE_URLS
export EXPECTATION_SCORER_MODEL
export EXPECTATION_SCORER_TIMEOUT_SECONDS
export SWANLAB_LOG_DIR
export SWANLAB_MODE
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/third_party/verl:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$TRAINING_CUDA_VISIBLE_DEVICES"

echo "Training physical GPUs: $TRAINING_CUDA_VISIBLE_DEVICES"
echo "Repository commit: $(git -C "$REPO_ROOT" rev-parse --short HEAD)"
echo "Trainer entrypoint: expectation_step_rl.verl_adapter.main_ppo"
echo "verl workers: $TRAINING_N_GPUS; rollout tensor parallel: $ROLLOUT_TENSOR_PARALLEL_SIZE"
echo "rollout replicas: $((TRAINING_N_GPUS / ROLLOUT_TENSOR_PARALLEL_SIZE)); agent-loop workers: $AGENT_LOOP_NUM_WORKERS"
echo "Frozen scorer request timeout: ${EXPECTATION_SCORER_TIMEOUT_SECONDS}s"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Checkpoint contents: $CHECKPOINT_SAVE_CONTENTS; resume mode: $TRAINER_RESUME_MODE"
echo "Tracking backends: $TRAINER_LOGGERS"
if [[ "$TRAINER_LOGGERS" == *swanlab* ]]; then
  echo "SwanLab log directory: $SWANLAB_LOG_DIR"
fi

"$TRAINING_VENV/bin/python" -m expectation_step_rl.preflight \
  --project-root "$PROJECT_DIR" \
  --train-data "$EXPECTATION_DATA_DIR/train.jsonl" \
  --test-data "$EXPECTATION_DATA_DIR/test.jsonl" \
  --calibration "$EXPECTATION_CALIBRATION_PATH" \
  --scorer-base-urls "$EXPECTATION_SCORER_BASE_URLS" \
  --scorer-api-key "$EXPECTATION_SCORER_API_KEY" \
  --scorer-model "$EXPECTATION_SCORER_MODEL" \
  --training-gpus "$TRAINING_N_GPUS" \
  --rollout-tensor-parallel-size "$ROLLOUT_TENSOR_PARALLEL_SIZE" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --rollout-n "$ROLLOUT_N" \
  --ppo-mini-batch-size "$PPO_MINI_BATCH_SIZE" \
  --ppo-micro-batch-size-per-gpu "$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  --agent-loop-workers "$AGENT_LOOP_NUM_WORKERS"

mkdir -p "$OUTPUT_DIR/rollouts" "$CHECKPOINT_DIR"
if [[ "$TRAINER_LOGGERS" == *swanlab* ]]; then
  mkdir -p "$SWANLAB_LOG_DIR"
fi

"$TRAINING_VENV/bin/python" -m expectation_step_rl.verl_adapter.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$EXPECTATION_DATA_DIR/train.jsonl" \
  data.val_files="$EXPECTATION_DATA_DIR/test.jsonl" \
  data.prompt_key=prompt \
  data.return_raw_chat=True \
  data.dataloader_num_workers=0 \
  data.train_max_samples="$TRAIN_MAX_SAMPLES" \
  data.val_max_samples="$VAL_MAX_SAMPLES" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length=16384 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.shuffle=True \
  data.seed=42 \
  data.validation_shuffle=False \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_rank="$LORA_RANK" \
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.checkpoint.save_contents="$CHECKPOINT_SAVE_CONTENTS" \
  actor_rollout_ref.actor.checkpoint.load_contents="$CHECKPOINT_LOAD_CONTENTS" \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TENSOR_PARALLEL_SIZE" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.agent.default_agent_loop=tau2_expectation_step \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$PROJECT_DIR/configs/agent_loops.yaml" \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_LOOP_NUM_WORKERS" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.project_name="$TRAINER_PROJECT_NAME" \
  trainer.experiment_name="$TRAINER_EXPERIMENT_NAME" \
  trainer.logger="$TRAINER_LOGGERS" \
  trainer.n_gpus_per_node="$TRAINING_N_GPUS" \
  trainer.nnodes=1 \
  trainer.use_legacy_worker_impl=disable \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.resume_mode="$TRAINER_RESUME_MODE" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.rollout_data_dir="$OUTPUT_DIR/rollouts" \
  "$@"
