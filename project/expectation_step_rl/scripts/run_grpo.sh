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
  echo "Unknown mode '$MODE'; expected smoke, pilot, or full" >&2
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
if [[ "$MODE" == "smoke" ]]; then
  DEFAULT_TRAINING_GPUS="1,2,3,4"
else
  DEFAULT_TRAINING_GPUS="4,5,6,7"
fi
TRAINING_CUDA_VISIBLE_DEVICES="${TRAINING_CUDA_VISIBLE_DEVICES:-$DEFAULT_TRAINING_GPUS}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/$MODE}"

IFS=',' read -r -a TRAINING_GPU_IDS <<< "$TRAINING_CUDA_VISIBLE_DEVICES"
if [[ "${#TRAINING_GPU_IDS[@]}" -ne "$TRAINING_N_GPUS" ]]; then
  echo "TRAINING_N_GPUS=$TRAINING_N_GPUS but CUDA_VISIBLE_DEVICES exposes ${#TRAINING_GPU_IDS[@]} GPUs" >&2
  exit 2
fi

export EXPECTATION_CALIBRATION_PATH
export EXPECTATION_SCORER_API_KEY
export EXPECTATION_SCORER_BASE_URLS
export EXPECTATION_SCORER_MODEL
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/third_party/verl:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$TRAINING_CUDA_VISIBLE_DEVICES"

echo "Training physical GPUs: $TRAINING_CUDA_VISIBLE_DEVICES"
echo "verl workers: $TRAINING_N_GPUS; rollout tensor parallel: $ROLLOUT_TENSOR_PARALLEL_SIZE"

"$TRAINING_VENV/bin/python" -m expectation_step_rl.preflight \
  --project-root "$PROJECT_DIR" \
  --train-data "$EXPECTATION_DATA_DIR/train.jsonl" \
  --test-data "$EXPECTATION_DATA_DIR/test.jsonl" \
  --calibration "$EXPECTATION_CALIBRATION_PATH" \
  --scorer-base-urls "$EXPECTATION_SCORER_BASE_URLS" \
  --scorer-api-key "$EXPECTATION_SCORER_API_KEY" \
  --scorer-model "$EXPECTATION_SCORER_MODEL"

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/rollouts"

"$TRAINING_VENV/bin/python" -m verl.trainer.main_ppo \
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
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.project_name=tau2_expectation_step_rl \
  trainer.experiment_name="qwen3_32b_${MODE}" \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node="$TRAINING_N_GPUS" \
  trainer.nnodes=1 \
  trainer.use_legacy_worker_impl=disable \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
  trainer.rollout_data_dir="$OUTPUT_DIR/rollouts" \
  "$@"
