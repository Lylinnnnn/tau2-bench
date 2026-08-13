from pathlib import Path

PROJECT_DIR = Path(__file__).parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"


def read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text()


def test_qwen3_32b_launcher_matches_vllm_025() -> None:
    script = read_script("start_qwen3_32b_vllm.sh")

    assert "/home/liuyanlin.lyl/.venvs/tau2-vllm/bin/vllm" in script
    assert "--enable-reasoning" not in script
    assert "--reasoning-parser qwen3" in script
    assert "--enable-auto-tool-choice" in script
    assert "--tool-call-parser hermes" in script


def test_qwen3_32b_entrypoints_use_managed_server() -> None:
    manager = "with_managed_qwen3_32b_vllm.sh"

    assert manager in read_script("run_qwen3_32b_model_preexperiment.sh")
    assert manager in read_script("run_qwen3_32b_retail_smoke.sh")


def test_model_launchers_use_runtime_preflight_module() -> None:
    module = "trace_to_micro.runtime.server_preflight"

    assert module in read_script("run_server_model_preexperiment.sh")
    assert module in read_script("run_qwen3_32b_retail_smoke.sh")


def test_managed_server_waits_for_models_endpoint() -> None:
    script = read_script("with_managed_qwen3_32b_vllm.sh")

    assert '"${base_url%/}/models"' in script
    assert '"${project_dir}/scripts/start_qwen3_32b_vllm.sh"' in script
    assert "trap cleanup EXIT" in script
    assert '--model-id "${served_model_name}"' in script
    assert 'wait_seconds="${TRACE_TO_MICRO_SERVER_WAIT_SECONDS:-1800}"' in script


def test_success_direction_stages_reuse_or_validate_existing_servers() -> None:
    launcher = read_script("run_qwen3_32b_success_direction.sh")
    hidden_manager = read_script("with_managed_qwen3_32b_hidden_vllm.sh")
    hidden_server = read_script("start_qwen3_32b_hidden_vllm.sh")

    assert "with_managed_qwen3_32b_vllm.sh" in launcher
    assert "with_managed_qwen3_32b_hidden_vllm.sh" in launcher
    assert "hidden_export_responds" in hidden_manager
    assert "Reusing the existing hidden-state model server" in hidden_manager
    assert "has a healthy generation server" in hidden_manager
    assert "--enforce-eager" in hidden_server
    assert "--no-enable-chunked-prefill" in hidden_server
    assert "/dev/shm/trace_to_micro_hidden_states" in hidden_server
    assert "VLLM_USE_V2_MODEL_RUNNER=0" in hidden_server


def test_success_direction_smoke_is_two_strict_stages() -> None:
    launcher = read_script("run_qwen3_32b_success_direction_smoke.sh")

    assert "success-smoke-trajectories" in launcher
    assert "success-smoke-activation-requests" in launcher
    assert "success-smoke-activations" in launcher
    assert "with_managed_qwen3_32b_vllm.sh" in launcher
    assert "with_managed_qwen3_32b_hidden_vllm.sh" in launcher


def test_success_direction_8gpu_uses_isolated_single_gpu_workers() -> None:
    launcher = read_script("run_qwen3_32b_success_direction_8gpu.sh")
    generation_server = read_script("start_qwen3_32b_vllm.sh")
    hidden_server = read_script("start_qwen3_32b_hidden_vllm.sh")

    assert 'num_shards="${NUM_SHARDS:-8}"' in launcher
    assert 'base_port="${BASE_PORT:-8100}"' in launcher
    assert 'internal_base_port="${INTERNAL_BASE_PORT:-20000}"' in launcher
    assert 'internal_port_stride="${INTERNAL_PORT_STRIDE:-1000}"' in launcher
    assert 'master_base_port="${MASTER_BASE_PORT:-40000}"' in launcher
    assert 'shard_ids_raw="${SHARD_IDS:-}"' in launcher
    assert 'for shard in "${selected_shards[@]}"' in launcher
    assert "trajectory-shards" in launcher
    assert "require_explicit_shards" in launcher
    assert 'export CUDA_VISIBLE_DEVICES="${gpu}"' in launcher
    assert 'export QWEN3_32B_HTTP_PORT="${port}"' in launcher
    assert 'export VLLM_PORT="${internal_port}"' in launcher
    assert "export VLLM_RPC_BASE_PATH=" in launcher
    assert 'export MASTER_PORT="${master_port}"' in launcher
    assert "validate_port_layout" in launcher
    assert "export TENSOR_PARALLEL_SIZE=1" in launcher
    assert "success-trajectory-shard" in launcher
    assert "success-merge-trajectories" in launcher
    assert "success-activation-shard" in launcher
    assert "success-merge-activations" in launcher
    assert "ensure_target_ports_are_free" in launcher
    assert 'http_port="${QWEN3_32B_HTTP_PORT:-8000}"' in generation_server
    assert 'http_port="${QWEN3_32B_HTTP_PORT:-8000}"' in hidden_server
    assert 'port="${VLLM_PORT:-8000}"' not in generation_server
    assert 'port="${VLLM_PORT:-8000}"' not in hidden_server


def test_hidden_server_allows_slow_parallel_startup() -> None:
    manager = read_script("with_managed_qwen3_32b_hidden_vllm.sh")

    assert 'wait_seconds="${TRACE_TO_MICRO_SERVER_WAIT_SECONDS:-1800}"' in manager


def test_merge_stage_skips_model_run_checks() -> None:
    launcher = read_script("run_qwen3_32b_success_direction_8gpu.sh")
    prefix, stage_gate, dispatch = launcher.split('\ncase "${stage}" in\n')

    assert "merge-trajectories | evaluate" in stage_gate
    assert "run_checks" in prefix
    assert "run_checks" in stage_gate
    assert "run_checks" not in dispatch


def test_local_consequence_launcher_reuses_traces_and_shards_hidden_exports() -> None:
    launcher = read_script("run_qwen3_32b_local_consequence.sh")

    assert "local-consequence-requests" in launcher
    assert "local-consequence-smoke" in launcher
    assert "local-consequence-activation-shard" in launcher
    assert "local-consequence-merge-activations" in launcher
    assert "local-consequence-evaluate" in launcher
    assert "consequence-expectation-evaluate" in launcher
    assert "expectation)" in launcher
    assert "success-trajectory-shard" not in launcher
    assert 'num_shards="${NUM_SHARDS:-8}"' in launcher
    assert 'export CUDA_VISIBLE_DEVICES="${gpu}"' in launcher
    assert "with_managed_qwen3_32b_hidden_vllm.sh" in launcher


def test_expectation_matching_launcher_reuses_healthy_servers_and_shards_scores() -> (
    None
):
    launcher = read_script("run_qwen3_32b_expectation_matching.sh")

    assert 'num_shards="${NUM_SHARDS:-8}"' in launcher
    assert 'base_port="${BASE_PORT:-8200}"' in launcher
    assert 'shard_zero_port="${SHARD_ZERO_PORT:-8000}"' in launcher
    assert 'internal_base_port="${INTERNAL_BASE_PORT:-22000}"' in launcher
    assert 'internal_port_stride="${INTERNAL_PORT_STRIDE:-1000}"' in launcher
    assert 'export CUDA_VISIBLE_DEVICES="${gpu}"' in launcher
    assert 'export VLLM_PORT="${internal_port}"' in launcher
    assert 'export MASTER_PORT="${master_port}"' in launcher
    assert "with_managed_qwen3_32b_vllm.sh" in launcher
    assert "expectation-matching-prepare" in launcher
    assert "expectation-matching-score-shard" in launcher
    assert "expectation-matching-merge" in launcher
    assert "expectation-matching-evaluate" in launcher
