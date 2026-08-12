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


def test_success_direction_stages_reuse_or_validate_existing_servers() -> None:
    launcher = read_script("run_qwen3_32b_success_direction.sh")
    hidden_manager = read_script("with_managed_qwen3_32b_hidden_vllm.sh")
    hidden_server = read_script("start_qwen3_32b_hidden_vllm.sh")

    assert "with_managed_qwen3_32b_vllm.sh" in launcher
    assert "with_managed_qwen3_32b_hidden_vllm.sh" in launcher
    assert "hidden_export_responds" in hidden_manager
    assert "Reusing the existing hidden-state model server" in hidden_manager
    assert "Port 8000 has a healthy generation server" in hidden_manager
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
