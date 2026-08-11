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
