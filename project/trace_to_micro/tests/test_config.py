from pathlib import Path

from trace_to_micro.config import (
    ExperimentConfig,
    ModelExperimentConfig,
    SuccessDirectionConfig,
)


def test_relative_output_directory_is_scoped_to_project(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "experiment.toml"
    config_path.write_text(
        """
[experiment]
domain = "telecom"
task_set = "telecom"
train_split = "train"
test_split = "test"
support_thresholds = [1, 3]
state_changing_only = true
output_dir = "outputs/pilot"
""".strip(),
        encoding="utf-8",
    )

    config = ExperimentConfig.load(config_path)

    assert config.output_dir == tmp_path / "outputs" / "pilot"


def test_model_experiment_paths_and_models_are_loaded(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "model.toml"
    config_path.write_text(
        """
[trajectory]
domain = "telecom"
task_set = "telecom"
task_split = "base"
agent = "llm_agent"
user = "user_simulator"
agent_llm = "openai/local-agent"
user_llm = "openai/local-user"
agent_llm_args = { temperature = 0.0 }
user_llm_args = { temperature = 0.0 }
num_trials = 1
max_steps = 200
max_errors = 10
max_concurrency = 2
seed = 300
save_to = "experiment"

[probe]
results_path = "../../data/simulations/experiment/results.json"
output_dir = "outputs/model"
train_split = "train"
splits = ["train", "test"]
max_snapshots_per_task = 3
support_thresholds = [1, 3, 5]
variants = ["long_raw", "structured_state", "clean_subtask"]
agent_llm = "openai/local-agent"
user_llm = "openai/local-user"
context_builder_llm = "openai/local-builder"
agent_llm_args = { temperature = 0.0 }
user_llm_args = { temperature = 0.0 }
context_builder_llm_args = { temperature = 0.0 }
seed = 300
""".strip(),
        encoding="utf-8",
    )

    config = ModelExperimentConfig.load(config_path)

    assert config.trajectory.agent_llm == "openai/local-agent"
    assert config.trajectory.timeout_seconds is None
    assert (
        config.probe.results_path
        == tmp_path.parent.parent / "data/simulations/experiment/results.json"
    )
    assert config.probe.output_dir == tmp_path / "outputs/model"
    assert config.probe.splits == ("train", "test")


def test_qwen3_32b_config_separates_thinking_agent_and_json_builder() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs/qwen3_32b_thinking_model_preexperiment.toml"
    )

    config = ModelExperimentConfig.load(config_path)

    assert config.trajectory.agent_llm == "openai/qwen3-32b"
    assert config.trajectory.agent_llm_args["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    assert config.probe.context_builder_llm_args["max_tokens"] == 1024
    assert config.probe.context_builder_llm_args["extra_body"][
        "chat_template_kwargs"
    ] == {"enable_thinking": False}
    assert "qwen3_32b" in str(config.probe.results_path)
    assert "hybrid_clean" in config.probe.variants


def test_success_direction_config_preregisters_primary_layer() -> None:
    config_path = Path(__file__).parents[1] / "configs/qwen3_32b_success_direction.toml"

    config = SuccessDirectionConfig.load(config_path)

    assert config.domains == ("airline", "retail")
    assert config.force_overwrite is True
    assert config.primary_layer_id == 47
    assert config.primary_layer_id in config.hidden_layer_ids
    assert config.hidden_size == 5120
    assert config.smoke_domain == "retail"
    assert config.smoke_task_id == "2"
    assert config.smoke_results_path().name == "results.json"
    assert config.smoke_output_dir().name == "smoke"
    assert config.bootstrap_samples == 2000
    assert config.permutation_samples == 5000
