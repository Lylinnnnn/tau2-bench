from pathlib import Path

from trace_to_micro.config import (
    ExpectationDeviationConfig,
    ExpectationMatchingConfig,
    ExperimentConfig,
    LocalConsequenceConfig,
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
    assert "shard_00_of_08" in config.shard_save_name("airline", 0, 8)
    assert config.activation_shard_path(7, 8).name == "shard_07_of_08.jsonl"
    assert config.bootstrap_samples == 2000
    assert config.permutation_samples == 5000


def test_local_consequence_config_reuses_existing_complete_trajectories() -> None:
    config_path = Path(__file__).parents[1] / "configs/qwen3_32b_local_consequence.toml"

    config = LocalConsequenceConfig.load(config_path)

    assert config.domains == ("airline", "retail")
    assert config.primary_layer_id == 47
    assert config.primary_moment == "action"
    assert config.random_seed == 300
    assert config.expectation_regularization == 1.0
    assert config.expectation_minimum_train_class_count == 2
    assert config.expected_agent_model == "openai/qwen3-32b"
    assert config.expected_user_model == "openai/qwen3-32b"
    assert config.expected_num_trials == 1
    assert config.results_path("airline").name == "results.json"
    assert "success_direction_airline_base" in str(config.results_path("airline"))
    assert config.smoke_domain == "retail"
    assert config.smoke_task_id == "0"


def test_expectation_matching_config_preregisters_target_free_controls() -> None:
    config_path = (
        Path(__file__).parents[1] / "configs/qwen3_32b_expectation_matching.toml"
    )

    config = ExpectationMatchingConfig.load(config_path)

    assert config.domains == ("airline", "retail")
    assert config.evaluation_split == "test"
    assert config.candidate_count == 4
    assert config.minimum_candidate_count == 4
    assert config.context_variants == ("full", "action_only", "shuffled")
    assert config.content_variants == ("raw", "identifier_masked")
    assert config.scoring_model == "qwen3-32b"
    assert "success_direction_retail_base" in str(config.results_path("retail"))
    assert config.score_shard_path(7, 8).name == "shard_07_of_08.jsonl"


def test_expectation_deviation_config_separates_train_calibration_and_test() -> None:
    config_path = (
        Path(__file__).parents[1] / "configs/qwen3_32b_expectation_deviation.toml"
    )

    config = ExpectationDeviationConfig.load(config_path)

    assert config.domains == ("airline", "retail")
    assert config.train_split == "train"
    assert config.test_split == "test"
    assert config.severity_levels == (1, 2, 3)
    assert config.max_length_delta_ratio == 0.1
    assert config.calibration_minimum_count == 5
    assert config.min_k_fraction == 0.1
    assert config.sigma_threshold == 3.0
    assert config.rematch_candidate_count == 4
    assert config.score_shard_path(7, 8).name == "shard_07_of_08.jsonl"
    assert config.min_k_score_shard_path(7, 8).parent.name == (
        "contextual_min_k_score_shards"
    )
