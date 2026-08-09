from pathlib import Path

from trace_to_micro.config import ExperimentConfig


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
