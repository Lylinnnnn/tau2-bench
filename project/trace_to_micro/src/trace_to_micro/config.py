"""Experiment configuration loading."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the deterministic LOCO preflight."""

    domain: str
    task_set: str
    train_split: str
    test_split: str
    support_thresholds: tuple[int, ...]
    state_changing_only: bool
    output_dir: Path

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load the `[experiment]` table from a TOML file."""

        with path.open("rb") as handle:
            values = tomllib.load(handle)["experiment"]
        output_dir = Path(values["output_dir"])
        if not output_dir.is_absolute():
            output_dir = path.parent.parent / output_dir
        return cls(
            domain=values["domain"],
            task_set=values["task_set"],
            train_split=values["train_split"],
            test_split=values["test_split"],
            support_thresholds=tuple(values["support_thresholds"]),
            state_changing_only=values["state_changing_only"],
            output_dir=output_dir,
        )
