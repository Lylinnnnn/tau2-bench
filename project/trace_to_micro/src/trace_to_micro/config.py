"""Experiment configuration loading."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent.parent / path).resolve()


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
        return cls(
            domain=values["domain"],
            task_set=values["task_set"],
            train_split=values["train_split"],
            test_split=values["test_split"],
            support_thresholds=tuple(values["support_thresholds"]),
            state_changing_only=values["state_changing_only"],
            output_dir=_project_path(path, values["output_dir"]),
        )


@dataclass(frozen=True)
class TrajectoryConfig:
    """Configuration for one-trace-per-task behavior-policy rollouts."""

    domain: str
    task_set: str
    task_split: str
    agent: str
    user: str
    agent_llm: str
    user_llm: str
    agent_llm_args: dict[str, Any]
    user_llm_args: dict[str, Any]
    num_trials: int
    max_steps: int
    max_errors: int
    max_concurrency: int
    seed: int
    timeout_seconds: float | None
    save_to: str


@dataclass(frozen=True)
class ProbeConfig:
    """Configuration for logged snapshot extraction and paired inference."""

    results_path: Path
    output_dir: Path
    train_split: str
    split: str
    max_snapshots_per_task: int
    support_thresholds: tuple[int, ...]
    variants: tuple[str, ...]
    agent_llm: str
    user_llm: str
    context_builder_llm: str
    agent_llm_args: dict[str, Any]
    user_llm_args: dict[str, Any]
    context_builder_llm_args: dict[str, Any]
    seed: int


@dataclass(frozen=True)
class ModelExperimentConfig:
    """Complete model-based, training-free experiment configuration."""

    trajectory: TrajectoryConfig
    probe: ProbeConfig

    @classmethod
    def load(cls, path: Path) -> "ModelExperimentConfig":
        """Load `[trajectory]` and `[probe]` from TOML."""

        with path.open("rb") as handle:
            values = tomllib.load(handle)
        trajectory = values["trajectory"]
        probe = values["probe"]
        return cls(
            trajectory=TrajectoryConfig(
                domain=trajectory["domain"],
                task_set=trajectory["task_set"],
                task_split=trajectory["task_split"],
                agent=trajectory["agent"],
                user=trajectory["user"],
                agent_llm=trajectory["agent_llm"],
                user_llm=trajectory["user_llm"],
                agent_llm_args=dict(trajectory["agent_llm_args"]),
                user_llm_args=dict(trajectory["user_llm_args"]),
                num_trials=trajectory["num_trials"],
                max_steps=trajectory["max_steps"],
                max_errors=trajectory["max_errors"],
                max_concurrency=trajectory["max_concurrency"],
                seed=trajectory["seed"],
                timeout_seconds=trajectory.get("timeout_seconds"),
                save_to=trajectory["save_to"],
            ),
            probe=ProbeConfig(
                results_path=_project_path(path, probe["results_path"]),
                output_dir=_project_path(path, probe["output_dir"]),
                train_split=probe["train_split"],
                split=probe["split"],
                max_snapshots_per_task=probe["max_snapshots_per_task"],
                support_thresholds=tuple(probe["support_thresholds"]),
                variants=tuple(probe["variants"]),
                agent_llm=probe["agent_llm"],
                user_llm=probe["user_llm"],
                context_builder_llm=probe["context_builder_llm"],
                agent_llm_args=dict(probe["agent_llm_args"]),
                user_llm_args=dict(probe["user_llm_args"]),
                context_builder_llm_args=dict(probe["context_builder_llm_args"]),
                seed=probe["seed"],
            ),
        )
