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
    splits: tuple[str, ...]
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
        splits = tuple(probe["splits"])
        if len(splits) != 2:
            raise ValueError("Model pre-experiment requires exactly train/test splits")
        if len(set(splits)) != len(splits):
            raise ValueError("Probe splits must be unique")
        if probe["train_split"] not in splits:
            raise ValueError("Probe splits must include train_split")
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
                splits=splits,
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


@dataclass(frozen=True)
class SuccessDirectionConfig:
    """Configuration for the Airline/Retail hidden-state direction probe."""

    domains: tuple[str, ...]
    task_split: str
    train_split: str
    test_split: str
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
    save_to_prefix: str
    results_dir: Path
    output_dir: Path
    hidden_base_url: str
    hidden_model: str
    hidden_layer_ids: tuple[int, ...]
    hidden_size: int
    primary_layer_id: int
    max_tool_decisions_per_trajectory: int
    bootstrap_samples: int
    permutation_samples: int
    force_overwrite: bool
    smoke_domain: str
    smoke_task_id: str

    @classmethod
    def load(cls, path: Path) -> "SuccessDirectionConfig":
        """Load and validate the `[success_direction]` table."""

        with path.open("rb") as handle:
            values = tomllib.load(handle)["success_direction"]
        domains = tuple(values["domains"])
        layer_ids = tuple(values["hidden_layer_ids"])
        if len(domains) != 2 or len(set(domains)) != len(domains):
            raise ValueError(
                "Success-direction domains must be unique and contain two domains"
            )
        if values["train_split"] == values["test_split"]:
            raise ValueError("Success-direction train/test splits must differ")
        if values["primary_layer_id"] not in layer_ids:
            raise ValueError("primary_layer_id must be included in hidden_layer_ids")
        if values["hidden_size"] <= 0:
            raise ValueError("hidden_size must be positive")
        if values["smoke_domain"] not in domains:
            raise ValueError("smoke_domain must be one of the configured domains")
        if values["max_tool_decisions_per_trajectory"] <= 0:
            raise ValueError("max_tool_decisions_per_trajectory must be positive")
        if values["bootstrap_samples"] <= 0 or values["permutation_samples"] <= 0:
            raise ValueError("Resampling counts must be positive")
        return cls(
            domains=domains,
            task_split=values["task_split"],
            train_split=values["train_split"],
            test_split=values["test_split"],
            agent=values["agent"],
            user=values["user"],
            agent_llm=values["agent_llm"],
            user_llm=values["user_llm"],
            agent_llm_args=dict(values["agent_llm_args"]),
            user_llm_args=dict(values["user_llm_args"]),
            num_trials=values["num_trials"],
            max_steps=values["max_steps"],
            max_errors=values["max_errors"],
            max_concurrency=values["max_concurrency"],
            seed=values["seed"],
            timeout_seconds=values.get("timeout_seconds"),
            save_to_prefix=values["save_to_prefix"],
            results_dir=_project_path(path, values["results_dir"]),
            output_dir=_project_path(path, values["output_dir"]),
            hidden_base_url=values["hidden_base_url"],
            hidden_model=values["hidden_model"],
            hidden_layer_ids=layer_ids,
            hidden_size=values["hidden_size"],
            primary_layer_id=values["primary_layer_id"],
            max_tool_decisions_per_trajectory=values[
                "max_tool_decisions_per_trajectory"
            ],
            bootstrap_samples=values["bootstrap_samples"],
            permutation_samples=values["permutation_samples"],
            force_overwrite=values["force_overwrite"],
            smoke_domain=values["smoke_domain"],
            smoke_task_id=str(values["smoke_task_id"]),
        )

    def save_name(self, domain: str) -> str:
        """Return the τ² result-directory name for one domain."""

        return f"{self.save_to_prefix}_{domain}_{self.task_split}"

    def results_path(self, domain: str) -> Path:
        """Return the expected τ² results file for one domain."""

        return self.results_dir / self.save_name(domain) / "results.json"

    def shard_save_name(
        self, domain: str, shard_index: int, num_shards: int
    ) -> str:
        """Return the isolated result-directory name for one task shard."""

        return (
            f"{self.save_name(domain)}_shard_{shard_index:02d}_of_{num_shards:02d}"
        )

    def shard_results_path(
        self, domain: str, shard_index: int, num_shards: int
    ) -> Path:
        """Return the result file written by one trajectory worker."""

        return (
            self.results_dir
            / self.shard_save_name(domain, shard_index, num_shards)
            / "results.json"
        )

    def activation_shard_path(self, shard_index: int, num_shards: int) -> Path:
        """Return the compact hidden-vector file for one activation worker."""

        return (
            self.output_dir
            / "activation_shards"
            / f"shard_{shard_index:02d}_of_{num_shards:02d}.jsonl"
        )

    def smoke_save_name(self) -> str:
        """Return the isolated one-task smoke result-directory name."""

        return (
            f"{self.save_to_prefix}_smoke_{self.smoke_domain}_task_{self.smoke_task_id}"
        )

    def smoke_results_path(self) -> Path:
        """Return the isolated one-task smoke results file."""

        return self.results_dir / self.smoke_save_name() / "results.json"

    def smoke_output_dir(self) -> Path:
        """Return the directory for smoke-only audit artifacts."""

        return self.output_dir / "smoke"
