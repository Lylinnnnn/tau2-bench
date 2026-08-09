"""Orchestration for the two model-based, training-free experiments."""

from collections import Counter
from pathlib import Path

from tau2.data_model.simulation import Results
from tau2.runner import load_task_splits
from trace_to_micro.config import ModelExperimentConfig
from trace_to_micro.evaluation import build_cross_split_report, build_paired_report
from trace_to_micro.io import read_jsonl, write_json, write_jsonl
from trace_to_micro.logged_audit import build_logged_audit
from trace_to_micro.logged_trace import (
    audit_results_completeness,
    extract_decision_snapshots,
)
from trace_to_micro.probe import run_paired_probe
from trace_to_micro.trajectory_analysis import (
    build_cross_split_trajectory_report,
    build_trajectory_report,
)
from trace_to_micro.trajectory_run import run_complete_trajectories

TRAJECTORY_ANALYSIS_DIR = "experiment_1_trajectory_analysis"
CONTEXT_PROBE_DIR = "experiment_2_context_probe"


def _trajectory_analysis_dir(config: ModelExperimentConfig) -> Path:
    return config.probe.output_dir / TRAJECTORY_ANALYSIS_DIR


def _context_probe_dir(config: ModelExperimentConfig, split: str) -> Path:
    return config.probe.output_dir / CONTEXT_PROBE_DIR / split


def _test_split(config: ModelExperimentConfig) -> str:
    return next(
        split for split in config.probe.splits if split != config.probe.train_split
    )


def _audit_model_results(config: ModelExperimentConfig) -> dict:
    split_map = load_task_splits(config.trajectory.task_set)
    return audit_results_completeness(
        config.probe.results_path,
        expected_task_ids=set(split_map[config.trajectory.task_split]),
        expected_num_trials=config.trajectory.num_trials,
    )


def run_trajectory_generation(
    config_path: Path,
    *,
    num_tasks: int | None = None,
    save_to: str | None = None,
) -> None:
    """Generate/resume the configured real behavior-policy trajectories."""

    config = ModelExperimentConfig.load(config_path)
    run_complete_trajectories(
        config.trajectory,
        num_tasks=num_tasks,
        save_to=save_to,
    )


def run_logged_snapshot_extraction(config_path: Path) -> dict[str, Path]:
    """Require complete results, then extract train/test decision pointers."""

    config = ModelExperimentConfig.load(config_path)
    completeness = _audit_model_results(config)
    output_dir = _trajectory_analysis_dir(config)
    completeness_path = output_dir / "trajectory_completeness.json"
    write_json(completeness_path, completeness)
    paths = {"trajectory_completeness": completeness_path}
    for split in config.probe.splits:
        snapshots = extract_decision_snapshots(
            config.probe.results_path,
            task_set=config.trajectory.task_set,
            split=split,
            max_snapshots_per_task=config.probe.max_snapshots_per_task,
        )
        if not snapshots:
            raise ValueError(f"No agent decision snapshots extracted for {split!r}")
        split_dir = output_dir / split
        snapshots_path = split_dir / "decision_snapshots.jsonl"
        report_path = split_dir / "snapshot_report.json"
        write_jsonl(snapshots_path, snapshots)
        write_json(
            report_path,
            {
                "snapshot_count": len(snapshots),
                "unique_tasks": len({row["task_id"] for row in snapshots}),
                "position_buckets": dict(
                    sorted(Counter(row["position_bucket"] for row in snapshots).items())
                ),
                "context_length_buckets": dict(
                    sorted(
                        Counter(
                            row["context_length_bucket"] for row in snapshots
                        ).items()
                    )
                ),
                "selection": "deterministic early/middle/late within each trajectory",
                "max_snapshots_per_task": config.probe.max_snapshots_per_task,
                "split": split,
            },
        )
        paths[f"{split}_snapshots"] = snapshots_path
        paths[f"{split}_snapshot_report"] = report_path
    return paths


def run_logged_transition_audit(config_path: Path) -> dict[str, Path]:
    """Replay actual model actions and report observational transition support."""

    config = ModelExperimentConfig.load(config_path)
    _audit_model_results(config)
    rows, report = build_logged_audit(
        results_path=config.probe.results_path,
        domain=config.trajectory.domain,
        task_set=config.trajectory.task_set,
        train_split=config.probe.train_split,
        test_split=_test_split(config),
        thresholds=config.probe.support_thresholds,
    )
    output_dir = _trajectory_analysis_dir(config)
    transitions_path = output_dir / "logged_transitions.jsonl"
    report_path = output_dir / "cross_split_support_report.json"
    write_jsonl(transitions_path, rows)
    write_json(report_path, report)
    return {"logged_transitions": transitions_path, "logged_report": report_path}


def run_trajectory_analysis(config_path: Path) -> dict[str, Path]:
    """Experiment 1: analyze complete logged trajectories on train and test."""

    config = ModelExperimentConfig.load(config_path)
    paths = run_logged_snapshot_extraction(config_path)
    paths.update(run_logged_transition_audit(config_path))
    transitions = read_jsonl(paths["logged_transitions"])
    split_map = load_task_splits(config.trajectory.task_set)
    simulations = list(Results.iter_simulations(config.probe.results_path))
    reports = {}
    for split in config.probe.splits:
        allowed = set(split_map[split])
        split_simulations = [
            simulation
            for simulation in simulations
            if str(simulation.task_id) in allowed
        ]
        split_dir = _trajectory_analysis_dir(config) / split
        snapshots = read_jsonl(split_dir / "decision_snapshots.jsonl")
        report = build_trajectory_report(
            split=split,
            simulations=split_simulations,
            snapshots=snapshots,
            transition_rows=transitions,
        )
        report_path = split_dir / "trajectory_report.json"
        write_json(report_path, report)
        reports[split] = report
        paths[f"{split}_trajectory_report"] = report_path
    cross_split_path = _trajectory_analysis_dir(config) / "cross_split_report.json"
    write_json(
        cross_split_path,
        build_cross_split_trajectory_report(
            reports, train_split=config.probe.train_split
        ),
    )
    paths["cross_split_trajectory_report"] = cross_split_path
    return paths


def run_model_probe(config_path: Path) -> dict[str, Path]:
    """Experiment 2: run same-state paired inference on train and test."""

    config = ModelExperimentConfig.load(config_path)
    _audit_model_results(config)
    paths = {}
    for split in config.probe.splits:
        snapshots_path = (
            _trajectory_analysis_dir(config) / split / "decision_snapshots.jsonl"
        )
        snapshots = read_jsonl(snapshots_path)
        if not snapshots:
            raise ValueError(
                f"No snapshots at {snapshots_path}; run analyze-trajectories first"
            )
        split_paths = run_paired_probe(
            config,
            snapshots,
            output_dir=_context_probe_dir(config, split),
        )
        paths.update({f"{split}_{name}": path for name, path in split_paths.items()})
    return paths


def run_probe_summary(config_path: Path) -> dict[str, Path]:
    """Aggregate within-split paired effects and their generalization gap."""

    config = ModelExperimentConfig.load(config_path)
    reports = {}
    paths = {}
    for split in config.probe.splits:
        split_dir = _context_probe_dir(config, split)
        rows = read_jsonl(split_dir / "paired_predictions.jsonl")
        snapshots = read_jsonl(
            _trajectory_analysis_dir(config) / split / "decision_snapshots.jsonl"
        )
        expected = len(snapshots) * len(config.probe.variants)
        if len(rows) != expected:
            raise ValueError(
                f"Paired probe for {split!r} is incomplete: expected {expected} "
                f"predictions, found {len(rows)}"
            )
        if not all(row["same_pre_state"] for row in rows):
            raise ValueError(f"Paired probe for {split!r} violated state identity")
        report = build_paired_report(rows)
        report["experiment"] = {
            "name": "same_state_different_context_single_step",
            "split": split,
            "variants": list(config.probe.variants),
            "agent_model": config.probe.agent_llm,
            "user_model": config.probe.user_llm,
            "context_builder_model": config.probe.context_builder_llm,
            "training": False,
            "cross_task_support_used_as_model_input": False,
            "original_rollouts_per_task": config.trajectory.num_trials,
        }
        report_path = split_dir / "paired_probe_report.json"
        write_json(report_path, report)
        reports[split] = report
        paths[f"{split}_paired_report"] = report_path
    cross_split = build_cross_split_report(
        reports, train_split=config.probe.train_split
    )
    cross_split_path = (
        config.probe.output_dir / CONTEXT_PROBE_DIR / "cross_split_report.json"
    )
    write_json(cross_split_path, cross_split)
    paths["cross_split_paired_report"] = cross_split_path
    return paths


def run_full_model_preexperiment(config_path: Path) -> dict[str, Path]:
    """Generate trajectories, run both experiments, and summarize."""

    run_trajectory_generation(config_path)
    paths = run_trajectory_analysis(config_path)
    paths.update(run_model_probe(config_path))
    paths.update(run_probe_summary(config_path))
    return paths
