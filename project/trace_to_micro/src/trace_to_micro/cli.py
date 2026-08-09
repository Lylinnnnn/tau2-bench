"""Command-line orchestration for deterministic pre-experiments."""

import argparse
import sys
from collections import Counter
from pathlib import Path

from tau2.runner import load_task_splits, load_tasks
from trace_to_micro.config import ExperimentConfig, ModelExperimentConfig
from trace_to_micro.evaluation import build_paired_report, build_support_report
from trace_to_micro.io import read_jsonl, write_events_jsonl, write_json, write_jsonl
from trace_to_micro.logged_audit import build_logged_audit
from trace_to_micro.logged_trace import (
    audit_results_completeness,
    extract_decision_snapshots,
)
from trace_to_micro.probe import run_paired_probe
from trace_to_micro.replay import replay_reference_tasks
from trace_to_micro.task_inventory import build_task_inventory
from trace_to_micro.trajectory_run import run_complete_trajectories


def _load_benchmark(config: ExperimentConfig):
    tasks = load_tasks(config.task_set, task_split_name=None)
    split_map = load_task_splits(config.task_set)
    tasks_by_id = {task.id: task for task in tasks}
    return tasks, split_map, tasks_by_id


def _resolve_output_dir(config: ExperimentConfig, output_dir: Path | None) -> Path:
    return output_dir if output_dir is not None else config.output_dir


def run_task_audit(config_path: Path, output_dir: Path | None = None) -> Path:
    """Write the deterministic task/split inventory."""

    config = ExperimentConfig.load(config_path)
    tasks, split_map, _ = _load_benchmark(config)
    inventory = build_task_inventory(
        tasks,
        split_map,
        train_split=config.train_split,
        test_split=config.test_split,
    )
    path = _resolve_output_dir(config, output_dir) / "task_inventory.json"
    write_json(path, inventory)
    return path


def run_oracle_preflight(
    config_path: Path, output_dir: Path | None = None
) -> dict[str, Path]:
    """Replay reference trajectories and write the LOCO upper-bound report."""

    config = ExperimentConfig.load(config_path)
    tasks, split_map, tasks_by_id = _load_benchmark(config)
    destination = _resolve_output_dir(config, output_dir)

    inventory = build_task_inventory(
        tasks,
        split_map,
        train_split=config.train_split,
        test_split=config.test_split,
    )
    train_tasks = [tasks_by_id[task_id] for task_id in split_map[config.train_split]]
    test_tasks = [tasks_by_id[task_id] for task_id in split_map[config.test_split]]
    train_events = replay_reference_tasks(
        train_tasks, split=config.train_split, domain=config.domain
    )
    test_events = replay_reference_tasks(
        test_tasks, split=config.test_split, domain=config.domain
    )
    report = build_support_report(
        train_events,
        test_events,
        thresholds=config.support_thresholds,
        state_changing_only=config.state_changing_only,
    )
    report["experiment"] = {
        "domain": config.domain,
        "task_set": config.task_set,
        "train_split": config.train_split,
        "test_split": config.test_split,
        "support_thresholds": list(config.support_thresholds),
        "data_source": "evaluation_criteria.actions",
        "claim_scope": "oracle structural upper bound; not off-policy evidence",
    }

    inventory_path = destination / "task_inventory.json"
    transitions_path = destination / "oracle_transitions.jsonl"
    report_path = destination / "oracle_support_report.json"
    write_json(inventory_path, inventory)
    write_events_jsonl(transitions_path, train_events + test_events)
    write_json(report_path, report)
    return {
        "inventory": inventory_path,
        "transitions": transitions_path,
        "report": report_path,
    }


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
    """Require complete results, then extract compact decision pointers."""

    config = ModelExperimentConfig.load(config_path)
    completeness = audit_results_completeness(config.probe.results_path)
    snapshots = extract_decision_snapshots(
        config.probe.results_path,
        task_set=config.trajectory.task_set,
        split=config.probe.split,
        max_snapshots_per_task=config.probe.max_snapshots_per_task,
    )
    if not snapshots:
        raise ValueError("No agent decision snapshots were extracted")
    output_dir = config.probe.output_dir
    completeness_path = output_dir / "trajectory_completeness.json"
    snapshots_path = output_dir / "decision_snapshots.jsonl"
    report_path = output_dir / "snapshot_report.json"
    write_json(completeness_path, completeness)
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
                    Counter(row["context_length_bucket"] for row in snapshots).items()
                )
            ),
            "selection": "deterministic early/middle/late within each trajectory",
            "max_snapshots_per_task": config.probe.max_snapshots_per_task,
            "split": config.probe.split,
        },
    )
    return {
        "completeness": completeness_path,
        "snapshots": snapshots_path,
        "report": report_path,
    }


def run_logged_transition_audit(config_path: Path) -> dict[str, Path]:
    """Replay actual model actions and report observational transition support."""

    config = ModelExperimentConfig.load(config_path)
    audit_results_completeness(config.probe.results_path)
    rows, report = build_logged_audit(
        results_path=config.probe.results_path,
        domain=config.trajectory.domain,
        task_set=config.trajectory.task_set,
        train_split=config.probe.train_split,
        test_split=config.probe.split,
        thresholds=config.probe.support_thresholds,
    )
    transitions_path = config.probe.output_dir / "logged_transitions.jsonl"
    report_path = config.probe.output_dir / "logged_support_report.json"
    write_jsonl(transitions_path, rows)
    write_json(report_path, report)
    return {"logged_transitions": transitions_path, "logged_report": report_path}


def run_model_probe(config_path: Path) -> dict[str, Path]:
    """Run/resume paired inference from an existing snapshot manifest."""

    config = ModelExperimentConfig.load(config_path)
    audit_results_completeness(config.probe.results_path)
    snapshots_path = config.probe.output_dir / "decision_snapshots.jsonl"
    snapshots = read_jsonl(snapshots_path)
    if not snapshots:
        raise ValueError(
            f"No snapshots at {snapshots_path}; run extract-snapshots first"
        )
    return run_paired_probe(config, snapshots)


def run_probe_summary(config_path: Path) -> Path:
    """Aggregate completed paired predictions into the final report."""

    config = ModelExperimentConfig.load(config_path)
    predictions_path = config.probe.output_dir / "paired_predictions.jsonl"
    rows = read_jsonl(predictions_path)
    expected = len(read_jsonl(config.probe.output_dir / "decision_snapshots.jsonl"))
    expected *= len(config.probe.variants)
    if len(rows) != expected:
        raise ValueError(
            f"Paired probe is incomplete: expected {expected} predictions, "
            f"found {len(rows)}"
        )
    report = build_paired_report(rows)
    report["experiment"] = {
        "variants": list(config.probe.variants),
        "agent_model": config.probe.agent_llm,
        "user_model": config.probe.user_llm,
        "context_builder_model": config.probe.context_builder_llm,
        "training": False,
        "original_rollouts_per_task": config.trajectory.num_trials,
    }
    path = config.probe.output_dir / "paired_probe_report.json"
    write_json(path, report)
    return path


def run_full_model_preexperiment(config_path: Path) -> dict[str, Path]:
    """Generate trajectories, extract snapshots, run probes, and summarize."""

    run_trajectory_generation(config_path)
    paths = run_logged_snapshot_extraction(config_path)
    paths.update(run_logged_transition_audit(config_path))
    paths.update(run_model_probe(config_path))
    paths["paired_report"] = run_probe_summary(config_path)
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trace-to-micro")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("task-audit", "oracle-preflight"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path)
    generate_parser = subparsers.add_parser("generate-trajectories")
    generate_parser.add_argument("--config", type=Path, required=True)
    generate_parser.add_argument("--num-tasks", type=int)
    generate_parser.add_argument("--save-to")
    for command in (
        "extract-snapshots",
        "logged-audit",
        "paired-probe",
        "summarize-probe",
        "model-preexperiment",
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run a configured pre-experiment command."""

    args = _parser().parse_args(argv)
    if args.command == "task-audit":
        path = run_task_audit(args.config, args.output_dir)
        print(path)
        return
    if args.command == "oracle-preflight":
        paths = run_oracle_preflight(args.config, args.output_dir)
    elif args.command == "generate-trajectories":
        run_trajectory_generation(
            args.config,
            num_tasks=args.num_tasks,
            save_to=args.save_to,
        )
        return
    elif args.command == "extract-snapshots":
        paths = run_logged_snapshot_extraction(args.config)
    elif args.command == "logged-audit":
        paths = run_logged_transition_audit(args.config)
    elif args.command == "paired-probe":
        paths = run_model_probe(args.config)
    elif args.command == "summarize-probe":
        print(run_probe_summary(args.config))
        return
    else:
        paths = run_full_model_preexperiment(args.config)
    for name, path in paths.items():
        print(f"{name}: {path}")


def task_audit_main() -> None:
    """Run the task-audit subcommand from its thin script wrapper."""

    main(["task-audit", *sys.argv[1:]])


def oracle_preflight_main() -> None:
    """Run the oracle-preflight subcommand from its thin script wrapper."""

    main(["oracle-preflight", *sys.argv[1:]])


if __name__ == "__main__":
    main()
