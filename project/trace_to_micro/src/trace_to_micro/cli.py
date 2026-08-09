"""Command-line orchestration for deterministic pre-experiments."""

import argparse
from pathlib import Path

from tau2.runner import load_task_splits, load_tasks

from trace_to_micro.config import ExperimentConfig
from trace_to_micro.evaluation import build_support_report
from trace_to_micro.io import write_events_jsonl, write_json
from trace_to_micro.replay import replay_reference_tasks
from trace_to_micro.task_inventory import build_task_inventory


def _load_benchmark(config: ExperimentConfig):
    tasks = load_tasks(config.task_set, task_split_name=None)
    split_map = load_task_splits(config.task_set)
    tasks_by_id = {task.id: task for task in tasks}
    return tasks, split_map, tasks_by_id


def _resolve_output_dir(
    config: ExperimentConfig, output_dir: Path | None
) -> Path:
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trace-to-micro")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("task-audit", "oracle-preflight"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run a configured pre-experiment command."""

    args = _parser().parse_args(argv)
    if args.command == "task-audit":
        path = run_task_audit(args.config, args.output_dir)
        print(path)
        return
    paths = run_oracle_preflight(args.config, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
