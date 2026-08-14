"""Command-line orchestration for deterministic pre-experiments."""

import argparse
import sys
from pathlib import Path

from tau2.runner import load_task_splits, load_tasks
from trace_to_micro.analysis.results import build_results_audit
from trace_to_micro.analysis.task_inventory import build_task_inventory
from trace_to_micro.config import ExperimentConfig
from trace_to_micro.evaluation import build_support_report
from trace_to_micro.replay.reference import replay_reference_tasks
from trace_to_micro.runner.consequence_expectation import (
    run_consequence_expectation_evaluation,
)
from trace_to_micro.runner.expectation_deviation import (
    build_expectation_deviation_dataset,
    merge_contextual_min_k_scores,
    merge_expectation_deviation_scores,
    run_contextual_min_k_evaluation,
    run_contextual_min_k_score_shard,
    run_contextual_min_k_smoke,
    run_expectation_deviation_evaluation,
    run_expectation_deviation_score_shard,
    run_expectation_deviation_smoke,
)
from trace_to_micro.runner.expectation_matching import (
    build_expectation_matching_dataset,
    merge_expectation_matching_scores,
    run_expectation_matching_evaluation,
    run_expectation_matching_score_shard,
    run_expectation_matching_smoke,
)
from trace_to_micro.runner.local_consequence import (
    build_local_consequence_requests,
    build_local_consequence_smoke_requests,
    merge_local_consequence_activations,
    run_local_consequence_activation_shard,
    run_local_consequence_evaluation,
    run_local_consequence_smoke,
)
from trace_to_micro.runner.model_experiment import (
    run_full_model_preexperiment,
    run_logged_snapshot_extraction,
    run_logged_transition_audit,
    run_model_probe,
    run_probe_summary,
    run_trajectory_analysis,
    run_trajectory_generation,
)
from trace_to_micro.runner.success_direction import (
    build_activation_requests,
    build_success_smoke_activation_requests,
    merge_activation_shards,
    merge_success_trajectory_shards,
    run_activation_extraction,
    run_activation_extraction_shard,
    run_success_direction_evaluation,
    run_success_official_metrics,
    run_success_smoke_activations,
    run_success_smoke_trajectory,
    run_success_trajectories,
    run_success_trajectory_shard,
)
from trace_to_micro.utils.io import write_events_jsonl, write_json


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
    audit_parser = subparsers.add_parser("audit-results")
    audit_parser.add_argument("--results", type=Path, required=True)
    audit_parser.add_argument("--domain", required=True)
    audit_parser.add_argument("--expected-task-ids", nargs="+", required=True)
    audit_parser.add_argument("--expected-agent-model", required=True)
    audit_parser.add_argument("--expected-user-model", required=True)
    audit_parser.add_argument("--expected-num-trials", type=int, default=1)
    audit_parser.add_argument("--output", type=Path)
    for command in (
        "extract-snapshots",
        "logged-audit",
        "analyze-trajectories",
        "paired-probe",
        "summarize-probe",
        "model-preexperiment",
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
    success_trajectories = subparsers.add_parser("success-trajectories")
    success_trajectories.add_argument("--config", type=Path, required=True)
    success_trajectories.add_argument("--force", action=argparse.BooleanOptionalAction)
    success_trajectory_shard = subparsers.add_parser("success-trajectory-shard")
    success_trajectory_shard.add_argument("--config", type=Path, required=True)
    success_trajectory_shard.add_argument("--shard-index", type=int, required=True)
    success_trajectory_shard.add_argument("--num-shards", type=int, required=True)
    success_trajectory_shard.add_argument("--base-url", required=True)
    success_trajectory_shard.add_argument(
        "--force", action=argparse.BooleanOptionalAction
    )
    success_merge_trajectories = subparsers.add_parser("success-merge-trajectories")
    success_merge_trajectories.add_argument("--config", type=Path, required=True)
    success_merge_trajectories.add_argument("--num-shards", type=int, required=True)
    success_activation_shard = subparsers.add_parser("success-activation-shard")
    success_activation_shard.add_argument("--config", type=Path, required=True)
    success_activation_shard.add_argument("--shard-index", type=int, required=True)
    success_activation_shard.add_argument("--num-shards", type=int, required=True)
    success_activation_shard.add_argument("--base-url", required=True)
    local_activation_shard = subparsers.add_parser("local-consequence-activation-shard")
    local_activation_shard.add_argument("--config", type=Path, required=True)
    local_activation_shard.add_argument("--shard-index", type=int, required=True)
    local_activation_shard.add_argument("--num-shards", type=int, required=True)
    local_activation_shard.add_argument("--base-url", required=True)
    local_merge = subparsers.add_parser("local-consequence-merge-activations")
    local_merge.add_argument("--config", type=Path, required=True)
    local_merge.add_argument("--num-shards", type=int, required=True)
    local_smoke = subparsers.add_parser("local-consequence-smoke")
    local_smoke.add_argument("--config", type=Path, required=True)
    local_smoke.add_argument("--base-url", required=True)
    matching_score = subparsers.add_parser("expectation-matching-score-shard")
    matching_score.add_argument("--config", type=Path, required=True)
    matching_score.add_argument("--shard-index", type=int, required=True)
    matching_score.add_argument("--num-shards", type=int, required=True)
    matching_score.add_argument("--base-url", required=True)
    matching_merge = subparsers.add_parser("expectation-matching-merge")
    matching_merge.add_argument("--config", type=Path, required=True)
    matching_merge.add_argument("--num-shards", type=int, required=True)
    matching_smoke = subparsers.add_parser("expectation-matching-smoke")
    matching_smoke.add_argument("--config", type=Path, required=True)
    matching_smoke.add_argument("--base-url", required=True)
    deviation_score = subparsers.add_parser("expectation-deviation-score-shard")
    deviation_score.add_argument("--config", type=Path, required=True)
    deviation_score.add_argument("--shard-index", type=int, required=True)
    deviation_score.add_argument("--num-shards", type=int, required=True)
    deviation_score.add_argument("--base-url", required=True)
    deviation_merge = subparsers.add_parser("expectation-deviation-merge")
    deviation_merge.add_argument("--config", type=Path, required=True)
    deviation_merge.add_argument("--num-shards", type=int, required=True)
    deviation_smoke = subparsers.add_parser("expectation-deviation-smoke")
    deviation_smoke.add_argument("--config", type=Path, required=True)
    deviation_smoke.add_argument("--base-url", required=True)
    min_k_score = subparsers.add_parser("contextual-min-k-score-shard")
    min_k_score.add_argument("--config", type=Path, required=True)
    min_k_score.add_argument("--shard-index", type=int, required=True)
    min_k_score.add_argument("--num-shards", type=int, required=True)
    min_k_score.add_argument("--base-url", required=True)
    min_k_merge = subparsers.add_parser("contextual-min-k-merge")
    min_k_merge.add_argument("--config", type=Path, required=True)
    min_k_merge.add_argument("--num-shards", type=int, required=True)
    min_k_smoke = subparsers.add_parser("contextual-min-k-smoke")
    min_k_smoke.add_argument("--config", type=Path, required=True)
    min_k_smoke.add_argument("--base-url", required=True)
    success_merge_activations = subparsers.add_parser("success-merge-activations")
    success_merge_activations.add_argument("--config", type=Path, required=True)
    success_merge_activations.add_argument("--num-shards", type=int, required=True)
    for command in (
        "success-official-metrics",
        "success-activation-requests",
        "success-activations",
        "success-evaluate",
        "success-smoke-trajectories",
        "success-smoke-activation-requests",
        "success-smoke-activations",
        "local-consequence-requests",
        "local-consequence-smoke-requests",
        "local-consequence-evaluate",
        "consequence-expectation-evaluate",
        "expectation-matching-prepare",
        "expectation-matching-evaluate",
        "expectation-deviation-prepare",
        "expectation-deviation-evaluate",
        "contextual-min-k-evaluate",
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
    elif args.command == "audit-results":
        report = build_results_audit(
            args.results,
            domain=args.domain,
            expected_task_ids=set(args.expected_task_ids),
            expected_agent_model=args.expected_agent_model,
            expected_user_model=args.expected_user_model,
            expected_num_trials=args.expected_num_trials,
        )
        output = args.output or args.results.parent / "integrity_report.json"
        write_json(output, report)
        print(
            f"{output}: valid_for_analysis="
            f"{report['data_integrity']['valid_for_analysis']}, "
            f"mean_reward={report['outcomes']['mean_reward']}, "
            "invalid_tool_calls="
            f"{len(report['tool_behavior']['invalid_calls'])}"
        )
        return
    elif args.command == "extract-snapshots":
        paths = run_logged_snapshot_extraction(args.config)
    elif args.command == "logged-audit":
        paths = run_logged_transition_audit(args.config)
    elif args.command == "analyze-trajectories":
        paths = run_trajectory_analysis(args.config)
    elif args.command == "paired-probe":
        paths = run_model_probe(args.config)
    elif args.command == "summarize-probe":
        paths = run_probe_summary(args.config)
    elif args.command == "success-trajectories":
        run_success_trajectories(args.config, force=args.force)
        return
    elif args.command == "success-trajectory-shard":
        run_success_trajectory_shard(
            args.config,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            base_url=args.base_url,
            force=args.force,
        )
        return
    elif args.command == "success-merge-trajectories":
        merge_success_trajectory_shards(args.config, num_shards=args.num_shards)
        return
    elif args.command == "success-activation-requests":
        print(build_activation_requests(args.config))
        return
    elif args.command == "success-official-metrics":
        print(run_success_official_metrics(args.config))
        return
    elif args.command == "success-activations":
        print(run_activation_extraction(args.config))
        return
    elif args.command == "success-activation-shard":
        print(
            run_activation_extraction_shard(
                args.config,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                base_url=args.base_url,
            )
        )
        return
    elif args.command == "success-merge-activations":
        print(merge_activation_shards(args.config, num_shards=args.num_shards))
        return
    elif args.command == "local-consequence-requests":
        print(build_local_consequence_requests(args.config))
        return
    elif args.command == "local-consequence-smoke-requests":
        print(build_local_consequence_smoke_requests(args.config))
        return
    elif args.command == "local-consequence-smoke":
        print(run_local_consequence_smoke(args.config, base_url=args.base_url))
        return
    elif args.command == "local-consequence-activation-shard":
        print(
            run_local_consequence_activation_shard(
                args.config,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                base_url=args.base_url,
            )
        )
        return
    elif args.command == "local-consequence-merge-activations":
        print(
            merge_local_consequence_activations(args.config, num_shards=args.num_shards)
        )
        return
    elif args.command == "local-consequence-evaluate":
        print(run_local_consequence_evaluation(args.config))
        return
    elif args.command == "consequence-expectation-evaluate":
        paths = run_consequence_expectation_evaluation(args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return
    elif args.command == "expectation-matching-prepare":
        paths = build_expectation_matching_dataset(args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return
    elif args.command == "expectation-matching-score-shard":
        print(
            run_expectation_matching_score_shard(
                args.config,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                base_url=args.base_url,
            )
        )
        return
    elif args.command == "expectation-matching-merge":
        print(
            merge_expectation_matching_scores(args.config, num_shards=args.num_shards)
        )
        return
    elif args.command == "expectation-matching-smoke":
        paths = run_expectation_matching_smoke(args.config, base_url=args.base_url)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return
    elif args.command == "expectation-matching-evaluate":
        paths = run_expectation_matching_evaluation(args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return
    elif args.command == "expectation-deviation-prepare":
        paths = build_expectation_deviation_dataset(args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return
    elif args.command == "expectation-deviation-score-shard":
        print(
            run_expectation_deviation_score_shard(
                args.config,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                base_url=args.base_url,
            )
        )
        return
    elif args.command == "expectation-deviation-merge":
        print(
            merge_expectation_deviation_scores(args.config, num_shards=args.num_shards)
        )
        return
    elif args.command == "expectation-deviation-smoke":
        paths = run_expectation_deviation_smoke(args.config, base_url=args.base_url)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return
    elif args.command == "expectation-deviation-evaluate":
        paths = run_expectation_deviation_evaluation(args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return
    elif args.command == "contextual-min-k-score-shard":
        print(
            run_contextual_min_k_score_shard(
                args.config,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                base_url=args.base_url,
            )
        )
        return
    elif args.command == "contextual-min-k-merge":
        print(merge_contextual_min_k_scores(args.config, num_shards=args.num_shards))
        return
    elif args.command == "contextual-min-k-smoke":
        paths = run_contextual_min_k_smoke(args.config, base_url=args.base_url)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return
    elif args.command == "contextual-min-k-evaluate":
        paths = run_contextual_min_k_evaluation(args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return
    elif args.command == "success-evaluate":
        print(run_success_direction_evaluation(args.config))
        return
    elif args.command == "success-smoke-trajectories":
        paths = run_success_smoke_trajectory(args.config)
    elif args.command == "success-smoke-activation-requests":
        print(build_success_smoke_activation_requests(args.config))
        return
    elif args.command == "success-smoke-activations":
        paths = run_success_smoke_activations(args.config)
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
