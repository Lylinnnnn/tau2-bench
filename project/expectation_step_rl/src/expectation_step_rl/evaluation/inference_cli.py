"""Command-line entry point for frozen-checkpoint tau2 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expectation_step_rl.evaluation.checkpoints import (
    discover_checkpoints,
    parse_steps,
)
from expectation_step_rl.evaluation.inference import merge_shards, run_shard


def _add_shared_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--domain", choices=("airline", "retail", "telecom"), required=True
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list-checkpoints")
    list_parser.add_argument("--checkpoint-root", type=Path, required=True)
    list_parser.add_argument("--steps")
    list_parser.add_argument("--format", choices=("json", "tsv"), default="json")

    shard_parser = subparsers.add_parser("run-shard")
    _add_shared_run_arguments(shard_parser)
    shard_parser.add_argument("--shard-index", type=int, required=True)
    shard_parser.add_argument("--api-base", required=True)
    shard_parser.add_argument("--max-concurrency", type=int, default=2)
    shard_parser.add_argument("--task-timeout", type=float, default=1800.0)

    merge_parser = subparsers.add_parser("merge")
    _add_shared_run_arguments(merge_parser)
    merge_parser.add_argument("--adapter-path")
    merge_parser.add_argument("--expected-model-keys", required=True)
    merge_parser.add_argument("--expected-domains", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list-checkpoints":
        checkpoints = discover_checkpoints(
            args.checkpoint_root,
            selected_steps=parse_steps(args.steps),
        )
        if args.format == "tsv":
            for checkpoint in checkpoints:
                print(
                    f"{checkpoint.step}\t{checkpoint.key}\t"
                    f"{checkpoint.served_model_name}\t{checkpoint.path}"
                )
            return
        print(
            json.dumps(
                [
                    {
                        "step": checkpoint.step,
                        "key": checkpoint.key,
                        "served_model_name": checkpoint.served_model_name,
                        "path": str(checkpoint.path),
                    }
                    for checkpoint in checkpoints
                ],
                indent=2,
            )
        )
    elif args.command == "run-shard":
        run_shard(args)
    else:
        merge_shards(args)


if __name__ == "__main__":
    main()
