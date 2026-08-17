"""CLI for evaluation manifests, canonical baselines, and the run registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expectation_step_rl.evaluation.baseline import (
    baseline_is_ready,
    materialize_baseline,
    publish_baseline,
)
from expectation_step_rl.evaluation.tracking import (
    prepare_run_manifest,
    rebuild_registry,
    update_run_status,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-run")
    prepare.add_argument("--repo-root", type=Path, required=True)
    prepare.add_argument("--evaluation-root", type=Path, required=True)
    prepare.add_argument("--run-tag", required=True)
    prepare.add_argument("--checkpoint-root", type=Path, required=True)
    prepare.add_argument("--checkpoint-steps")
    prepare.add_argument("--source-model-path", type=Path, required=True)
    prepare.add_argument("--base-model-name", required=True)
    prepare.add_argument("--domains", required=True)
    prepare.add_argument("--split", required=True)
    prepare.add_argument("--num-trials", type=int, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--max-tasks-per-domain", type=int, required=True)
    prepare.add_argument("--include-baseline", type=int, choices=(0, 1), required=True)
    prepare.add_argument(
        "--local-staging-enabled", type=int, choices=(0, 1), required=True
    )

    ready = commands.add_parser("baseline-ready")
    ready.add_argument("--run-root", type=Path, required=True)
    ready.add_argument("--baseline-root", type=Path, required=True)

    for command in ("publish-baseline", "materialize-baseline"):
        baseline = commands.add_parser(command)
        baseline.add_argument("--run-root", type=Path, required=True)
        baseline.add_argument("--baseline-root", type=Path, required=True)
        baseline.add_argument("--domain", required=True)

    status = commands.add_parser("set-status")
    status.add_argument("--run-root", type=Path, required=True)
    status.add_argument("--status", required=True)
    status.add_argument("--error")

    steps = commands.add_parser("manifest-steps")
    steps.add_argument("--run-root", type=Path, required=True)

    registry = commands.add_parser("rebuild-registry")
    registry.add_argument("--evaluation-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-run":
        manifest = prepare_run_manifest(
            repo_root=args.repo_root,
            evaluation_root=args.evaluation_root,
            run_tag=args.run_tag,
            checkpoint_root=args.checkpoint_root,
            checkpoint_steps=args.checkpoint_steps,
            source_model_path=args.source_model_path,
            base_model_name=args.base_model_name,
            domains=args.domains.split(","),
            split=args.split,
            num_trials=args.num_trials,
            seed=args.seed,
            max_tasks_per_domain=args.max_tasks_per_domain,
            include_baseline=bool(args.include_baseline),
            local_staging_enabled=bool(args.local_staging_enabled),
        )
        print(manifest["baseline_id"])
    elif args.command == "baseline-ready":
        manifest = json.loads((args.run_root / "experiment_manifest.json").read_text())
        if not baseline_is_ready(args.baseline_root, manifest):
            raise SystemExit(1)
        print(manifest["baseline_id"])
    elif args.command == "publish-baseline":
        print(
            publish_baseline(
                run_root=args.run_root,
                baseline_root=args.baseline_root,
                domain=args.domain,
            )
        )
    elif args.command == "materialize-baseline":
        print(
            materialize_baseline(
                run_root=args.run_root,
                baseline_root=args.baseline_root,
                domain=args.domain,
            )
        )
    elif args.command == "set-status":
        manifest = update_run_status(
            args.run_root, status=args.status, error=args.error
        )
        print(
            json.dumps({"run_tag": manifest["run_tag"], "status": manifest["status"]})
        )
    elif args.command == "manifest-steps":
        manifest = json.loads((args.run_root / "experiment_manifest.json").read_text())
        print(",".join(str(value["step"]) for value in manifest["checkpoints"]))
    else:
        registry = rebuild_registry(args.evaluation_root)
        print(json.dumps({"runs": len(registry["runs"])}))


if __name__ == "__main__":
    main()
