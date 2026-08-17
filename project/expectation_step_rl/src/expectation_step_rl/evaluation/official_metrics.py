"""Aggregate complete frozen-checkpoint trajectories with tau2 metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from expectation_step_rl.evaluation.tracking import update_run_status
from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import compute_metrics


def summarize_results(results: Results) -> dict[str, Any]:
    """Return the official task-aggregated reward and pass^k metrics."""

    metrics = compute_metrics(results)
    if metrics.infra_error_count:
        raise ValueError(
            f"Official result contains {metrics.infra_error_count} infrastructure errors"
        )
    if 1 not in metrics.pass_hat_ks:
        raise ValueError("tau2 official metrics did not produce pass^1")
    terminations = Counter(
        simulation.termination_reason.value for simulation in results.simulations
    )
    return {
        "implementation": "tau2.metrics.agent_metrics.compute_metrics",
        "task_aggregation": "per-task pass^k, then mean across tasks",
        "total_tasks": metrics.total_tasks,
        "total_simulations": metrics.total_simulations,
        "configured_num_trials": results.info.num_trials,
        "average_reward": metrics.avg_reward,
        "pass^1": metrics.pass_hat_ks[1],
        "pass^k": {str(k): value for k, value in sorted(metrics.pass_hat_ks.items())},
        "termination_reasons": dict(sorted(terminations.items())),
    }


def build_report(evaluation_run_root: Path) -> dict[str, Any]:
    """Audit canonical inference outputs and build model/domain summaries."""

    trajectory_root = evaluation_run_root / "trajectories"
    audit_paths = sorted(trajectory_root.glob("*/*/*/inference_audit.json"))
    if not audit_paths:
        raise ValueError(f"No merged inference outputs found under {trajectory_root}")
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    expected_model_keys: list[str] | None = None
    expected_domain_splits: list[str] | None = None
    for audit_path in audit_paths:
        audit = json.loads(audit_path.read_text())
        if audit.get("complete") is not True:
            raise ValueError(f"Incomplete inference audit: {audit_path}")
        audit_model_keys = [str(value) for value in audit["expected_model_keys"]]
        audit_domain_splits = [
            f"{domain}/{audit['split']}" for domain in audit["expected_domains"]
        ]
        if expected_model_keys is None:
            expected_model_keys = audit_model_keys
            expected_domain_splits = audit_domain_splits
        elif (
            audit_model_keys != expected_model_keys
            or audit_domain_splits != expected_domain_splits
        ):
            raise ValueError(f"Inconsistent evaluation scope: {audit_path}")
        results_path = Path(audit["results_path"])
        results = Results.load(results_path)
        model_key = str(audit["model_key"])
        domain = str(audit["domain"])
        split = str(audit["split"])
        runs.setdefault(model_key, {})[f"{domain}/{split}"] = {
            **summarize_results(results),
            "full_split": bool(audit["full_split"]),
            "results_path": str(results_path),
            "adapter_path": audit.get("adapter_path"),
        }
    assert expected_model_keys is not None and expected_domain_splits is not None
    expected_runs = {
        (model_key, domain_split)
        for model_key in expected_model_keys
        for domain_split in expected_domain_splits
    }
    actual_runs = {
        (model_key, domain_split)
        for model_key, model_runs in runs.items()
        for domain_split in model_runs
    }
    if actual_runs != expected_runs:
        raise ValueError(
            f"Evaluation matrix incomplete: missing={sorted(expected_runs - actual_runs)}, "
            f"extra={sorted(actual_runs - expected_runs)}"
        )
    summaries = {}
    for model_key, model_runs in runs.items():
        by_split: dict[str, list[float]] = {}
        for domain_split, metrics in model_runs.items():
            split = domain_split.rsplit("/", 1)[1]
            by_split.setdefault(split, []).append(float(metrics["pass^1"]))
        summaries[model_key] = {
            split: {
                "domain_macro_average_pass^1": sum(values) / len(values),
                "num_domains": len(values),
            }
            for split, values in sorted(by_split.items())
        }
    comparisons = {}
    if "base" in runs:
        for model_key, model_runs in runs.items():
            if model_key == "base":
                continue
            comparisons[model_key] = {
                domain_split: {
                    "pass^1_delta_vs_base": metrics["pass^1"]
                    - runs["base"][domain_split]["pass^1"]
                }
                for domain_split, metrics in model_runs.items()
                if domain_split in runs["base"]
            }
    report = {
        "primary_metric": "pass^1",
        "official_implementation": "tau2.metrics.agent_metrics.compute_metrics",
        "evaluation_scope": {
            "model_keys": expected_model_keys,
            "domain_splits": expected_domain_splits,
        },
        "runs": runs,
        "domain_macro_summaries": summaries,
        "comparisons": comparisons,
    }
    manifest_path = evaluation_run_root / "experiment_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        report["experiment"] = {
            "run_tag": manifest["run_tag"],
            "training_run_id": manifest["training_run_id"],
            "checkpoint_steps": [
                checkpoint["step"] for checkpoint in manifest["checkpoints"]
            ],
            "baseline_id": manifest["baseline_id"],
            "protocol_sha256": manifest["protocol_sha256"],
            "evaluation_git_commit": manifest["evaluation_git_commit"],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.evaluation_run_root)
    output = args.output or args.evaluation_run_root / "official_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    update_run_status(args.evaluation_run_root, status="official_metrics_complete")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
