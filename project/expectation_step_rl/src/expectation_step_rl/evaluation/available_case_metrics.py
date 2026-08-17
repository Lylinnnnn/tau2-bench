"""Compute diagnostic tau2 metrics after excluding infrastructure failures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from expectation_step_rl.evaluation.inference import result_name
from tau2.data_model.simulation import Results, TerminationReason
from tau2.metrics.agent_metrics import compute_metrics


def summarize_available_cases(results: Results) -> dict[str, Any]:
    """Use tau2 metrics on runnable simulations and expose the reduced denominator."""

    infrastructure = [
        simulation
        for simulation in results.simulations
        if simulation.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    ]
    missing_non_infrastructure = [
        simulation.id
        for simulation in results.simulations
        if simulation.reward_info is None
        and simulation.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR
    ]
    if missing_non_infrastructure:
        raise ValueError(
            "Missing rewards outside infrastructure failures: "
            f"{sorted(missing_non_infrastructure)}"
        )

    metrics = compute_metrics(results)
    if 1 not in metrics.pass_hat_ks:
        raise ValueError("tau2 metrics did not produce pass^1")
    expected_task_ids = {str(task.id) for task in results.tasks}
    evaluated_task_ids = {
        str(simulation.task_id)
        for simulation in results.simulations
        if simulation.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR
    }
    return {
        "status": "provisional_available_case",
        "publication_ready": False,
        "implementation": "tau2.metrics.agent_metrics.compute_metrics",
        "exclusion_rule": "termination_reason == infrastructure_error",
        "configured_num_trials": results.info.num_trials,
        "expected_task_count": len(expected_task_ids),
        "evaluated_task_count": len(evaluated_task_ids),
        "observed_simulation_count": len(results.simulations),
        "evaluated_simulation_count": metrics.total_simulations,
        "excluded_infrastructure_count": len(infrastructure),
        "excluded_simulation_ids": sorted(
            simulation.id for simulation in infrastructure
        ),
        "excluded_task_ids": sorted(
            {str(simulation.task_id) for simulation in infrastructure}
        ),
        "average_reward": metrics.avg_reward,
        "pass^1": metrics.pass_hat_ks[1],
        "pass^k": {str(k): value for k, value in sorted(metrics.pass_hat_ks.items())},
        "termination_reasons_before_exclusion": dict(
            sorted(
                Counter(
                    simulation.termination_reason.value
                    for simulation in results.simulations
                ).items()
            )
        ),
    }


def _result_path(
    *,
    repo_root: Path,
    evaluation_root: Path,
    run_tag: str,
    model_key: str,
    domain: str,
    split: str,
    num_trials: int,
    seed: int,
) -> Path:
    canonical = (
        evaluation_root
        / run_tag
        / "trajectories"
        / model_key
        / domain
        / split
        / "results.json"
    )
    if canonical.is_file():
        return canonical
    raw_name = result_name(
        run_tag=run_tag,
        model_key=model_key,
        domain=domain,
        split=split,
        num_trials=num_trials,
        seed=seed,
        shard_index=0,
        num_shards=1,
    )
    raw = repo_root / "data" / "simulations" / raw_name / "results.json"
    if raw.is_file():
        return raw
    raise FileNotFoundError(
        f"No canonical or raw trajectory found for {model_key}/{domain}: "
        f"canonical={canonical}, raw={raw}"
    )


def build_report(
    *,
    repo_root: Path,
    evaluation_root: Path,
    run_tag: str,
    model_keys: list[str],
    domains: list[str],
    split: str,
    num_trials: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize the requested model/domain matrix without mutating trajectories."""

    runs: dict[str, dict[str, Any]] = {}
    for model_key in model_keys:
        runs[model_key] = {}
        for domain in domains:
            path = _result_path(
                repo_root=repo_root,
                evaluation_root=evaluation_root,
                run_tag=run_tag,
                model_key=model_key,
                domain=domain,
                split=split,
                num_trials=num_trials,
                seed=seed,
            )
            runs[model_key][domain] = {
                **summarize_available_cases(Results.load(path)),
                "results_path": str(path),
            }

    domain_macro = {
        model_key: sum(float(value["pass^1"]) for value in domain_runs.values())
        / len(domain_runs)
        for model_key, domain_runs in runs.items()
    }
    return {
        "status": "provisional_available_case",
        "publication_ready": False,
        "warning": (
            "Infrastructure failures were excluded. These values are diagnostic and "
            "are not complete official benchmark results."
        ),
        "run_tag": run_tag,
        "split": split,
        "models": model_keys,
        "domains": domains,
        "runs": runs,
        "domain_macro_average_pass^1": domain_macro,
    }


def _csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--model-keys", default="base,step_10,step_30")
    parser.add_argument("--domains", default="airline,retail")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(
        repo_root=args.repo_root.resolve(),
        evaluation_root=args.evaluation_root.resolve(),
        run_tag=args.run_tag,
        model_keys=_csv(args.model_keys),
        domains=_csv(args.domains),
        split=args.split,
        num_trials=args.num_trials,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
