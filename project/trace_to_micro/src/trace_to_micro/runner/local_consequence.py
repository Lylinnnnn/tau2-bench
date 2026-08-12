"""Orchestrate the offline local-consequence hidden-state experiment."""

from __future__ import annotations

import os
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tau2.data_model.simulation import Results
from tau2.runner import load_task_splits
from trace_to_micro.analysis.logged_trace import (
    audit_results_completeness,
    tool_decision_records,
)
from trace_to_micro.config import LocalConsequenceConfig
from trace_to_micro.evaluation.local_consequence import (
    build_local_consequence_report,
)
from trace_to_micro.evaluation.success_direction import build_activation_smoke_report
from trace_to_micro.replay.consequence import (
    InvalidReferenceTargetError,
    replay_local_consequences,
    target_snapshot,
)
from trace_to_micro.runtime.activations import (
    activation_request_fingerprint,
    extract_request_activation,
)
from trace_to_micro.utils.io import (
    append_jsonl,
    read_jsonl,
    write_json,
    write_jsonl,
)


def _validate_shard(shard_index: int, num_shards: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards})")


def activation_requests_for_shard(
    requests: list[dict[str, Any]], shard_index: int, num_shards: int
) -> list[dict[str, Any]]:
    """Assign complete three-moment decision groups to one worker."""

    _validate_shard(shard_index, num_shards)
    decision_ids = sorted({row["decision_id"] for row in requests})
    assigned = {
        decision_id
        for position, decision_id in enumerate(decision_ids)
        if position % num_shards == shard_index
    }
    return [row for row in requests if row["decision_id"] in assigned]


def _exception_details(error: Exception) -> dict[str, Any]:
    return {
        "exception_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(traceback.format_exception(error)),
    }


def _trajectory_exclusion(
    *,
    domain: str,
    split: str,
    simulation,
    stage: str,
    reason: str,
    error: Exception | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "scope": "trajectory",
        "domain": domain,
        "split": split,
        "task_id": str(simulation.task_id),
        "simulation_id": simulation.id,
        "trial": simulation.trial,
        "stage": stage,
        "reason": reason,
        "excluded_trajectory_count": 1,
    }
    if error is not None:
        row.update(_exception_details(error))
    if details is not None:
        row["details"] = details
    return row


def _completeness_exclusions(
    *,
    domain: str,
    simulations: list,
    split_by_task: dict[str, str],
    audit: dict[str, Any],
) -> tuple[set[str], list[dict[str, Any]]]:
    reasons_by_id: dict[str, list[str]] = {}

    def mark(simulation_ids: list[str], reason: str) -> None:
        for simulation_id in simulation_ids:
            reasons_by_id.setdefault(simulation_id, []).append(reason)

    mark(audit["empty_simulation_ids"], "empty_trajectory")
    mark(audit["missing_reward_simulation_ids"], "missing_reward")
    mark(audit["infrastructure_error_simulation_ids"], "infrastructure_error")
    if audit["agent_model_mismatch"]:
        mark(
            [simulation.id for simulation in simulations],
            "agent_model_mismatch",
        )
    if audit["user_model_mismatch"]:
        mark(
            [simulation.id for simulation in simulations],
            "user_model_mismatch",
        )
    duplicate_keys = {tuple(value) for value in audit["duplicate_task_trials"]}
    unexpected_keys = {tuple(value) for value in audit["unexpected_task_trials"]}
    for simulation in simulations:
        key = (str(simulation.task_id), simulation.trial)
        if key in duplicate_keys:
            mark([simulation.id], "duplicate_task_trial")
        if key in unexpected_keys:
            mark([simulation.id], "unexpected_task_trial")
    by_id = {simulation.id: simulation for simulation in simulations}
    exclusions = []
    for simulation_id, reasons in sorted(reasons_by_id.items()):
        simulation = by_id[simulation_id]
        exclusions.append(
            _trajectory_exclusion(
                domain=domain,
                split=split_by_task.get(str(simulation.task_id), "outside_split"),
                simulation=simulation,
                stage="source_completeness",
                reason="trajectory_completeness_failed",
                details={"reasons": sorted(reasons)},
            )
        )
    for task_id, trial in audit["missing_task_trials"]:
        exclusions.append(
            {
                "scope": "task_trial",
                "domain": domain,
                "split": split_by_task.get(str(task_id), "outside_split"),
                "task_id": str(task_id),
                "trial": trial,
                "stage": "source_completeness",
                "reason": "missing_task_trial",
                "excluded_trajectory_count": 0,
                "excluded_trajectory_ids": [],
            }
        )
    return set(reasons_by_id), exclusions


def _excluded_trajectory_ids(exclusions: list[dict[str, Any]]) -> set[str]:
    return {
        simulation_id
        for row in exclusions
        for simulation_id in row.get("excluded_trajectory_ids", [])
    } | {
        row["simulation_id"]
        for row in exclusions
        if row.get("simulation_id") is not None
    }


def _request_validation_errors(
    consequences: list[dict[str, Any]], requests: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Return per-trajectory errors without accepting incomplete triples."""

    requests_by_decision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in requests:
        requests_by_decision[row["decision_id"]].append(row)
    errors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    consequence_by_decision = {row["decision_id"]: row for row in consequences}
    for decision_id, consequence in consequence_by_decision.items():
        decision_requests = requests_by_decision.get(decision_id, [])
        moments = [row.get("moment") for row in decision_requests]
        if len(decision_requests) != 3 or set(moments) != {
            "before",
            "action",
            "result",
        }:
            errors[consequence["simulation_id"]].append(
                {
                    "decision_id": decision_id,
                    "expected_moments": ["before", "action", "result"],
                    "observed_moments": moments,
                }
            )
    for decision_id, decision_requests in requests_by_decision.items():
        if decision_id not in consequence_by_decision:
            errors[decision_requests[0]["simulation_id"]].append(
                {
                    "decision_id": decision_id,
                    "reason": "request_without_consequence",
                }
            )
    duplicate_sample_ids = {
        sample_id
        for sample_id, count in Counter(row["sample_id"] for row in requests).items()
        if count > 1
    }
    for row in requests:
        if row["sample_id"] in duplicate_sample_ids:
            errors[row["simulation_id"]].append(
                {
                    "sample_id": row["sample_id"],
                    "reason": "duplicate_sample_id",
                }
            )
    return dict(errors)


def _domain_consequences(
    config: LocalConsequenceConfig,
    domain: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    results = Results.load(config.results_path(domain))
    tasks = {str(task.id): task for task in results.tasks}
    split_map = load_task_splits(domain)
    if split_map is None:
        raise ValueError(f"Domain {domain!r} has no train/test split")
    base_task_ids = set(split_map[config.train_split]) | set(
        split_map[config.test_split]
    )
    split_by_task = {
        str(task_id): split
        for split in (config.train_split, config.test_split)
        for task_id in split_map[split]
    }
    completeness = audit_results_completeness(
        config.results_path(domain),
        expected_task_ids=base_task_ids,
        expected_num_trials=config.expected_num_trials,
        expected_agent_model=config.expected_agent_model,
        expected_user_model=config.expected_user_model,
        raise_on_incomplete=False,
    )
    preexcluded_ids, exclusions = _completeness_exclusions(
        domain=domain,
        simulations=results.simulations,
        split_by_task=split_by_task,
        audit=completeness,
    )
    simulations_by_task = Counter(
        str(simulation.task_id) for simulation in results.simulations
    )
    simulation_ids_by_task: dict[str, list[str]] = {}
    for simulation in results.simulations:
        simulation_ids_by_task.setdefault(str(simulation.task_id), []).append(
            simulation.id
        )
    goals = {}
    for task_id in sorted(base_task_ids, key=str):
        task_id = str(task_id)
        if task_id not in tasks:
            exclusions.append(
                {
                    "scope": "task",
                    "domain": domain,
                    "split": split_by_task[task_id],
                    "task_id": task_id,
                    "stage": "target_snapshot",
                    "reason": "missing_task_definition",
                    "excluded_trajectory_count": simulations_by_task[task_id],
                    "excluded_trajectory_ids": simulation_ids_by_task.get(task_id, []),
                }
            )
            continue
        try:
            goals[task_id] = target_snapshot(domain, tasks[task_id])
        except Exception as error:
            exclusion = {
                "scope": "task",
                "domain": domain,
                "split": split_by_task[task_id],
                "task_id": task_id,
                "stage": "target_snapshot",
                "reason": "target_snapshot_failed",
                "excluded_trajectory_count": simulations_by_task[task_id],
                "excluded_trajectory_ids": simulation_ids_by_task.get(task_id, []),
                **_exception_details(error),
            }
            if isinstance(error, InvalidReferenceTargetError):
                exclusion.update(
                    {
                        "reference_action_id": error.action_id,
                        "reference_action_name": error.action_name,
                        "reference_action_requestor": error.requestor,
                        "reference_action_arguments": error.arguments,
                    }
                )
            exclusions.append(exclusion)
    rows = []
    for simulation in results.simulations:
        task_id = str(simulation.task_id)
        split = split_by_task.get(task_id)
        if split is None:
            continue
        if simulation.id in preexcluded_ids:
            continue
        if task_id not in goals:
            continue
        try:
            simulation_rows = replay_local_consequences(
                simulation=simulation,
                task=tasks[task_id],
                split=split,
                domain=domain,
                goal=goals[task_id],
            )
        except Exception as error:
            exclusions.append(
                _trajectory_exclusion(
                    domain=domain,
                    split=split,
                    simulation=simulation,
                    stage="trajectory_replay",
                    reason="trajectory_replay_failed",
                    error=error,
                )
            )
            continue
        rows.extend(simulation_rows)
    return rows, exclusions, completeness


def _support_report(
    rows: list[dict[str, Any]],
    *,
    exclusions: list[dict[str, Any]],
    completeness: dict[str, dict[str, Any]],
    domains: tuple[str, ...],
    splits: tuple[str, str],
) -> dict[str, Any]:
    report = {}
    for domain in domains:
        report[domain] = {}
        for split in splits:
            selected = [
                row for row in rows if row["domain"] == domain and row["split"] == split
            ]
            excluded = [
                row
                for row in exclusions
                if row["domain"] == domain and row["split"] == split
            ]
            tool_progress = {}
            for tool_name in sorted({row["tool_name"] for row in selected}):
                tool_rows = [row for row in selected if row["tool_name"] == tool_name]
                tool_progress[tool_name] = dict(
                    Counter(str(row["goal_progress"]).lower() for row in tool_rows)
                )
            report[domain][split] = {
                "decision_count": len(selected),
                "trajectory_count": len({row["simulation_id"] for row in selected}),
                "tool_success": dict(
                    Counter(str(row["tool_success"]).lower() for row in selected)
                ),
                "state_changed": dict(
                    Counter(str(row["state_changed"]).lower() for row in selected)
                ),
                "goal_progress": dict(
                    Counter(
                        str(row["goal_progress"]).lower()
                        for row in selected
                        if row["goal_progress"] is not None
                    )
                ),
                "goal_progress_eligible_count": sum(
                    row["goal_progress"] is not None for row in selected
                ),
                "goal_progress_by_tool": tool_progress,
                "within_tool_eligible_tools": sorted(
                    tool_name
                    for tool_name, counts in tool_progress.items()
                    if counts.get("true", 0) and counts.get("false", 0)
                ),
                "exclusion_count": len(excluded),
                "excluded_task_count": len({row["task_id"] for row in excluded}),
                "excluded_trajectory_count": sum(
                    1 for _ in _excluded_trajectory_ids(excluded)
                ),
                "excluded_task_ids": sorted({row["task_id"] for row in excluded}),
            }
    return {
        "audit_completed": True,
        "label_provenance": {
            "tool_success": "recorded result checked against deterministic replay",
            "state_changed": "leaf diff between replayed s_t and s_{t+1}",
            "goal_progress": (
                "for mutating tools only: distance(s_{t+1}, official target) "
                "< distance(s_t, official target)"
            ),
            "invalid_reference_target": (
                "exclude the entire task when target construction fails; never "
                "compile labels against a partial target"
            ),
            "invalid_trajectory": (
                "exclude the entire trajectory when completeness, replay, or "
                "request construction fails; never retain partial rows"
            ),
        },
        "exclusion_summary": {
            "issue_count": len(exclusions),
            "task_count": len({(row["domain"], row["task_id"]) for row in exclusions}),
            "trajectory_count": len(_excluded_trajectory_ids(exclusions)),
            "by_stage": dict(Counter(row["stage"] for row in exclusions)),
            "by_reason": dict(Counter(row["reason"] for row in exclusions)),
        },
        "source_completeness": completeness,
        "exclusions": exclusions,
        "domains": report,
    }


def build_local_consequence_requests(config_path: Path) -> Path:
    """Compile every logged assistant tool call into three factual moments."""

    config = LocalConsequenceConfig.load(config_path)
    consequences = []
    exclusions = []
    completeness = {}
    for domain in config.domains:
        domain_rows, domain_exclusions, domain_completeness = _domain_consequences(
            config, domain
        )
        consequences.extend(row for row in domain_rows if row["goal_progress_eligible"])
        exclusions.extend(domain_exclusions)
        completeness[domain] = domain_completeness
    duplicate_decision_ids = {
        decision_id
        for decision_id, count in Counter(
            row["decision_id"] for row in consequences
        ).items()
        if count > 1
    }
    if duplicate_decision_ids:
        duplicate_simulations = {
            row["simulation_id"]
            for row in consequences
            if row["decision_id"] in duplicate_decision_ids
        }
        first_by_simulation = {
            row["simulation_id"]: row
            for row in consequences
            if row["simulation_id"] in duplicate_simulations
        }
        for simulation_id in sorted(duplicate_simulations):
            row = first_by_simulation[simulation_id]
            exclusions.append(
                {
                    "scope": "trajectory",
                    "domain": row["domain"],
                    "split": row["split"],
                    "task_id": row["task_id"],
                    "simulation_id": simulation_id,
                    "trial": row["trial"],
                    "stage": "consequence_index",
                    "reason": "duplicate_decision_id",
                    "excluded_trajectory_count": 1,
                    "details": {
                        "decision_ids": sorted(
                            decision_id
                            for decision_id in duplicate_decision_ids
                            if decision_id.startswith(f"{simulation_id}:")
                        )
                    },
                }
            )
        consequences = [
            row
            for row in consequences
            if row["simulation_id"] not in duplicate_simulations
        ]
    by_decision = {row["decision_id"]: row for row in consequences}
    requests = []
    request_failed_simulations = set()
    for domain in config.domains:
        domain_ids = {
            row["decision_id"] for row in consequences if row["domain"] == domain
        }
        split_map = load_task_splits(domain)
        if split_map is None:
            raise ValueError(f"Domain {domain!r} has no train/test split")
        request_split_by_task = {
            str(task_id): split
            for split in (config.train_split, config.test_split)
            for task_id in split_map[split]
        }

        def record_request_error(simulation, error: Exception) -> None:
            request_failed_simulations.add(simulation.id)
            exclusions.append(
                _trajectory_exclusion(
                    domain=domain,
                    split=request_split_by_task.get(
                        str(simulation.task_id), "outside_split"
                    ),
                    simulation=simulation,
                    stage="activation_request_construction",
                    reason="activation_request_construction_failed",
                    error=error,
                )
            )

        requests.extend(
            tool_decision_records(
                config.results_path(domain),
                domain=domain,
                task_set=domain,
                train_split=config.train_split,
                test_split=config.test_split,
                max_per_trajectory=None,
                consequences=by_decision,
                include_decision_ids=domain_ids,
                on_simulation_error=record_request_error,
            )
        )
    if request_failed_simulations:
        consequences = [
            row
            for row in consequences
            if row["simulation_id"] not in request_failed_simulations
        ]
    fingerprinted = []
    for simulation_id in sorted({row["simulation_id"] for row in requests}):
        simulation_requests = [
            row for row in requests if row["simulation_id"] == simulation_id
        ]
        try:
            for row in simulation_requests:
                row["request_fingerprint"] = activation_request_fingerprint(
                    row,
                    model=config.hidden_model,
                    layer_ids=config.hidden_layer_ids,
                )
        except Exception as error:
            first = simulation_requests[0]
            exclusions.append(
                {
                    "scope": "trajectory",
                    "domain": first["domain"],
                    "split": first["split"],
                    "task_id": first["task_id"],
                    "simulation_id": simulation_id,
                    "trial": first["trial"],
                    "stage": "request_fingerprint",
                    "reason": "request_fingerprint_failed",
                    "excluded_trajectory_count": 1,
                    **_exception_details(error),
                }
            )
            request_failed_simulations.add(simulation_id)
            continue
        fingerprinted.extend(simulation_requests)
    requests = fingerprinted
    if request_failed_simulations:
        consequences = [
            row
            for row in consequences
            if row["simulation_id"] not in request_failed_simulations
        ]
    validation_errors = _request_validation_errors(consequences, requests)
    if validation_errors:
        first_by_simulation = {
            row["simulation_id"]: row
            for row in consequences
            if row["simulation_id"] in validation_errors
        }
        for simulation_id, errors in sorted(validation_errors.items()):
            first = first_by_simulation.get(simulation_id)
            if first is None:
                first = next(
                    row for row in requests if row["simulation_id"] == simulation_id
                )
            exclusions.append(
                {
                    "scope": "trajectory",
                    "domain": first["domain"],
                    "split": first["split"],
                    "task_id": first["task_id"],
                    "simulation_id": simulation_id,
                    "trial": first["trial"],
                    "stage": "request_validation",
                    "reason": "incomplete_or_duplicate_activation_requests",
                    "excluded_trajectory_count": 1,
                    "details": {"errors": errors},
                }
            )
        consequences = [
            row for row in consequences if row["simulation_id"] not in validation_errors
        ]
        requests = [
            row for row in requests if row["simulation_id"] not in validation_errors
        ]
    write_jsonl(config.output_dir / "local_consequences.jsonl", consequences)
    write_json(
        config.output_dir / "local_consequence_support.json",
        _support_report(
            consequences,
            exclusions=exclusions,
            completeness=completeness,
            domains=config.domains,
            splits=(config.train_split, config.test_split),
        ),
    )
    write_jsonl(config.output_dir / "local_consequence_exclusions.jsonl", exclusions)
    path = config.output_dir / "activation_requests.jsonl"
    write_jsonl(path, requests)
    return path


def build_local_consequence_smoke_requests(config_path: Path) -> Path:
    """Build one real mutating-decision triple without calling the model."""

    config = LocalConsequenceConfig.load(config_path)
    rows, _, _ = _domain_consequences(config, config.smoke_domain)
    selected = next(
        row
        for row in rows
        if row["task_id"] == config.smoke_task_id and row["declared_mutating"]
    )
    consequence = {selected["decision_id"]: selected}
    requests = tool_decision_records(
        config.results_path(config.smoke_domain),
        domain=config.smoke_domain,
        task_set=config.smoke_domain,
        train_split=config.train_split,
        test_split=config.test_split,
        max_per_trajectory=None,
        consequences=consequence,
        include_decision_ids={selected["decision_id"]},
    )
    if len(requests) != 3:
        raise ValueError(
            f"Local consequence smoke requires 3 moments, got {len(requests)}"
        )
    for row in requests:
        row["request_fingerprint"] = activation_request_fingerprint(
            row,
            model=config.hidden_model,
            layer_ids=config.hidden_layer_ids,
        )
    path = config.smoke_output_dir() / "activation_requests.jsonl"
    write_jsonl(path, requests)
    write_jsonl(config.smoke_output_dir() / "activations.jsonl", [])
    report_path = config.smoke_output_dir() / "smoke_report.json"
    if report_path.exists():
        report_path.unlink()
    return path


def _extract_requests(
    requests: list[dict[str, Any]], output_path: Path, *, base_url: str, config
) -> Path:
    completed = {
        (row["sample_id"], row.get("request_fingerprint"))
        for row in read_jsonl(output_path)
    }
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be non-empty")
    for request in requests:
        key = (request["sample_id"], request["request_fingerprint"])
        if key in completed:
            continue
        append_jsonl(
            output_path,
            extract_request_activation(
                request,
                base_url=base_url,
                api_key=api_key,
                model=config.hidden_model,
                layer_ids=config.hidden_layer_ids,
            ),
        )
    return output_path


def run_local_consequence_smoke(config_path: Path, *, base_url: str) -> Path:
    """Extract and strictly validate one real local-consequence triple."""

    config = LocalConsequenceConfig.load(config_path)
    request_path = config.smoke_output_dir() / "activation_requests.jsonl"
    if not request_path.is_file():
        request_path = build_local_consequence_smoke_requests(config_path)
    requests = read_jsonl(request_path)
    output = config.smoke_output_dir() / "activations.jsonl"
    write_jsonl(output, [])
    _extract_requests(requests, output, base_url=base_url, config=config)
    activations = read_jsonl(output)
    report = {
        **build_activation_smoke_report(
            requests,
            activations,
            layer_ids=config.hidden_layer_ids,
            hidden_size=config.hidden_size,
            hidden_model=config.hidden_model,
        ),
        "pipeline_valid": True,
        "decision_id": requests[0]["decision_id"],
        "domain": requests[0]["domain"],
        "task_id": requests[0]["task_id"],
        "tool_name": requests[0]["tool_name"],
        "labels": {
            key: requests[0][key]
            for key in ("tool_success", "state_changed", "goal_progress")
        },
        "moments": [row["moment"] for row in activations],
        "prompt_tokens": {row["moment"]: row["prompt_tokens"] for row in activations},
    }
    path = config.smoke_output_dir() / "smoke_report.json"
    write_json(path, report)
    return path


def run_local_consequence_activation_shard(
    config_path: Path,
    *,
    shard_index: int,
    num_shards: int,
    base_url: str,
) -> Path:
    """Export one disjoint activation shard from prebuilt requests."""

    config = LocalConsequenceConfig.load(config_path)
    request_path = config.output_dir / "activation_requests.jsonl"
    if not request_path.is_file():
        raise FileNotFoundError("Build local consequence requests first")
    requests = activation_requests_for_shard(
        read_jsonl(request_path), shard_index, num_shards
    )
    if not requests:
        raise ValueError(f"Local consequence activation shard {shard_index} is empty")
    output = config.activation_shard_path(shard_index, num_shards)
    current = {
        (request["sample_id"], request["request_fingerprint"]) for request in requests
    }
    retained = [
        row
        for row in read_jsonl(output)
        if (row["sample_id"], row.get("request_fingerprint")) in current
    ]
    write_jsonl(output, retained)
    return _extract_requests(requests, output, base_url=base_url, config=config)


def merge_local_consequence_activations(config_path: Path, *, num_shards: int) -> Path:
    """Strictly merge local-consequence activation shards."""

    config = LocalConsequenceConfig.load(config_path)
    requests = read_jsonl(config.output_dir / "activation_requests.jsonl")
    order = {
        (row["sample_id"], row["request_fingerprint"]): position
        for position, row in enumerate(requests)
    }
    rows = [
        row
        for shard_index in range(num_shards)
        for row in read_jsonl(config.activation_shard_path(shard_index, num_shards))
    ]
    keys = [(row["sample_id"], row.get("request_fingerprint")) for row in rows]
    if len(set(keys)) != len(keys) or set(keys) != set(order):
        raise ValueError("Local consequence activation shards are incomplete or stale")
    rows.sort(key=lambda row: order[(row["sample_id"], row["request_fingerprint"])])
    path = config.output_dir / "activations.jsonl"
    write_jsonl(path, rows)
    return path


def run_local_consequence_evaluation(config_path: Path) -> Path:
    """Evaluate held-out local-consequence directions over complete exports."""

    config = LocalConsequenceConfig.load(config_path)
    requests = read_jsonl(config.output_dir / "activation_requests.jsonl")
    rows = read_jsonl(config.output_dir / "activations.jsonl")
    expected = {(row["sample_id"], row["request_fingerprint"]) for row in requests}
    observed = {(row["sample_id"], row.get("request_fingerprint")) for row in rows}
    if len(rows) != len(requests) or observed != expected:
        raise ValueError(
            "Local consequence activation extraction is incomplete or stale"
        )
    report = build_local_consequence_report(
        rows,
        domains=config.domains,
        train_split=config.train_split,
        test_split=config.test_split,
        layer_ids=config.hidden_layer_ids,
        primary_layer_id=config.primary_layer_id,
        primary_moment=config.primary_moment,
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed,
    )
    path = config.output_dir / "local_consequence_report.json"
    write_json(path, report)
    return path
