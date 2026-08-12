"""Orchestrate the offline local-consequence hidden-state experiment."""

from __future__ import annotations

import os
from collections import Counter
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
from trace_to_micro.replay.consequence import replay_local_consequences
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


def _domain_consequences(
    config: LocalConsequenceConfig,
    domain: str,
) -> list[dict[str, Any]]:
    results = Results.load(config.results_path(domain))
    tasks = {str(task.id): task for task in results.tasks}
    split_map = load_task_splits(domain)
    if split_map is None:
        raise ValueError(f"Domain {domain!r} has no train/test split")
    base_task_ids = set(split_map[config.train_split]) | set(
        split_map[config.test_split]
    )
    audit_results_completeness(
        config.results_path(domain),
        expected_task_ids=base_task_ids,
        expected_num_trials=config.expected_num_trials,
        expected_agent_model=config.expected_agent_model,
        expected_user_model=config.expected_user_model,
    )
    split_by_task = {
        str(task_id): split
        for split in (config.train_split, config.test_split)
        for task_id in split_map[split]
    }
    rows = []
    for simulation in results.simulations:
        task_id = str(simulation.task_id)
        split = split_by_task.get(task_id)
        if split is None:
            continue
        rows.extend(
            replay_local_consequences(
                simulation=simulation,
                task=tasks[task_id],
                split=split,
                domain=domain,
            )
        )
    return rows


def _support_report(
    rows: list[dict[str, Any]], *, splits: tuple[str, str]
) -> dict[str, Any]:
    report = {}
    for domain in sorted({row["domain"] for row in rows}):
        report[domain] = {}
        for split in splits:
            selected = [
                row for row in rows if row["domain"] == domain and row["split"] == split
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
            }
    return {
        "label_provenance": {
            "tool_success": "recorded result checked against deterministic replay",
            "state_changed": "leaf diff between replayed s_t and s_{t+1}",
            "goal_progress": (
                "for mutating tools only: distance(s_{t+1}, official target) "
                "< distance(s_t, official target)"
            ),
        },
        "domains": report,
    }


def build_local_consequence_requests(config_path: Path) -> Path:
    """Compile every logged assistant tool call into three factual moments."""

    config = LocalConsequenceConfig.load(config_path)
    consequences = [
        row
        for domain in config.domains
        for row in _domain_consequences(config, domain)
        if row["goal_progress_eligible"]
    ]
    by_decision = {row["decision_id"]: row for row in consequences}
    if len(by_decision) != len(consequences):
        raise ValueError("Local consequence decision ids must be unique")
    requests = []
    for domain in config.domains:
        domain_ids = {
            row["decision_id"] for row in consequences if row["domain"] == domain
        }
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
            )
        )
    for row in requests:
        row["request_fingerprint"] = activation_request_fingerprint(
            row,
            model=config.hidden_model,
            layer_ids=config.hidden_layer_ids,
        )
    expected = len(consequences) * 3
    if len(requests) != expected:
        raise ValueError(f"Expected {expected} moment rows, got {len(requests)}")
    write_jsonl(config.output_dir / "local_consequences.jsonl", consequences)
    write_json(
        config.output_dir / "local_consequence_support.json",
        _support_report(consequences, splits=(config.train_split, config.test_split)),
    )
    path = config.output_dir / "activation_requests.jsonl"
    write_jsonl(path, requests)
    return path


def build_local_consequence_smoke_requests(config_path: Path) -> Path:
    """Build one real mutating-decision triple without calling the model."""

    config = LocalConsequenceConfig.load(config_path)
    rows = _domain_consequences(config, config.smoke_domain)
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
