"""Orchestration for the Airline/Retail success-direction experiment."""

from __future__ import annotations

import hashlib
import json
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
from trace_to_micro.config import SuccessDirectionConfig, TrajectoryConfig
from trace_to_micro.evaluation.success_direction import build_success_direction_report
from trace_to_micro.runner.trajectory import (
    remove_existing_run,
    run_complete_trajectories,
)
from trace_to_micro.runtime.activations import extract_request_activation
from trace_to_micro.utils.io import append_jsonl, read_jsonl, write_json, write_jsonl


def _trajectory_config(config: SuccessDirectionConfig, domain: str) -> TrajectoryConfig:
    return TrajectoryConfig(
        domain=domain,
        task_set=domain,
        task_split=config.task_split,
        agent=config.agent,
        user=config.user,
        agent_llm=config.agent_llm,
        user_llm=config.user_llm,
        agent_llm_args=config.agent_llm_args,
        user_llm_args=config.user_llm_args,
        num_trials=config.num_trials,
        max_steps=config.max_steps,
        max_errors=config.max_errors,
        max_concurrency=config.max_concurrency,
        seed=config.seed,
        timeout_seconds=config.timeout_seconds,
        save_to=config.save_name(domain),
    )


def _require_complete(config: SuccessDirectionConfig, domain: str) -> dict[str, Any]:
    split_map = load_task_splits(domain)
    if split_map is None:
        raise ValueError(f"Domain {domain!r} has no task split")
    return audit_results_completeness(
        config.results_path(domain),
        expected_task_ids=set(split_map[config.task_split]),
        expected_num_trials=config.num_trials,
        expected_agent_model=config.agent_llm,
        expected_user_model=config.user_llm,
    )


def run_success_trajectories(config_path: Path, *, force: bool | None = None) -> None:
    """Generate fresh official base trajectories for both configured domains."""

    config = SuccessDirectionConfig.load(config_path)
    overwrite = config.force_overwrite if force is None else force
    for domain in config.domains:
        if overwrite:
            remove_existing_run(config.results_path(domain))
        run_complete_trajectories(
            _trajectory_config(config, domain),
            auto_resume=not overwrite,
        )
        _require_complete(config, domain)


def _request_fingerprint(row: dict[str, Any], config: SuccessDirectionConfig) -> str:
    payload = {
        "request_record": row,
        "hidden_model": config.hidden_model,
        "hidden_layer_ids": config.hidden_layer_ids,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _request_coverage(
    config: SuccessDirectionConfig, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    selected = {(row["domain"], row["simulation_id"]) for row in rows}
    report: dict[str, Any] = {}
    for domain in config.domains:
        split_map = load_task_splits(domain)
        if split_map is None:
            raise ValueError(f"Domain {domain!r} has no task split")
        split_by_task = {
            str(task_id): split
            for split in (config.train_split, config.test_split)
            for task_id in split_map[split]
        }
        domain_report = {}
        for split in (config.train_split, config.test_split):
            simulations = [
                simulation
                for simulation in Results.iter_simulations(config.results_path(domain))
                if split_by_task.get(str(simulation.task_id)) == split
            ]
            covered = [
                simulation
                for simulation in simulations
                if (domain, simulation.id) in selected
            ]
            outcomes = Counter(
                "success"
                if simulation.reward_info is not None
                and simulation.reward_info.reward == 1.0
                else "failure"
                for simulation in simulations
            )
            covered_outcomes = Counter(
                "success"
                if simulation.reward_info is not None
                and simulation.reward_info.reward == 1.0
                else "failure"
                for simulation in covered
            )
            domain_report[split] = {
                "official_trajectories": len(simulations),
                "with_assistant_tool_decision": len(covered),
                "omitted_without_assistant_tool_decision": (
                    len(simulations) - len(covered)
                ),
                "official_outcomes": dict(sorted(outcomes.items())),
                "covered_outcomes": dict(sorted(covered_outcomes.items())),
            }
        report[domain] = domain_report
    return {
        "selection": "first assistant tool decision in each trajectory",
        "moments_per_decision": 3,
        "domains": report,
    }


def build_activation_requests(config_path: Path) -> Path:
    """Compile factual three-moment requests without calling the model."""

    config = SuccessDirectionConfig.load(config_path)
    rows = []
    completeness = {}
    for domain in config.domains:
        completeness[domain] = _require_complete(config, domain)
        rows.extend(
            tool_decision_records(
                config.results_path(domain),
                domain=domain,
                task_set=domain,
                train_split=config.train_split,
                test_split=config.test_split,
                max_per_trajectory=config.max_tool_decisions_per_trajectory,
            )
        )
    for row in rows:
        row["request_fingerprint"] = _request_fingerprint(row, config)
    request_path = config.output_dir / "activation_requests.jsonl"
    write_jsonl(request_path, rows)
    activation_path = config.output_dir / "activations.jsonl"
    if activation_path.exists():
        current = {(row["sample_id"], row["request_fingerprint"]) for row in rows}
        retained = [
            row
            for row in read_jsonl(activation_path)
            if (row["sample_id"], row.get("request_fingerprint")) in current
        ]
        write_jsonl(activation_path, retained)
    write_json(config.output_dir / "trajectory_completeness.json", completeness)
    write_json(
        config.output_dir / "activation_request_coverage.json",
        _request_coverage(config, rows),
    )
    return request_path


def run_activation_extraction(config_path: Path) -> Path:
    """Export compact hidden vectors, resuming at the sample level."""

    config = SuccessDirectionConfig.load(config_path)
    request_path = config.output_dir / "activation_requests.jsonl"
    if not request_path.exists():
        request_path = build_activation_requests(config_path)
    output_path = config.output_dir / "activations.jsonl"
    completed = {
        (row["sample_id"], row.get("request_fingerprint"))
        for row in read_jsonl(output_path)
    }
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be non-empty")
    for request in read_jsonl(request_path):
        sample_key = (request["sample_id"], request["request_fingerprint"])
        if sample_key in completed:
            continue
        row = extract_request_activation(
            request,
            base_url=config.hidden_base_url,
            api_key=api_key,
            model=config.hidden_model,
            layer_ids=config.hidden_layer_ids,
        )
        append_jsonl(output_path, row)
    return output_path


def run_success_direction_evaluation(config_path: Path) -> Path:
    """Evaluate held-out within-domain and cross-domain success directions."""

    config = SuccessDirectionConfig.load(config_path)
    rows = read_jsonl(config.output_dir / "activations.jsonl")
    expected = read_jsonl(config.output_dir / "activation_requests.jsonl")
    observed_keys = {(row["sample_id"], row.get("request_fingerprint")) for row in rows}
    expected_keys = {(row["sample_id"], row["request_fingerprint"]) for row in expected}
    if observed_keys != expected_keys or len(rows) != len(expected):
        raise ValueError(
            "Activation extraction incomplete or stale: "
            f"expected {len(expected)} current requests, got {len(rows)} rows"
        )
    report = build_success_direction_report(
        rows,
        domains=config.domains,
        train_split=config.train_split,
        test_split=config.test_split,
        layer_ids=config.hidden_layer_ids,
        primary_layer_id=config.primary_layer_id,
        random_seed=config.seed,
        bootstrap_samples=config.bootstrap_samples,
        permutation_samples=config.permutation_samples,
    )
    path = config.output_dir / "success_direction_report.json"
    write_json(path, report)
    return path
