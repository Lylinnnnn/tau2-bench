"""Orchestration for the Airline/Retail success-direction experiment."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import is_successful
from tau2.runner import load_task_splits, load_tasks
from trace_to_micro.analysis.logged_trace import (
    audit_results_completeness,
    tool_decision_records,
)
from trace_to_micro.analysis.results import build_official_metrics
from trace_to_micro.config import SuccessDirectionConfig, TrajectoryConfig
from trace_to_micro.evaluation.success_direction import (
    build_activation_smoke_report,
    build_success_direction_report,
)
from trace_to_micro.runner.trajectory import (
    remove_existing_run,
    run_complete_trajectories,
)
from trace_to_micro.runtime.activations import extract_request_activation
from trace_to_micro.utils.io import (
    append_jsonl,
    read_jsonl,
    write_json,
    write_jsonl,
)


def _trajectory_config(
    config: SuccessDirectionConfig,
    domain: str,
    *,
    base_url: str | None = None,
) -> TrajectoryConfig:
    agent_llm_args = dict(config.agent_llm_args)
    user_llm_args = dict(config.user_llm_args)
    if base_url is not None:
        agent_llm_args["api_base"] = base_url
        user_llm_args["api_base"] = base_url
    return TrajectoryConfig(
        domain=domain,
        task_set=domain,
        task_split=config.task_split,
        agent=config.agent,
        user=config.user,
        agent_llm=config.agent_llm,
        user_llm=config.user_llm,
        agent_llm_args=agent_llm_args,
        user_llm_args=user_llm_args,
        num_trials=config.num_trials,
        max_steps=config.max_steps,
        max_errors=config.max_errors,
        max_concurrency=config.max_concurrency,
        seed=config.seed,
        timeout_seconds=config.timeout_seconds,
        save_to=config.save_name(domain),
    )


def _validate_shard(shard_index: int, num_shards: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")


def task_ids_for_shard(
    config: SuccessDirectionConfig,
    domain: str,
    shard_index: int,
    num_shards: int,
) -> list[str]:
    """Assign the global Airline/Retail task list round-robin to one worker."""

    _validate_shard(shard_index, num_shards)
    if domain not in config.domains:
        raise ValueError(f"Domain {domain!r} is not configured")
    ordered_tasks: list[tuple[str, str]] = []
    for configured_domain in config.domains:
        split_map = load_task_splits(configured_domain)
        if split_map is None or config.task_split not in split_map:
            raise ValueError(
                f"Domain {configured_domain!r} has no {config.task_split!r} split"
            )
        ordered_tasks.extend(
            (configured_domain, str(task_id))
            for task_id in split_map[config.task_split]
        )
    return [
        task_id
        for position, (task_domain, task_id) in enumerate(ordered_tasks)
        if position % num_shards == shard_index and task_domain == domain
    ]


def activation_requests_for_shard(
    requests: list[dict[str, Any]],
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    """Keep every three-moment decision group on exactly one worker."""

    _validate_shard(shard_index, num_shards)
    decision_ids = sorted({row["decision_id"] for row in requests})
    assigned = {
        decision_id
        for position, decision_id in enumerate(decision_ids)
        if position % num_shards == shard_index
    }
    return [row for row in requests if row["decision_id"] in assigned]


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
    run_success_official_metrics(config_path)


def run_success_trajectory_shard(
    config_path: Path,
    *,
    shard_index: int,
    num_shards: int,
    base_url: str,
    force: bool | None = None,
) -> None:
    """Generate one disjoint task shard against one model endpoint."""

    config = SuccessDirectionConfig.load(config_path)
    overwrite = config.force_overwrite if force is None else force
    for domain in config.domains:
        task_ids = task_ids_for_shard(config, domain, shard_index, num_shards)
        if not task_ids:
            continue
        results_path = config.shard_results_path(domain, shard_index, num_shards)
        if overwrite:
            remove_existing_run(results_path)
        run_complete_trajectories(
            _trajectory_config(config, domain, base_url=base_url),
            task_ids=task_ids,
            save_to=config.shard_save_name(domain, shard_index, num_shards),
            auto_resume=not overwrite,
        )
        audit_results_completeness(
            results_path,
            expected_task_ids=set(task_ids),
            expected_num_trials=config.num_trials,
            expected_agent_model=config.agent_llm,
            expected_user_model=config.user_llm,
        )


def merge_success_trajectory_shards(config_path: Path, *, num_shards: int) -> None:
    """Strictly merge disjoint trajectory shards into the canonical result files."""

    _validate_shard(0, num_shards)
    config = SuccessDirectionConfig.load(config_path)
    manifest: dict[str, Any] = {
        "num_shards": num_shards,
        "assignment": "round-robin over the combined Airline/Retail base task list",
        "domains": {},
    }
    for domain in config.domains:
        shard_results = []
        domain_shards = []
        for shard_index in range(num_shards):
            expected_task_ids = task_ids_for_shard(
                config, domain, shard_index, num_shards
            )
            if not expected_task_ids:
                continue
            path = config.shard_results_path(domain, shard_index, num_shards)
            audit = audit_results_completeness(
                path,
                expected_task_ids=set(expected_task_ids),
                expected_num_trials=config.num_trials,
                expected_agent_model=config.agent_llm,
                expected_user_model=config.user_llm,
            )
            result = Results.load(path)
            shard_results.append(result)
            domain_shards.append(
                {
                    "shard_index": shard_index,
                    "task_ids": expected_task_ids,
                    "results_path": str(path),
                    "api_base": result.info.agent_info.llm_args.get("api_base"),
                    "complete": audit["complete"],
                }
            )
        if not shard_results:
            raise ValueError(f"No trajectory shards found for domain {domain!r}")
        task_order = {
            str(task.id): position
            for position, task in enumerate(load_tasks(domain, config.task_split))
        }
        simulations = [
            simulation for result in shard_results for simulation in result.simulations
        ]
        simulations.sort(
            key=lambda simulation: (
                task_order[str(simulation.task_id)],
                simulation.trial,
            )
        )
        merged_info = shard_results[0].info.model_copy(deep=True)
        for model_info in (merged_info.agent_info, merged_info.user_info):
            llm_args = dict(model_info.llm_args or {})
            llm_args["api_base"] = (
                "sharded endpoints; see trajectory_shard_manifest.json"
            )
            model_info.llm_args = llm_args
        merged = Results(
            info=merged_info,
            tasks=load_tasks(domain, config.task_split),
            simulations=simulations,
        )
        merged.save(config.results_path(domain))
        _require_complete(config, domain)
        manifest["domains"][domain] = domain_shards
    write_json(config.output_dir / "trajectory_shard_manifest.json", manifest)
    run_success_official_metrics(config_path)


def run_success_official_metrics(config_path: Path) -> Path:
    """Audit complete trajectories and compute each domain's official metrics."""

    config = SuccessDirectionConfig.load(config_path)
    completeness = {
        domain: _require_complete(config, domain) for domain in config.domains
    }
    write_json(config.output_dir / "trajectory_completeness.json", completeness)
    path = config.output_dir / "official_metrics.json"
    write_json(
        path,
        {
            domain: build_official_metrics(config.results_path(domain))
            for domain in config.domains
        },
    )
    return path


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
            missing_rewards = [
                simulation.id
                for simulation in simulations
                if simulation.reward_info is None
            ]
            if missing_rewards:
                raise ValueError(f"Coverage requires rewards: {missing_rewards}")
            outcomes = Counter(
                "success" if is_successful(simulation.reward_info.reward) else "failure"
                for simulation in simulations
            )
            covered_outcomes = Counter(
                "success" if is_successful(simulation.reward_info.reward) else "failure"
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
    if not rows:
        raise ValueError(
            "No assistant tool decisions were found in complete trajectories"
        )
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
    requests = read_jsonl(request_path)
    if not requests:
        raise ValueError(f"Activation request file is empty: {request_path}")
    output_path = config.output_dir / "activations.jsonl"
    completed = {
        (row["sample_id"], row.get("request_fingerprint"))
        for row in read_jsonl(output_path)
    }
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be non-empty")
    for request in requests:
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


def run_activation_extraction_shard(
    config_path: Path,
    *,
    shard_index: int,
    num_shards: int,
    base_url: str,
) -> Path:
    """Export one disjoint set of decision triples to one isolated JSONL file."""

    _validate_shard(shard_index, num_shards)
    config = SuccessDirectionConfig.load(config_path)
    request_path = config.output_dir / "activation_requests.jsonl"
    if not request_path.is_file():
        raise FileNotFoundError(
            "Build activation requests before starting sharded extraction"
        )
    all_requests = read_jsonl(request_path)
    requests = activation_requests_for_shard(all_requests, shard_index, num_shards)
    if not requests:
        raise ValueError(f"Activation shard {shard_index} has no requests")
    output_path = config.activation_shard_path(shard_index, num_shards)
    current = {
        (request["sample_id"], request["request_fingerprint"]) for request in requests
    }
    retained = [
        row
        for row in read_jsonl(output_path)
        if (row["sample_id"], row.get("request_fingerprint")) in current
    ]
    write_jsonl(output_path, retained)
    completed = {(row["sample_id"], row.get("request_fingerprint")) for row in retained}
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be non-empty")
    for request in requests:
        sample_key = (request["sample_id"], request["request_fingerprint"])
        if sample_key in completed:
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


def merge_activation_shards(config_path: Path, *, num_shards: int) -> Path:
    """Merge current activation shards and reject missing, duplicate, or stale rows."""

    _validate_shard(0, num_shards)
    config = SuccessDirectionConfig.load(config_path)
    request_path = config.output_dir / "activation_requests.jsonl"
    if not request_path.is_file():
        raise FileNotFoundError(f"Activation requests are missing: {request_path}")
    requests = read_jsonl(request_path)
    request_order = {
        (row["sample_id"], row["request_fingerprint"]): position
        for position, row in enumerate(requests)
    }
    rows = [
        row
        for shard_index in range(num_shards)
        for row in read_jsonl(config.activation_shard_path(shard_index, num_shards))
    ]
    observed = [(row["sample_id"], row.get("request_fingerprint")) for row in rows]
    if len(set(observed)) != len(observed):
        raise ValueError("Activation shards contain duplicate current sample keys")
    if set(observed) != set(request_order):
        missing = sorted(set(request_order) - set(observed))
        unexpected = sorted(set(observed) - set(request_order))
        raise ValueError(
            "Activation shards do not match current requests: "
            f"missing={missing}, unexpected={unexpected}"
        )
    rows.sort(
        key=lambda row: request_order[(row["sample_id"], row["request_fingerprint"])]
    )
    output_path = config.output_dir / "activations.jsonl"
    write_jsonl(output_path, rows)
    return output_path


def run_success_direction_evaluation(config_path: Path) -> Path:
    """Evaluate held-out within-domain and cross-domain success directions."""

    config = SuccessDirectionConfig.load(config_path)
    activation_path = config.output_dir / "activations.jsonl"
    request_path = config.output_dir / "activation_requests.jsonl"
    if not activation_path.is_file() or not request_path.is_file():
        raise FileNotFoundError(
            "Run the activation stage before evaluation: "
            f"{request_path}, {activation_path}"
        )
    rows = read_jsonl(activation_path)
    expected = read_jsonl(request_path)
    if not rows or not expected:
        raise ValueError("Activation evaluation inputs must be non-empty")
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


def _require_smoke_complete(config: SuccessDirectionConfig) -> dict[str, Any]:
    return audit_results_completeness(
        config.smoke_results_path(),
        expected_task_ids={config.smoke_task_id},
        expected_num_trials=config.num_trials,
        expected_agent_model=config.agent_llm,
        expected_user_model=config.user_llm,
    )


def run_success_smoke_trajectory(config_path: Path) -> dict[str, Path]:
    """Force one isolated official task through generation and official scoring."""

    config = SuccessDirectionConfig.load(config_path)
    split_map = load_task_splits(config.smoke_domain)
    if split_map is None or config.smoke_task_id not in split_map[config.task_split]:
        raise ValueError(
            f"Smoke task {config.smoke_task_id!r} is not in "
            f"{config.smoke_domain}/{config.task_split}"
        )
    remove_existing_run(config.smoke_results_path())
    for filename in (
        "trajectory_completeness.json",
        "official_metrics.json",
        "activation_requests.jsonl",
        "activations.jsonl",
        "smoke_report.json",
    ):
        path = config.smoke_output_dir() / filename
        if path.exists():
            path.unlink()
    run_complete_trajectories(
        _trajectory_config(config, config.smoke_domain),
        task_ids=[config.smoke_task_id],
        save_to=config.smoke_save_name(),
        auto_resume=False,
    )
    completeness = _require_smoke_complete(config)
    metrics = build_official_metrics(config.smoke_results_path())
    completeness_path = config.smoke_output_dir() / "trajectory_completeness.json"
    metrics_path = config.smoke_output_dir() / "official_metrics.json"
    write_json(completeness_path, completeness)
    write_json(metrics_path, metrics)
    return {
        "results": config.smoke_results_path(),
        "trajectory_completeness": completeness_path,
        "official_metrics": metrics_path,
    }


def build_success_smoke_activation_requests(config_path: Path) -> Path:
    """Build exactly three factual moments from the one smoke trajectory."""

    config = SuccessDirectionConfig.load(config_path)
    _require_smoke_complete(config)
    rows = tool_decision_records(
        config.smoke_results_path(),
        domain=config.smoke_domain,
        task_set=config.smoke_domain,
        train_split=config.train_split,
        test_split=config.test_split,
        max_per_trajectory=1,
    )
    if len(rows) != 3 or {row["moment"] for row in rows} != {
        "before",
        "action",
        "result",
    }:
        raise ValueError(
            "Smoke task must contain one assistant tool decision and exactly three "
            f"probe moments; got {len(rows)} rows"
        )
    if len({row["decision_id"] for row in rows}) != 1:
        raise ValueError("Smoke moments do not come from the same tool decision")
    for row in rows:
        row["request_fingerprint"] = _request_fingerprint(row, config)
    request_path = config.smoke_output_dir() / "activation_requests.jsonl"
    write_jsonl(request_path, rows)
    write_jsonl(config.smoke_output_dir() / "activations.jsonl", [])
    report_path = config.smoke_output_dir() / "smoke_report.json"
    if report_path.exists():
        report_path.unlink()
    return request_path


def build_success_smoke_report(config_path: Path) -> Path:
    """Fail fast on mismatched exports and summarize the real one-task pipeline."""

    config = SuccessDirectionConfig.load(config_path)
    output_dir = config.smoke_output_dir()
    request_path = output_dir / "activation_requests.jsonl"
    activation_path = output_dir / "activations.jsonl"
    if not request_path.is_file() or not activation_path.is_file():
        raise FileNotFoundError(
            f"Smoke activation inputs are missing: {request_path}, {activation_path}"
        )
    requests = read_jsonl(request_path)
    activations = read_jsonl(activation_path)
    request_by_moment = {row["moment"]: row for row in requests}
    activation_report = build_activation_smoke_report(
        requests,
        activations,
        layer_ids=config.hidden_layer_ids,
        hidden_size=config.hidden_size,
        hidden_model=config.hidden_model,
    )
    results = Results.load(config.smoke_results_path())
    simulation = results.simulations[0]
    metrics = build_official_metrics(config.smoke_results_path())
    report = {
        "hypothesis_evaluable": False,
        "hypothesis_evaluable_reason": (
            "one trajectory validates extraction and metrics, but cannot fit or "
            "evaluate a success direction"
        ),
        "task": {
            "domain": config.smoke_domain,
            "task_id": config.smoke_task_id,
            "official_split": request_by_moment["before"]["split"],
            "simulation_id": simulation.id,
            "termination_reason": str(simulation.termination_reason),
            "official_reward": simulation.reward_info.reward,
            "official_success": is_successful(simulation.reward_info.reward),
            "logged_action": request_by_moment["before"]["logged_action"],
        },
        "official_metrics": metrics,
        **activation_report,
    }
    report_path = output_dir / "smoke_report.json"
    write_json(report_path, report)
    return report_path


def run_success_smoke_activations(config_path: Path) -> dict[str, Path]:
    """Extract all smoke activations and immediately run strict validation."""

    config = SuccessDirectionConfig.load(config_path)
    request_path = config.smoke_output_dir() / "activation_requests.jsonl"
    if not request_path.is_file():
        raise FileNotFoundError(
            "Build smoke activation requests before starting hidden extraction"
        )
    requests = read_jsonl(request_path)
    if len(requests) != 3:
        raise ValueError(
            f"Smoke activation request count must be 3, got {len(requests)}"
        )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be non-empty")
    activation_path = config.smoke_output_dir() / "activations.jsonl"
    write_jsonl(activation_path, [])
    for request in requests:
        append_jsonl(
            activation_path,
            extract_request_activation(
                request,
                base_url=config.hidden_base_url,
                api_key=api_key,
                model=config.hidden_model,
                layer_ids=config.hidden_layer_ids,
            ),
        )
    report_path = build_success_smoke_report(config_path)
    return {"activations": activation_path, "smoke_report": report_path}
