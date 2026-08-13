"""Run target-free expected-result matching over saved trajectories."""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from tau2.runner import load_task_splits
from trace_to_micro.analysis.logged_trace import (
    audit_results_completeness,
    tool_decision_records,
)
from trace_to_micro.config import ExpectationMatchingConfig
from trace_to_micro.evaluation.expectation_matching import (
    build_expectation_matching_report,
    build_matching_queries,
    mask_identifiers,
    matching_record,
)
from trace_to_micro.runtime.activations import score_chat_suffix
from trace_to_micro.utils.io import (
    append_jsonl,
    read_jsonl,
    write_json,
    write_jsonl,
)


def _matching_paths(config: ExpectationMatchingConfig) -> dict[str, Path]:
    return {
        "records": config.output_dir / "expectation_matching_records.jsonl",
        "queries": config.output_dir / "expectation_matching_queries.jsonl",
        "audit": config.output_dir / "expectation_matching_audit.json",
        "scores": config.output_dir / "expectation_matching_scores.jsonl",
        "measurements": config.output_dir / "expectation_matching_measurements.jsonl",
        "report": config.output_dir / "expectation_matching_report.json",
    }


def _matching_exclusion(
    *, domain: str, split: str, simulation, error: Exception
) -> dict[str, Any]:
    return {
        "scope": "trajectory",
        "domain": domain,
        "split": split,
        "task_id": str(simulation.task_id),
        "simulation_id": simulation.id,
        "trial": simulation.trial,
        "stage": "logged_result_extraction",
        "reason": "trajectory_extraction_failed",
        "exception_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(traceback.format_exception(error)),
    }


def build_expectation_matching_dataset(config_path: Path) -> dict[str, Path]:
    """Compile factual Test tool results and target-free hard negatives."""

    config = ExpectationMatchingConfig.load(config_path)
    paths = _matching_paths(config)
    records = []
    exclusions = []
    completeness = {}
    source_counts = {}
    for domain in config.domains:
        split_map = load_task_splits(domain)
        if split_map is None:
            raise ValueError(f"Domain {domain!r} has no Train/Test split")
        split_by_task = {
            str(task_id): split
            for split in (config.train_split, config.test_split)
            for task_id in split_map[split]
        }
        results_path = config.results_path(domain)
        completeness[domain] = audit_results_completeness(
            results_path,
            expected_task_ids=set(split_map[config.train_split])
            | set(split_map[config.test_split]),
            expected_num_trials=config.expected_num_trials,
            expected_agent_model=config.expected_agent_model,
            expected_user_model=config.expected_user_model,
            raise_on_incomplete=False,
        )
        invalid_simulation_ids = set(
            completeness[domain]["empty_simulation_ids"]
            + completeness[domain]["missing_reward_simulation_ids"]
            + completeness[domain]["infrastructure_error_simulation_ids"]
        )
        if (
            completeness[domain]["agent_model_mismatch"]
            or completeness[domain]["user_model_mismatch"]
        ):
            raise ValueError(f"{domain} trajectories come from a different model")
        if completeness[domain]["duplicate_task_trials"]:
            raise ValueError(f"{domain} trajectories contain duplicate task/trials")

        def record_error(simulation, error: Exception) -> None:
            exclusions.append(
                _matching_exclusion(
                    domain=domain,
                    split=split_by_task.get(str(simulation.task_id), "outside_split"),
                    simulation=simulation,
                    error=error,
                )
            )

        rows = tool_decision_records(
            results_path,
            domain=domain,
            task_set=domain,
            train_split=config.train_split,
            test_split=config.test_split,
            max_per_trajectory=None,
            on_simulation_error=record_error,
        )
        action_rows = [
            row
            for row in rows
            if row["moment"] == "action"
            and row["split"] == config.evaluation_split
            and row["simulation_id"] not in invalid_simulation_ids
        ]
        successful_rows = [
            row
            for row in action_rows
            if row["tool_success"]
            and len(row["messages"][-1].get("tool_calls") or []) == 1
            and len(row["tool_result_contents"]) == 1
        ]
        domain_records = [matching_record(row) for row in successful_rows]
        records.extend(domain_records)
        source_counts[domain] = {
            "all_logged_tool_decisions_on_evaluation_split": len(action_rows),
            "successful_single_call_decisions": len(domain_records),
            "tools": dict(Counter(row["tool_name"] for row in domain_records)),
        }
    duplicate_ids = {
        decision_id
        for decision_id, count in Counter(row["decision_id"] for row in records).items()
        if count > 1
    }
    if duplicate_ids:
        raise ValueError(f"Duplicate matching decision IDs: {sorted(duplicate_ids)}")
    queries, query_exclusions = build_matching_queries(
        records,
        candidate_count=config.candidate_count,
        minimum_candidate_count=config.minimum_candidate_count,
        random_seed=config.random_seed,
    )
    if not queries:
        raise ValueError("Expectation matching produced no eligible query")
    write_jsonl(paths["records"], records)
    write_jsonl(paths["queries"], queries)
    write_json(
        paths["audit"],
        {
            "completed": True,
            "data_source": "saved behavior-policy trajectories",
            "new_rollout_used": False,
            "official_goal_used": False,
            "reference_action_used": False,
            "evaluation_split": config.evaluation_split,
            "source_completeness": completeness,
            "source_counts": source_counts,
            "eligible_query_count": len(queries),
            "eligible_task_count": len(
                {(row["domain"], row["task_id"]) for row in queries}
            ),
            "query_tool_support": dict(Counter(row["tool_name"] for row in queries)),
            "trajectory_exclusions": exclusions,
            "candidate_exclusions": query_exclusions,
        },
    )
    return {key: paths[key] for key in ("records", "queries", "audit")}


def _validate_shard(shard_index: int, num_shards: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards})")


def messages_for_context(
    query: dict[str, Any],
    records: dict[str, dict[str, Any]],
    context_variant: str,
) -> list[dict[str, Any]]:
    source = records[query["query_id"]]
    action = source["messages"][-1]
    if context_variant == "full":
        return deepcopy(source["messages"])
    if context_variant == "action_only":
        return [deepcopy(source["messages"][0]), deepcopy(action)]
    if context_variant == "shuffled":
        shuffled = deepcopy(records[query["shuffled_source_decision_id"]]["messages"])
        shuffled[-1] = deepcopy(action)
        return shuffled
    raise ValueError(f"Unknown context variant {context_variant!r}")


def _score_fingerprint(
    query: dict[str, Any],
    *,
    model: str,
    context_variants: tuple[str, ...],
    content_variants: tuple[str, ...],
) -> str:
    payload = {
        "query": query,
        "model": model,
        "context_variants": context_variants,
        "content_variants": content_variants,
        "scoring": "mean log probability of full serialized tool-result suffix",
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _score_matching_query(
    query: dict[str, Any],
    records: dict[str, dict[str, Any]],
    *,
    config: ExpectationMatchingConfig,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    source = records[query["query_id"]]
    fingerprint = _score_fingerprint(
        query,
        model=config.scoring_model,
        context_variants=config.context_variants,
        content_variants=config.content_variants,
    )
    rows = []
    for context_variant in config.context_variants:
        raw_messages = messages_for_context(query, records, context_variant)
        for content_variant in config.content_variants:
            messages = (
                raw_messages
                if content_variant == "raw"
                else mask_identifiers(raw_messages)
            )
            tools = (
                source["tools"]
                if content_variant == "raw"
                else mask_identifiers(source["tools"])
            )
            for candidate_id in query["candidate_decision_ids"]:
                candidate = records[candidate_id]
                result_content = candidate["result_content"]
                if content_variant == "identifier_masked":
                    result_content = mask_identifiers(result_content)
                score = score_chat_suffix(
                    prefix_messages=messages,
                    suffix_message={
                        "role": "tool",
                        "content": result_content,
                        "tool_call_id": source["tool_call_id"],
                    },
                    tools=tools,
                    chat_template_kwargs=source["chat_template_kwargs"],
                    base_url=base_url,
                    api_key=api_key,
                    model=config.scoring_model,
                )
                rows.append(
                    {
                        "query_id": query["query_id"],
                        "query_fingerprint": query["query_fingerprint"],
                        "score_fingerprint": fingerprint,
                        "domain": query["domain"],
                        "split": query["split"],
                        "task_id": query["task_id"],
                        "simulation_id": query["simulation_id"],
                        "trial": query["trial"],
                        "tool_name": query["tool_name"],
                        "candidate_decision_id": candidate_id,
                        "candidate_task_id": candidate["task_id"],
                        "is_positive": candidate_id == query["positive_decision_id"],
                        "context_variant": context_variant,
                        "content_variant": content_variant,
                        **score,
                    }
                )
    return rows


def _load_matching_inputs(
    config: ExpectationMatchingConfig,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    paths = _matching_paths(config)
    records = read_jsonl(paths["records"])
    queries = read_jsonl(paths["queries"])
    if not records or not queries:
        raise ValueError("Run expectation-matching-prepare before scoring")
    record_by_id = {row["decision_id"]: row for row in records}
    if len(record_by_id) != len(records):
        raise ValueError("Expectation matching records contain duplicate IDs")
    required_ids = {
        decision_id
        for query in queries
        for decision_id in (
            *query["candidate_decision_ids"],
            query["shuffled_source_decision_id"],
        )
    }
    missing = required_ids - set(record_by_id)
    if missing:
        raise ValueError(
            f"Matching queries reference missing records: {sorted(missing)}"
        )
    for query in queries:
        for decision_id, expected in query["record_fingerprints"].items():
            observed = hashlib.sha256(
                json.dumps(
                    record_by_id[decision_id],
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if observed != expected:
                raise ValueError(
                    f"Prepared record changed after candidate construction: {decision_id}"
                )
    return record_by_id, queries


def run_expectation_matching_score_shard(
    config_path: Path,
    *,
    shard_index: int,
    num_shards: int,
    base_url: str,
) -> Path:
    """Score one resumable shard; model-interface errors remain fatal."""

    _validate_shard(shard_index, num_shards)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be non-empty")
    config = ExpectationMatchingConfig.load(config_path)
    records, queries = _load_matching_inputs(config)
    assigned = [
        query
        for position, query in enumerate(
            sorted(queries, key=lambda row: row["query_id"])
        )
        if position % num_shards == shard_index
    ]
    output = config.score_shard_path(shard_index, num_shards)
    existing = read_jsonl(output)
    completed = {row["query_id"] for row in existing}
    expected_rows_per_query = len(config.context_variants) * len(
        config.content_variants
    )
    counts = Counter(row["query_id"] for row in existing)
    candidate_counts = {
        query["query_id"]: len(query["candidate_decision_ids"]) for query in assigned
    }
    assigned_by_id = {query["query_id"]: query for query in assigned}
    incomplete = {
        query_id
        for query_id, count in counts.items()
        if query_id not in candidate_counts
        or count != candidate_counts[query_id] * expected_rows_per_query
    }
    stale = {
        row["query_id"]
        for row in existing
        if row["query_id"] in assigned_by_id
        and row["score_fingerprint"]
        != _score_fingerprint(
            assigned_by_id[row["query_id"]],
            model=config.scoring_model,
            context_variants=config.context_variants,
            content_variants=config.content_variants,
        )
    }
    if stale:
        raise ValueError(f"Shard contains stale score groups: {sorted(stale)}")
    if incomplete:
        raise ValueError(
            f"Shard contains partial query groups; remove and rerun it: {sorted(incomplete)}"
        )
    for query in assigned:
        if query["query_id"] in completed:
            continue
        for row in _score_matching_query(
            query,
            records,
            config=config,
            base_url=base_url,
            api_key=api_key,
        ):
            append_jsonl(output, row)
    return output


def merge_expectation_matching_scores(config_path: Path, *, num_shards: int) -> Path:
    """Merge complete query groups from all likelihood-score shards."""

    _validate_shard(0, num_shards)
    config = ExpectationMatchingConfig.load(config_path)
    _, queries = _load_matching_inputs(config)
    rows = []
    for shard_index in range(num_shards):
        path = config.score_shard_path(shard_index, num_shards)
        if not path.is_file():
            raise ValueError(f"Missing score shard {path}")
        rows.extend(read_jsonl(path))
    condition_count = len(config.context_variants) * len(config.content_variants)
    counts = Counter(row["query_id"] for row in rows)
    expected_ids = {row["query_id"] for row in queries}
    if set(counts) != expected_ids:
        raise ValueError("Merged score shards do not cover the prepared queries")
    candidate_counts = {
        query["query_id"]: len(query["candidate_decision_ids"]) for query in queries
    }
    bad = {
        query_id: count
        for query_id, count in counts.items()
        if count != candidate_counts[query_id] * condition_count
    }
    if bad:
        raise ValueError(f"Merged score query groups are incomplete: {bad}")
    output = _matching_paths(config)["scores"]
    write_jsonl(
        output,
        sorted(
            rows,
            key=lambda row: (
                row["query_id"],
                row["context_variant"],
                row["content_variant"],
                not row["is_positive"],
                row["candidate_decision_id"],
            ),
        ),
    )
    return output


def run_expectation_matching_evaluation(config_path: Path) -> dict[str, Path]:
    """Aggregate ranking metrics and task-cluster uncertainty."""

    config = ExpectationMatchingConfig.load(config_path)
    paths = _matching_paths(config)
    scores = read_jsonl(paths["scores"])
    if not scores:
        raise ValueError("Run expectation-matching-merge before evaluation")
    report, measurements = build_expectation_matching_report(
        scores,
        domains=config.domains,
        evaluation_split=config.evaluation_split,
        context_variants=config.context_variants,
        content_variants=config.content_variants,
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed,
    )
    write_jsonl(paths["measurements"], measurements)
    write_json(paths["report"], report)
    return {"measurements": paths["measurements"], "report": paths["report"]}


def run_expectation_matching_smoke(
    config_path: Path, *, base_url: str
) -> dict[str, Path]:
    """Run one real query through all six model-scoring conditions."""

    config = ExpectationMatchingConfig.load(config_path)
    records, queries = _load_matching_inputs(config)
    eligible = [
        query
        for query in queries
        if query["domain"] == config.smoke_domain
        and query["task_id"] == config.smoke_task_id
    ]
    if not eligible:
        eligible = [
            query for query in queries if query["domain"] == config.smoke_domain
        ]
    if not eligible:
        raise ValueError(f"No smoke query for domain {config.smoke_domain}")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be non-empty")
    query = sorted(eligible, key=lambda row: row["query_id"])[0]
    rows = _score_matching_query(
        query,
        records,
        config=config,
        base_url=base_url,
        api_key=api_key,
    )
    output_dir = config.smoke_output_dir()
    scores_path = output_dir / "expectation_matching_scores.jsonl"
    report_path = output_dir / "smoke_report.json"
    write_jsonl(scores_path, rows)
    measurements = build_expectation_matching_report(
        rows,
        domains=(config.smoke_domain,),
        evaluation_split=config.evaluation_split,
        context_variants=config.context_variants,
        content_variants=config.content_variants,
        bootstrap_samples=max(10, min(100, config.bootstrap_samples)),
        random_seed=config.random_seed,
    )[1]
    write_json(
        report_path,
        {
            "completed": True,
            "query": query,
            "score_row_count": len(rows),
            "expected_score_row_count": (
                config.candidate_count
                * len(config.context_variants)
                * len(config.content_variants)
            ),
            "conditions": measurements,
        },
    )
    return {"scores": scores_path, "report": report_path}
