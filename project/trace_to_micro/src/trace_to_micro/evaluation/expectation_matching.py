"""Build and evaluate target-free expected-result matching sets."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any

import numpy as np

_IDENTIFIER_PATTERNS = (
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"#[A-Za-z0-9_-]+"),
    re.compile(r"\b[A-Za-z]+(?:_[A-Za-z]+)+_\d+\b"),
    re.compile(r"\b(?=[A-Z0-9]{5,}\b)(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]+\b"),
    re.compile(r"\b\d{5,}\b"),
)


def mask_identifiers(value: Any) -> Any:
    """Mechanically remove high-cardinality identifiers without semantic labels."""

    if isinstance(value, dict):
        return {key: mask_identifiers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_identifiers(item) for item in value]
    if not isinstance(value, str):
        return value
    masked = value
    for pattern in _IDENTIFIER_PATTERNS:
        masked = pattern.sub("<IDENTIFIER>", masked)
    return masked


def _value_paths(value: Any, prefix: str = "$") -> set[str]:
    if isinstance(value, dict):
        paths = {f"{prefix}:object"}
        for key, item in value.items():
            paths.update(_value_paths(item, f"{prefix}.{key}"))
        return paths
    if isinstance(value, list):
        paths = {f"{prefix}:array"}
        for item in value:
            paths.update(_value_paths(item, f"{prefix}[]"))
        return paths
    return {f"{prefix}:{type(value).__name__}"}


def result_structure(content: str) -> list[str]:
    """Describe a tool result by JSON field/type paths, never by its values."""

    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        value = content
    return sorted(_value_paths(value))


def _candidate_similarity(source: dict[str, Any], candidate: dict[str, Any]) -> tuple:
    source_paths = set(source["result_structure"])
    candidate_paths = set(candidate["result_structure"])
    union = source_paths | candidate_paths
    jaccard = len(source_paths & candidate_paths) / len(union)
    source_length = max(1, len(source["result_content"]))
    candidate_length = max(1, len(candidate["result_content"]))
    length_gap = abs(math.log(source_length / candidate_length))
    return (
        source_paths == candidate_paths,
        jaccard,
        -length_gap,
        candidate["decision_id"],
    )


def matching_record(action_row: dict[str, Any]) -> dict[str, Any]:
    """Compile one successful, single-call action/result pair for matching."""

    if action_row["moment"] != "action":
        raise ValueError("Expectation matching records require action moments")
    calls = action_row["messages"][-1].get("tool_calls") or []
    contents = action_row["tool_result_contents"]
    if len(calls) != 1 or len(contents) != 1:
        raise ValueError(
            f"{action_row['decision_id']} is not a single-call/single-result decision"
        )
    if not action_row["tool_success"]:
        raise ValueError(f"{action_row['decision_id']} is not a successful tool call")
    content = contents[0]
    if not isinstance(content, str):
        raise ValueError(f"{action_row['decision_id']} has a non-text tool result")
    return {
        key: action_row[key]
        for key in (
            "decision_id",
            "simulation_id",
            "domain",
            "split",
            "task_id",
            "trial",
            "message_index",
            "decision_position",
            "tool_name",
            "messages",
            "tools",
            "chat_template_kwargs",
        )
    } | {
        "tool_call_id": calls[0]["id"],
        "result_content": content,
        "result_structure": result_structure(content),
    }


def build_matching_queries(
    records: list[dict[str, Any]],
    *,
    candidate_count: int,
    minimum_candidate_count: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build same-tool, cross-task hard-negative sets and shuffled controls."""

    if candidate_count < minimum_candidate_count or minimum_candidate_count < 2:
        raise ValueError("Candidate-count constraints are inconsistent")
    ordered = sorted(records, key=lambda row: row["decision_id"])
    by_decision_id = {row["decision_id"]: row for row in ordered}
    if len(by_decision_id) != len(ordered):
        raise ValueError("Matching records contain duplicate decision IDs")
    queries = []
    exclusions = []
    for source in ordered:
        pool = [
            candidate
            for candidate in ordered
            if candidate["domain"] == source["domain"]
            and candidate["split"] == source["split"]
            and candidate["tool_name"] == source["tool_name"]
            and candidate["task_id"] != source["task_id"]
            and candidate["result_content"] != source["result_content"]
        ]
        pool.sort(key=lambda row: _candidate_similarity(source, row), reverse=True)
        negatives = []
        seen_results = {source["result_content"]}
        for candidate in pool:
            if candidate["result_content"] in seen_results:
                continue
            negatives.append(candidate)
            seen_results.add(candidate["result_content"])
            if len(negatives) == candidate_count - 1:
                break
        if len(negatives) + 1 < minimum_candidate_count:
            exclusions.append(
                {
                    "decision_id": source["decision_id"],
                    "domain": source["domain"],
                    "task_id": source["task_id"],
                    "tool_name": source["tool_name"],
                    "reason": "insufficient_distinct_same_tool_cross_task_results",
                    "available_candidate_count": len(negatives) + 1,
                }
            )
            continue
        shuffled_pool = [
            candidate
            for candidate in ordered
            if candidate["domain"] == source["domain"]
            and candidate["split"] == source["split"]
            and candidate["tool_name"] == source["tool_name"]
            and candidate["task_id"] != source["task_id"]
        ]
        shuffled_pool.sort(key=lambda row: row["decision_id"])
        if not shuffled_pool:
            raise ValueError(
                f"Eligible query has no shuffled context: {source['decision_id']}"
            )
        offset = int(
            hashlib.sha256(
                f"{random_seed}:{source['decision_id']}".encode()
            ).hexdigest(),
            16,
        ) % len(shuffled_pool)
        candidate_ids = [
            source["decision_id"],
            *[row["decision_id"] for row in negatives],
        ]
        query = {
            "query_id": source["decision_id"],
            "domain": source["domain"],
            "split": source["split"],
            "task_id": source["task_id"],
            "simulation_id": source["simulation_id"],
            "trial": source["trial"],
            "tool_name": source["tool_name"],
            "positive_decision_id": source["decision_id"],
            "candidate_decision_ids": candidate_ids,
            "shuffled_source_decision_id": shuffled_pool[offset]["decision_id"],
        }
        queries.append(query)
    selected_ids = {
        decision_id
        for query in queries
        for decision_id in (
            query["query_id"],
            query["shuffled_source_decision_id"],
            *query["candidate_decision_ids"],
        )
    }
    selected_records = [by_decision_id[decision_id] for decision_id in selected_ids]
    record_fingerprints = {
        row["decision_id"]: hashlib.sha256(
            json.dumps(
                row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest()
        for row in selected_records
    }
    for query in queries:
        query["record_fingerprints"] = {
            decision_id: record_fingerprints[decision_id]
            for decision_id in (
                query["query_id"],
                query["shuffled_source_decision_id"],
                *query["candidate_decision_ids"],
            )
        }
        query["query_fingerprint"] = hashlib.sha256(
            json.dumps(
                {
                    key: value
                    for key, value in query.items()
                    if key != "query_fingerprint"
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    return queries, exclusions


def _rank_measurements(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in score_rows:
        key = (row["query_id"], row["context_variant"], row["content_variant"])
        grouped.setdefault(key, []).append(row)
    measurements = []
    for (query_id, context, content), rows in sorted(grouped.items()):
        positives = [row for row in rows if row["is_positive"]]
        if len(positives) != 1:
            raise ValueError(f"{query_id}/{context}/{content} has != 1 positive")
        positive = positives[0]
        negatives = [row for row in rows if not row["is_positive"]]
        if not negatives:
            raise ValueError(f"{query_id}/{context}/{content} has no negatives")
        positive_score = positive["mean_logprob"]
        negative_scores = [row["mean_logprob"] for row in negatives]
        greater = sum(score > positive_score for score in negative_scores)
        ties = sum(score == positive_score for score in negative_scores)
        rank = 1 + greater + ties / 2
        pairwise = sum(
            1.0 if positive_score > score else 0.5 if positive_score == score else 0.0
            for score in negative_scores
        ) / len(negative_scores)
        measurements.append(
            {
                "query_id": query_id,
                "domain": positive["domain"],
                "task_id": positive["task_id"],
                "tool_name": positive["tool_name"],
                "context_variant": context,
                "content_variant": content,
                "candidate_count": len(rows),
                "positive_rank": rank,
                "hit_at_1": float(positive_score > max(negative_scores)),
                "reciprocal_rank": 1 / rank,
                "pairwise_accuracy": pairwise,
                "top_margin": positive_score - max(negative_scores),
            }
        )
    return measurements


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "query_count": len(rows),
        "mean_candidate_count": float(
            np.mean([row["candidate_count"] for row in rows])
        ),
        "pairwise_accuracy": float(np.mean([row["pairwise_accuracy"] for row in rows])),
        "hit_at_1": float(np.mean([row["hit_at_1"] for row in rows])),
        "mean_reciprocal_rank": float(
            np.mean([row["reciprocal_rank"] for row in rows])
        ),
        "mean_top_margin": float(np.mean([row["top_margin"] for row in rows])),
    }


def _task_bootstrap_interval(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    samples: int,
    random_seed: int,
) -> list[float]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    task_ids = sorted(by_task)
    rng = np.random.default_rng(random_seed)
    estimates = []
    for _ in range(samples):
        chosen = rng.choice(task_ids, size=len(task_ids), replace=True)
        values = [item[metric] for task_id in chosen for item in by_task[task_id]]
        estimates.append(float(np.mean(values)))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def _task_bootstrap_delta(
    full: list[dict[str, Any]],
    shuffled: list[dict[str, Any]],
    *,
    samples: int,
    random_seed: int,
) -> list[float]:
    full_by_id = {row["query_id"]: row for row in full}
    shuffled_by_id = {row["query_id"]: row for row in shuffled}
    if set(full_by_id) != set(shuffled_by_id):
        raise ValueError("Full and shuffled controls do not contain the same queries")
    deltas = [
        {
            "task_id": row["task_id"],
            "delta": row["pairwise_accuracy"]
            - shuffled_by_id[query_id]["pairwise_accuracy"],
        }
        for query_id, row in full_by_id.items()
    ]
    by_task: dict[str, list[float]] = {}
    for row in deltas:
        by_task.setdefault(row["task_id"], []).append(row["delta"])
    task_ids = sorted(by_task)
    rng = np.random.default_rng(random_seed)
    estimates = []
    for _ in range(samples):
        chosen = rng.choice(task_ids, size=len(task_ids), replace=True)
        estimates.append(
            float(np.mean([value for task_id in chosen for value in by_task[task_id]]))
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def build_expectation_matching_report(
    score_rows: list[dict[str, Any]],
    *,
    domains: tuple[str, ...],
    evaluation_split: str,
    context_variants: tuple[str, ...],
    content_variants: tuple[str, ...],
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate whether pre-result context ranks the factual result highest."""

    measurements = _rank_measurements(score_rows)
    reports = {}
    evidence = []
    for domain_index, domain in enumerate(domains):
        domain_rows = [row for row in measurements if row["domain"] == domain]
        condition_reports = {}
        for context in context_variants:
            for content in content_variants:
                selected = [
                    row
                    for row in domain_rows
                    if row["context_variant"] == context
                    and row["content_variant"] == content
                ]
                if not selected:
                    raise ValueError(f"No scores for {domain}/{context}/{content}")
                values = _mean_metrics(selected)
                values["pairwise_accuracy_task_bootstrap_95_ci"] = (
                    _task_bootstrap_interval(
                        selected,
                        metric="pairwise_accuracy",
                        samples=bootstrap_samples,
                        random_seed=random_seed + domain_index,
                    )
                )
                condition_reports[f"{context}:{content}"] = values
        full_raw = [
            row
            for row in domain_rows
            if row["context_variant"] == "full" and row["content_variant"] == "raw"
        ]
        shuffled_raw = [
            row
            for row in domain_rows
            if row["context_variant"] == "shuffled" and row["content_variant"] == "raw"
        ]
        full_masked = [
            row
            for row in domain_rows
            if row["context_variant"] == "full"
            and row["content_variant"] == "identifier_masked"
        ]
        raw_delta = (
            _mean_metrics(full_raw)["pairwise_accuracy"]
            - _mean_metrics(shuffled_raw)["pairwise_accuracy"]
        )
        delta_interval = _task_bootstrap_delta(
            full_raw,
            shuffled_raw,
            samples=bootstrap_samples,
            random_seed=random_seed + 100 + domain_index,
        )
        primary_interval = condition_reports["full:raw"][
            "pairwise_accuracy_task_bootstrap_95_ci"
        ]
        identifier_robust = _mean_metrics(full_masked)["pairwise_accuracy"] > 0.5
        domain_pass = (
            primary_interval[0] > 0.5
            and raw_delta > 0
            and delta_interval[0] > 0
            and identifier_robust
        )
        evidence.append(domain_pass)
        reports[domain] = {
            "conditions": condition_reports,
            "full_raw_minus_shuffled_raw_pairwise_accuracy": raw_delta,
            "delta_task_bootstrap_95_ci": delta_interval,
            "identifier_masked_above_chance": identifier_robust,
            "primary_rule_passed": domain_pass,
            "tool_support": dict(Counter(row["tool_name"] for row in full_raw)),
        }
    status = (
        "reliable_contextual_expectation_signal"
        if all(evidence)
        else "partial_contextual_expectation_signal"
        if any(evidence)
        else "no_reliable_contextual_expectation_signal"
    )
    return (
        {
            "experiment": {
                "name": "Experiment A: target-free expected-result matching",
                "hypothesis": (
                    "After selecting a tool action but before observing its result, "
                    "the behavior model assigns higher conditional likelihood to "
                    "the factual result than to structurally similar same-tool "
                    "results from other tasks."
                ),
                "official_goal_used": False,
                "reference_action_used": False,
                "trained_probe_used": False,
                "new_rollout_used": False,
                "evaluation_split": evaluation_split,
                "negative_source": (
                    "successful same-tool results from different tasks in the same "
                    "domain and official split"
                ),
                "uncertainty_cluster": "task_id",
            },
            "primary_test": {
                "metric": "per-query pairwise ranking accuracy",
                "chance": 0.5,
                "evidence_rule": (
                    "In both domains, full/raw has a task-bootstrap lower bound "
                    "> 0.5, exceeds shuffled/raw with a positive lower-bound "
                    "delta, and full/identifier-masked remains above chance."
                ),
                "evidence_status": status,
            },
            "domains": reports,
        },
        measurements,
    )
