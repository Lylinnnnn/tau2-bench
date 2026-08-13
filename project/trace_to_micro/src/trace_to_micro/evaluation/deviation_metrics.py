"""Train-only calibration and Test metrics for result-deviation scores."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np


def calibration_key(row: dict[str, Any], level: str) -> str:
    """Return a stable calibration group key."""

    prefix = f"{row['domain']}|{row['context_variant']}|{row['tool_name']}"
    if level == "structure":
        return f"{prefix}|{row['structure_key']}"
    if level == "tool":
        return prefix
    raise ValueError(f"Unknown calibration level {level!r}")


def fit_clean_calibration(
    rows: list[dict[str, Any]], *, minimum_count: int
) -> dict[str, Any]:
    """Fit Train-clean score centers and scales by structure, then by tool."""

    if minimum_count < 2:
        raise ValueError("Calibration minimum_count must be at least two")
    output = {"minimum_count": minimum_count, "structure": {}, "tool": {}}
    for level in ("structure", "tool"):
        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            groups[calibration_key(row, level)].append(row["mean_logprob"])
        for key, values in sorted(groups.items()):
            if len(values) < minimum_count:
                continue
            std = float(np.std(values, ddof=1))
            if std <= 0:
                continue
            output[level][key] = {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": std,
            }
    return output


def calibrated_anomaly_score(
    row: dict[str, Any], calibration: dict[str, Any]
) -> tuple[float | None, str | None, int | None]:
    """Convert likelihood to Train-normalized surprisal using a fixed hierarchy."""

    for level in ("structure", "tool"):
        group = calibration[level].get(calibration_key(row, level))
        if group is not None:
            score = (group["mean"] - row["mean_logprob"]) / group["std"]
            return float(score), level, int(group["count"])
    return None, None, None


def _average_ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2
        ranks[order[position:end]] = rank
        position = end
    return ranks


def roc_auc(labels: list[int], scores: list[float]) -> float:
    """Compute binary AUROC with half credit for tied scores."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes")
    ranks = _average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    )


def _bootstrap_auc(
    rows: list[dict[str, Any]], *, samples: int, random_seed: int
) -> list[float]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    task_ids = sorted(by_task)
    rng = np.random.default_rng(random_seed)
    estimates = []
    for _ in range(samples):
        chosen = rng.choice(task_ids, size=len(task_ids), replace=True)
        sample_rows = [row for task_id in chosen for row in by_task[task_id]]
        estimates.append(
            roc_auc(
                [int(row["severity"] > 0) for row in sample_rows],
                [row["anomaly_score"] for row in sample_rows],
            )
        )
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _condition_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    per_severity = {}
    pair_values = []
    monotonic = []
    severity_order = []
    for severity in sorted({row["severity"] for row in rows if row["severity"] > 0}):
        comparisons = []
        margins = []
        for query_rows in by_query.values():
            clean = next(row for row in query_rows if row["severity"] == 0)
            anomaly = next(row for row in query_rows if row["severity"] == severity)
            comparisons.append(
                1.0
                if clean["mean_logprob"] > anomaly["mean_logprob"]
                else 0.5
                if clean["mean_logprob"] == anomaly["mean_logprob"]
                else 0.0
            )
            margins.append(clean["mean_logprob"] - anomaly["mean_logprob"])
        pair_values.extend(comparisons)
        per_severity[str(severity)] = {
            "query_count": len(comparisons),
            "clean_preferred_rate": float(np.mean(comparisons)),
            "mean_clean_minus_anomaly_logprob": float(np.mean(margins)),
        }
    for query_rows in by_query.values():
        ordered = sorted(query_rows, key=lambda row: row["severity"])
        scores = [row["mean_logprob"] for row in ordered]
        monotonic.append(float(all(a > b for a, b in zip(scores, scores[1:]))))
        for left in range(len(scores)):
            for right in range(left + 1, len(scores)):
                severity_order.append(
                    1.0
                    if scores[left] > scores[right]
                    else 0.5
                    if scores[left] == scores[right]
                    else 0.0
                )
    labels = [int(row["severity"] > 0) for row in rows]
    raw_scores = [-row["mean_logprob"] for row in rows]
    paired_rows = []
    for query_rows in by_query.values():
        clean_score = next(
            row["mean_logprob"] for row in query_rows if row["severity"] == 0
        )
        paired_rows.extend(
            {
                **row,
                "anomaly_score": clean_score - row["mean_logprob"],
            }
            for row in query_rows
        )
    mean_logprob_by_severity = {
        str(severity): float(
            np.mean(
                [row["mean_logprob"] for row in rows if row["severity"] == severity]
            )
        )
        for severity in sorted({row["severity"] for row in rows})
    }
    return {
        "query_count": len(by_query),
        "raw_surprisal_auroc": roc_auc(labels, raw_scores),
        "paired_clean_margin_auroc": roc_auc(
            [int(row["severity"] > 0) for row in paired_rows],
            [row["anomaly_score"] for row in paired_rows],
        ),
        "all_clean_vs_anomaly_preferred_rate": float(np.mean(pair_values)),
        "strictly_monotonic_query_rate": float(np.mean(monotonic)),
        "severity_order_accuracy": float(np.mean(severity_order)),
        "mean_logprob_by_severity": mean_logprob_by_severity,
        "per_severity": per_severity,
    }


def build_deviation_report(
    score_rows: list[dict[str, Any]],
    *,
    domains: tuple[str, ...],
    context_variants: tuple[str, ...],
    calibration_minimum_count: int,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Calibrate Train-clean scores and evaluate controlled Test anomalies."""

    train_rows = [row for row in score_rows if row["track"] == "calibration_clean"]
    test_rows = [row for row in score_rows if row["track"] == "controlled_anomaly"]
    calibration = fit_clean_calibration(
        train_rows, minimum_count=calibration_minimum_count
    )
    enriched = []
    for row in test_rows:
        calibrated, level, count = calibrated_anomaly_score(row, calibration)
        enriched.append(
            {
                **row,
                "anomaly_score": -row["mean_logprob"],
                "calibrated_anomaly_score": calibrated,
                "calibration_level": level,
                "calibration_count": count,
            }
        )
    reports = {}
    for domain_index, domain in enumerate(domains):
        conditions = {}
        for context in context_variants:
            selected = [
                row
                for row in enriched
                if row["domain"] == domain and row["context_variant"] == context
            ]
            condition = _condition_report(selected)
            condition["raw_surprisal_task_bootstrap_95_ci"] = _bootstrap_auc(
                selected,
                samples=bootstrap_samples,
                random_seed=random_seed + domain_index,
            )
            calibrated_rows = [
                {**row, "anomaly_score": row["calibrated_anomaly_score"]}
                for row in selected
                if row["calibrated_anomaly_score"] is not None
            ]
            condition["calibrated_row_count"] = len(calibrated_rows)
            condition["calibrated_query_count"] = len(
                {row["query_id"] for row in calibrated_rows}
            )
            if calibrated_rows:
                labels = [int(row["severity"] > 0) for row in calibrated_rows]
                scores = [row["anomaly_score"] for row in calibrated_rows]
                condition["calibrated_surprisal_auroc"] = roc_auc(labels, scores)
                condition["calibrated_surprisal_task_bootstrap_95_ci"] = _bootstrap_auc(
                    calibrated_rows,
                    samples=bootstrap_samples,
                    random_seed=random_seed + 100 + domain_index,
                )
            conditions[context] = condition
        full_rows = [
            row
            for row in enriched
            if row["domain"] == domain and row["context_variant"] == "full"
        ]
        tool_reports = {}
        for tool_name in sorted({row["tool_name"] for row in full_rows}):
            selected = [row for row in full_rows if row["tool_name"] == tool_name]
            condition = _condition_report(selected)
            calibrated = [
                row for row in selected if row["calibrated_anomaly_score"] is not None
            ]
            condition["calibrated_row_count"] = len(calibrated)
            if calibrated:
                condition["calibrated_surprisal_auroc"] = roc_auc(
                    [int(row["severity"] > 0) for row in calibrated],
                    [row["calibrated_anomaly_score"] for row in calibrated],
                )
            tool_reports[tool_name] = condition
        reports[domain] = {
            "conditions": conditions,
            "full_context_tool_breakdown": tool_reports,
            "calibration_level_counts": dict(
                Counter(row["calibration_level"] or "unavailable" for row in full_rows)
            ),
        }
    return (
        {
            "experiment": {
                "name": "Experiment B: controlled expected-result deviation",
                "new_rollout_used": False,
                "official_goal_used": False,
                "reference_action_used": False,
                "calibration_source": "official Train clean tool results",
                "evaluation_source": "official Test controlled corruptions",
            },
            "domains": reports,
        },
        enriched,
        calibration,
    )


def build_consistent_rematch_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank factual cross-task candidates after identity-preserving anonymization."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["query_id"], row["context_variant"])].append(row)
    by_domain_context: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for (_, context), candidates in grouped.items():
        positive = next(row for row in candidates if row["is_positive"])
        negatives = [row for row in candidates if not row["is_positive"]]
        comparisons = [
            1.0
            if positive["mean_logprob"] > row["mean_logprob"]
            else 0.5
            if positive["mean_logprob"] == row["mean_logprob"]
            else 0.0
            for row in negatives
        ]
        by_domain_context[(positive["domain"], context)].append(
            {
                "pairwise": float(np.mean(comparisons)),
                "hit_at_1": float(
                    positive["mean_logprob"]
                    > max(row["mean_logprob"] for row in negatives)
                ),
            }
        )
    report = {}
    for (domain, context), values in sorted(by_domain_context.items()):
        report.setdefault(domain, {})[context] = {
            "query_count": len(values),
            "pairwise_accuracy": float(np.mean([row["pairwise"] for row in values])),
            "hit_at_1": float(np.mean([row["hit_at_1"] for row in values])),
        }
    return report
