"""Train-only calibration and Test metrics for result-deviation scores."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

PROBABILITY_PROBE_SCORE_FIELDS = (
    "full_mean_surprisal",
    "full_min_k_surprisal",
    "mean_contextual_deviation",
    "contextual_min_k_deviation",
)
PRIMARY_PROBABILITY_PROBE_SCORE = "contextual_min_k_deviation"


def contextual_min_k_statistics(
    full_score: dict[str, Any],
    action_only_score: dict[str, Any],
    *,
    fraction: float,
) -> dict[str, float | int]:
    """Reduce aligned token likelihoods to absolute and contextual tail scores."""

    if not 0 < fraction <= 1:
        raise ValueError("Min-K fraction must be in (0, 1]")
    full_ids = full_score["suffix_token_ids"]
    action_ids = action_only_score["suffix_token_ids"]
    if full_ids != action_ids:
        raise ValueError("Full and action-only suffix token IDs do not match")
    full = np.asarray(full_score["token_logprobs"], dtype=float)
    action = np.asarray(action_only_score["token_logprobs"], dtype=float)
    if len(full) == 0 or len(full) != len(action) or len(full) != len(full_ids):
        raise ValueError("Aligned token likelihood vectors have invalid lengths")
    count = max(1, int(np.ceil(len(full) * fraction)))
    full_surprisal = -full
    contextual_deviation = action - full
    return {
        "suffix_token_count": len(full),
        "min_k_token_count": count,
        "full_mean_surprisal": float(np.mean(full_surprisal)),
        "full_min_k_surprisal": float(np.mean(np.sort(full_surprisal)[-count:])),
        "mean_contextual_deviation": float(np.mean(contextual_deviation)),
        "contextual_min_k_deviation": float(
            np.mean(np.sort(contextual_deviation)[-count:])
        ),
    }


def probability_probe_calibration_key(row: dict[str, Any]) -> str:
    """Group one probability probe row by domain and selected tool."""

    return f"{row['domain']}|{row['tool_name']}"


def fit_probability_probe_calibration(
    rows: list[dict[str, Any]], *, minimum_count: int
) -> dict[str, Any]:
    """Fit tool-conditional centers and scales from Train clean results only."""

    if minimum_count < 2:
        raise ValueError("Calibration minimum_count must be at least two")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[probability_probe_calibration_key(row)].append(row)
    groups = {}
    for key, values in sorted(grouped.items()):
        if len(values) < minimum_count:
            continue
        statistics = {}
        for field in PROBABILITY_PROBE_SCORE_FIELDS:
            scores = np.asarray([row[field] for row in values], dtype=float)
            std = float(np.std(scores, ddof=1))
            if std > 0:
                statistics[field] = {
                    "mean": float(np.mean(scores)),
                    "std": std,
                }
        if PRIMARY_PROBABILITY_PROBE_SCORE in statistics:
            groups[key] = {
                "count": len(values),
                "statistics": statistics,
            }
    return {
        "source": "official Train clean tool results",
        "group_by": ["domain", "tool_name"],
        "minimum_count": minimum_count,
        "score_fields": list(PROBABILITY_PROBE_SCORE_FIELDS),
        "groups": groups,
    }


def calibrate_probability_probe_row(
    row: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    """Attach Train-standardized probability probe scores to one Test row."""

    group = calibration["groups"].get(probability_probe_calibration_key(row))
    calibrated = {
        "calibration_level": None,
        "calibration_count": None,
    }
    for field in PROBABILITY_PROBE_SCORE_FIELDS:
        calibrated[f"{field}_z"] = None
    if group is None:
        return calibrated
    calibrated["calibration_level"] = "tool"
    calibrated["calibration_count"] = int(group["count"])
    for field, statistics in group["statistics"].items():
        calibrated[f"{field}_z"] = float(
            (row[field] - statistics["mean"]) / statistics["std"]
        )
    return calibrated


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


def average_precision(labels: list[int], scores: list[float]) -> float:
    """Compute threshold-grouped average precision with deterministic ties."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Average precision requires both classes")
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    previous_recall = 0.0
    result = 0.0
    for threshold in sorted(set(scores), reverse=True):
        predicted = scores_array >= threshold
        true_positives = int(np.sum(labels_array[predicted]))
        precision = true_positives / int(np.sum(predicted))
        recall = true_positives / positives
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return float(result)


def _probability_probe_bootstrap(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    samples: int,
    random_seed: int,
) -> dict[str, list[float]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    task_ids = sorted(by_task)
    rng = np.random.default_rng(random_seed)
    auc_estimates = []
    ap_estimates = []
    for _ in range(samples):
        chosen = rng.choice(task_ids, size=len(task_ids), replace=True)
        selected = [row for task_id in chosen for row in by_task[task_id]]
        labels = [int(row["severity"] > 0) for row in selected]
        scores = [row[score_field] for row in selected]
        auc_estimates.append(roc_auc(labels, scores))
        ap_estimates.append(average_precision(labels, scores))
    return {
        "auroc": [float(value) for value in np.quantile(auc_estimates, [0.025, 0.975])],
        "average_precision": [
            float(value) for value in np.quantile(ap_estimates, [0.025, 0.975])
        ],
    }


def _probability_score_report(
    rows: list[dict[str, Any]], *, score_field: str
) -> dict[str, float]:
    labels = [int(row["severity"] > 0) for row in rows]
    scores = [row[score_field] for row in rows]
    return {
        "auroc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "positive_prevalence": float(np.mean(labels)),
    }


def _probability_probe_ordering(
    rows: list[dict[str, Any]], *, score_field: str
) -> dict[str, Any]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    per_severity = {}
    all_clean_comparisons = []
    strictly_monotonic = []
    severity_comparisons = []
    for severity in sorted({row["severity"] for row in rows if row["severity"] > 0}):
        preferred = []
        margins = []
        for query_rows in by_query.values():
            clean = next(row for row in query_rows if row["severity"] == 0)
            anomaly = next(row for row in query_rows if row["severity"] == severity)
            difference = anomaly[score_field] - clean[score_field]
            preferred.append(1.0 if difference > 0 else 0.5 if difference == 0 else 0.0)
            margins.append(difference)
        all_clean_comparisons.extend(preferred)
        per_severity[str(severity)] = {
            "query_count": len(preferred),
            "anomaly_score_higher_rate": float(np.mean(preferred)),
            "mean_anomaly_minus_clean": float(np.mean(margins)),
        }
    for query_rows in by_query.values():
        ordered = sorted(query_rows, key=lambda row: row["severity"])
        scores = [row[score_field] for row in ordered]
        strictly_monotonic.append(float(all(a < b for a, b in zip(scores, scores[1:]))))
        for left in range(len(scores)):
            for right in range(left + 1, len(scores)):
                severity_comparisons.append(
                    1.0
                    if scores[left] < scores[right]
                    else 0.5
                    if scores[left] == scores[right]
                    else 0.0
                )
    return {
        "all_clean_vs_anomaly_preferred_rate": float(np.mean(all_clean_comparisons)),
        "strictly_monotonic_query_rate": float(np.mean(strictly_monotonic)),
        "severity_order_accuracy": float(np.mean(severity_comparisons)),
        "per_severity": per_severity,
    }


def _three_sigma_report(
    rows: list[dict[str, Any]], *, score_field: str, threshold: float
) -> dict[str, Any]:
    clean = [row for row in rows if row["severity"] == 0]
    anomalies = [row for row in rows if row["severity"] > 0]
    false_positives = sum(row[score_field] > threshold for row in clean)
    true_positives = sum(row[score_field] > threshold for row in anomalies)
    predicted_positives = false_positives + true_positives
    true_negatives = len(clean) - false_positives
    false_negatives = len(anomalies) - true_positives
    precision = true_positives / predicted_positives if predicted_positives else 0.0
    recall = true_positives / len(anomalies)
    return {
        "threshold": threshold,
        "comparison": "score > threshold",
        "clean_count": len(clean),
        "anomaly_count": len(anomalies),
        "true_positive": true_positives,
        "false_positive": false_positives,
        "true_negative": true_negatives,
        "false_negative": false_negatives,
        "false_positive_rate": false_positives / len(clean),
        "true_positive_rate": recall,
        "specificity": true_negatives / len(clean),
        "balanced_accuracy": (recall + true_negatives / len(clean)) / 2,
        "precision": precision,
        "f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "per_severity": {
            str(severity): {
                "count": sum(row["severity"] == severity for row in anomalies),
                "true_positive_rate": float(
                    np.mean(
                        [
                            row[score_field] > threshold
                            for row in anomalies
                            if row["severity"] == severity
                        ]
                    )
                ),
            }
            for severity in sorted({row["severity"] for row in anomalies})
        },
    }


def _severity_summary(
    rows: list[dict[str, Any]], *, score_field: str
) -> dict[str, dict[str, float | int]]:
    return {
        str(severity): {
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
        }
        for severity in sorted({row["severity"] for row in rows})
        if (values := [row[score_field] for row in rows if row["severity"] == severity])
    }


def build_contextual_min_k_report(
    score_rows: list[dict[str, Any]],
    *,
    domains: tuple[str, ...],
    calibration_minimum_count: int,
    min_k_fraction: float,
    sigma_threshold: float,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one contextual Min-K probe with Train-only tool calibration."""

    train_rows = [row for row in score_rows if row["track"] == "min_k_calibration"]
    test_rows = [row for row in score_rows if row["track"] == "min_k_controlled"]
    calibration = fit_probability_probe_calibration(
        train_rows, minimum_count=calibration_minimum_count
    )
    enriched = [
        {**row, **calibrate_probability_probe_row(row, calibration)}
        for row in test_rows
    ]
    domain_reports = {}
    for domain_index, domain in enumerate(domains):
        selected = [row for row in enriched if row["domain"] == domain]
        score_metrics = {}
        for field in PROBABILITY_PROBE_SCORE_FIELDS:
            calibrated_field = f"{field}_z"
            calibrated = [row for row in selected if row[calibrated_field] is not None]
            score_metrics[field] = {
                "raw": _probability_score_report(selected, score_field=field),
                "train_calibrated": _probability_score_report(
                    calibrated, score_field=calibrated_field
                ),
                "calibrated_row_count": len(calibrated),
            }
        primary_z = f"{PRIMARY_PROBABILITY_PROBE_SCORE}_z"
        primary_rows = [row for row in selected if row[primary_z] is not None]
        primary = {
            "score_field": PRIMARY_PROBABILITY_PROBE_SCORE,
            "raw": _probability_score_report(
                selected, score_field=PRIMARY_PROBABILITY_PROBE_SCORE
            ),
            "train_calibrated": _probability_score_report(
                primary_rows, score_field=primary_z
            ),
            "train_calibrated_task_bootstrap_95_ci": _probability_probe_bootstrap(
                primary_rows,
                score_field=primary_z,
                samples=bootstrap_samples,
                random_seed=random_seed + domain_index,
            ),
            "paired_ordering": _probability_probe_ordering(
                selected, score_field=PRIMARY_PROBABILITY_PROBE_SCORE
            ),
            "raw_by_severity": _severity_summary(
                selected, score_field=PRIMARY_PROBABILITY_PROBE_SCORE
            ),
            "calibrated_z_by_severity": _severity_summary(
                primary_rows, score_field=primary_z
            ),
            "three_sigma": _three_sigma_report(
                primary_rows,
                score_field=primary_z,
                threshold=sigma_threshold,
            ),
        }
        tool_reports = {}
        for tool_name in sorted({row["tool_name"] for row in primary_rows}):
            tool_rows = [row for row in primary_rows if row["tool_name"] == tool_name]
            tool_reports[tool_name] = {
                "query_count": len({row["query_id"] for row in tool_rows}),
                "train_calibrated": _probability_score_report(
                    tool_rows, score_field=primary_z
                ),
                "paired_ordering": _probability_probe_ordering(
                    tool_rows, score_field=PRIMARY_PROBABILITY_PROBE_SCORE
                ),
                "three_sigma": _three_sigma_report(
                    tool_rows,
                    score_field=primary_z,
                    threshold=sigma_threshold,
                ),
            }
        domain_reports[domain] = {
            "row_count": len(selected),
            "query_count": len({row["query_id"] for row in selected}),
            "task_count": len({row["task_id"] for row in selected}),
            "calibrated_row_count": len(primary_rows),
            "calibrated_query_count": len({row["query_id"] for row in primary_rows}),
            "uncalibrated_tools": sorted(
                {row["tool_name"] for row in selected if row[primary_z] is None}
            ),
            "score_metrics": score_metrics,
            "primary": primary,
            "tool_breakdown": tool_reports,
        }
    return (
        {
            "experiment": {
                "name": "Contextual Min-K probability probe",
                "new_rollout_used": False,
                "official_goal_used": False,
                "reference_action_used": False,
                "min_k_fraction": min_k_fraction,
                "primary_score": PRIMARY_PROBABILITY_PROBE_SCORE,
                "calibration_source": "official Train clean tool results only",
                "calibration_group_by": ["domain", "tool_name"],
                "evaluation_source": "official Test controlled corruptions",
                "sigma_threshold": sigma_threshold,
            },
            "domains": domain_reports,
        },
        enriched,
        calibration,
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
