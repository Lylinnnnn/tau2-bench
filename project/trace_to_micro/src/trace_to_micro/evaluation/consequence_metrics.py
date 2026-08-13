"""Metrics and task-cluster uncertainty for expected consequences."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np


def target_group(target: str) -> str:
    """Map one consequence head to an equal-weighted consequence family."""

    if target.startswith("execution."):
        return "execution"
    if target.startswith("output."):
        return "tool_output"
    return "state_transition"


def auc(labels: list[bool], scores: list[float]) -> float | None:
    """Compute tie-aware binary AUROC without an external ML dependency."""

    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = sum(left > right for left in positives for right in negatives)
    ties = sum(left == right for left in positives for right in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def _average_precision(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    grouped: dict[float, Counter] = defaultdict(Counter)
    for label, score in zip(labels, scores):
        grouped[float(score)][bool(label)] += 1
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    for score in sorted(grouped, reverse=True):
        true_positives += grouped[score][True]
        false_positives += grouped[score][False]
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def _within_tool_auc(
    labels: list[bool], scores: list[float], tools: list[str]
) -> tuple[float | None, int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, tool in enumerate(tools):
        grouped[tool].append(index)
    wins = 0.0
    pairs = 0
    for indices in grouped.values():
        positive = [scores[index] for index in indices if labels[index]]
        negative = [scores[index] for index in indices if not labels[index]]
        wins += sum(left > right for left in positive for right in negative)
        wins += 0.5 * sum(left == right for left in positive for right in negative)
        pairs += len(positive) * len(negative)
    return (wins / pairs if pairs else None), pairs


def head_metrics(
    actual: np.ndarray, predicted: np.ndarray, target_names: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Report per-head and equal-family-weighted consequence metrics."""

    heads = {}
    for index, target in enumerate(target_names):
        labels = actual[:, index].astype(bool).tolist()
        scores = predicted[:, index].tolist()
        heads[target] = {
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
            "auroc": auc(labels, scores),
            "average_precision": _average_precision(labels, scores),
            "brier": float(np.mean((predicted[:, index] - actual[:, index]) ** 2)),
        }
    eligible = [row for row in heads.values() if row["auroc"] is not None]
    macro = {
        "evaluable_head_count": len(eligible),
        "macro_auroc": (
            float(np.mean([row["auroc"] for row in eligible])) if eligible else None
        ),
        "macro_average_precision": (
            float(np.mean([row["average_precision"] for row in eligible]))
            if eligible
            else None
        ),
        "macro_brier": (
            float(np.mean([row["brier"] for row in eligible])) if eligible else None
        ),
    }
    groups = {}
    for group in ("execution", "state_transition", "tool_output"):
        selected = [
            heads[target] for target in target_names if target_group(target) == group
        ]
        eligible_group = [row for row in selected if row["auroc"] is not None]
        groups[group] = {
            "trained_head_count": len(selected),
            "evaluable_head_count": len(eligible_group),
            "macro_auroc": (
                float(np.mean([row["auroc"] for row in eligible_group]))
                if eligible_group
                else None
            ),
            "macro_brier": (
                float(np.mean([row["brier"] for row in selected])) if selected else None
            ),
        }
    evaluable_groups = [
        row for row in groups.values() if row["macro_auroc"] is not None
    ]
    macro["consequence_family_balanced_auroc"] = (
        float(np.mean([row["macro_auroc"] for row in evaluable_groups]))
        if evaluable_groups
        else None
    )
    macro["consequence_family_balanced_brier"] = float(
        np.mean(
            [
                row["macro_brier"]
                for row in groups.values()
                if row["macro_brier"] is not None
            ]
        )
    )
    return heads, {"all_heads": macro, "by_consequence_family": groups}


def surprise(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Compute row-level mean squared expected-versus-observed deviation."""

    return np.mean((actual - predicted) ** 2, axis=1)


def _family_balanced_metric(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_names: list[str],
    *,
    metric: str,
) -> float | None:
    group_values = []
    for group in ("execution", "state_transition", "tool_output"):
        values = []
        for index, target in enumerate(target_names):
            if target_group(target) != group:
                continue
            labels = actual[:, index].astype(bool).tolist()
            if metric == "auroc":
                value = auc(labels, predicted[:, index].tolist())
            else:
                value = float(np.mean((predicted[:, index] - actual[:, index]) ** 2))
            if value is not None:
                values.append(value)
        if values:
            group_values.append(float(np.mean(values)))
    return float(np.mean(group_values)) if group_values else None


def cluster_bootstrap_delta(
    actual: np.ndarray,
    combined: np.ndarray,
    surface: np.ndarray,
    target_names: list[str],
    task_ids: list[str],
    *,
    metric: str,
    samples: int,
    random_seed: int,
) -> list[float] | None:
    """Bootstrap a combined-minus-surface metric delta by task id."""

    by_task: dict[str, list[int]] = defaultdict(list)
    for index, task_id in enumerate(task_ids):
        by_task[task_id].append(index)
    task_keys = sorted(by_task)
    rng = np.random.default_rng(random_seed)
    estimates = []
    for _ in range(samples):
        indices = [
            index
            for task_id in rng.choice(task_keys, size=len(task_keys), replace=True)
            for index in by_task[str(task_id)]
        ]
        combined_value = _family_balanced_metric(
            actual[indices], combined[indices], target_names, metric=metric
        )
        surface_value = _family_balanced_metric(
            actual[indices], surface[indices], target_names, metric=metric
        )
        if combined_value is not None and surface_value is not None:
            estimates.append(combined_value - surface_value)
    if not estimates:
        return None
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _error_detection_metrics(
    rows: list[dict[str, Any]], surprises: np.ndarray
) -> dict[str, Any]:
    errors = [not bool(row["tool_success"]) for row in rows]
    scores = surprises.tolist()
    within_tool, pairs = _within_tool_auc(
        errors, scores, [row["tool_name"] for row in rows]
    )
    return {
        "error_count": sum(errors),
        "non_error_count": len(errors) - sum(errors),
        "error_detection_auroc": auc(errors, scores),
        "error_detection_average_precision": _average_precision(errors, scores),
        "within_tool_error_detection_auroc": within_tool,
        "within_tool_positive_negative_pairs": pairs,
        "mean_surprise_for_error": (
            float(np.mean([score for score, error in zip(scores, errors) if error]))
            if any(errors)
            else None
        ),
        "mean_surprise_for_non_error": (
            float(np.mean([score for score, error in zip(scores, errors) if not error]))
            if not all(errors)
            else None
        ),
    }


def feedback_metrics(
    rows: list[dict[str, Any]],
    actual: np.ndarray,
    predicted: np.ndarray,
    target_names: list[str],
) -> dict[str, Any]:
    """Score whether expected-versus-observed deviation exposes tool errors."""

    all_head_surprise = surprise(actual, predicted)
    semantic_indices = [
        index
        for index, target in enumerate(target_names)
        if target != "execution.success"
    ]
    semantic_surprise = surprise(
        actual[:, semantic_indices], predicted[:, semantic_indices]
    )
    return {
        "definition": (
            "mean squared deviation between the pre-result expected "
            "consequence and the observed consequence"
        ),
        "all_heads": _error_detection_metrics(rows, all_head_surprise),
        "without_execution_success_head": _error_detection_metrics(
            rows, semantic_surprise
        ),
    }


def optional_delta(left: float | None, right: float | None) -> float | None:
    """Subtract two optional metrics only when both are evaluable."""

    if left is None or right is None:
        return None
    return left - right
