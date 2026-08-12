"""Held-out directions for immediate consequences of logged tool decisions."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from trace_to_micro.utils.io import decode_float16_vector

LABELS = ("goal_progress", "tool_success", "state_changed")
MOMENTS = ("before", "action", "result")


class InsufficientWithinToolClassesError(ValueError):
    """Raised when Train has no tool with both local outcome classes."""


def _auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = sum(left > right for left in positives for right in negatives)
    ties = sum(left == right for left in positives for right in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def _within_group_auc(
    labels: list[bool], scores: list[float], groups: list[str]
) -> tuple[float | None, int, int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[group].append(index)
    wins = 0.0
    pairs = 0
    eligible_groups = 0
    for indices in grouped.values():
        positives = [scores[index] for index in indices if labels[index]]
        negatives = [scores[index] for index in indices if not labels[index]]
        if not positives or not negatives:
            continue
        eligible_groups += 1
        wins += sum(left > right for left in positives for right in negatives)
        wins += 0.5 * sum(left == right for left in positives for right in negatives)
        pairs += len(positives) * len(negatives)
    return (wins / pairs if pairs else None), eligible_groups, pairs


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero activation vector")
    return vector / norm


def _eligible(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get(label) is not None]


def _fit_direction(
    rows: list[dict[str, Any]], layer_id: int, label: str
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[tuple[str, bool], list[np.ndarray]] = defaultdict(list)
    for row in _eligible(rows, label):
        grouped[(row["simulation_id"], bool(row[label]))].append(
            _unit(decode_float16_vector(row["activations"][str(layer_id)]))
        )
    class_vectors: dict[bool, list[np.ndarray]] = {False: [], True: []}
    for (_, value), vectors in grouped.items():
        class_vectors[value].append(np.mean(vectors, axis=0))
    if not class_vectors[False] or not class_vectors[True]:
        raise ValueError("Both local consequence classes are required in training")
    negative = np.mean(class_vectors[False], axis=0)
    positive = np.mean(class_vectors[True], axis=0)
    return _unit(positive - negative), 0.5 * (positive + negative)


def _fit_tool_centered_direction(
    rows: list[dict[str, Any]], layer_id: int, label: str
) -> tuple[np.ndarray, np.ndarray, set[str]]:
    selected = _eligible(rows, label)
    by_tool: dict[str, list[tuple[bool, np.ndarray]]] = defaultdict(list)
    for row in selected:
        vector = _unit(decode_float16_vector(row["activations"][str(layer_id)]))
        by_tool[row["tool_name"]].append((bool(row[label]), vector))
    eligible_tools = {
        tool
        for tool, values in by_tool.items()
        if {label_value for label_value, _ in values} == {False, True}
    }
    if not eligible_tools:
        raise InsufficientWithinToolClassesError(
            "Train has no tool with both local consequence classes"
        )
    class_vectors: dict[bool, list[np.ndarray]] = {False: [], True: []}
    for tool in eligible_tools:
        values = by_tool[tool]
        center = np.mean([vector for _, vector in values], axis=0)
        for value, vector in values:
            class_vectors[value].append(vector - center)
    negative = np.mean(class_vectors[False], axis=0)
    positive = np.mean(class_vectors[True], axis=0)
    return (
        _unit(positive - negative),
        0.5 * (positive + negative),
        eligible_tools,
    )


def _scores(
    rows: list[dict[str, Any]],
    *,
    layer_id: int,
    label: str,
    direction: np.ndarray,
    midpoint: np.ndarray,
) -> tuple[list[bool], list[float], list[str]]:
    selected = _eligible(rows, label)
    labels = [bool(row[label]) for row in selected]
    scores = [
        float(
            (_unit(decode_float16_vector(row["activations"][str(layer_id)])) - midpoint)
            @ direction
        )
        for row in selected
    ]
    return labels, scores, [row["simulation_id"] for row in selected]


def _tool_centered_scores(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    layer_id: int,
    label: str,
    direction: np.ndarray,
    midpoint: np.ndarray,
    eligible_tools: set[str],
) -> tuple[list[bool], list[float], list[str], list[str]]:
    train = _eligible(train, label)
    test = [row for row in _eligible(test, label) if row["tool_name"] in eligible_tools]
    train_by_tool: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in train:
        train_by_tool[row["tool_name"]].append(
            _unit(decode_float16_vector(row["activations"][str(layer_id)]))
        )
    centers = {
        tool: np.mean(vectors, axis=0) for tool, vectors in train_by_tool.items()
    }
    labels = []
    scores = []
    simulation_ids = []
    tools = []
    for row in test:
        vector = _unit(decode_float16_vector(row["activations"][str(layer_id)]))
        centered = vector - centers[row["tool_name"]]
        labels.append(bool(row[label]))
        scores.append(float((centered - midpoint) @ direction))
        simulation_ids.append(row["simulation_id"])
        tools.append(row["tool_name"])
    return labels, scores, simulation_ids, tools


def _cluster_bootstrap_interval(
    labels: list[bool],
    scores: list[float],
    simulation_ids: list[str],
    *,
    samples: int,
    random_seed: int,
) -> list[float] | None:
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, simulation_id in enumerate(simulation_ids):
        clusters[simulation_id].append(index)
    ids = sorted(clusters)
    if _auc(labels, scores) is None:
        return None
    rng = np.random.default_rng(random_seed)
    estimates = []
    for _ in range(samples):
        sampled_ids = rng.choice(ids, size=len(ids), replace=True)
        indices = [index for key in sampled_ids for index in clusters[str(key)]]
        estimate = _auc(
            [labels[index] for index in indices],
            [scores[index] for index in indices],
        )
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        return None
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _cluster_bootstrap_within_group_interval(
    labels: list[bool],
    scores: list[float],
    groups: list[str],
    simulation_ids: list[str],
    *,
    samples: int,
    random_seed: int,
) -> list[float] | None:
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, simulation_id in enumerate(simulation_ids):
        clusters[simulation_id].append(index)
    ids = sorted(clusters)
    if _within_group_auc(labels, scores, groups)[0] is None:
        return None
    rng = np.random.default_rng(random_seed)
    estimates = []
    for _ in range(samples):
        sampled_ids = rng.choice(ids, size=len(ids), replace=True)
        indices = [index for key in sampled_ids for index in clusters[str(key)]]
        estimate = _within_group_auc(
            [labels[index] for index in indices],
            [scores[index] for index in indices],
            [groups[index] for index in indices],
        )[0]
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        return None
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _tool_prior_baseline(
    train: list[dict[str, Any]], test: list[dict[str, Any]], label: str
) -> dict[str, Any]:
    train = _eligible(train, label)
    test = _eligible(test, label)
    global_rate = sum(bool(row[label]) for row in train) / len(train)
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in train:
        counts[row["tool_name"]][bool(row[label])] += 1
    scores = [
        (
            counts[row["tool_name"]][True] / sum(counts[row["tool_name"]].values())
            if row["tool_name"] in counts
            else global_rate
        )
        for row in test
    ]
    return {
        "name": "train tool-name positive rate",
        "auroc": _auc([bool(row[label]) for row in test], scores),
        "unseen_test_decisions": sum(row["tool_name"] not in counts for row in test),
    }


def _scalar_baseline(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    label: str,
    field: str,
) -> dict[str, Any]:
    train = [row for row in _eligible(train, label) if row.get(field) is not None]
    test = [row for row in _eligible(test, label) if row.get(field) is not None]
    train_auc = _auc(
        [bool(row[label]) for row in train], [float(row[field]) for row in train]
    )
    sign = -1.0 if train_auc is not None and train_auc < 0.5 else 1.0
    return {
        "field": field,
        "direction_selected_on_train": "negative" if sign < 0 else "positive",
        "auroc": _auc(
            [bool(row[label]) for row in test],
            [sign * float(row[field]) for row in test],
        ),
    }


def _evaluate(
    rows: list[dict[str, Any]],
    *,
    domain: str,
    train_split: str,
    test_split: str,
    moment: str,
    label: str,
    layer_id: int,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    train = [
        row
        for row in rows
        if row["domain"] == domain
        and row["split"] == train_split
        and row["moment"] == moment
    ]
    test = [
        row
        for row in rows
        if row["domain"] == domain
        and row["split"] == test_split
        and row["moment"] == moment
    ]
    train_eligible = _eligible(train, label)
    test_eligible = _eligible(test, label)
    train_counts = Counter(bool(row[label]) for row in train_eligible)
    test_counts = Counter(bool(row[label]) for row in test_eligible)
    if len(train_counts) < 2 or len(test_counts) < 2:
        return {
            "eligible": False,
            "reason": "train or test split does not contain both label classes",
            "training_positive_count": train_counts[True],
            "training_negative_count": train_counts[False],
            "positive_count": test_counts[True],
            "negative_count": test_counts[False],
        }
    direction, midpoint = _fit_direction(train, layer_id, label)
    labels, scores, simulation_ids = _scores(
        test,
        layer_id=layer_id,
        label=label,
        direction=direction,
        midpoint=midpoint,
    )
    tool_names = [row["tool_name"] for row in test_eligible]
    within_tool_auc, eligible_tool_count, within_tool_pairs = _within_group_auc(
        labels, scores, tool_names
    )
    centered_fields = {
        "tool_centered_within_tool_auroc": None,
        "tool_centered_train_eligible_tools": [],
        "tool_centered_test_eligible_tool_count": 0,
        "tool_centered_positive_negative_pairs": 0,
        "tool_centered_trajectory_cluster_bootstrap_95_ci": None,
    }
    try:
        (
            centered_direction,
            centered_midpoint,
            centered_train_tools,
        ) = _fit_tool_centered_direction(train, layer_id, label)
    except InsufficientWithinToolClassesError:
        pass
    else:
        (
            centered_labels,
            centered_scores,
            centered_simulation_ids,
            centered_tools,
        ) = _tool_centered_scores(
            train,
            test,
            layer_id=layer_id,
            label=label,
            direction=centered_direction,
            midpoint=centered_midpoint,
            eligible_tools=centered_train_tools,
        )
        centered_auc, centered_tool_count, centered_pairs = _within_group_auc(
            centered_labels, centered_scores, centered_tools
        )
        centered_fields = {
            "tool_centered_within_tool_auroc": centered_auc,
            "tool_centered_train_eligible_tools": sorted(centered_train_tools),
            "tool_centered_test_eligible_tool_count": centered_tool_count,
            "tool_centered_positive_negative_pairs": centered_pairs,
            "tool_centered_trajectory_cluster_bootstrap_95_ci": (
                _cluster_bootstrap_within_group_interval(
                    centered_labels,
                    centered_scores,
                    centered_tools,
                    centered_simulation_ids,
                    samples=bootstrap_samples,
                    random_seed=random_seed,
                )
            ),
        }
    return {
        "eligible": True,
        "training_decision_count": len(train_eligible),
        "training_trajectory_count": len(
            {row["simulation_id"] for row in train_eligible}
        ),
        "training_positive_count": train_counts[True],
        "training_negative_count": train_counts[False],
        "decision_count": len(test_eligible),
        "trajectory_count": len({row["simulation_id"] for row in test_eligible}),
        "positive_count": test_counts[True],
        "negative_count": test_counts[False],
        "auroc": _auc(labels, scores),
        "trajectory_cluster_bootstrap_95_ci": _cluster_bootstrap_interval(
            labels,
            scores,
            simulation_ids,
            samples=bootstrap_samples,
            random_seed=random_seed,
        ),
        "within_tool_auroc": within_tool_auc,
        "within_tool_eligible_tool_count": eligible_tool_count,
        "within_tool_positive_negative_pairs": within_tool_pairs,
        "within_tool_trajectory_cluster_bootstrap_95_ci": (
            _cluster_bootstrap_within_group_interval(
                labels,
                scores,
                tool_names,
                simulation_ids,
                samples=bootstrap_samples,
                random_seed=random_seed,
            )
        ),
        **centered_fields,
        "baselines": [
            _tool_prior_baseline(train, test, label),
            *[
                _scalar_baseline(train, test, label, field)
                for field in (
                    "target_distance_before",
                    "decision_position",
                    "prior_tool_errors",
                    "prompt_tokens",
                )
            ],
        ],
    }


def build_local_consequence_report(
    rows: list[dict[str, Any]],
    *,
    domains: tuple[str, ...],
    train_split: str,
    test_split: str,
    layer_ids: tuple[int, ...],
    primary_layer_id: int,
    primary_moment: str,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    """Evaluate preregistered local progress and auxiliary consequence labels."""

    evaluations = {}
    for domain in domains:
        for moment in MOMENTS:
            for label in LABELS:
                for layer_id in layer_ids:
                    key = f"{domain}:{moment}:{label}:layer_{layer_id}"
                    is_primary = (
                        moment == primary_moment
                        and label == "goal_progress"
                        and layer_id == primary_layer_id
                    )
                    evaluations[key] = _evaluate(
                        rows,
                        domain=domain,
                        train_split=train_split,
                        test_split=test_split,
                        moment=moment,
                        label=label,
                        layer_id=layer_id,
                        bootstrap_samples=(bootstrap_samples if is_primary else 0),
                        random_seed=random_seed,
                    )
    primary_keys = [
        f"{domain}:{primary_moment}:goal_progress:layer_{primary_layer_id}"
        for domain in domains
    ]
    primary = {key: evaluations[key] for key in primary_keys}
    evaluable = [
        value["eligible"] and value.get("tool_centered_within_tool_auroc") is not None
        for value in primary.values()
    ]
    passes = []
    for value in primary.values():
        interval = value.get("tool_centered_trajectory_cluster_bootstrap_95_ci")
        passes.append(
            value["eligible"]
            and value.get("tool_centered_within_tool_auroc") is not None
            and value["tool_centered_within_tool_auroc"] > 0.5
            and interval is not None
            and interval[0] > 0.5
        )
    if not all(evaluable):
        status = "not_evaluable_within_tool_in_both_domains"
    elif all(passes):
        status = "consistent_local_progress_direction"
    elif any(passes):
        status = "partial_local_progress_direction"
    else:
        status = "no_detectable_local_progress_direction"
    return {
        "experiment": {
            "hypothesis": (
                "Given the logged state and emitted mutating tool call but before "
                "its result, a simple hidden-state direction predicts whether the "
                "real next state is closer to the official target state."
            ),
            "data_source": "complete logged trajectories; no new agent rollout",
            "train_split": train_split,
            "test_split": test_split,
            "primary_moment": primary_moment,
            "primary_label": "goal_progress",
            "primary_layer_id": primary_layer_id,
            "primary_metric": "tool-centered within-tool AUROC",
            "unit": "tool decision; uncertainty is clustered by trajectory",
        },
        "primary_test": {
            "evidence_rule": (
                "Both domains require tool-centered within-tool AUROC > 0.5 "
                "and a trajectory-cluster bootstrap 95% lower bound > 0.5."
            ),
            "evidence_status": status,
            "evaluations": primary,
        },
        "evaluations": evaluations,
    }
