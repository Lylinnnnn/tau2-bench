"""Non-parametric success-direction evaluation over frozen hidden states."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from trace_to_micro.utils.io import decode_float16_vector

LABELS = ("overall_success", "db_success", "communication_success")
MOMENTS = ("before", "action", "result")


class InsufficientClassesError(ValueError):
    """Raised when a direction cannot be fitted from both outcome classes."""


def _auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = sum(left > right for left in positives for right in negatives)
    ties = sum(left == right for left in positives for right in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def _balanced_accuracy(labels: list[bool], scores: list[float]) -> float | None:
    positives = [score >= 0 for label, score in zip(labels, scores) if label]
    negatives = [score < 0 for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    return 0.5 * (sum(positives) / len(positives) + sum(negatives) / len(negatives))


def _bootstrap_auc_interval(
    labels: list[bool],
    scores: list[float],
    *,
    samples: int,
    random_seed: int,
) -> list[float] | None:
    if samples <= 0:
        return None
    positive_indices = [index for index, label in enumerate(labels) if label]
    negative_indices = [index for index, label in enumerate(labels) if not label]
    if not positive_indices or not negative_indices:
        return None
    rng = np.random.default_rng(random_seed)
    estimates = []
    for _ in range(samples):
        indices = rng.choice(
            positive_indices, size=len(positive_indices), replace=True
        ).tolist()
        indices += rng.choice(
            negative_indices, size=len(negative_indices), replace=True
        ).tolist()
        estimate = _auc(
            [labels[index] for index in indices],
            [scores[index] for index in indices],
        )
        assert estimate is not None
        estimates.append(estimate)
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _permutation_p_value(
    labels: list[bool],
    scores: list[float],
    *,
    samples: int,
    random_seed: int,
) -> float | None:
    if samples <= 0:
        return None
    observed = _auc(labels, scores)
    if observed is None:
        return None
    rng = np.random.default_rng(random_seed)
    shuffled = np.asarray(labels, dtype=bool)
    exceedances = 0
    for _ in range(samples):
        candidate = rng.permutation(shuffled).tolist()
        estimate = _auc(candidate, scores)
        assert estimate is not None
        exceedances += estimate >= observed
    return (exceedances + 1) / (samples + 1)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero activation vector")
    return vector / norm


def _fit_direction(rows: list[dict[str, Any]], layer_id: int, label: str) -> tuple:
    by_trajectory: dict[str, list[np.ndarray]] = defaultdict(list)
    trajectory_labels: dict[str, bool] = {}
    for row in rows:
        value = row[label]
        if value is not None:
            by_trajectory[row["simulation_id"]].append(
                _unit(decode_float16_vector(row["activations"][str(layer_id)]))
            )
            trajectory_labels[row["simulation_id"]] = bool(value)
    groups: dict[bool, list[np.ndarray]] = {False: [], True: []}
    for simulation_id, vectors in by_trajectory.items():
        groups[trajectory_labels[simulation_id]].append(_unit(np.mean(vectors, axis=0)))
    if not groups[False] or not groups[True]:
        raise InsufficientClassesError("Both success and failure examples are required")
    failure = np.mean(groups[False], axis=0)
    success = np.mean(groups[True], axis=0)
    direction = _unit(success - failure)
    midpoint = 0.5 * (success + failure)
    return direction, midpoint


def _scalar_baseline(
    train: list[dict[str, Any]], test: list[dict[str, Any]], label: str, field: str
) -> dict[str, Any]:
    def aggregate(rows: list[dict[str, Any]]) -> tuple[list[bool], list[float]]:
        values: dict[str, list[float]] = defaultdict(list)
        labels: dict[str, bool] = {}
        for row in rows:
            if row[label] is None or row.get(field) is None:
                continue
            values[row["simulation_id"]].append(float(row[field]))
            labels[row["simulation_id"]] = bool(row[label])
        ids = sorted(values)
        return (
            [labels[key] for key in ids],
            [float(np.mean(values[key])) for key in ids],
        )

    train_labels, train_scores = aggregate(train)
    train_auc = _auc(train_labels, train_scores)
    sign = -1.0 if train_auc is not None and train_auc < 0.5 else 1.0
    test_labels, test_scores = aggregate(test)
    adjusted = [sign * score for score in test_scores]
    return {
        "field": field,
        "direction_selected_on_train": "negative" if sign < 0 else "positive",
        "trajectory_count": len(test_labels),
        "auroc": _auc(test_labels, adjusted),
    }


def _trajectory_scores(
    rows: list[dict[str, Any]],
    *,
    layer_id: int,
    label: str,
    direction: np.ndarray,
    midpoint: np.ndarray,
) -> tuple[list[bool], list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, bool] = {}
    for row in rows:
        value = row[label]
        if value is None:
            continue
        vector = _unit(decode_float16_vector(row["activations"][str(layer_id)]))
        grouped[row["simulation_id"]].append(float((vector - midpoint) @ direction))
        labels[row["simulation_id"]] = bool(value)
    ids = sorted(grouped)
    return [labels[key] for key in ids], [float(np.mean(grouped[key])) for key in ids]


def _evaluate_transfer(
    rows: list[dict[str, Any]],
    *,
    source_domain: str,
    target_domain: str,
    train_split: str,
    test_split: str,
    layer_id: int,
    moment: str,
    label: str,
    bootstrap_samples: int,
    permutation_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    train = [
        row
        for row in rows
        if row["domain"] == source_domain
        and row["split"] == train_split
        and row["moment"] == moment
    ]
    test = [
        row
        for row in rows
        if row["domain"] == target_domain
        and row["split"] == test_split
        and row["moment"] == moment
    ]
    try:
        direction, midpoint = _fit_direction(train, layer_id, label)
    except InsufficientClassesError:
        return {
            "eligible": False,
            "reason": "training split does not contain both label classes",
        }
    labels, scores = _trajectory_scores(
        test,
        layer_id=layer_id,
        label=label,
        direction=direction,
        midpoint=midpoint,
    )
    positives = sum(labels)
    negatives = len(labels) - positives
    train_labels, _ = _trajectory_scores(
        train,
        layer_id=layer_id,
        label=label,
        direction=direction,
        midpoint=midpoint,
    )
    auc = _auc(labels, scores)
    return {
        "eligible": bool(positives and negatives),
        "training_trajectory_count": len(train_labels),
        "training_positive_count": sum(train_labels),
        "training_negative_count": len(train_labels) - sum(train_labels),
        "trajectory_count": len(labels),
        "positive_count": positives,
        "negative_count": negatives,
        "auroc": auc,
        "auroc_stratified_bootstrap_95_ci": _bootstrap_auc_interval(
            labels,
            scores,
            samples=bootstrap_samples,
            random_seed=random_seed,
        ),
        "one_sided_label_permutation_p": _permutation_p_value(
            labels,
            scores,
            samples=permutation_samples,
            random_seed=random_seed,
        ),
        "balanced_accuracy_at_centroid_midpoint": _balanced_accuracy(labels, scores),
        "surface_baselines": [
            _scalar_baseline(train, test, label, field)
            for field in (
                "prefix_char_count",
                "prefix_message_count",
                "decision_position",
                "prior_tool_errors",
                "prompt_tokens",
            )
        ],
    }


def _pairwise_hidden_diagnostics(
    rows_by_moment: dict[str, dict[str, Any]], layer_id: int
) -> dict[str, dict[str, float]]:
    vectors = {
        moment: decode_float16_vector(row["activations"][str(layer_id)])
        for moment, row in rows_by_moment.items()
    }
    diagnostics = {}
    for left, right in (
        ("before", "action"),
        ("action", "result"),
        ("before", "result"),
    ):
        left_vector = vectors[left]
        right_vector = vectors[right]
        distance = float(np.linalg.norm(left_vector - right_vector))
        if distance == 0.0:
            raise ValueError(
                f"Layer {layer_id} returned identical {left}/{right} hidden vectors"
            )
        diagnostics[f"{left}_vs_{right}"] = {
            "cosine_similarity": float(
                left_vector
                @ right_vector
                / (np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
            ),
            "l2_distance": distance,
        }
    return diagnostics


def build_activation_smoke_report(
    requests: list[dict[str, Any]],
    activations: list[dict[str, Any]],
    *,
    layer_ids: tuple[int, ...],
    hidden_size: int,
    hidden_model: str,
) -> dict[str, Any]:
    """Strictly validate one decision's three exported hidden states."""

    if len(requests) != 3 or len(activations) != 3:
        raise ValueError(
            f"Smoke requires 3 requests and 3 activations, got "
            f"{len(requests)} and {len(activations)}"
        )
    expected_keys = {(row["sample_id"], row["request_fingerprint"]) for row in requests}
    observed_keys = {
        (row["sample_id"], row.get("request_fingerprint")) for row in activations
    }
    if expected_keys != observed_keys:
        raise ValueError("Smoke activation rows are incomplete, duplicated, or stale")
    request_by_moment = {row["moment"]: row for row in requests}
    rows_by_moment = {row["moment"]: row for row in activations}
    required_moments = set(MOMENTS)
    if (
        set(request_by_moment) != required_moments
        or set(rows_by_moment) != required_moments
    ):
        raise ValueError("Smoke moments must be exactly before/action/result")
    before_messages = request_by_moment["before"]["messages"]
    action_messages = request_by_moment["action"]["messages"]
    result_messages = request_by_moment["result"]["messages"]
    if action_messages[: len(before_messages)] != before_messages:
        raise ValueError("Action context does not extend the exact before context")
    if result_messages[: len(action_messages)] != action_messages:
        raise ValueError("Result context does not extend the exact action context")
    if len(action_messages) != len(before_messages) + 1:
        raise ValueError(
            "Action context must add exactly one assistant tool-call message"
        )
    if len(result_messages) <= len(action_messages):
        raise ValueError("Result context must add at least one tool-result message")
    prompt_tokens = [rows_by_moment[moment]["prompt_tokens"] for moment in MOMENTS]
    if not prompt_tokens[0] < prompt_tokens[1] < prompt_tokens[2]:
        raise ValueError(
            "Prompt token counts must strictly increase before/action/result: "
            f"{prompt_tokens}"
        )
    last_token_ids = {
        moment: rows_by_moment[moment]["last_token_id"] for moment in MOMENTS
    }
    if last_token_ids["before"] != last_token_ids["result"]:
        raise ValueError(
            "Before/result next-decision readouts must end on the same token type: "
            f"{last_token_ids}"
        )
    vector_summaries = {}
    expected_layers = {str(layer_id) for layer_id in layer_ids}
    for moment, row in rows_by_moment.items():
        if set(row["activations"]) != expected_layers:
            raise ValueError(
                f"{moment} layers do not match configuration: {set(row['activations'])}"
            )
        layer_summaries = {}
        for layer_id in layer_ids:
            vector = decode_float16_vector(row["activations"][str(layer_id)])
            if vector.shape != (hidden_size,):
                raise ValueError(
                    f"{moment}/layer {layer_id} expected hidden size "
                    f"{hidden_size}, got {vector.shape}"
                )
            if not np.isfinite(vector).all():
                raise ValueError(
                    f"{moment}/layer {layer_id} contains non-finite values"
                )
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                raise ValueError(f"{moment}/layer {layer_id} is a zero vector")
            layer_summaries[str(layer_id)] = {
                "dimension": int(vector.size),
                "l2_norm": norm,
                "mean": float(np.mean(vector)),
                "standard_deviation": float(np.std(vector)),
            }
        vector_summaries[moment] = layer_summaries
    return {
        "pipeline_valid": True,
        "context_chain": {
            "moments": list(MOMENTS),
            "message_counts": {
                moment: len(request_by_moment[moment]["messages"]) for moment in MOMENTS
            },
            "prompt_token_counts": {
                moment: rows_by_moment[moment]["prompt_tokens"] for moment in MOMENTS
            },
            "exported_token_ids_matched_submitted_prompt": True,
            "prompt_token_sha256": {
                moment: rows_by_moment[moment]["prompt_token_sha256"]
                for moment in MOMENTS
            },
            "last_token_ids": last_token_ids,
            "readout_definition": {
                "before": "last token of the next-assistant generation prompt",
                "action": "last token of the logged assistant tool-call message",
                "result": "last token of the next-assistant generation prompt",
            },
        },
        "hidden_states": {
            "model": hidden_model,
            "layers": list(layer_ids),
            "expected_dimension": hidden_size,
            "vectors": vector_summaries,
            "pairwise_diagnostics": {
                str(layer_id): _pairwise_hidden_diagnostics(rows_by_moment, layer_id)
                for layer_id in layer_ids
            },
        },
    }


def build_success_direction_report(
    rows: list[dict[str, Any]],
    *,
    domains: tuple[str, ...],
    train_split: str,
    test_split: str,
    layer_ids: tuple[int, ...],
    primary_layer_id: int,
    random_seed: int = 300,
    bootstrap_samples: int = 2_000,
    permutation_samples: int = 5_000,
) -> dict[str, Any]:
    """Evaluate within-domain and cross-domain centroid directions."""

    evaluations = {}
    for source in domains:
        for target in domains:
            for moment in MOMENTS:
                for label in LABELS:
                    for layer_id in layer_ids:
                        key = f"{source}_to_{target}:{moment}:{label}:layer_{layer_id}"
                        is_primary = (
                            source == target
                            and moment == "before"
                            and label == "overall_success"
                            and layer_id == primary_layer_id
                        )
                        evaluations[key] = _evaluate_transfer(
                            rows,
                            source_domain=source,
                            target_domain=target,
                            train_split=train_split,
                            test_split=test_split,
                            layer_id=layer_id,
                            moment=moment,
                            label=label,
                            bootstrap_samples=(bootstrap_samples if is_primary else 0),
                            permutation_samples=(
                                permutation_samples if is_primary else 0
                            ),
                            random_seed=random_seed,
                        )
    primary_keys = [
        f"{domain}_to_{domain}:before:overall_success:layer_{primary_layer_id}"
        for domain in domains
    ]
    primary_results = {key: evaluations[key] for key in primary_keys}
    eligible_domains = [
        domain
        for domain, key in zip(domains, primary_keys)
        if primary_results[key].get("eligible")
    ]
    passing_domains = [
        domain
        for domain, key in zip(domains, primary_keys)
        if primary_results[key].get("eligible")
        and primary_results[key]["auroc"] > 0.5
        and primary_results[key]["one_sided_label_permutation_p"] < 0.05
    ]
    if not eligible_domains:
        evidence_status = "not_evaluable"
    elif len(eligible_domains) < len(domains) and not passing_domains:
        evidence_status = "not_evaluable_in_all_domains"
    elif len(passing_domains) == len(domains):
        evidence_status = "consistent_evidence_in_both_domains"
    elif passing_domains:
        evidence_status = "partial_evidence_in_one_domain"
    else:
        evidence_status = "no_detectable_held_out_direction"
    return {
        "experiment": "success direction existence",
        "primary_test": {
            "definition": (
                "pre-generation, overall official success, preregistered layer, "
                "trajectory-level mean over selected tool decisions"
            ),
            "layer_id": primary_layer_id,
            "evidence_rule": (
                "A domain passes only if held-out AUROC > 0.5 and the one-sided "
                "label-permutation p-value is < 0.05. Both domains passing is "
                "the preregistered consistency criterion."
            ),
            "evidence_status": evidence_status,
            "domains_passing": passing_domains,
            "results": primary_results,
        },
        "evaluations": evaluations,
        "metric_notes": {
            "auroc": "0.5 is random; computed on held-out official test tasks",
            "balanced_accuracy_at_centroid_midpoint": (
                "no threshold tuning; zero is the midpoint of train centroids"
            ),
            "unit_of_analysis": "one trajectory, not one decision",
            "bootstrap_samples": bootstrap_samples,
            "permutation_samples": permutation_samples,
            "random_seed": random_seed,
        },
    }
