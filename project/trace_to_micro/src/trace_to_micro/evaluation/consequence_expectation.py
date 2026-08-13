"""Orchestrate expected-consequence probes and post-result surprise."""

from __future__ import annotations

from collections import Counter
from typing import Any

from trace_to_micro.evaluation.consequence_metrics import (
    cluster_bootstrap_delta,
    feedback_metrics,
    head_metrics,
    optional_delta,
    surprise,
)
from trace_to_micro.evaluation.consequence_probe import (
    FEATURE_SETS,
    feature_matrices,
    fit_ridge,
    predict_ridge,
    target_matrix,
)


def _evaluate_domain(
    rows: list[dict[str, Any]],
    consequence_by_id: dict[str, dict[str, Any]],
    *,
    domain: str,
    train_split: str,
    test_split: str,
    layer_id: int,
    regularization: float,
    minimum_train_class_count: int,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train = [
        row for row in rows if row["domain"] == domain and row["split"] == train_split
    ]
    test = [
        row for row in rows if row["domain"] == domain and row["split"] == test_split
    ]
    if not train or not test:
        raise ValueError(f"{domain} requires non-empty Train and Test decisions")
    train_consequences = [consequence_by_id[row["decision_id"]] for row in train]
    test_consequences = [consequence_by_id[row["decision_id"]] for row in test]
    target_names = sorted(
        {
            target
            for consequence in train_consequences
            for target in consequence["targets"]
        }
    )
    all_train_targets = target_matrix(train_consequences, target_names)
    trained_indices = [
        index
        for index in range(len(target_names))
        if all_train_targets[:, index].sum() >= minimum_train_class_count
        and (1 - all_train_targets[:, index]).sum() >= minimum_train_class_count
    ]
    trained_targets = [target_names[index] for index in trained_indices]
    if not trained_targets:
        raise ValueError(f"No consequence head has both classes in {domain} Train")
    semantic_indices = [
        index
        for index, target in enumerate(trained_targets)
        if target != "execution.success"
    ]
    if not semantic_indices:
        raise ValueError(f"{domain} has no trainable non-execution consequence head")
    train_y = all_train_targets[:, trained_indices]
    test_y = target_matrix(test_consequences, trained_targets)
    features, surface_metadata = feature_matrices(train, test, layer_id)
    predictions = {}
    metrics = {}
    for feature_set, (train_x, test_x) in features.items():
        prediction = predict_ridge(
            test_x,
            fit_ridge(train_x, train_y, regularization),
        )
        predictions[feature_set] = prediction
        heads, macro = head_metrics(test_y, prediction, trained_targets)
        metrics[feature_set] = {
            "feature_dimension": train_x.shape[1],
            "heads": heads,
            "macro": macro,
            "feedback": feedback_metrics(test, test_y, prediction, trained_targets),
        }

    combined_macro = metrics["surface_plus_hidden"]["macro"]
    surface_macro = metrics["surface"]["macro"]
    combined_feedback = metrics["surface_plus_hidden"]["feedback"]
    surface_feedback = metrics["surface"]["feedback"]
    comparison = {
        "consequence_family_balanced_auroc_delta": (
            combined_macro["all_heads"]["consequence_family_balanced_auroc"]
            - surface_macro["all_heads"]["consequence_family_balanced_auroc"]
        ),
        "consequence_family_balanced_brier_delta": (
            combined_macro["all_heads"]["consequence_family_balanced_brier"]
            - surface_macro["all_heads"]["consequence_family_balanced_brier"]
        ),
        "auroc_delta_task_bootstrap_95_ci": cluster_bootstrap_delta(
            test_y,
            predictions["surface_plus_hidden"],
            predictions["surface"],
            trained_targets,
            [row["task_id"] for row in test],
            metric="auroc",
            samples=bootstrap_samples,
            random_seed=random_seed,
        ),
        "brier_delta_task_bootstrap_95_ci": cluster_bootstrap_delta(
            test_y,
            predictions["surface_plus_hidden"],
            predictions["surface"],
            trained_targets,
            [row["task_id"] for row in test],
            metric="brier",
            samples=bootstrap_samples,
            random_seed=random_seed + 1,
        ),
        "semantic_surprise_error_detection_auroc_delta": optional_delta(
            combined_feedback["without_execution_success_head"][
                "error_detection_auroc"
            ],
            surface_feedback["without_execution_success_head"]["error_detection_auroc"],
        ),
        "semantic_surprise_within_tool_error_detection_auroc_delta": (
            optional_delta(
                combined_feedback["without_execution_success_head"][
                    "within_tool_error_detection_auroc"
                ],
                surface_feedback["without_execution_success_head"][
                    "within_tool_error_detection_auroc"
                ],
            )
        ),
    }
    prediction_rows = []
    for index, row in enumerate(test):
        prediction_rows.append(
            {
                "decision_id": row["decision_id"],
                "simulation_id": row["simulation_id"],
                "domain": domain,
                "split": row["split"],
                "task_id": row["task_id"],
                "trial": row["trial"],
                "tool_name": row["tool_name"],
                "actual": {
                    target: bool(test_y[index, target_index])
                    for target_index, target in enumerate(trained_targets)
                },
                "predicted": {
                    feature_set: {
                        target: float(predictions[feature_set][index, target_index])
                        for target_index, target in enumerate(trained_targets)
                    }
                    for feature_set in FEATURE_SETS
                },
                "surprise": {
                    feature_set: float(
                        surprise(
                            test_y[index : index + 1],
                            predictions[feature_set][index : index + 1],
                        )[0]
                    )
                    for feature_set in FEATURE_SETS
                },
                "semantic_surprise_without_execution_success": {
                    feature_set: float(
                        surprise(
                            test_y[index : index + 1, semantic_indices],
                            predictions[feature_set][
                                index : index + 1, semantic_indices
                            ],
                        )[0]
                    )
                    for feature_set in FEATURE_SETS
                },
            }
        )
    return (
        {
            "training_decision_count": len(train),
            "test_decision_count": len(test),
            "training_task_count": len({row["task_id"] for row in train}),
            "test_task_count": len({row["task_id"] for row in test}),
            "trained_target_count": len(trained_targets),
            "trained_targets": trained_targets,
            "training_target_support": {
                target: {
                    "positive_count": int(train_y[:, index].sum()),
                    "negative_count": int(len(train_y) - train_y[:, index].sum()),
                }
                for index, target in enumerate(trained_targets)
            },
            "untrained_targets": sorted(set(target_names) - set(trained_targets)),
            "surface_features": surface_metadata,
            "feature_sets": metrics,
            "incremental_hidden_signal": comparison,
        },
        prediction_rows,
    )


def build_consequence_expectation_report(
    rows: list[dict[str, Any]],
    consequences: list[dict[str, Any]],
    *,
    domains: tuple[str, ...],
    train_split: str,
    test_split: str,
    layer_id: int,
    regularization: float,
    minimum_train_class_count: int,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit target-free consequence probes and score held-out surprise."""

    action_rows = [row for row in rows if row["moment"] == "action"]
    counts = Counter(row["decision_id"] for row in action_rows)
    if set(counts.values()) != {1}:
        raise ValueError("Each decision must have exactly one action activation")
    consequence_by_id = {row["decision_id"]: row for row in consequences}
    if len(consequence_by_id) != len(consequences):
        raise ValueError("Abstract consequences contain duplicate decisions")
    if set(counts) != set(consequence_by_id):
        raise ValueError("Action activations and abstract consequences do not match")
    reports = {}
    predictions = []
    for domain in domains:
        reports[domain], domain_predictions = _evaluate_domain(
            action_rows,
            consequence_by_id,
            domain=domain,
            train_split=train_split,
            test_split=test_split,
            layer_id=layer_id,
            regularization=regularization,
            minimum_train_class_count=minimum_train_class_count,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
        )
        predictions.extend(domain_predictions)
    passes = []
    for domain in domains:
        result = reports[domain]["incremental_hidden_signal"]
        interval = result["auroc_delta_task_bootstrap_95_ci"]
        passes.append(
            result["consequence_family_balanced_auroc_delta"] > 0
            and interval is not None
            and interval[0] > 0
        )
    status = (
        "reliable_incremental_expectation_signal"
        if all(passes)
        else "partial_incremental_expectation_signal"
        if any(passes)
        else "no_reliable_incremental_expectation_signal"
    )
    return (
        {
            "experiment": {
                "hypothesis": (
                    "The action-time hidden state predicts a target-free "
                    "abstract tool consequence beyond tool name and "
                    "trajectory-position surfaces."
                ),
                "actual_consequence_source": (
                    "deterministic tool result plus leaf diff from logged s_t to s_t+1"
                ),
                "official_goal_used": False,
                "moment": "action",
                "layer_id": layer_id,
                "probe": "closed-form multi-target linear ridge",
                "regularization": regularization,
                "train_split": train_split,
                "test_split": test_split,
                "uncertainty_cluster": "task_id",
            },
            "primary_test": {
                "metric": (
                    "consequence-family-balanced AUROC(surface + hidden) "
                    "- AUROC(surface)"
                ),
                "evidence_rule": (
                    "Both domains require a positive task-bootstrap 95% lower bound."
                ),
                "evidence_status": status,
            },
            "domains": reports,
        },
        predictions,
    )
