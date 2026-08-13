import numpy as np

from trace_to_micro.evaluation.consequence_expectation import (
    build_consequence_expectation_report,
)
from trace_to_micro.utils.io import encode_float16_vector


def _activation(
    domain: str,
    split: str,
    index: int,
    positive: bool,
) -> dict:
    decision_id = f"{domain}-{split}-{index}"
    return {
        "decision_id": decision_id,
        "simulation_id": decision_id,
        "domain": domain,
        "split": split,
        "task_id": decision_id,
        "trial": 0,
        "moment": "action",
        "tool_name": "update_record",
        "tool_success": positive,
        "decision_position": 1,
        "prior_tool_errors": 0,
        "prompt_tokens": 100,
        "logged_action": {
            "kind": "tool",
            "calls": [
                {
                    "name": "update_record",
                    "arguments": {"record_id": "same"},
                }
            ],
        },
        "activations": {
            "47": encode_float16_vector(np.asarray([1.0 if positive else -1.0, 0.25]))
        },
    }


def _consequence(row: dict) -> dict:
    positive = row["tool_success"]
    return {
        "decision_id": row["decision_id"],
        "simulation_id": row["simulation_id"],
        "domain": row["domain"],
        "split": row["split"],
        "task_id": row["task_id"],
        "trial": 0,
        "tool_name": row["tool_name"],
        "tool_success": positive,
        "targets": {
            "execution.success": positive,
            "effect.changed": positive,
        },
    }


def test_expected_consequence_requires_increment_over_surface_baseline() -> None:
    rows = []
    for domain in ("airline", "retail"):
        for split, count in (("train", 8), ("test", 20)):
            rows.extend(
                _activation(domain, split, index, index % 2 == 0)
                for index in range(count)
            )
    consequences = [_consequence(row) for row in rows]

    report, predictions = build_consequence_expectation_report(
        rows,
        consequences,
        domains=("airline", "retail"),
        train_split="train",
        test_split="test",
        layer_id=47,
        regularization=0.1,
        minimum_train_class_count=2,
        bootstrap_samples=100,
        random_seed=300,
    )

    airline = report["domains"]["airline"]
    assert (
        airline["feature_sets"]["surface"]["macro"]["all_heads"][
            "consequence_family_balanced_auroc"
        ]
        == 0.5
    )
    assert (
        airline["feature_sets"]["hidden"]["macro"]["all_heads"][
            "consequence_family_balanced_auroc"
        ]
        == 1.0
    )
    assert (
        airline["incremental_hidden_signal"]["consequence_family_balanced_auroc_delta"]
        == 0.5
    )
    assert (
        "semantic_surprise_error_detection_auroc_delta"
        in airline["incremental_hidden_signal"]
    )
    assert report["primary_test"]["evidence_status"] == (
        "reliable_incremental_expectation_signal"
    )
    assert len(predictions) == 40


def test_expected_consequence_rejects_unmatched_decisions() -> None:
    row = _activation("retail", "train", 0, True)

    try:
        build_consequence_expectation_report(
            [row],
            [],
            domains=("airline", "retail"),
            train_split="train",
            test_split="test",
            layer_id=47,
            regularization=1.0,
            minimum_train_class_count=2,
            bootstrap_samples=10,
            random_seed=300,
        )
    except ValueError as error:
        assert "do not match" in str(error)
    else:
        raise AssertionError("Mismatched decisions must be rejected")
