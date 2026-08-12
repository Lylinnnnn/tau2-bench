import numpy as np
import pytest

from trace_to_micro.evaluation.success_direction import (
    build_activation_smoke_report,
    build_success_direction_report,
)
from trace_to_micro.utils.io import encode_float16_vector


def _row(domain: str, split: str, simulation_id: str, success: bool, x: float) -> dict:
    return {
        "domain": domain,
        "split": split,
        "simulation_id": simulation_id,
        "moment": "before",
        "overall_success": success,
        "db_success": success,
        "language_success": success,
        "prefix_char_count": 10,
        "prefix_message_count": 2,
        "decision_position": 0,
        "prior_tool_errors": 0,
        "prompt_tokens": 5,
        "activations": {"47": encode_float16_vector(np.array([x, 1.0]))},
    }


def test_success_direction_uses_held_out_trajectory_scores() -> None:
    rows = []
    for domain in ("airline", "retail"):
        rows.extend(
            [
                _row(domain, "train", f"{domain}-train-f", False, -2),
                _row(domain, "train", f"{domain}-train-s", True, 2),
            ]
        )
        for index in range(10):
            rows.append(_row(domain, "test", f"{domain}-test-f-{index}", False, -1))
            rows.append(_row(domain, "test", f"{domain}-test-s-{index}", True, 1))

    report = build_success_direction_report(
        rows,
        domains=("airline", "retail"),
        train_split="train",
        test_split="test",
        layer_ids=(47,),
        primary_layer_id=47,
        bootstrap_samples=20,
        permutation_samples=100,
    )

    key = "airline_to_retail:before:overall_success:layer_47"
    assert report["evaluations"][key]["auroc"] == 1.0
    assert report["evaluations"][key]["trajectory_count"] == 20
    assert report["primary_test"]["evidence_status"] == (
        "consistent_evidence_in_both_domains"
    )


def test_zero_activation_is_not_hidden_as_class_imbalance() -> None:
    rows = [
        _row("airline", "train", "failure", False, -1),
        _row("airline", "train", "success", True, 0),
    ]
    rows[1]["activations"]["47"] = encode_float16_vector(np.zeros(2))

    with pytest.raises(ValueError, match="zero activation vector"):
        build_success_direction_report(
            rows,
            domains=("airline", "retail"),
            train_split="train",
            test_split="test",
            layer_ids=(47,),
            primary_layer_id=47,
            bootstrap_samples=20,
            permutation_samples=20,
        )


def _smoke_row(moment: str, prompt_tokens: int, value: float) -> dict:
    return {
        "sample_id": f"decision:{moment}",
        "request_fingerprint": f"fingerprint-{moment}",
        "moment": moment,
        "prompt_tokens": prompt_tokens,
        "last_token_id": 1 if moment != "action" else 2,
        "prompt_token_sha256": f"sha-{moment}",
        "activations": {
            "31": encode_float16_vector(np.array([value, 1.0])),
            "47": encode_float16_vector(np.array([1.0, value])),
        },
    }


def test_activation_smoke_requires_a_real_context_chain() -> None:
    system = {"role": "system", "content": "policy"}
    user = {"role": "user", "content": "help"}
    action = {"role": "assistant", "tool_calls": [{"id": "call-1"}]}
    result = {"role": "tool", "tool_call_id": "call-1", "content": "ok"}
    requests = [
        {
            **_smoke_row("before", 10, 1.0),
            "messages": [system, user],
        },
        {
            **_smoke_row("action", 12, 2.0),
            "messages": [system, user, action],
        },
        {
            **_smoke_row("result", 14, 3.0),
            "messages": [system, user, action, result],
        },
    ]
    activations = [
        {key: value for key, value in request.items() if key != "messages"}
        for request in requests
    ]

    report = build_activation_smoke_report(
        requests,
        activations,
        layer_ids=(31, 47),
        hidden_size=2,
        hidden_model="model",
    )

    assert report["pipeline_valid"] is True
    assert report["context_chain"]["prompt_token_counts"] == {
        "before": 10,
        "action": 12,
        "result": 14,
    }
    assert report["hidden_states"]["vectors"]["before"]["31"]["dimension"] == 2


def test_activation_smoke_rejects_wrong_vector_dimension() -> None:
    system = {"role": "system", "content": "policy"}
    action = {"role": "assistant", "content": "call"}
    result = {"role": "tool", "content": "ok"}
    requests = [
        {**_smoke_row("before", 10, 1.0), "messages": [system]},
        {**_smoke_row("action", 12, 2.0), "messages": [system, action]},
        {
            **_smoke_row("result", 14, 3.0),
            "messages": [system, action, result],
        },
    ]
    activations = [
        {key: value for key, value in request.items() if key != "messages"}
        for request in requests
    ]

    with pytest.raises(ValueError, match="expected hidden size 5120"):
        build_activation_smoke_report(
            requests,
            activations,
            layer_ids=(31, 47),
            hidden_size=5120,
            hidden_model="model",
        )
