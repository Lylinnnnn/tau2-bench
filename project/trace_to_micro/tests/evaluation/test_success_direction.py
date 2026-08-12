import numpy as np

from trace_to_micro.evaluation.success_direction import build_success_direction_report
from trace_to_micro.utils.io import encode_float16_vector


def _row(domain: str, split: str, simulation_id: str, success: bool, x: float) -> dict:
    return {
        "domain": domain,
        "split": split,
        "simulation_id": simulation_id,
        "moment": "before",
        "overall_success": success,
        "db_success": success,
        "communication_success": success,
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
