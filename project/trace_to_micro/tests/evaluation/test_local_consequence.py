import numpy as np

from trace_to_micro.evaluation.local_consequence import (
    build_local_consequence_report,
)
from trace_to_micro.utils.io import encode_float16_vector


def _row(
    domain: str,
    split: str,
    simulation_id: str,
    progress: bool,
    value: float,
) -> dict:
    return {
        "domain": domain,
        "split": split,
        "simulation_id": simulation_id,
        "moment": "action",
        "goal_progress": progress,
        "tool_success": True,
        "state_changed": progress,
        "tool_name": "update",
        "target_distance_before": 2,
        "decision_position": 1,
        "prior_tool_errors": 0,
        "prompt_tokens": 100,
        "activations": {"47": encode_float16_vector(np.array([value, 1.0]))},
    }


def test_local_consequence_direction_uses_held_out_decisions() -> None:
    rows = []
    for domain in ("airline", "retail"):
        rows.extend(
            [
                _row(domain, "train", f"{domain}-train-f", False, -2),
                _row(domain, "train", f"{domain}-train-s", True, 2),
            ]
        )
        for index in range(10):
            rows.append(
                _row(domain, "test", f"{domain}-test-f-{index}", False, -1)
            )
            rows.append(
                _row(domain, "test", f"{domain}-test-s-{index}", True, 1)
            )

    report = build_local_consequence_report(
        rows,
        domains=("airline", "retail"),
        train_split="train",
        test_split="test",
        layer_ids=(47,),
        primary_layer_id=47,
        primary_moment="action",
        bootstrap_samples=20,
        random_seed=300,
    )

    result = report["evaluations"]["airline:action:goal_progress:layer_47"]
    assert result["auroc"] == 1.0
    assert result["within_tool_auroc"] == 1.0
    assert result["tool_centered_within_tool_auroc"] == 1.0
    assert result["decision_count"] == 20
    assert report["primary_test"]["evidence_status"] == (
        "consistent_local_progress_direction"
    )
