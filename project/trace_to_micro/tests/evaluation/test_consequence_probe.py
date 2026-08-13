import numpy as np

from trace_to_micro.evaluation.consequence_probe import (
    feature_matrices,
    fit_ridge,
    predict_ridge,
    target_matrix,
)
from trace_to_micro.utils.io import encode_float16_vector


def _row(value: float) -> dict:
    return {
        "tool_name": "update",
        "decision_position": 1,
        "prior_tool_errors": 0,
        "prompt_tokens": 100,
        "logged_action": {"calls": [{"arguments": {"id": "same"}}]},
        "activations": {"47": encode_float16_vector(np.asarray([value, 0.25]))},
    }


def test_closed_form_probe_predicts_held_out_hidden_direction() -> None:
    train = [_row(-2), _row(-1), _row(1), _row(2)]
    test = [_row(-0.5), _row(0.5)]
    matrices, metadata = feature_matrices(train, test, 47)
    train_y = np.asarray([[0], [0], [1], [1]], dtype=np.float64)

    prediction = predict_ridge(
        matrices["hidden"][1],
        fit_ridge(matrices["hidden"][0], train_y, regularization=0.1),
    )

    assert prediction[0, 0] < prediction[1, 0]
    assert metadata["numeric_fields"][-1] == "argument_count"
    assert target_matrix(
        [{"targets": {"effect.changed": True}}], ["effect.changed"]
    ).tolist() == [[1.0]]
