"""Feature compilation and closed-form probes for expected consequences."""

from __future__ import annotations

from typing import Any

import numpy as np

from trace_to_micro.utils.io import decode_float16_vector

SURFACE_FIELDS = (
    "decision_position",
    "prior_tool_errors",
    "prompt_tokens",
)
FEATURE_SETS = ("surface", "hidden", "surface_plus_hidden")


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or (norms == 0).any():
        raise ValueError("Probe features must be finite and non-zero")
    return values / norms


def _hidden_matrix(rows: list[dict[str, Any]], layer_id: int) -> np.ndarray:
    return _normalize_rows(
        np.stack(
            [
                decode_float16_vector(row["activations"][str(layer_id)]).astype(
                    np.float64
                )
                for row in rows
            ]
        )
    )


def _argument_count(row: dict[str, Any]) -> int:
    calls = row.get("logged_action", {}).get("calls", [])
    return sum(len(call.get("arguments", {})) for call in calls)


def _surface_matrices(
    train: list[dict[str, Any]], test: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tools = sorted({row["tool_name"] for row in train})
    tool_index = {tool: index for index, tool in enumerate(tools)}

    def scalars(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [
                [
                    *[float(row[field]) for field in SURFACE_FIELDS],
                    float(_argument_count(row)),
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

    train_scalars = scalars(train)
    test_scalars = scalars(test)
    mean = train_scalars.mean(axis=0)
    scale = train_scalars.std(axis=0)
    scale[scale == 0] = 1.0

    def build(rows: list[dict[str, Any]], values: np.ndarray) -> np.ndarray:
        one_hot = np.zeros((len(rows), len(tools) + 1), dtype=np.float64)
        for index, row in enumerate(rows):
            one_hot[index, tool_index.get(row["tool_name"], len(tools))] = 1.0
        return _normalize_rows(
            np.concatenate([one_hot, (values - mean) / scale], axis=1)
        )

    return (
        build(train, train_scalars),
        build(test, test_scalars),
        {
            "tool_vocabulary": tools,
            "unseen_test_decisions": sum(
                row["tool_name"] not in tool_index for row in test
            ),
            "numeric_fields": [*SURFACE_FIELDS, "argument_count"],
        },
    )


def feature_matrices(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    layer_id: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Build ordinary-information, hidden, and combined probe inputs."""

    surface_train, surface_test, metadata = _surface_matrices(train, test)
    hidden_train = _hidden_matrix(train, layer_id)
    hidden_test = _hidden_matrix(test, layer_id)
    scale = np.sqrt(2.0)
    return (
        {
            "surface": (surface_train, surface_test),
            "hidden": (hidden_train, hidden_test),
            "surface_plus_hidden": (
                np.concatenate([surface_train / scale, hidden_train / scale], axis=1),
                np.concatenate([surface_test / scale, hidden_test / scale], axis=1),
            ),
        },
        metadata,
    )


def fit_ridge(
    train_x: np.ndarray, train_y: np.ndarray, regularization: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a multi-target ridge probe through its small dual system."""

    mean_x = train_x.mean(axis=0)
    mean_y = train_y.mean(axis=0)
    centered_x = train_x - mean_x
    centered_y = train_y - mean_y
    gram = np.einsum("nd,md->nm", centered_x, centered_x)
    dual = np.linalg.solve(
        gram + regularization * np.eye(len(centered_x), dtype=np.float64),
        centered_y,
    )
    weights = np.einsum("nd,nk->dk", centered_x, dual)
    if not np.isfinite(weights).all():
        raise ValueError("Ridge fit produced non-finite weights")
    return weights, mean_x, mean_y


def predict_ridge(
    values: np.ndarray,
    model: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Predict bounded multi-target scores from a fitted ridge probe."""

    weights, mean_x, mean_y = model
    predicted = np.einsum("nd,dk->nk", values - mean_x, weights) + mean_y
    if not np.isfinite(predicted).all():
        raise ValueError("Ridge prediction produced non-finite values")
    return np.clip(predicted, 1e-6, 1 - 1e-6)


def target_matrix(
    consequences: list[dict[str, Any]], target_names: list[str]
) -> np.ndarray:
    """Materialize binary abstract-consequence targets in a stable order."""

    return np.asarray(
        [
            [float(row["targets"].get(target, False)) for target in target_names]
            for row in consequences
        ],
        dtype=np.float64,
    )
