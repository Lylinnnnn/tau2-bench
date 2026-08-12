import json
import struct
from pathlib import Path

import numpy as np
import pytest

from trace_to_micro.runtime.activations import (
    load_hidden_state_export,
    load_last_token_vectors,
)


def _write_safetensor(path: Path, name: str, value: np.ndarray, dtype: str) -> None:
    data = value.tobytes()
    header = json.dumps(
        {
            name: {
                "dtype": dtype,
                "shape": list(value.shape),
                "data_offsets": [0, len(data)],
            }
        }
    ).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        handle.write(data)
    Path(f"{path}.lock").touch()


def _write_hidden_export(
    path: Path, hidden_states: np.ndarray, token_ids: np.ndarray
) -> None:
    hidden_data = hidden_states.tobytes()
    token_data = token_ids.tobytes()
    header = json.dumps(
        {
            "hidden_states": {
                "dtype": "F32",
                "shape": list(hidden_states.shape),
                "data_offsets": [0, len(hidden_data)],
            },
            "token_ids": {
                "dtype": "I64",
                "shape": list(token_ids.shape),
                "data_offsets": [
                    len(hidden_data),
                    len(hidden_data) + len(token_data),
                ],
            },
        }
    ).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        handle.write(hidden_data)
        handle.write(token_data)
    Path(f"{path}.lock").touch()


def test_load_last_token_vectors_preserves_layer_order(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    path = tmp_path / "hidden.safetensors"
    _write_safetensor(path, "hidden_states", values, "F32")

    vectors = load_last_token_vectors(path, (31, 47))

    np.testing.assert_array_equal(vectors[31], values[-1, 0])
    np.testing.assert_array_equal(vectors[47], values[-1, 1])


def test_load_hidden_export_matches_submitted_prompt_tokens(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    token_ids = np.array([7, 8, 9], dtype=np.int64)
    path = tmp_path / "hidden.safetensors"
    _write_hidden_export(path, values, token_ids)

    vectors = load_hidden_state_export(path, (31, 47), [7, 8, 9])

    np.testing.assert_array_equal(vectors[47], values[-1, 1])


def test_load_hidden_export_rejects_other_prompt_tokens(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    path = tmp_path / "hidden.safetensors"
    _write_hidden_export(path, values, np.array([7, 8, 9], dtype=np.int64))

    with pytest.raises(ValueError, match="do not match"):
        load_hidden_state_export(path, (31, 47), [7, 8, 10])
