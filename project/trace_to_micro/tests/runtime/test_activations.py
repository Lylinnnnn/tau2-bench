import json
import struct
from pathlib import Path

import numpy as np

from trace_to_micro.runtime.activations import load_last_token_vectors


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


def test_load_last_token_vectors_preserves_layer_order(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    path = tmp_path / "hidden.safetensors"
    _write_safetensor(path, "hidden_states", values, "F32")

    vectors = load_last_token_vectors(path, (31, 47))

    np.testing.assert_array_equal(vectors[31], values[-1, 0])
    np.testing.assert_array_equal(vectors[47], values[-1, 1])
