"""Small JSON output helpers for pre-experiment artifacts."""

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from trace_to_micro.data_model import TransitionEvent


def write_json(path: Path, value: Any) -> None:
    """Write a human-readable JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_events_jsonl(path: Path, events: list[TransitionEvent]) -> None:
    """Write one transition event per line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False))
            handle.write("\n")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    """Write arbitrary dictionaries as JSON Lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False))
            handle.write("\n")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    """Append one dictionary to a resumable JSON Lines artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False))
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines artifact, returning an empty list if absent."""

    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encode_float16_vector(vector: np.ndarray) -> str:
    """Encode one numeric vector as compact base64 float16 bytes."""

    return base64.b64encode(vector.astype("<f2").tobytes()).decode("ascii")


def decode_float16_vector(value: str) -> np.ndarray:
    """Decode one compact vector into float32 for stable evaluation."""

    return np.frombuffer(base64.b64decode(value), dtype="<f2").astype(np.float32)
