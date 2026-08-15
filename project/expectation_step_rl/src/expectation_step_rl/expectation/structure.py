"""Value-free structural keys for tool results."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _value_paths(value: Any, prefix: str = "$") -> set[str]:
    if isinstance(value, dict):
        paths = {f"{prefix}:object"}
        for key, item in value.items():
            paths.update(_value_paths(item, f"{prefix}.{key}"))
        return paths
    if isinstance(value, list):
        paths = {f"{prefix}:array"}
        for item in value:
            paths.update(_value_paths(item, f"{prefix}[]"))
        return paths
    return {f"{prefix}:{type(value).__name__}"}


def result_structure(content: str) -> list[str]:
    """Describe a result using JSON paths and types, never field values."""

    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        value = content
    return sorted(_value_paths(value))


def result_structure_key(content: str) -> str:
    """Return a compact stable key for a result structure."""

    encoded = json.dumps(result_structure(content), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]
