"""Environment snapshots, entity-value normalization, and leaf-level diffs."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from tau2.environment.environment import Environment
from trace_to_micro.data_model import StateChange

IDENTIFIER_FIELDS = {
    "account_id",
    "bill_id",
    "customer_id",
    "line_id",
    "payment_method_id",
    "phone_number",
    "plan_id",
}
PHONE_PATTERN = re.compile(r"^\+?[\d\-() ]{7,}$")


def snapshot_environment(environment: Environment) -> dict[str, Any]:
    """Serialize every stateful side available in an environment."""

    return {
        "assistant": (
            environment.tools.db.model_dump(mode="json")
            if environment.tools is not None and environment.tools.db is not None
            else None
        ),
        "user": (
            environment.user_tools.db.model_dump(mode="json")
            if environment.user_tools is not None
            and environment.user_tools.db is not None
            else None
        ),
    }


def canonicalize_value(value: Any, field_name: str | None = None) -> Any:
    """Replace entity-specific scalar values while preserving task semantics."""

    if isinstance(value, Mapping):
        return {
            key: canonicalize_value(child, key) for key, child in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [canonicalize_value(child) for child in value]
    if field_name in IDENTIFIER_FIELDS or (
        field_name is not None and field_name.endswith("_id")
    ):
        return f"<{field_name}>" if value is not None else None
    if isinstance(value, str) and PHONE_PATTERN.fullmatch(value):
        return "<phone_number>"
    return value


def snapshot_hash(snapshot: dict[str, Any]) -> str:
    """Return a stable fingerprint for a canonicalized environment snapshot."""

    payload = json.dumps(
        canonicalize_value(snapshot), sort_keys=True, ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def flatten_state(value: Any, path: tuple[str, ...] = ()) -> dict[str, Any]:
    """Flatten a nested JSON value into dot-separated leaf paths."""

    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key, child in sorted(value.items()):
            flattened.update(flatten_state(child, (*path, str(key))))
        return flattened
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        flattened = {}
        for index, child in enumerate(value):
            flattened.update(flatten_state(child, (*path, str(index))))
        return flattened
    return {".".join(path): value}


def diff_snapshots(
    before: dict[str, Any], after: dict[str, Any]
) -> tuple[StateChange, ...]:
    """Return canonical leaf-level changes between two snapshots."""

    before_flat = flatten_state(before)
    after_flat = flatten_state(after)
    changes = []
    for path in sorted(set(before_flat) | set(after_flat)):
        before_present = path in before_flat
        after_present = path in after_flat
        before_value = before_flat.get(path)
        after_value = after_flat.get(path)
        if before_present == after_present and before_value == after_value:
            continue
        field_name = path.rsplit(".", maxsplit=1)[-1]
        changes.append(
            StateChange(
                path=path,
                before=canonicalize_value(before_value, field_name),
                after=canonicalize_value(after_value, field_name),
                before_present=before_present,
                after_present=after_present,
            )
        )
    return tuple(changes)


def snapshot_distance(left: dict[str, Any], right: dict[str, Any]) -> int:
    """Count leaf values or presences that differ between two snapshots."""

    left_flat = flatten_state(left)
    right_flat = flatten_state(right)
    return sum(
        path not in left_flat
        or path not in right_flat
        or left_flat[path] != right_flat[path]
        for path in set(left_flat) | set(right_flat)
    )
