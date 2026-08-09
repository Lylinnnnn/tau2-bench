"""Serializable data structures shared across the pre-experiment."""

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TaskIdentity:
    """Structured fields encoded in a Telecom task ID."""

    intent: str
    atoms: tuple[str, ...]
    persona: str | None


@dataclass(frozen=True)
class StateChange:
    """One leaf-level difference between environment snapshots."""

    path: str
    before: Any
    after: Any
    before_present: bool = True
    after_present: bool = True

    def signature(self) -> tuple[str, str, str, bool, bool]:
        """Return a stable, hashable representation of this change."""

        return (
            self.path,
            json.dumps(self.before, sort_keys=True, ensure_ascii=False),
            json.dumps(self.after, sort_keys=True, ensure_ascii=False),
            self.before_present,
            self.after_present,
        )


@dataclass(frozen=True)
class TransitionEvent:
    """An observed action and its factual environment effect."""

    task_id: str
    split: str
    step_index: int
    actor: str
    action_name: str
    arguments: dict[str, Any]
    declared_mutating: bool
    tool_result: str
    changes: tuple[StateChange, ...]

    @property
    def changed_state(self) -> bool:
        """Whether the two serialized environment snapshots differ."""

        return bool(self.changes)

    def action_key(self) -> str:
        """Canonical action key used by the support audit."""

        args = json.dumps(self.arguments, sort_keys=True, ensure_ascii=False)
        return f"{self.actor}:{self.action_name}:{args}"

    def effect_key(self) -> tuple[tuple[str, str, str, bool, bool], ...]:
        """Canonical exact-effect key used by the oracle upper bound."""

        return tuple(change.signature() for change in self.changes)

    def to_dict(self) -> dict[str, Any]:
        """Convert the event to JSON-serializable primitives."""

        result = asdict(self)
        result["changed_state"] = self.changed_state
        return result
