"""Small JSON output helpers for pre-experiment artifacts."""

import json
from pathlib import Path
from typing import Any

from trace_to_micro.models import TransitionEvent


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
