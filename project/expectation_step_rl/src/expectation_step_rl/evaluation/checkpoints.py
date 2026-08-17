"""Discover inference-only LoRA adapters exported by formal training."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_STEP_PATTERN = re.compile(r"global_step_(\d+)")


@dataclass(frozen=True)
class AdapterCheckpoint:
    """One complete inference-only training checkpoint."""

    step: int
    path: Path

    @property
    def key(self) -> str:
        return f"step_{self.step}"

    @property
    def served_model_name(self) -> str:
        return f"expectation-step-{self.step}"


def parse_steps(value: str | None) -> set[int] | None:
    """Parse a comma-separated checkpoint selection."""

    if value is None or not value.strip():
        return None
    steps = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("Checkpoint steps must be positive integers")
    return steps


def _validate_adapter(path: Path) -> None:
    model_path = path / "adapter_model.safetensors"
    config_path = path / "adapter_config.json"
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise ValueError(f"Missing non-empty LoRA weights: {model_path}")
    if not config_path.is_file() or config_path.stat().st_size == 0:
        raise ValueError(f"Missing non-empty LoRA config: {config_path}")
    config = json.loads(config_path.read_text())
    if int(config.get("r", 0)) <= 0:
        raise ValueError(f"LoRA config has no positive rank: {config_path}")


def discover_checkpoints(
    checkpoint_root: Path,
    *,
    selected_steps: set[int] | None = None,
) -> list[AdapterCheckpoint]:
    """Return complete adapters ordered by training step."""

    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {checkpoint_root}")
    checkpoints = []
    for step_dir in checkpoint_root.glob("global_step_*"):
        match = _STEP_PATTERN.fullmatch(step_dir.name)
        if match is None:
            continue
        step = int(match.group(1))
        if selected_steps is not None and step not in selected_steps:
            continue
        adapter_path = step_dir / "actor" / "lora_adapter"
        _validate_adapter(adapter_path)
        checkpoints.append(AdapterCheckpoint(step=step, path=adapter_path))
    checkpoints.sort(key=lambda checkpoint: checkpoint.step)
    discovered_steps = {checkpoint.step for checkpoint in checkpoints}
    if selected_steps is not None and discovered_steps != selected_steps:
        missing = sorted(selected_steps - discovered_steps)
        raise ValueError(f"Requested checkpoints are absent or incomplete: {missing}")
    if not checkpoints:
        raise ValueError(f"No complete LoRA adapters found under {checkpoint_root}")
    return checkpoints
