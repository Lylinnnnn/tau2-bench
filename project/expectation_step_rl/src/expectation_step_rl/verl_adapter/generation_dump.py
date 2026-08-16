"""JSONL generation dumping compatible with NumPy reward diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """Encode NumPy containers and scalars without hiding unsupported types."""

    def default(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return super().default(value)


def dump_generations(
    trainer: Any,
    inputs: list[Any],
    outputs: list[Any],
    gts: list[Any],
    scores: list[Any],
    reward_extra_infos_dict: dict[str, Any],
    dump_path: str,
) -> None:
    """Replace the pinned verl dumper while preserving its JSONL schema."""

    output_dir = Path(dump_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{trainer.global_steps}.jsonl"

    row_count = len(inputs)
    columns = {
        "input": inputs,
        "output": outputs,
        "gts": gts,
        "score": scores,
        "step": [trainer.global_steps] * row_count,
    }
    for key, values in reward_extra_infos_dict.items():
        if len(values) == row_count:
            columns[key] = values

    lines = []
    for index in range(row_count):
        row = {key: values[index] for key, values in columns.items()}
        lines.append(json.dumps(row, ensure_ascii=False, cls=NumpyJSONEncoder))
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Dumped generations to {output_path}")


def install_generation_dump_patch() -> None:
    """Install the narrow compatibility override in the Ray trainer process."""

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    RayPPOTrainer._dump_generations = dump_generations
    print("Installed NumPy-compatible verl generation dumper")
