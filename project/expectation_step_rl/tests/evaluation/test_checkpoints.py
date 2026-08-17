import json
from pathlib import Path

import pytest

from expectation_step_rl.evaluation.checkpoints import (
    discover_checkpoints,
    parse_steps,
)


def _write_adapter(checkpoint_root: Path, step: int) -> Path:
    adapter = checkpoint_root / f"global_step_{step}" / "actor" / "lora_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 32}))
    return adapter


def test_discover_checkpoints_is_sorted_and_selectable(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    step_30 = _write_adapter(checkpoint_root, 30)
    _write_adapter(checkpoint_root, 10)

    all_checkpoints = discover_checkpoints(checkpoint_root)
    selected = discover_checkpoints(checkpoint_root, selected_steps={30})

    assert [checkpoint.step for checkpoint in all_checkpoints] == [10, 30]
    assert selected[0].path == step_30
    assert selected[0].served_model_name == "expectation-step-30"


def test_discover_checkpoints_rejects_missing_requested_step(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    _write_adapter(checkpoint_root, 10)

    with pytest.raises(ValueError, match="absent or incomplete"):
        discover_checkpoints(checkpoint_root, selected_steps={20})


def test_parse_steps_requires_positive_integers() -> None:
    assert parse_steps("10, 30,10") == {10, 30}
    assert parse_steps("") is None
    with pytest.raises(ValueError, match="positive"):
        parse_steps("0")
