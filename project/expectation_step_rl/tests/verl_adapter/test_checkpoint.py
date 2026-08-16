from pathlib import Path

import pytest

from expectation_step_rl.verl_adapter.checkpoint import verify_checkpoint


def test_verify_checkpoint_requires_complete_verl_layout(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    actor_dir = checkpoint_dir / "global_step_1" / "actor"
    adapter_dir = actor_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter_dir / "adapter_config.json").write_text('{"r": 32}')
    (checkpoint_dir / "global_step_1" / "data.pt").write_bytes(b"data")
    (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text("1")

    report = verify_checkpoint(checkpoint_dir, expected_step=1)

    assert report["step"] == 1
    assert report["actor_file_count"] == 2
    assert report["adapter_bytes"] == 16


def test_verify_checkpoint_rejects_missing_actor_files(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    step_dir = checkpoint_dir / "global_step_1"
    step_dir.mkdir(parents=True)
    (step_dir / "data.pt").write_bytes(b"data")
    (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text("1")

    with pytest.raises(ValueError, match="LoRA adapter"):
        verify_checkpoint(checkpoint_dir, expected_step=1)


def test_verify_checkpoint_rejects_heavy_train_state(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    actor_dir = checkpoint_dir / "global_step_1" / "actor"
    adapter_dir = actor_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter_dir / "adapter_config.json").write_text('{"r": 32}')
    (actor_dir / "model_world_size_6_rank_0.pt").write_bytes(b"full model")
    (checkpoint_dir / "global_step_1" / "data.pt").write_bytes(b"data")
    (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text("1")

    with pytest.raises(ValueError, match="heavy training state"):
        verify_checkpoint(checkpoint_dir, expected_step=1)
