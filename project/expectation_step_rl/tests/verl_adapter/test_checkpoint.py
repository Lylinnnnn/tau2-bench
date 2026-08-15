from pathlib import Path

import pytest

from expectation_step_rl.verl_adapter.checkpoint import verify_checkpoint


def test_verify_checkpoint_requires_complete_verl_layout(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    actor_dir = checkpoint_dir / "global_step_1" / "actor"
    actor_dir.mkdir(parents=True)
    (actor_dir / "model.pt").write_bytes(b"actor")
    (checkpoint_dir / "global_step_1" / "data.pt").write_bytes(b"data")
    (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text("1")

    report = verify_checkpoint(checkpoint_dir, expected_step=1)

    assert report["step"] == 1
    assert report["actor_file_count"] == 1
    assert report["actor_bytes"] == 5


def test_verify_checkpoint_rejects_missing_actor_files(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    step_dir = checkpoint_dir / "global_step_1"
    step_dir.mkdir(parents=True)
    (step_dir / "data.pt").write_bytes(b"data")
    (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text("1")

    with pytest.raises(ValueError, match="actor checkpoint"):
        verify_checkpoint(checkpoint_dir, expected_step=1)
