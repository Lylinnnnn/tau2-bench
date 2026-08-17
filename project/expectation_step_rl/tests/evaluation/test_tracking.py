import json
from pathlib import Path

import pytest

from expectation_step_rl.evaluation import tracking
from expectation_step_rl.evaluation.checkpoints import AdapterCheckpoint


def _checkpoint(tmp_path: Path, step: int) -> AdapterCheckpoint:
    adapter = tmp_path / f"global_step_{step}" / "actor" / "lora_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text('{"r": 32}')
    return AdapterCheckpoint(step, adapter)


def _prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, run_tag: str = "run-a"
) -> dict:
    checkpoint = _checkpoint(tmp_path / "checkpoints", 10)
    protocol = {
        "base_model": {"served_name": "qwen3-32b"},
        "domains": ["airline", "retail"],
        "split": "test",
        "num_trials": 1,
        "seed": 300,
    }
    monkeypatch.setattr(
        tracking, "discover_checkpoints", lambda *args, **kwargs: [checkpoint]
    )
    monkeypatch.setattr(
        tracking, "build_evaluation_protocol", lambda **kwargs: protocol
    )
    monkeypatch.setattr(
        tracking.subprocess, "check_output", lambda *args, **kwargs: "abc123\n"
    )
    return tracking.prepare_run_manifest(
        repo_root=tmp_path,
        evaluation_root=tmp_path / "evaluation" / "outputs",
        run_tag=run_tag,
        checkpoint_root=tmp_path / "checkpoints",
        checkpoint_steps=None,
        source_model_path=tmp_path / "model",
        base_model_name="qwen3-32b",
        domains=["airline", "retail"],
        split="test",
        num_trials=1,
        seed=300,
        max_tasks_per_domain=0,
        include_baseline=True,
        local_staging_enabled=True,
    )


def test_manifest_and_registry_link_checkpoint_to_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _prepare(tmp_path, monkeypatch)
    registry_path = tmp_path / "evaluation" / "outputs" / "experiment_registry.json"
    registry = json.loads(registry_path.read_text())

    assert manifest["expected_model_keys"] == ["base", "step_10"]
    assert manifest["checkpoint_root"].endswith("checkpoints")
    assert manifest["local_staging_enabled"] is True
    assert registry["runs"][0]["baseline_id"] == manifest["baseline_id"]
    assert registry["runs"][0]["checkpoint_steps"] == [10]


def test_run_tag_cannot_be_reused_for_different_checkpoint_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path, monkeypatch)
    different = _checkpoint(tmp_path / "other-checkpoints", 20)
    monkeypatch.setattr(
        tracking, "discover_checkpoints", lambda *args, **kwargs: [different]
    )

    with pytest.raises(ValueError, match="different evaluation"):
        tracking.prepare_run_manifest(
            repo_root=tmp_path,
            evaluation_root=tmp_path / "evaluation" / "outputs",
            run_tag="run-a",
            checkpoint_root=tmp_path / "other-checkpoints",
            checkpoint_steps=None,
            source_model_path=tmp_path / "model",
            base_model_name="qwen3-32b",
            domains=["airline", "retail"],
            split="test",
            num_trials=1,
            seed=300,
            max_tasks_per_domain=0,
            include_baseline=True,
            local_staging_enabled=True,
        )
