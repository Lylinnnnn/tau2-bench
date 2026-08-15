import json
from pathlib import Path

import pytest

from expectation_step_rl.preflight import (
    check_calibration,
    check_dataset,
    check_training_packages,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row(split: str, decision_id: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": "policy"}],
        "agent_name": "tau2_expectation_step",
        "domain": "retail",
        "split": split,
        "task_id": "1",
        "decision_id": decision_id,
        "replay_prefix_json": "[]",
        "tool_schemas_json": "[]",
    }


def test_dataset_check_detects_decision_leakage(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    _write_jsonl(train, [_row("train", "same")])
    _write_jsonl(test, [_row("test", "same")])

    with pytest.raises(ValueError, match="leakage"):
        check_dataset(train, test)


def test_calibration_requires_train_provenance(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"source": "Test", "groups": {"x": {}}}))

    with pytest.raises(ValueError, match="Train"):
        check_calibration(path)


def test_calibration_requires_logged_clean_train_coverage(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "source": "official Train controlled corruption subset",
                "source_filter": {"track": "min_k_controlled"},
                "scored_train_records": 100,
                "groups": {"retail|lookup": {}},
                "domain_fallbacks": {"retail": {}},
            }
        )
    )

    with pytest.raises(ValueError, match="logged clean"):
        check_calibration(path)


def test_dataset_check_rejects_official_reward_leakage(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    leaked = {**_row("train", "train-1"), "reward": 1.0}
    _write_jsonl(train, [leaked])
    _write_jsonl(test, [_row("test", "test-1")])

    with pytest.raises(ValueError, match="forbidden supervision"):
        check_dataset(train, test)


def test_training_package_check_rejects_version_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "FRAMEWORK.lock"
    lock.write_text(
        "training_vllm=0.11.0\n"
        "training_torch=2.8.0\n"
        "training_flash_attn=2.8.1\n"
        "training_flashinfer=0.3.1\n"
        "training_transformers=4.57.1\n"
        "training_huggingface_hub=0.36.0\n"
        "training_tokenizers=0.22.1\n"
        "training_swanlab=0.9.1\n"
    )
    versions = {
        "vllm": "0.11.0",
        "torch": "2.8.0",
        "flash-attn": "2.8.0",
        "flashinfer-python": "0.3.1",
        "transformers": "4.57.1",
        "huggingface-hub": "0.36.0",
        "tokenizers": "0.22.1",
        "swanlab": "0.9.1",
    }
    monkeypatch.setattr(
        "expectation_step_rl.preflight.importlib.metadata.version",
        versions.__getitem__,
    )

    with pytest.raises(ValueError, match="flash-attn"):
        check_training_packages(tmp_path)
