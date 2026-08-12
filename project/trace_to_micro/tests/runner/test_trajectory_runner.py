from pathlib import Path

import pytest

from trace_to_micro.config import TrajectoryConfig
from trace_to_micro.runner.trajectory import build_run_config, remove_existing_run


def test_build_run_config_selects_exact_smoke_task() -> None:
    config = TrajectoryConfig(
        domain="retail",
        task_set="retail",
        task_split="base",
        agent="llm_agent",
        user="user_simulator",
        agent_llm="agent-model",
        user_llm="user-model",
        agent_llm_args={},
        user_llm_args={},
        num_trials=1,
        max_steps=10,
        max_errors=2,
        max_concurrency=1,
        seed=300,
        timeout_seconds=30,
        save_to="smoke",
    )

    run_config = build_run_config(config, task_ids=["2"], auto_resume=False)

    assert run_config.task_ids == ["2"]
    assert run_config.num_tasks is None
    assert run_config.auto_resume is False


def test_remove_existing_run_unlinks_only_exact_results_file(tmp_path: Path) -> None:
    results = tmp_path / "data" / "simulations" / "run" / "results.json"
    results.parent.mkdir(parents=True)
    results.write_text("{}", encoding="utf-8")

    assert remove_existing_run(results) is True
    assert not results.exists()
    assert remove_existing_run(results) is False


def test_remove_existing_run_rejects_unexpected_path(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        remove_existing_run(results)
