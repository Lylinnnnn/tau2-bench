from types import SimpleNamespace

import pytest

from expectation_step_rl.evaluation.audit import audit_results
from expectation_step_rl.evaluation.inference import task_ids_for_shard
from tau2.data_model.simulation import TerminationReason


def _results(task_ids: list[str], num_trials: int = 1) -> SimpleNamespace:
    simulations = [
        SimpleNamespace(
            id=f"{task_id}-{trial}",
            task_id=task_id,
            trial=trial,
            reward_info=SimpleNamespace(reward=0.0),
            termination_reason=TerminationReason.USER_STOP,
        )
        for task_id in task_ids
        for trial in range(num_trials)
    ]
    return SimpleNamespace(
        info=SimpleNamespace(
            num_trials=num_trials,
            agent_info=SimpleNamespace(llm="openai/adapter"),
            user_info=SimpleNamespace(llm="openai/base"),
        ),
        simulations=simulations,
    )


def test_round_robin_shards_are_disjoint_and_exhaustive() -> None:
    task_ids = [str(index) for index in range(11)]
    shards = [task_ids_for_shard(task_ids, index, 4) for index in range(4)]

    flattened = [task_id for shard in shards for task_id in shard]
    assert sorted(flattened) == sorted(task_ids)
    assert len(flattened) == len(set(flattened))


def test_audit_accepts_normal_task_failures() -> None:
    report = audit_results(
        _results(["a", "b"], num_trials=2),
        expected_task_ids=["a", "b"],
        expected_num_trials=2,
        expected_agent_model="openai/adapter",
        expected_user_model="openai/base",
    )

    assert report["complete"] is True
    assert report["simulation_count"] == 4


def test_audit_rejects_missing_simulation_atomically() -> None:
    results = _results(["a", "b"])
    results.simulations.pop()

    with pytest.raises(ValueError, match="missing_pairs"):
        audit_results(
            results,
            expected_task_ids=["a", "b"],
            expected_num_trials=1,
            expected_agent_model="openai/adapter",
            expected_user_model="openai/base",
        )


def test_audit_rejects_infrastructure_error() -> None:
    results = _results(["a"])
    results.simulations[0].termination_reason = TerminationReason.INFRASTRUCTURE_ERROR

    with pytest.raises(ValueError, match="infrastructure_errors"):
        audit_results(
            results,
            expected_task_ids=["a"],
            expected_num_trials=1,
            expected_agent_model="openai/adapter",
            expected_user_model="openai/base",
        )
