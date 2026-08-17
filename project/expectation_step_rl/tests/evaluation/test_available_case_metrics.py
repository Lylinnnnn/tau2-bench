from types import SimpleNamespace

import pytest

from expectation_step_rl.evaluation import available_case_metrics
from tau2.data_model.simulation import TerminationReason


def _simulation(
    simulation_id: str,
    task_id: str,
    termination_reason: TerminationReason,
    reward: float | None,
) -> SimpleNamespace:
    reward_info = None if reward is None else SimpleNamespace(reward=reward)
    return SimpleNamespace(
        id=simulation_id,
        task_id=task_id,
        termination_reason=termination_reason,
        reward_info=reward_info,
    )


def test_available_case_summary_exposes_reduced_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = SimpleNamespace(
        info=SimpleNamespace(num_trials=1),
        tasks=[SimpleNamespace(id="1"), SimpleNamespace(id="62")],
        simulations=[
            _simulation("good", "1", TerminationReason.AGENT_STOP, 1.0),
            _simulation("infra", "62", TerminationReason.INFRASTRUCTURE_ERROR, None),
        ],
    )
    monkeypatch.setattr(
        available_case_metrics,
        "compute_metrics",
        lambda value: SimpleNamespace(
            total_simulations=1,
            avg_reward=1.0,
            pass_hat_ks={1: 1.0},
        ),
    )

    report = available_case_metrics.summarize_available_cases(results)

    assert report["publication_ready"] is False
    assert report["expected_task_count"] == 2
    assert report["evaluated_task_count"] == 1
    assert report["excluded_task_ids"] == ["62"]
    assert report["pass^1"] == 1.0


def test_available_case_summary_rejects_other_missing_rewards() -> None:
    results = SimpleNamespace(
        info=SimpleNamespace(num_trials=1),
        tasks=[SimpleNamespace(id="1")],
        simulations=[_simulation("bad", "1", TerminationReason.USER_STOP, None)],
    )

    with pytest.raises(ValueError, match="outside infrastructure failures"):
        available_case_metrics.summarize_available_cases(results)
