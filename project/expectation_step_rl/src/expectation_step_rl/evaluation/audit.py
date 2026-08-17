"""Completeness checks for generated tau2 evaluation trajectories."""

from __future__ import annotations

from collections import Counter
from typing import Any

from tau2.data_model.simulation import Results, TerminationReason


def audit_results(
    results: Results,
    *,
    expected_task_ids: list[str],
    expected_num_trials: int,
    expected_agent_model: str,
    expected_user_model: str,
) -> dict[str, Any]:
    """Reject incomplete or infrastructure-corrupted official trajectories.

    Normal task failures remain valid benchmark observations. Only missing,
    duplicated, model-mismatched, or infrastructure-corrupted simulations make
    an evaluation run incomplete.
    """

    expected_pairs = {
        (task_id, trial)
        for task_id in expected_task_ids
        for trial in range(expected_num_trials)
    }
    pair_counts = Counter(
        (str(simulation.task_id), simulation.trial)
        for simulation in results.simulations
    )
    actual_pairs = set(pair_counts)
    duplicate_pairs = sorted(pair for pair, count in pair_counts.items() if count > 1)
    missing_rewards = sorted(
        simulation.id
        for simulation in results.simulations
        if simulation.reward_info is None
    )
    infrastructure_errors = sorted(
        simulation.id
        for simulation in results.simulations
        if simulation.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    )

    errors = []
    if results.info.num_trials != expected_num_trials:
        errors.append(
            f"num_trials={results.info.num_trials}, expected={expected_num_trials}"
        )
    if results.info.agent_info.llm != expected_agent_model:
        errors.append(
            f"agent_model={results.info.agent_info.llm!r}, "
            f"expected={expected_agent_model!r}"
        )
    if results.info.user_info.llm != expected_user_model:
        errors.append(
            f"user_model={results.info.user_info.llm!r}, "
            f"expected={expected_user_model!r}"
        )
    if actual_pairs != expected_pairs:
        errors.append(
            f"missing_pairs={sorted(expected_pairs - actual_pairs)}, "
            f"extra_pairs={sorted(actual_pairs - expected_pairs)}"
        )
    if duplicate_pairs:
        errors.append(f"duplicate_pairs={duplicate_pairs}")
    if missing_rewards:
        errors.append(f"missing_rewards={missing_rewards}")
    if infrastructure_errors:
        errors.append(f"infrastructure_errors={infrastructure_errors}")
    if errors:
        raise ValueError("; ".join(errors))

    return {
        "complete": True,
        "task_count": len(expected_task_ids),
        "simulation_count": len(results.simulations),
        "num_trials": expected_num_trials,
        "agent_model": expected_agent_model,
        "user_model": expected_user_model,
        "infrastructure_error_count": 0,
    }
