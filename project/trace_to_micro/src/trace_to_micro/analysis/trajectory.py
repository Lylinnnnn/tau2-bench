"""Descriptive analysis of already-generated behavior-policy trajectories."""

from collections import Counter, defaultdict
from typing import Any

from tau2.data_model.simulation import SimulationRun
from tau2.metrics.agent_metrics import is_successful
from trace_to_micro.analysis.logged_trace import decision_indices


def _mean(values: list[float | int]) -> float | None:
    return sum(values) / len(values) if values else None


def _horizon_bucket(decision_count: int) -> str:
    if decision_count <= 3:
        return "short_0_3"
    if decision_count <= 6:
        return "medium_4_6"
    return "long_7_plus"


def _success_groups(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        if record["success"] is not None:
            groups[str(record[field])].append(record["success"])
    return {
        key: {"count": len(values), "pass_1": sum(values) / len(values)}
        for key, values in sorted(groups.items())
    }


def _recovery_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_simulation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_simulation[row["simulation_id"]].append(row)
    error_count = 0
    recovered_count = 0
    for simulation_rows in by_simulation.values():
        ordered = sorted(simulation_rows, key=lambda row: row["step_index"])
        for index, row in enumerate(ordered):
            if not row["replay_error"]:
                continue
            error_count += 1
            recovered_count += any(
                not later["replay_error"] and bool(later["changes"])
                for later in ordered[index + 1 :]
            )
    return {
        "error_transition_count": error_count,
        "followed_by_later_state_change": recovered_count,
        "observed_recovery_rate": (
            recovered_count / error_count if error_count else None
        ),
    }


def build_trajectory_report(
    *,
    split: str,
    simulations: list[SimulationRun],
    snapshots: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize outcomes, horizons, contexts, and observed transition quality."""

    records = []
    for simulation in simulations:
        reward = (
            simulation.reward_info.reward
            if simulation.reward_info is not None
            else None
        )
        decision_count = len(decision_indices(simulation))
        records.append(
            {
                "task_id": str(simulation.task_id),
                "reward": reward,
                "success": is_successful(reward) if reward is not None else None,
                "decision_count": decision_count,
                "horizon_bucket": _horizon_bucket(decision_count),
                "message_count": len(simulation.get_messages()),
            }
        )
    observed_rewards = [
        record["reward"] for record in records if record["reward"] is not None
    ]
    observed_success = [
        record["success"] for record in records if record["success"] is not None
    ]
    split_rows = [row for row in transition_rows if row["split"] == split]
    mutation_noops = [
        row
        for row in split_rows
        if row["declared_mutating"] and not row["changes"] and not row["replay_error"]
    ]
    return {
        "experiment": "already_generated_trajectory_analysis",
        "split": split,
        "simulation_count": len(simulations),
        "unique_task_count": len({str(sim.task_id) for sim in simulations}),
        "outcomes": {
            "reward_count": len(observed_rewards),
            "missing_reward_count": len(records) - len(observed_rewards),
            "mean_reward": _mean(observed_rewards),
            "pass_1": (
                sum(observed_success) / len(observed_success)
                if observed_success
                else None
            ),
            "termination_reasons": dict(
                sorted(
                    Counter(str(sim.termination_reason) for sim in simulations).items()
                )
            ),
        },
        "trajectory_length": {
            "mean_message_count": _mean(
                [record["message_count"] for record in records]
            ),
            "mean_agent_decision_count": _mean(
                [record["decision_count"] for record in records]
            ),
            "agent_decision_count_histogram": {
                str(key): value
                for key, value in sorted(
                    Counter(record["decision_count"] for record in records).items()
                )
            },
            "pass_1_by_agent_decision_count": _success_groups(
                records, "decision_count"
            ),
            "pass_1_by_horizon_bucket": _success_groups(records, "horizon_bucket"),
        },
        "decision_snapshots": {
            "count": len(snapshots),
            "position_buckets": dict(
                sorted(Counter(row["position_bucket"] for row in snapshots).items())
            ),
            "context_length_buckets": dict(
                sorted(
                    Counter(row["context_length_bucket"] for row in snapshots).items()
                )
            ),
            "mean_prefix_message_count": _mean(
                [row["prefix_message_count"] for row in snapshots]
            ),
            "mean_prefix_char_count": _mean(
                [row["prefix_char_count"] for row in snapshots]
            ),
        },
        "observed_transitions": {
            "count": len(split_rows),
            "actor_counts": dict(
                sorted(Counter(row["actor"] for row in split_rows).items())
            ),
            "action_counts": dict(
                sorted(Counter(row["action_name"] for row in split_rows).items())
            ),
            "state_changing_count": sum(bool(row["changes"]) for row in split_rows),
            "replay_error_count": sum(row["replay_error"] for row in split_rows),
            "mutation_noop_count": len(mutation_noops),
            "recovery": _recovery_summary(split_rows),
        },
        "interpretation": {
            "outcome_by_horizon": (
                "observational association from complete logged trajectories; "
                "not a same-state context intervention"
            ),
            "transition_support": (
                "reported separately with train leave-one-task-out and train-to-test"
            ),
        },
    }


def build_cross_split_trajectory_report(
    reports: dict[str, dict[str, Any]], *, train_split: str
) -> dict[str, Any]:
    """Compare descriptive trajectory statistics without mixing split records."""

    train = reports[train_split]
    gaps = {}
    for split, report in reports.items():
        if split == train_split:
            continue
        gaps[f"{split}_minus_{train_split}"] = {
            "pass_1": report["outcomes"]["pass_1"] - train["outcomes"]["pass_1"],
            "mean_reward": report["outcomes"]["mean_reward"]
            - train["outcomes"]["mean_reward"],
            "mean_agent_decision_count": report["trajectory_length"][
                "mean_agent_decision_count"
            ]
            - train["trajectory_length"]["mean_agent_decision_count"],
        }
    return {
        "experiment": "already_generated_trajectory_analysis",
        "train_split": train_split,
        "split_reports": reports,
        "descriptive_gaps": gaps,
        "interpretation": (
            "These are observational train/test differences. Causal context effects "
            "are reported only by experiment_2_context_probe."
        ),
    }
