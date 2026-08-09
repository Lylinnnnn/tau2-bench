"""Transition and natural-overlap audit over real behavior-policy trajectories."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import Results, SimulationRun
from tau2.data_model.tasks import Task
from tau2.runner import build_environment, load_task_splits
from trace_to_micro.evaluation import build_support_report
from trace_to_micro.models import StateChange, TransitionEvent
from trace_to_micro.state_diff import (
    canonicalize_value,
    diff_snapshots,
    snapshot_environment,
    snapshot_hash,
)


def _local_pre_state(event: TransitionEvent) -> str | None:
    if not event.changes:
        return None
    value = [
        {
            "path": change.path,
            "before": change.before,
            "before_present": change.before_present,
        }
        for change in event.changes
    ]
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def replay_logged_simulation(
    *,
    simulation: SimulationRun,
    task: Task,
    split: str,
    domain: str,
) -> list[dict[str, Any]]:
    """Replay every observed tool call while retaining failures and pre-state keys."""

    environment = build_environment(domain)
    initial_state = task.initial_state
    environment.set_state(
        initialization_data=(
            initial_state.initialization_data if initial_state is not None else None
        ),
        initialization_actions=(
            initial_state.initialization_actions if initial_state is not None else None
        ),
        message_history=[],
    )
    messages = simulation.get_messages()
    recorded_results = {
        message.id: message for message in messages if isinstance(message, ToolMessage)
    }
    rows = []
    step_index = 0
    for message in messages:
        if not isinstance(message, (AssistantMessage, UserMessage)):
            continue
        for call in message.tool_calls or []:
            before = snapshot_environment(environment)
            declared_mutating = environment._is_mutating_tool(call.name)
            replayed = environment.get_response(call)
            after = snapshot_environment(environment)
            event = TransitionEvent(
                task_id=str(task.id),
                split=split,
                step_index=step_index,
                actor=call.requestor,
                action_name=call.name,
                arguments=canonicalize_value(call.arguments),
                declared_mutating=declared_mutating,
                tool_result=replayed.content,
                changes=diff_snapshots(before, after),
            )
            recorded = recorded_results.get(call.id)
            row = event.to_dict()
            row.update(
                {
                    "simulation_id": simulation.id,
                    "trial": simulation.trial,
                    "pre_state_hash": snapshot_hash(before),
                    "effect_local_pre_state": _local_pre_state(event),
                    "replay_error": replayed.error,
                    "recorded_tool_result": (
                        recorded.content if recorded is not None else None
                    ),
                    "recorded_error": (
                        recorded.error if recorded is not None else None
                    ),
                }
            )
            rows.append(row)
            step_index += 1
    return rows


def _to_event(row: dict[str, Any]) -> TransitionEvent:
    return TransitionEvent(
        task_id=row["task_id"],
        split=row["split"],
        step_index=row["step_index"],
        actor=row["actor"],
        action_name=row["action_name"],
        arguments=row["arguments"],
        declared_mutating=row["declared_mutating"],
        tool_result=row["tool_result"],
        changes=tuple(StateChange(**change) for change in row["changes"]),
    )


def _overlap_at_threshold(
    rows: list[dict[str, Any]], state_field: str, threshold: int
) -> dict[str, Any]:
    groups: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in rows:
        state_key = row[state_field]
        if state_key is None:
            continue
        action_key = json.dumps(
            [row["action_name"], row["arguments"]],
            sort_keys=True,
            ensure_ascii=False,
        )
        groups[(row["actor"], state_key)][action_key].add(row["task_id"])
    eligible = []
    for (actor, state_key), actions in groups.items():
        supported = {
            action: task_ids
            for action, task_ids in actions.items()
            if len(task_ids) >= threshold
        }
        if len(supported) >= 2:
            eligible.append(
                {
                    "actor": actor,
                    "state_key": state_key,
                    "actions": {
                        action: sorted(task_ids)
                        for action, task_ids in sorted(supported.items())
                    },
                }
            )
    return {
        "eligible_state_count": len(eligible),
        "examples": eligible[:20],
    }


def _state_recurrence(
    train_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    thresholds: tuple[int, ...],
    *,
    leave_one_task_out: bool,
) -> dict[str, Any]:
    train_support: dict[str, set[str]] = defaultdict(set)
    for row in train_rows:
        train_support[row["pre_state_hash"]].add(row["task_id"])
    support_counts = []
    for row in query_rows:
        task_ids = train_support[row["pre_state_hash"]]
        if leave_one_task_out:
            task_ids = task_ids - {row["task_id"]}
        support_counts.append(len(task_ids))
    return {
        str(threshold): {
            "covered": sum(count >= threshold for count in support_counts),
            "total": len(query_rows),
            "rate": (
                sum(count >= threshold for count in support_counts) / len(query_rows)
                if query_rows
                else None
            ),
        }
        for threshold in thresholds
    }


def build_logged_audit(
    *,
    results_path: Path,
    domain: str,
    task_set: str,
    train_split: str,
    test_split: str,
    thresholds: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay logged transitions and build observational support diagnostics."""

    metadata = Results.load_metadata(results_path)
    tasks = {str(task.id): task for task in metadata.tasks}
    split_map = load_task_splits(task_set)
    if split_map is None:
        raise ValueError(f"Task set {task_set!r} has no split map")
    train_ids = set(split_map[train_split])
    test_ids = set(split_map[test_split])
    rows = []
    for simulation in Results.iter_simulations(results_path):
        task_id = str(simulation.task_id)
        if task_id in train_ids:
            split = train_split
        elif task_id in test_ids:
            split = test_split
        else:
            continue
        rows.extend(
            replay_logged_simulation(
                simulation=simulation,
                task=tasks[task_id],
                split=split,
                domain=domain,
            )
        )
    train_rows = [row for row in rows if row["split"] == train_split]
    test_rows = [row for row in rows if row["split"] == test_split]
    train_loto_support = build_support_report(
        [_to_event(row) for row in train_rows],
        [_to_event(row) for row in train_rows],
        thresholds=thresholds,
        state_changing_only=False,
        leave_one_task_out=True,
    )
    test_transfer_support = build_support_report(
        [_to_event(row) for row in train_rows],
        [_to_event(row) for row in test_rows],
        thresholds=thresholds,
        state_changing_only=False,
    )
    report = {
        "experiment": {
            "data_source": "real SimulationRun.messages",
            "behavior_agent_model": metadata.info.agent_info.llm,
            "behavior_user_model": metadata.info.user_info.llm,
            "one_trajectory_per_task": metadata.info.num_trials == 1,
            "claim_scope": "observational logged support; no unobserved counterfactual claims",
        },
        "transition_count": len(rows),
        "train_transition_count": len(train_rows),
        "test_transition_count": len(test_rows),
        "replay_error_count": sum(row["replay_error"] for row in rows),
        "split_protocol": {
            "train": "train-to-train with the query task excluded from support",
            "test": "train-to-test; test transitions never enter support",
            "train_test_task_overlap": len(train_ids & test_ids),
        },
        "state_recurrence": {
            "train_loto": _state_recurrence(
                train_rows,
                train_rows,
                thresholds,
                leave_one_task_out=True,
            ),
            "test_transfer": _state_recurrence(
                train_rows,
                test_rows,
                thresholds,
                leave_one_task_out=False,
            ),
        },
        "transition_support": {
            "train_loto": train_loto_support,
            "test_transfer": test_transfer_support,
        },
        "natural_action_overlap": {
            "strict_full_pre_state": {
                str(threshold): _overlap_at_threshold(
                    train_rows, "pre_state_hash", threshold
                )
                for threshold in thresholds
            },
            "effect_local_pre_state": {
                str(threshold): _overlap_at_threshold(
                    train_rows, "effect_local_pre_state", threshold
                )
                for threshold in thresholds
            },
            "interpretation": (
                "Only groups with at least two naturally observed actions are eligible "
                "for action comparison; otherwise the method must abstain."
            ),
        },
    }
    return rows, report
