"""Compile observed tool decisions into deterministic local consequences."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.runner import build_environment
from trace_to_micro.replay.state import (
    diff_snapshots,
    snapshot_distance,
    snapshot_environment,
)


def _initialize(environment, task: Task) -> None:
    initial_state = task.initial_state
    message_history = (
        initial_state.message_history or [] if initial_state is not None else []
    )
    environment.set_state(
        initialization_data=(
            initial_state.initialization_data if initial_state is not None else None
        ),
        initialization_actions=(
            initial_state.initialization_actions if initial_state is not None else None
        ),
        message_history=message_history,
    )


def target_snapshot(domain: str, task: Task) -> dict[str, Any]:
    """Build the official DB target from state-mutating reference actions.

    The official evaluator compares final database hashes. Read-only reference
    actions cannot affect that target and may contain stale lookup arguments, so
    replaying them only introduces failures that are irrelevant to the target.
    Mutating reference actions remain strict because ignoring one would silently
    corrupt the local goal-progress label.
    """

    environment = build_environment(domain)
    _initialize(environment, task)
    actions = task.evaluation_criteria.actions if task.evaluation_criteria else []
    for action in actions or []:
        if not environment._is_mutating_tool(action.name):
            continue
        response = environment.get_response(
            ToolCall(
                id=action.action_id,
                requestor=action.requestor,
                name=action.name,
                arguments=action.arguments,
            )
        )
        if response.error:
            raise RuntimeError(
                f"Mutating reference action failed for {domain}/{task.id}: "
                f"{action.name}: {response.content}"
            )
    return snapshot_environment(environment)


def _tool_type(environment, requestor: str, tool_name: str) -> str:
    toolkit = environment.user_tools if requestor == "user" else environment.tools
    if toolkit is None or not toolkit.has_tool(tool_name):
        return "unknown"
    return toolkit.tool_type(tool_name).value


def _content_value(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def replay_local_consequences(
    *,
    simulation: SimulationRun,
    task: Task,
    split: str,
    domain: str,
) -> list[dict[str, Any]]:
    """Replay one logged trace and label each assistant tool decision locally."""

    environment = build_environment(domain)
    _initialize(environment, task)
    goal = target_snapshot(domain, task)
    messages = simulation.get_messages()
    recorded_results = {
        message.id: message
        for message in messages
        if isinstance(message, ToolMessage)
    }
    rows = []
    assistant_position = 0
    for message_index, message in enumerate(messages):
        if not isinstance(message, (AssistantMessage, UserMessage)):
            continue
        calls = message.tool_calls or []
        if isinstance(message, AssistantMessage) and len(calls) > 1:
            raise ValueError(
                "Local consequence extraction requires one assistant call per "
                f"message, got {len(calls)} at {simulation.id}:{message_index}"
            )
        for call in calls:
            before = snapshot_environment(environment)
            distance_before = snapshot_distance(before, goal)
            declared_mutating = environment._is_mutating_tool(call.name)
            tool_type = _tool_type(environment, call.requestor, call.name)
            progress_eligible = declared_mutating and tool_type != "unknown"
            replayed = environment.get_response(call)
            recorded = recorded_results.get(call.id)
            if recorded is None:
                raise ValueError(
                    f"Missing recorded result for {simulation.id}:{message_index}:"
                    f"{call.id}"
                )
            if recorded.error != replayed.error:
                raise ValueError(
                    "Recorded/replayed error mismatch at "
                    f"{simulation.id}:{message_index}:{call.name}"
                )
            if _content_value(recorded.content) != _content_value(replayed.content):
                raise ValueError(
                    "Recorded/replayed tool result mismatch at "
                    f"{simulation.id}:{message_index}:{call.name}"
                )
            after = snapshot_environment(environment)
            distance_after = snapshot_distance(after, goal)
            changes = diff_snapshots(before, after)
            if isinstance(message, AssistantMessage):
                rows.append(
                    {
                        "decision_id": f"{simulation.id}:{message_index}",
                        "simulation_id": simulation.id,
                        "domain": domain,
                        "split": split,
                        "task_id": str(task.id),
                        "trial": simulation.trial,
                        "message_index": message_index,
                        "decision_position": assistant_position,
                        "tool_name": call.name,
                        "tool_type": tool_type,
                        "declared_mutating": declared_mutating,
                        "tool_success": not replayed.error,
                        "state_changed": bool(changes),
                        "goal_progress_eligible": progress_eligible,
                        "goal_progress": (
                            not replayed.error and distance_after < distance_before
                            if progress_eligible
                            else None
                        ),
                        "target_distance_before": distance_before,
                        "target_distance_after": distance_after,
                        "target_distance_reduction": (
                            distance_before - distance_after
                        ),
                        "changes": [asdict(change) for change in changes],
                    }
                )
                assistant_position += 1
    return rows
