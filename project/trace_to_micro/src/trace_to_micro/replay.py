"""Deterministic replay of task reference trajectories."""

from collections.abc import Iterable

from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import Task
from tau2.runner import build_environment
from trace_to_micro.models import TransitionEvent
from trace_to_micro.state_diff import (
    canonicalize_value,
    diff_snapshots,
    snapshot_environment,
)


def _initialize_task(environment, task: Task) -> None:
    initial_state = task.initial_state
    environment.set_state(
        initialization_data=initial_state.initialization_data,
        initialization_actions=initial_state.initialization_actions,
        message_history=initial_state.message_history or [],
    )


def replay_reference_task(
    task: Task, split: str, domain: str = "telecom"
) -> list[TransitionEvent]:
    """Replay one reference trajectory and capture every action effect."""

    environment = build_environment(domain)
    _initialize_task(environment, task)
    actions = task.evaluation_criteria.actions or []
    events = []
    for step_index, action in enumerate(actions):
        tool_call = ToolCall(
            id=action.action_id,
            requestor=action.requestor,
            name=action.name,
            arguments=action.arguments,
        )
        before = snapshot_environment(environment)
        declared_mutating = environment._is_mutating_tool(action.name)
        result = environment.get_response(tool_call)
        if result.error:
            raise RuntimeError(
                f"Reference replay failed for task={task.id}, "
                f"step={step_index}, action={action.name}: {result.content}"
            )
        after = snapshot_environment(environment)
        events.append(
            TransitionEvent(
                task_id=task.id,
                split=split,
                step_index=step_index,
                actor=action.requestor,
                action_name=action.name,
                arguments=canonicalize_value(action.arguments),
                declared_mutating=declared_mutating,
                tool_result=result.content,
                changes=diff_snapshots(before, after),
            )
        )
    return events


def replay_reference_tasks(
    tasks: Iterable[Task], split: str, domain: str = "telecom"
) -> list[TransitionEvent]:
    """Replay reference trajectories for a task collection."""

    return [
        event
        for task in tasks
        for event in replay_reference_task(task, split=split, domain=domain)
    ]
