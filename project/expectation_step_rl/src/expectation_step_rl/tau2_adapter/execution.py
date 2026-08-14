"""Reconstruct one logged tau2 state and execute one generated tool call."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.runner import build_environment, get_tasks

_MESSAGE_TYPES = {
    "system": SystemMessage,
    "assistant": AssistantMessage,
    "user": UserMessage,
    "tool": ToolMessage,
}


def serialize_messages(messages: list[Message]) -> str:
    """Serialize a replay prefix without provider-specific Python objects."""

    return json.dumps(
        [message.model_dump(mode="json") for message in messages],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def deserialize_messages(payload: str) -> list[Message]:
    """Restore role-specific tau2 message classes from a dataset field."""

    records = json.loads(payload)
    return [_MESSAGE_TYPES[record["role"]].model_validate(record) for record in records]


@dataclass(frozen=True)
class ToolExecution:
    """Observable result of one generated tool call."""

    content: str
    error: bool
    tool_name: str
    arguments: dict[str, Any]
    call_id: str


@lru_cache(maxsize=None)
def _tasks_by_id(domain: str):
    return {str(task.id): task for task in get_tasks(domain)}


def execute_one_tool_call(
    *,
    domain: str,
    task_id: str,
    replay_prefix_json: str,
    tool_name: str,
    arguments: dict[str, Any],
    call_id: str,
) -> ToolExecution:
    """Replay the factual prefix, then execute one assistant-side call."""

    task = _tasks_by_id(domain)[str(task_id)]
    environment = build_environment(domain)
    initial_state = task.initial_state
    environment.set_state(
        initialization_data=(
            initial_state.initialization_data if initial_state is not None else None
        ),
        initialization_actions=(
            initial_state.initialization_actions if initial_state is not None else None
        ),
        message_history=deserialize_messages(replay_prefix_json),
        strict=False,
    )
    response = environment.get_response(
        ToolCall(
            id=call_id,
            name=tool_name,
            arguments=arguments,
            requestor="assistant",
        )
    )
    if response.content is None:
        raise ValueError("tau2 returned a tool message without content")
    return ToolExecution(
        content=response.content,
        error=response.error,
        tool_name=tool_name,
        arguments=arguments,
        call_id=call_id,
    )
