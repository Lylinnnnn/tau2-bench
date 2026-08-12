"""Stable message/action serialization for logged-trajectory experiments."""

import json
from typing import Any

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.utils.llm_utils import to_litellm_messages
from trace_to_micro.replay.state import canonicalize_value


def to_openai_messages(
    messages: list[Message], system_prompt: str
) -> list[dict[str, Any]]:
    """Serialize exactly the agent-visible prefix used by τ² generation."""

    agent_history = [
        message
        for message in messages
        if not isinstance(message, ToolMessage) or message.requestor == "assistant"
    ]
    system = SystemMessage(role="system", content=system_prompt)
    return to_litellm_messages([system, *agent_history])


def action_record(message: AssistantMessage | UserMessage) -> dict[str, Any]:
    """Represent a participant message without provider-specific metadata."""

    if message.tool_calls:
        return {
            "kind": "tool",
            "calls": [
                {
                    "requestor": call.requestor,
                    "name": call.name,
                    "arguments": canonicalize_value(call.arguments),
                }
                for call in message.tool_calls
            ],
        }
    return {
        "kind": "text",
        "content": (message.content or "").strip(),
    }


def visible_message_record(message: Message) -> dict[str, Any]:
    """Serialize only content visible to a conversation participant."""

    if isinstance(message, (AssistantMessage, UserMessage)):
        record = {"role": message.role, "content": message.content}
        if message.tool_calls:
            record["tool_calls"] = [
                {
                    "name": call.name,
                    "arguments": call.arguments,
                    "requestor": call.requestor,
                }
                for call in message.tool_calls
            ]
        return record
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "requestor": message.requestor,
            "content": message.content,
            "error": message.error,
        }
    raise TypeError(f"Unsupported message type: {type(message)}")


def transcript_json(messages: list[Message]) -> str:
    """Return a compact, deterministic JSON transcript."""

    records = [visible_message_record(message) for message in messages]
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"))


def prompt_tokens(message: AssistantMessage) -> int | None:
    """Read prompt-token usage recorded by the behavior model, if present."""

    if message.usage is None:
        return None
    return message.usage.get("prompt_tokens")
