"""Deterministic, agent-visible evidence extraction for hybrid clean contexts."""

import json
import re
from typing import Any

from tau2.agent.base_agent import is_valid_agent_history_message
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

MAX_RECENT_DIALOGUE = 8
MAX_TOOL_EVENTS = 16
MAX_TEXT_CHARS = 800
MAX_TOOL_RESULT_CHARS = 1_200

PHONE_PATTERN = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
ENTITY_PATTERNS = {
    "customer_id": re.compile(r"\bC\d+\b"),
    "line_id": re.compile(r"\bL\d+\b"),
    "bill_id": re.compile(r"\bB\d+\b"),
    "plan_id": re.compile(r"\bP\d+\b"),
    "device_id": re.compile(r"\bD\d+\b"),
}
ROLE_INVERSION_PATTERNS = (
    re.compile(r"\bwould you like me to (?:change|check|guide|help|reset)", re.I),
    re.compile(r"\blet me (?:check|verify) (?:your|the) (?:account|line|plan)", re.I),
    re.compile(r"\bi recommend we\b", re.I),
    re.compile(r"\bthe agent (?:has|will|needs to)\b", re.I),
)
RESOLUTION_PATTERNS = (
    re.compile(r"\b(?:can now|able to) (?:send|use|connect)", re.I),
    re.compile(r"\b(?:is|are) working now\b", re.I),
    re.compile(r"\b(?:issue|problem) (?:is|has been) resolved\b", re.I),
    re.compile(r"\bgreat news\b", re.I),
)


def _tool_names(tools: list[Tool]) -> list[str]:
    return sorted(tool.name for tool in tools)


def _short_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split())[:MAX_TEXT_CHARS]


def _add_entity(
    entities: dict[str, list[dict[str, str]]],
    field: str,
    value: str,
    source: str,
) -> None:
    record = {"value": value, "source": source}
    if record not in entities.setdefault(field, []):
        entities[field].append(record)


def _entity_field(key: str) -> str | None:
    if key == "phone_number":
        return key
    if key.endswith("_ids"):
        return f"{key[:-4]}_id"
    if key.endswith("_id"):
        return key
    return None


def _collect_entities(
    value: Any,
    *,
    source: str,
    entities: dict[str, list[dict[str, str]]],
    field: str | None = None,
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _collect_entities(
                child,
                source=source,
                entities=entities,
                field=_entity_field(key),
            )
        return
    if isinstance(value, list):
        for child in value:
            _collect_entities(
                child,
                source=source,
                entities=entities,
                field=field,
            )
        return
    if field is not None and isinstance(value, (str, int)):
        _add_entity(entities, field, str(value), source)


def _collect_text_entities(
    text: str,
    *,
    source: str,
    entities: dict[str, list[dict[str, str]]],
) -> None:
    for phone in PHONE_PATTERN.findall(text):
        _add_entity(entities, "phone_number", phone, source)
    for field, pattern in ENTITY_PATTERNS.items():
        for value in pattern.findall(text):
            _add_entity(entities, field, value, source)


def _result_value(content: str | None) -> Any:
    if content is None:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return _short_text(content)


def _compact_result(value: Any) -> Any:
    if not isinstance(value, (dict, list)):
        return value
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    if len(serialized) <= MAX_TOOL_RESULT_CHARS:
        return value
    return {
        "truncated": True,
        "json_prefix": serialized[:MAX_TOOL_RESULT_CHARS],
        "original_chars": len(serialized),
    }


def _executor_owner(
    name: str, agent_tool_names: set[str], user_tool_names: set[str]
) -> str:
    if name in agent_tool_names:
        return "assistant"
    if name in user_tool_names:
        return "user"
    return "unknown"


def _extract_dialogue_and_events(
    prefix: list[Message],
    *,
    agent_tool_names: set[str],
    user_tool_names: set[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[dict[str, str]]],
]:
    dialogue = []
    events = []
    events_by_call_id: dict[str, dict[str, Any]] = {}
    entities: dict[str, list[dict[str, str]]] = {}

    for index, message in enumerate(prefix):
        if isinstance(message, UserMessage) and not message.is_tool_call():
            evidence_id = f"msg_{index}"
            content = _short_text(message.content)
            dialogue.append(
                {"evidence_id": evidence_id, "role": "user", "content": content}
            )
            _collect_text_entities(content, source=evidence_id, entities=entities)
        elif isinstance(message, AssistantMessage):
            if message.content:
                dialogue.append(
                    {
                        "evidence_id": f"msg_{index}",
                        "role": "assistant",
                        "content": _short_text(message.content),
                    }
                )
            for call_index, call in enumerate(message.tool_calls or []):
                evidence_id = f"tool_{index}_{call_index}"
                owner = _executor_owner(call.name, agent_tool_names, user_tool_names)
                event = {
                    "evidence_id": evidence_id,
                    "actor": call.requestor,
                    "owner": owner,
                    "executor_valid": call.requestor == owner,
                    "name": call.name,
                    "arguments": call.arguments,
                    "status": "pending",
                    "result": None,
                }
                events.append(event)
                events_by_call_id[call.id] = event
                _collect_entities(
                    call.arguments,
                    source=evidence_id,
                    entities=entities,
                )
        elif (
            isinstance(message, ToolMessage)
            and message.requestor == "assistant"
            and message.id in events_by_call_id
        ):
            event = events_by_call_id[message.id]
            event["status"] = "error" if message.error else "success"
            full_result = _result_value(message.content)
            event["result"] = _compact_result(full_result)
            _collect_entities(
                full_result,
                source=event["evidence_id"],
                entities=entities,
            )

    return dialogue[-MAX_RECENT_DIALOGUE:], events[-MAX_TOOL_EVENTS:], entities


def _quality_flags(dialogue: list[dict[str, Any]]) -> list[str]:
    user_text = "\n".join(row["content"] for row in dialogue if row["role"] == "user")
    flags = []
    if any(pattern.search(user_text) for pattern in ROLE_INVERSION_PATTERNS):
        flags.append("suspected_user_role_inversion")
    return flags


def _latest_user_reports_resolution(dialogue: list[dict[str, Any]]) -> bool:
    latest = next(
        (row["content"] for row in reversed(dialogue) if row["role"] == "user"),
        "",
    )
    return any(pattern.search(latest) for pattern in RESOLUTION_PATTERNS)


def _interaction_phase(
    *,
    dialogue: list[dict[str, Any]],
    events: list[dict[str, Any]],
    entities: dict[str, list[dict[str, str]]],
) -> str:
    if any(
        event["name"] == "transfer_to_human_agents" and event["status"] == "success"
        for event in events
    ):
        return "transfer_completed"
    if _latest_user_reports_resolution(dialogue):
        return "resolution_reported"
    if not entities.get("customer_id") and not entities.get("phone_number"):
        return "identity_collection"
    if any(event["status"] == "error" for event in events):
        return "error_recovery"
    return "troubleshooting"


def build_verified_context(
    *,
    prefix: list[Message],
    agent_tools: list[Tool],
    user_tools: list[Tool],
) -> dict[str, Any]:
    """Build a compact evidence ledger without calling an LLM."""

    visible_prefix = [
        message for message in prefix if is_valid_agent_history_message(message)
    ]
    agent_names = set(_tool_names(agent_tools))
    user_names = set(_tool_names(user_tools))
    dialogue, events, entities = _extract_dialogue_and_events(
        visible_prefix,
        agent_tool_names=agent_names,
        user_tool_names=user_names,
    )
    phase = _interaction_phase(
        dialogue=dialogue,
        events=events,
        entities=entities,
    )
    return {
        "schema_version": "hybrid_clean_v1",
        "interaction_phase": phase,
        "terminal_state": {
            "transfer_completed": phase == "transfer_completed",
            "resolution_reported": phase == "resolution_reported",
        },
        "recent_dialogue": dialogue,
        "tool_events": events,
        "entities": entities,
        "tool_ownership": {
            "assistant": sorted(agent_names),
            "user": sorted(user_names),
        },
        "quality_flags": _quality_flags(dialogue),
    }


def evidence_ids(verified_context: dict[str, Any]) -> set[str]:
    """Return every evidence identifier available to the semantic builder."""

    direct_ids = {
        row["evidence_id"]
        for field in ("recent_dialogue", "tool_events")
        for row in verified_context[field]
    }
    entity_source_ids = {
        row["source"]
        for records in verified_context["entities"].values()
        for row in records
    }
    return direct_ids | entity_source_ids
