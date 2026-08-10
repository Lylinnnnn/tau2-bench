"""Compile a deterministic execution contract from a semantic brief."""

from typing import Any

from tau2.environment.tool import Tool


def _required_parameters(tool: Tool) -> list[str]:
    schema = tool.openai_schema["function"]["parameters"]
    return list(schema.get("required", []))


def _entity_values(verified_context: dict[str, Any], field: str) -> list[str]:
    return sorted({row["value"] for row in verified_context["entities"].get(field, [])})


def _semantic_text(brief: dict[str, Any]) -> str:
    return " ".join(
        str(value)
        for key, value in brief.items()
        if key not in {"proposed_action", "evidence_ids"}
    )


def _bind_generic_id(
    *, brief: dict[str, Any], verified_context: dict[str, Any]
) -> str | None:
    candidates = sorted(
        {
            row["value"]
            for field, records in verified_context["entities"].items()
            if field.endswith("_id")
            for row in records
        }
    )
    semantic_text = _semantic_text(brief)
    mentioned = [value for value in candidates if value in semantic_text]
    if len(mentioned) == 1:
        return mentioned[0]
    return candidates[0] if len(candidates) == 1 else None


def _bind_parameter(
    *,
    name: str,
    brief: dict[str, Any],
    verified_context: dict[str, Any],
) -> Any | None:
    if name == "summary":
        return brief["situation_summary"]
    if name == "id":
        return _bind_generic_id(
            brief=brief,
            verified_context=verified_context,
        )
    values = _entity_values(verified_context, name)
    return values[0] if len(values) == 1 else None


def _failed_invalid_tools(verified_context: dict[str, Any]) -> list[str]:
    return sorted(
        {
            event["name"]
            for event in verified_context["tool_events"]
            if event["status"] == "error" and not event["executor_valid"]
        }
    )


def compile_action_contract(
    *,
    semantic_brief: dict[str, Any],
    verified_context: dict[str, Any],
    agent_tools: list[Tool],
) -> dict[str, Any]:
    """Turn a semantic proposal into executor, mode, and argument constraints."""

    proposed = semantic_brief["proposed_action"]
    action_name = proposed["name"]
    proposed_executor = proposed["executor"]
    ownership = verified_context["tool_ownership"]
    agent_by_name = {tool.name: tool for tool in agent_tools}
    repairs = []
    forbidden = _failed_invalid_tools(verified_context)

    if verified_context["terminal_state"]["transfer_completed"]:
        return {
            "next_executor": "assistant",
            "agent_response_mode": "text",
            "allowed_agent_tools": [],
            "required_inputs": [],
            "bound_arguments": {},
            "forbidden_actions": sorted(set(forbidden) | {"transfer_to_human_agents"}),
            "completion_condition": "Send the policy-required transfer confirmation.",
            "repairs": ["terminal_transfer_confirmation"],
        }

    if action_name is None:
        return {
            "next_executor": "assistant",
            "agent_response_mode": "text",
            "allowed_agent_tools": [],
            "required_inputs": [],
            "bound_arguments": {},
            "forbidden_actions": forbidden,
            "completion_condition": semantic_brief["success_evidence"],
            "repairs": repairs,
        }

    if action_name in ownership["user"]:
        if proposed_executor != "user":
            repairs.append("corrected_executor_to_user")
        return {
            "next_executor": "user",
            "agent_response_mode": "instruction",
            "allowed_agent_tools": [],
            "required_inputs": [],
            "bound_arguments": {},
            "forbidden_actions": sorted(set(forbidden) | {action_name}),
            "completion_condition": semantic_brief["success_evidence"],
            "repairs": repairs,
        }

    tool = agent_by_name[action_name]
    if proposed_executor != "assistant":
        repairs.append("corrected_executor_to_assistant")
    bound = {}
    missing = []
    for parameter in _required_parameters(tool):
        value = _bind_parameter(
            name=parameter,
            brief=semantic_brief,
            verified_context=verified_context,
        )
        if value is None:
            missing.append(parameter)
        else:
            bound[parameter] = value
    if missing:
        return {
            "next_executor": "assistant",
            "agent_response_mode": "request_information",
            "allowed_agent_tools": [],
            "required_inputs": missing,
            "bound_arguments": bound,
            "forbidden_actions": forbidden,
            "completion_condition": (
                "Obtain the missing inputs before attempting the proposed action."
            ),
            "repairs": [*repairs, "blocked_on_required_inputs"],
        }
    return {
        "next_executor": "assistant",
        "agent_response_mode": "tool",
        "allowed_agent_tools": [action_name],
        "required_inputs": [],
        "bound_arguments": bound,
        "forbidden_actions": forbidden,
        "completion_condition": semantic_brief["success_evidence"],
        "repairs": repairs,
    }
