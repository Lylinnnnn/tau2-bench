"""Evaluation-only quality checks for logged source actions."""

from typing import Any

from tau2.data_model.message import AssistantMessage
from tau2.environment.tool import Tool


def audit_source_action(
    *,
    target: AssistantMessage,
    actual_branch: dict[str, Any],
    agent_tools: list[Tool],
    user_tools: list[Tool],
) -> dict[str, Any]:
    """Audit a logged target without exposing the result to context generation."""

    agent_names = {tool.name for tool in agent_tools}
    all_names = agent_names | {tool.name for tool in user_tools}
    calls = target.tool_calls or []
    tool_exists = all(call.name in all_names for call in calls)
    executor_valid = all(
        call.requestor == "assistant" and call.name in agent_names for call in calls
    )
    assistant_responses = [
        row
        for row in actual_branch["tool_responses"]
        if row["tool_call"]["requestor"] == "assistant"
    ]
    execution_success = not any(row["error"] for row in assistant_responses)
    valid = tool_exists and executor_valid and execution_success
    return {
        "valid": valid,
        "tool_exists": tool_exists,
        "executor_valid": executor_valid,
        "execution_success": execution_success,
    }


def attach_source_audit(
    row: dict[str, Any],
    *,
    audit: dict[str, Any],
    context_eligible: bool = True,
) -> bool:
    """Attach new audit fields to a resumed legacy prediction row."""

    changed = False
    if "source_action_audit" not in row:
        row["source_action_audit"] = audit
        changed = True
    if "training_eligible" not in row:
        row["training_eligible"] = audit["valid"] and context_eligible
        changed = True
    return changed
