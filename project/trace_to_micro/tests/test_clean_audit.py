from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.environment.tool import as_tool
from trace_to_micro.clean.audit import attach_source_audit, audit_source_action


def get_customer_by_phone(phone_number: str) -> str:
    """Look up a customer by phone number."""

    return phone_number


def toggle_airplane_mode() -> str:
    """Toggle airplane mode on the user's device."""

    return "ok"


def test_source_audit_rejects_logged_user_tool_called_by_agent() -> None:
    target = AssistantMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                name="toggle_airplane_mode",
                requestor="assistant",
                arguments={},
            )
        ],
    )
    actual_branch = {
        "tool_responses": [
            {
                "tool_call": {
                    "requestor": "assistant",
                    "name": "toggle_airplane_mode",
                },
                "error": True,
            }
        ]
    }

    audit = audit_source_action(
        target=target,
        actual_branch=actual_branch,
        agent_tools=[as_tool(get_customer_by_phone)],
        user_tools=[as_tool(toggle_airplane_mode)],
    )

    assert audit == {
        "valid": False,
        "tool_exists": True,
        "executor_valid": False,
        "execution_success": False,
    }


def test_resumed_prediction_gets_audit_without_changing_model_output() -> None:
    row = {"snapshot_id": "s1", "variant": "long_raw", "metrics": {}}
    audit = {
        "valid": False,
        "tool_exists": True,
        "executor_valid": False,
        "execution_success": False,
    }

    changed = attach_source_audit(row, audit=audit)

    assert changed is True
    assert row["source_action_audit"] == audit
    assert row["training_eligible"] is False
    assert attach_source_audit(row, audit=audit) is False
