import json

from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import as_tool
from trace_to_micro.clean.evidence import build_verified_context, evidence_ids


def transfer_to_human_agents(summary: str) -> str:
    """Transfer the current conversation."""

    return summary


def get_customer_by_phone(phone_number: str) -> str:
    """Look up a customer by phone number."""

    return phone_number


def toggle_airplane_mode() -> str:
    """Toggle airplane mode on the user's device."""

    return "ok"


AGENT_TOOLS = [as_tool(transfer_to_human_agents), as_tool(get_customer_by_phone)]
USER_TOOLS = [as_tool(toggle_airplane_mode)]


def test_verified_context_tracks_completed_transfer() -> None:
    prefix = [
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="transfer-1",
                    name="transfer_to_human_agents",
                    arguments={"summary": "unresolved"},
                )
            ],
        ),
        ToolMessage(
            id="transfer-1",
            role="tool",
            requestor="assistant",
            content="Transfer successful",
        ),
    ]

    context = build_verified_context(
        prefix=prefix,
        agent_tools=AGENT_TOOLS,
        user_tools=USER_TOOLS,
    )

    assert context["interaction_phase"] == "transfer_completed"
    assert context["terminal_state"]["transfer_completed"] is True
    assert context["tool_events"][0]["status"] == "success"


def test_verified_context_extracts_entities_and_flags_role_inversion() -> None:
    prefix = [
        UserMessage(
            role="user",
            content=(
                "My number is 555-123-2002. Would you like me to change "
                "the network mode for you?"
            ),
        )
    ]

    context = build_verified_context(
        prefix=prefix,
        agent_tools=AGENT_TOOLS,
        user_tools=USER_TOOLS,
    )

    assert context["entities"]["phone_number"][0]["value"] == "555-123-2002"
    assert context["quality_flags"] == ["suspected_user_role_inversion"]


def test_entity_source_remains_valid_evidence_after_event_windowing() -> None:
    prefix = [UserMessage(role="user", content="My number is 555-123-2002")]
    prefix.extend(
        UserMessage(role="user", content=f"follow-up {index}") for index in range(10)
    )

    context = build_verified_context(
        prefix=prefix,
        agent_tools=AGENT_TOOLS,
        user_tools=USER_TOOLS,
    )

    assert "msg_0" not in {row["evidence_id"] for row in context["recent_dialogue"]}
    assert "msg_0" in evidence_ids(context)


def test_large_tool_result_is_compact_but_all_entity_ids_are_extracted() -> None:
    customer = {
        "customer_id": "C1001",
        "line_ids": [f"L{index:04d}" for index in range(100)],
        "notes": "x" * 5_000,
    }
    prefix = [
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="customer-1",
                    name="get_customer_by_phone",
                    arguments={"phone_number": "555-123-2002"},
                )
            ],
        ),
        ToolMessage(
            id="customer-1",
            role="tool",
            requestor="assistant",
            content=json.dumps(customer),
        ),
    ]

    context = build_verified_context(
        prefix=prefix,
        agent_tools=AGENT_TOOLS,
        user_tools=USER_TOOLS,
    )

    assert context["tool_events"][0]["result"]["truncated"] is True
    assert len(context["entities"]["line_id"]) == 100
