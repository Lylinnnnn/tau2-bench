from tau2.environment.tool import as_tool
from trace_to_micro.clean.contract import compile_action_contract


def get_customer_by_phone(phone_number: str) -> str:
    """Look up a customer by phone number."""

    return phone_number


def get_details_by_id(id: str) -> str:
    """Look up any supported entity identifier."""

    return id


def toggle_airplane_mode() -> str:
    """Toggle airplane mode on the user's device."""

    return "ok"


AGENT_TOOLS = [as_tool(get_customer_by_phone), as_tool(get_details_by_id)]


def _verified(*, phone: str | None = None, transfer: bool = False) -> dict:
    entities = {}
    if phone is not None:
        entities["phone_number"] = [{"value": phone, "source": "msg_1"}]
    return {
        "terminal_state": {
            "transfer_completed": transfer,
            "resolution_reported": False,
        },
        "entities": entities,
        "tool_events": [],
        "tool_ownership": {
            "assistant": ["get_customer_by_phone", "get_details_by_id"],
            "user": ["toggle_airplane_mode"],
        },
    }


def _brief(name: str | None, executor: str) -> dict:
    return {
        "situation_summary": "Mobile data is unavailable.",
        "proposed_action": {"name": name, "executor": executor},
        "success_evidence": "The user confirms the setting changed.",
    }


def test_user_tool_is_compiled_to_instruction_not_agent_call() -> None:
    contract = compile_action_contract(
        semantic_brief=_brief("toggle_airplane_mode", "assistant"),
        verified_context=_verified(),
        agent_tools=AGENT_TOOLS,
    )

    assert contract["next_executor"] == "user"
    assert contract["agent_response_mode"] == "instruction"
    assert contract["allowed_agent_tools"] == []
    assert contract["repairs"] == ["corrected_executor_to_user"]


def test_missing_phone_blocks_customer_lookup() -> None:
    contract = compile_action_contract(
        semantic_brief=_brief("get_customer_by_phone", "assistant"),
        verified_context=_verified(),
        agent_tools=AGENT_TOOLS,
    )

    assert contract["agent_response_mode"] == "request_information"
    assert contract["required_inputs"] == ["phone_number"]
    assert contract["allowed_agent_tools"] == []


def test_known_phone_is_bound_by_code() -> None:
    contract = compile_action_contract(
        semantic_brief=_brief("get_customer_by_phone", "assistant"),
        verified_context=_verified(phone="555-123-2002"),
        agent_tools=AGENT_TOOLS,
    )

    assert contract["agent_response_mode"] == "tool"
    assert contract["bound_arguments"] == {"phone_number": "555-123-2002"}


def test_completed_transfer_forces_confirmation_text() -> None:
    contract = compile_action_contract(
        semantic_brief=_brief(None, "none"),
        verified_context=_verified(transfer=True),
        agent_tools=AGENT_TOOLS,
    )

    assert contract["agent_response_mode"] == "text"
    assert "transfer_to_human_agents" in contract["forbidden_actions"]


def test_generic_id_prefers_the_single_identifier_named_by_the_brief() -> None:
    verified = _verified()
    verified["entities"] = {
        "customer_id": [{"value": "C1001", "source": "tool_1_0"}],
        "line_id": [{"value": "L1001", "source": "tool_1_0"}],
    }
    brief = _brief("get_details_by_id", "assistant")
    brief["situation_summary"] = "Inspect line L1001 before proceeding."

    contract = compile_action_contract(
        semantic_brief=brief,
        verified_context=verified,
        agent_tools=AGENT_TOOLS,
    )

    assert contract["agent_response_mode"] == "tool"
    assert contract["bound_arguments"] == {"id": "L1001"}
