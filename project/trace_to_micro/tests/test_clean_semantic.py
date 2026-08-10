import json

from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.environment.tool import as_tool
from trace_to_micro.clean.semantic import (
    MAX_SEMANTIC_RETRIES,
    build_hybrid_clean_context,
    validate_semantic_brief,
)


def get_customer_by_phone(phone_number: str) -> str:
    """Look up a customer by phone number."""

    return phone_number


def toggle_airplane_mode() -> str:
    """Toggle airplane mode on the user's device."""

    return "ok"


AGENT_TOOLS = [as_tool(get_customer_by_phone)]
USER_TOOLS = [as_tool(toggle_airplane_mode)]


def _brief(evidence: str) -> dict:
    return {
        "user_goal": "Restore mobile data.",
        "active_issue": "Airplane mode may be enabled.",
        "situation_summary": "The user reports unavailable mobile data.",
        "proposed_subgoal": "Ask the user to check Airplane Mode.",
        "proposed_action": {
            "name": "toggle_airplane_mode",
            "executor": "user",
        },
        "success_evidence": "The user confirms Airplane Mode is disabled.",
        "evidence_ids": [evidence],
    }


def test_semantic_gate_rejects_unknown_evidence() -> None:
    verified = {
        "recent_dialogue": [
            {"evidence_id": "msg_0", "role": "user", "content": "No data"}
        ],
        "tool_events": [],
        "entities": {},
        "tool_ownership": {
            "assistant": ["get_customer_by_phone"],
            "user": ["toggle_airplane_mode"],
        },
        "terminal_state": {
            "transfer_completed": False,
            "resolution_reported": False,
        },
    }

    errors = validate_semantic_brief(_brief("missing"), verified)

    assert {error["code"] for error in errors} == {"UNKNOWN_EVIDENCE"}


def test_semantic_builder_allows_at_most_two_retries() -> None:
    calls = 0

    def always_bad(**kwargs):
        nonlocal calls
        calls += 1
        return AssistantMessage(role="assistant", content=json.dumps(_brief("missing")))

    context, metadata = build_hybrid_clean_context(
        model="fake",
        llm_args={},
        prefix=[UserMessage(role="user", content="My mobile data is unavailable")],
        agent_tools=AGENT_TOOLS,
        user_tools=USER_TOOLS,
        generate_fn=always_bad,
    )

    assert calls == 1 + MAX_SEMANTIC_RETRIES
    assert metadata["retry_count"] == MAX_SEMANTIC_RETRIES
    assert metadata["status"] == "rule_fallback"
    assert context["training_eligible"] is False


def test_semantic_builder_accepts_a_valid_retry_patch() -> None:
    responses = iter(
        [
            _brief("missing"),
            {"evidence_ids": ["msg_0"]},
        ]
    )

    context, metadata = build_hybrid_clean_context(
        model="fake",
        llm_args={},
        prefix=[UserMessage(role="user", content="My mobile data is unavailable")],
        agent_tools=AGENT_TOOLS,
        user_tools=USER_TOOLS,
        generate_fn=lambda **_: AssistantMessage(
            role="assistant", content=json.dumps(next(responses))
        ),
    )

    assert metadata["attempt_count"] == 2
    assert metadata["retry_count"] == 1
    assert metadata["status"] == "passed"
    assert context["semantic_brief"]["evidence_ids"] == ["msg_0"]
    assert context["action_contract"]["agent_response_mode"] == "instruction"
