import json

from tau2.data_model.message import AssistantMessage, UserMessage
from trace_to_micro.context_builder import (
    ContextBuildError,
    build_structured_context,
    make_probe_messages,
)


def _context() -> dict:
    return {
        "user_goal": "restore data",
        "observed_facts": ["data is unavailable"],
        "completed_steps": [],
        "tool_observations": [],
        "open_questions": ["is data enabled"],
        "local_subgoal": "determine whether mobile data is enabled",
        "success_condition": "the setting is reported",
    }


def test_context_builder_sees_prefix_but_not_future_target() -> None:
    prefix = [UserMessage(role="user", content="VISIBLE")]

    def fake_generate(**kwargs):
        request = kwargs["messages"][-1]
        assert "VISIBLE" in request.content
        assert "FUTURE" not in request.content
        return AssistantMessage(role="assistant", content=json.dumps(_context()))

    context, metadata = build_structured_context(
        model="fake",
        llm_args={},
        policy="policy",
        tools=[],
        prefix=prefix,
        generate_fn=fake_generate,
    )

    assert context["local_subgoal"].startswith("determine")
    assert metadata["attempt"] == 1
    assert metadata["usage"] is None


def test_context_builder_retries_invalid_json_once() -> None:
    responses = iter(
        [
            AssistantMessage(
                role="assistant",
                content='{"user_goal": "truncated',
                raw_data={"choices": [{"finish_reason": "length"}]},
            ),
            AssistantMessage(
                role="assistant",
                content=json.dumps(_context()),
                raw_data={"choices": [{"finish_reason": "stop"}]},
            ),
        ]
    )

    context, metadata = build_structured_context(
        model="fake",
        llm_args={},
        policy="policy",
        tools=[],
        prefix=[],
        generate_fn=lambda **_: next(responses),
    )

    assert context == _context()
    assert metadata["attempt"] == 2
    assert metadata["finish_reason"] == "stop"


def test_context_builder_exposes_bounded_failure_diagnostics() -> None:
    content = "{" + "x" * 5_000

    def invalid_response(**kwargs):
        return AssistantMessage(
            role="assistant",
            content=content,
            usage={"completion_tokens": 100},
            raw_data={"choices": [{"finish_reason": "length"}]},
        )

    try:
        build_structured_context(
            model="fake",
            llm_args={},
            policy="policy",
            tools=[],
            prefix=[],
            generate_fn=invalid_response,
        )
    except ContextBuildError as error:
        assert len(error.diagnostics) == 2
        assert error.diagnostics[-1]["finish_reason"] == "length"
        assert error.diagnostics[-1]["response_chars"] == len(content)
        assert len(error.diagnostics[-1]["response_head"]) == 2_000
        assert len(error.diagnostics[-1]["response_tail"]) == 2_000
    else:
        raise AssertionError("Expected invalid JSON to raise ContextBuildError")


def test_structured_state_removes_explicit_subgoal_fields() -> None:
    messages = make_probe_messages(
        variant="structured_state",
        agent_system_prompt="system",
        prefix=[],
        structured_context=_context(),
    )

    assert "local_subgoal" not in messages[-1].content
    assert "observed_facts" in messages[-1].content


def test_long_raw_preserves_visible_prefix() -> None:
    prefix = [UserMessage(role="user", content="raw request")]

    messages = make_probe_messages(
        variant="long_raw",
        agent_system_prompt="system",
        prefix=prefix,
        structured_context=None,
    )

    assert messages[1] is prefix[0]


def test_hybrid_clean_opens_a_new_window_with_action_contract() -> None:
    context = {
        "verified_context": {"interaction_phase": "identity_collection"},
        "semantic_brief": {"proposed_subgoal": "request phone number"},
        "action_contract": {
            "agent_response_mode": "request_information",
            "allowed_agent_tools": [],
        },
    }
    prefix = [UserMessage(role="user", content="raw history must be omitted")]

    messages = make_probe_messages(
        variant="hybrid_clean",
        agent_system_prompt="system",
        prefix=prefix,
        structured_context=context,
    )

    assert len(messages) == 2
    assert "raw history must be omitted" not in messages[-1].content
    assert "request_information" in messages[-1].content
