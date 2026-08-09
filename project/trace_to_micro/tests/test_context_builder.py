import json

from tau2.data_model.message import AssistantMessage, UserMessage
from trace_to_micro.context_builder import (
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

    context, usage = build_structured_context(
        model="fake",
        llm_args={},
        policy="policy",
        tools=[],
        prefix=prefix,
        generate_fn=fake_generate,
    )

    assert context["local_subgoal"].startswith("determine")
    assert usage is None


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
