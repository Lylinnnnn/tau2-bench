from expectation_step_rl.tau2_adapter.execution import (
    deserialize_messages,
    serialize_messages,
)
from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def test_replay_message_round_trip_preserves_role_specific_fields() -> None:
    messages = [
        UserMessage(role="user", content="Where is order #A12345?"),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="chatcmpl-tool-abc123",
                    name="get_order_details",
                    arguments={"order_id": "#A12345"},
                )
            ],
        ),
        ToolMessage(
            role="tool",
            id="chatcmpl-tool-abc123",
            requestor="assistant",
            content='{"status":"delivered"}',
        ),
    ]

    restored = deserialize_messages(serialize_messages(messages))

    assert restored == messages
    assert restored[1].tool_calls[0].arguments == {"order_id": "#A12345"}
