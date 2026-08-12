from types import SimpleNamespace

from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from trace_to_micro.replay.consequence import replay_local_consequences


class _DB:
    def __init__(self) -> None:
        self.count = 0

    def model_dump(self, mode: str) -> dict:
        assert mode == "json"
        return {"count": self.count}


class _Tools:
    def __init__(self, db: _DB) -> None:
        self.db = db

    def has_tool(self, name: str) -> bool:
        return name == "increment"

    def tool_type(self, name: str):
        assert name == "increment"
        return SimpleNamespace(value="write")


class _Environment:
    def __init__(self) -> None:
        self.tools = _Tools(_DB())
        self.user_tools = None

    def set_state(self, **kwargs) -> None:
        assert kwargs["message_history"] == []

    def _is_mutating_tool(self, name: str) -> bool:
        return name == "increment"

    def get_response(self, call: ToolCall) -> ToolMessage:
        self.tools.db.count += 1
        return ToolMessage(
            id=call.id,
            role="tool",
            requestor="assistant",
            content="1",
            error=False,
        )


def test_replay_local_consequence_uses_real_next_state_and_official_target(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "trace_to_micro.replay.consequence.build_environment",
        lambda domain: _Environment(),
    )
    action = SimpleNamespace(
        action_id="gold-1",
        requestor="assistant",
        name="increment",
        arguments={},
    )
    task = SimpleNamespace(
        id="task-1",
        initial_state=None,
        evaluation_criteria=SimpleNamespace(actions=[action]),
    )
    simulation = SimulationRun(
        id="sim-1",
        task_id="task-1",
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60,
        termination_reason=TerminationReason.USER_STOP,
        messages=[
            AssistantMessage(role="assistant", content="Hello"),
            UserMessage(role="user", content="Please continue"),
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        requestor="assistant",
                        name="increment",
                        arguments={},
                    )
                ],
            ),
            ToolMessage(
                id="call-1",
                role="tool",
                requestor="assistant",
                content="1",
                error=False,
            ),
        ],
        trial=0,
        seed=300,
    )

    rows = replay_local_consequences(
        simulation=simulation,
        task=task,
        split="train",
        domain="retail",
    )

    assert len(rows) == 1
    assert rows[0]["decision_id"] == "sim-1:2"
    assert rows[0]["tool_success"] is True
    assert rows[0]["state_changed"] is True
    assert rows[0]["goal_progress"] is True
    assert rows[0]["target_distance_before"] == 1
    assert rows[0]["target_distance_after"] == 0
