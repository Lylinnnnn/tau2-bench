from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from trace_to_micro.logged_trace import decision_indices, stratified_indices


def _simulation() -> SimulationRun:
    return SimulationRun(
        id="sim-1",
        task_id="task-1",
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=[
            AssistantMessage(role="assistant", content="Hello"),
            UserMessage(role="user", content="Please help"),
            AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="a1", name="lookup", arguments={})],
            ),
            ToolMessage(id="a1", role="tool", requestor="assistant", content="ok"),
            AssistantMessage(role="assistant", content="Done"),
        ],
        trial=0,
        seed=300,
    )


def test_decision_indices_exclude_static_greeting() -> None:
    assert decision_indices(_simulation()) == [2, 4]


def test_stratified_indices_keep_early_middle_late() -> None:
    assert stratified_indices([2, 4, 6, 8, 10], 3) == [2, 6, 10]
    assert stratified_indices([2, 4, 6], 1) == [6]
