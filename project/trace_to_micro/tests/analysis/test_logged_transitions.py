from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.runner import load_tasks
from trace_to_micro.analysis.logged_transitions import replay_logged_simulation

TASK_ID = "[mobile_data_issue]data_mode_off[PERSONA:None]"


def test_logged_replay_records_behavior_action_and_state_effect() -> None:
    task = next(task for task in load_tasks("telecom", "small") if task.id == TASK_ID)
    messages = [
        AssistantMessage(role="assistant", content="How can I help?"),
        UserMessage(role="user", content="Mobile data is unavailable."),
        AssistantMessage(role="assistant", content="Please enable mobile data."),
        UserMessage(
            role="user",
            tool_calls=[
                ToolCall(
                    id="u1",
                    requestor="user",
                    name="toggle_data",
                    arguments={},
                )
            ],
        ),
        ToolMessage(
            id="u1",
            role="tool",
            requestor="user",
            content="Mobile data is now enabled",
        ),
    ]
    simulation = SimulationRun(
        id="sim-1",
        task_id=task.id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=messages,
        trial=0,
        seed=300,
    )

    rows = replay_logged_simulation(
        simulation=simulation,
        task=task,
        split="train",
        domain="telecom",
    )

    assert len(rows) == 1
    assert rows[0]["actor"] == "user"
    assert rows[0]["action_name"] == "toggle_data"
    assert rows[0]["changes"][0]["path"] == "user.device.data_enabled"
    assert rows[0]["replay_error"] is False
