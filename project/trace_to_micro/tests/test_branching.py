from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.runner import load_tasks
from trace_to_micro.branching import execute_macro_step

TASK_ID = "[mobile_data_issue]data_mode_off[PERSONA:None]"


def test_logged_user_handoff_executes_real_state_change_without_llm() -> None:
    task = next(task for task in load_tasks("telecom", "small") if task.id == TASK_ID)
    prefix = [
        AssistantMessage(role="assistant", content="How can I help?"),
        UserMessage(role="user", content="My mobile data is not working."),
    ]
    target = AssistantMessage(role="assistant", content="Please enable mobile data.")
    user_action = UserMessage(
        role="user",
        tool_calls=[
            ToolCall(
                id="u1",
                requestor="user",
                name="toggle_data",
                arguments={},
            )
        ],
    )

    branch = execute_macro_step(
        domain="telecom",
        task=task,
        prefix=prefix,
        assistant_message=target,
        user_name="unused",
        user_llm="unused",
        user_llm_args={},
        seed=300,
        logged_messages=[*prefix, target, user_action],
        logged_target_index=2,
    )

    assert branch["user_continuation"]["calls"][0]["name"] == "toggle_data"
    assert branch["tool_error"] is False
    assert branch["changes"][0]["path"] == "user.device.data_enabled"
    assert branch["changes"][0]["after"] is True
