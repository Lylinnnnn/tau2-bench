from tau2.runner import load_tasks
from trace_to_micro.replay.reference import replay_reference_task

TASK_ID = "[mobile_data_issue]data_mode_off[PERSONA:None]"


def test_reference_replay_extracts_user_state_change() -> None:
    task = next(task for task in load_tasks("telecom", "small") if task.id == TASK_ID)

    events = replay_reference_task(task, split="small")

    toggle_event = next(event for event in events if event.action_name == "toggle_data")
    changed_paths = {change.path for change in toggle_event.changes}
    assert toggle_event.actor == "user"
    assert toggle_event.declared_mutating is True
    assert "user.device.data_enabled" in changed_paths
    assert toggle_event.changed_state is True


def test_reference_replay_starts_from_initialized_fault_state() -> None:
    task = next(task for task in load_tasks("telecom", "small") if task.id == TASK_ID)

    events = replay_reference_task(task, split="small")

    assert len(events) == 1
    data_change = next(
        change
        for change in events[0].changes
        if change.path == "user.device.data_enabled"
    )
    assert data_change.before is False
    assert data_change.after is True
