from tau2.runner import load_task_splits, load_tasks
from trace_to_micro.models import TaskIdentity
from trace_to_micro.task_inventory import build_task_inventory, parse_task_id


def test_parse_composed_task_id() -> None:
    task_id = (
        "[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:Hard]"
    )

    assert parse_task_id(task_id) == TaskIdentity(
        intent="mobile_data_issue",
        atoms=("airplane_mode_on", "data_mode_off"),
        persona="Hard",
    )


def test_parse_none_persona_as_missing_value() -> None:
    identity = parse_task_id(
        "[mobile_data_issue]data_mode_off[PERSONA:None]"
    )

    assert identity.persona is None


def test_current_telecom_split_is_compositional_ood() -> None:
    tasks = load_tasks("telecom", task_split_name=None)
    split_map = load_task_splits("telecom")

    inventory = build_task_inventory(tasks, split_map, "train", "test")

    assert inventory["split_sizes"]["train"] == 74
    assert inventory["split_sizes"]["test"] == 40
    assert inventory["split_sizes"]["base"] == 114
    assert inventory["exact_task_overlap"] == []
    assert len(inventory["shared_atoms"]) == 19
    assert inventory["train_only_atoms"] == []
    assert inventory["test_only_atoms"] == []
    assert inventory["unique_initialized_users"] == [
        {"name": "John Smith", "phone_number": "555-123-2002"}
    ]
    assert inventory["unique_ticket_count"] == 5
