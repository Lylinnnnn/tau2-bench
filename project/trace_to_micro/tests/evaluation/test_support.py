from trace_to_micro.data_model import StateChange, TransitionEvent
from trace_to_micro.evaluation import build_support_report


def _event(
    task_id: str,
    split: str,
    *,
    after: bool,
    declared_mutating: bool = True,
) -> TransitionEvent:
    return TransitionEvent(
        task_id=task_id,
        split=split,
        step_index=0,
        actor="user",
        action_name="toggle_data",
        arguments={},
        declared_mutating=declared_mutating,
        tool_result="ok",
        changes=(
            StateChange(
                path="user.device.data_enabled",
                before=not after,
                after=after,
            ),
        ),
    )


def test_support_report_uses_distinct_tasks_and_exact_effects() -> None:
    train_events = [
        _event("task-1", "train", after=True),
        _event("task-2", "train", after=True),
        _event("task-3", "train", after=False),
    ]
    test_events = [_event("task-4", "test", after=True)]

    report = build_support_report(
        train_events,
        test_events,
        thresholds=(1, 2, 3),
    )

    assert report["coverage"]["1"]["action"]["rate"] == 1.0
    assert report["coverage"]["2"]["exact_effect"]["rate"] == 1.0
    assert report["coverage"]["3"]["exact_effect"]["rate"] == 0.0
    assert report["train_effect_consistency"]["action_count"] == 1
    assert len(report["train_effect_consistency"]["ambiguous_actions"]) == 1


def test_support_report_surfaces_mutation_annotation_mismatch() -> None:
    event = _event(
        "task-1",
        "train",
        after=True,
        declared_mutating=False,
    )

    report = build_support_report([event], [], thresholds=(1,))

    assert report["declaration_mismatches"]["declared_non_mutating_with_change"] == 1


def test_train_loto_never_counts_query_task_as_support() -> None:
    train_events = [
        _event("task-1", "train", after=True),
        _event("task-2", "train", after=True),
    ]

    report = build_support_report(
        train_events,
        train_events,
        thresholds=(1, 2),
        leave_one_task_out=True,
    )

    assert report["coverage"]["1"]["exact_effect"]["rate"] == 1.0
    assert report["coverage"]["2"]["exact_effect"]["rate"] == 0.0
