from pathlib import Path
from types import SimpleNamespace

import pytest

from trace_to_micro.replay.consequence import InvalidReferenceTargetError
from trace_to_micro.runner.local_consequence import (
    _domain_consequences,
    _support_report,
)


def test_domain_consequences_excludes_invalid_target_and_continues(
    monkeypatch,
) -> None:
    tasks = [SimpleNamespace(id="good"), SimpleNamespace(id="bad")]
    simulations = [
        SimpleNamespace(id="sim-good", task_id="good"),
        SimpleNamespace(id="sim-bad", task_id="bad"),
    ]
    results = SimpleNamespace(tasks=tasks, simulations=simulations)
    config = SimpleNamespace(
        train_split="train",
        test_split="test",
        expected_num_trials=1,
        expected_agent_model="agent-model",
        expected_user_model="user-model",
        results_path=lambda domain: Path(f"{domain}/results.json"),
    )
    action = SimpleNamespace(
        action_id="bad-write",
        requestor="assistant",
        name="exchange_delivered_order_items",
        arguments={"order_id": "pending-order"},
    )
    monkeypatch.setattr(
        "trace_to_micro.runner.local_consequence.Results.load",
        lambda _: results,
    )
    monkeypatch.setattr(
        "trace_to_micro.runner.local_consequence.load_task_splits",
        lambda _: {"train": ["good"], "test": ["bad"]},
    )
    monkeypatch.setattr(
        "trace_to_micro.runner.local_consequence.audit_results_completeness",
        lambda *args, **kwargs: None,
    )

    def target(domain, task):
        if task.id == "bad":
            raise InvalidReferenceTargetError(
                domain=domain,
                task_id=task.id,
                action=action,
                error="Non-delivered order cannot be exchanged",
            )
        return {"assistant": {"valid": True}, "user": None}

    monkeypatch.setattr(
        "trace_to_micro.runner.local_consequence.target_snapshot", target
    )
    monkeypatch.setattr(
        "trace_to_micro.runner.local_consequence.replay_local_consequences",
        lambda **kwargs: [{"task_id": str(kwargs["task"].id)}],
    )

    rows, exclusions = _domain_consequences(config, "retail")

    assert rows == [{"task_id": "good"}]
    assert exclusions == [
        {
            "domain": "retail",
            "split": "test",
            "task_id": "bad",
            "reason": "invalid_reference_target",
            "reference_action_id": "bad-write",
            "reference_action_name": "exchange_delivered_order_items",
            "reference_action_requestor": "assistant",
            "reference_action_arguments": {"order_id": "pending-order"},
            "error": "Non-delivered order cannot be exchanged",
            "excluded_trajectory_count": 1,
        }
    ]


def test_support_report_surfaces_excluded_tasks() -> None:
    exclusion = {
        "domain": "retail",
        "split": "test",
        "task_id": "64",
        "reason": "invalid_reference_target",
        "reference_action_id": "64_6",
        "reference_action_name": "exchange_delivered_order_items",
        "reference_action_requestor": "assistant",
        "reference_action_arguments": {"order_id": "#W7464385"},
        "error": "Non-delivered order cannot be exchanged",
        "excluded_trajectory_count": 1,
    }

    report = _support_report(
        [],
        exclusions=[exclusion],
        domains=("retail",),
        splits=("train", "test"),
    )

    assert report["exclusion_summary"] == {
        "task_count": 1,
        "trajectory_count": 1,
    }
    assert report["excluded_tasks"] == [exclusion]
    assert report["domains"]["retail"]["test"]["excluded_task_ids"] == ["64"]


def test_domain_consequences_does_not_hide_unrelated_errors(monkeypatch) -> None:
    task = SimpleNamespace(id="task")
    results = SimpleNamespace(
        tasks=[task],
        simulations=[SimpleNamespace(id="sim", task_id="task")],
    )
    config = SimpleNamespace(
        train_split="train",
        test_split="test",
        expected_num_trials=1,
        expected_agent_model="agent-model",
        expected_user_model="user-model",
        results_path=lambda domain: Path(f"{domain}/results.json"),
    )
    monkeypatch.setattr(
        "trace_to_micro.runner.local_consequence.Results.load",
        lambda _: results,
    )
    monkeypatch.setattr(
        "trace_to_micro.runner.local_consequence.load_task_splits",
        lambda _: {"train": ["task"], "test": []},
    )
    monkeypatch.setattr(
        "trace_to_micro.runner.local_consequence.audit_results_completeness",
        lambda *args, **kwargs: None,
    )

    def fail_target(*args):
        raise ValueError("unexpected code error")

    monkeypatch.setattr(
        "trace_to_micro.runner.local_consequence.target_snapshot",
        fail_target,
    )

    with pytest.raises(ValueError, match="unexpected code error"):
        _domain_consequences(config, "retail")
