from pathlib import Path
from types import SimpleNamespace

from trace_to_micro.replay.consequence import InvalidReferenceTargetError
from trace_to_micro.runner.local_consequence import (
    _domain_consequences,
    _request_validation_errors,
    _support_report,
)


def _complete_audit() -> dict:
    return {
        "empty_simulation_ids": [],
        "missing_reward_simulation_ids": [],
        "infrastructure_error_simulation_ids": [],
        "duplicate_task_trials": [],
        "unexpected_task_trials": [],
        "missing_task_trials": [],
        "agent_model_mismatch": False,
        "user_model_mismatch": False,
        "complete": True,
    }


def test_domain_consequences_excludes_invalid_target_and_continues(
    monkeypatch,
) -> None:
    tasks = [SimpleNamespace(id="good"), SimpleNamespace(id="bad")]
    simulations = [
        SimpleNamespace(id="sim-good", task_id="good", trial=0),
        SimpleNamespace(id="sim-bad", task_id="bad", trial=0),
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
        lambda *args, **kwargs: _complete_audit(),
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

    rows, exclusions, completeness = _domain_consequences(config, "retail")

    assert rows == [{"task_id": "good"}]
    assert completeness["complete"] is True
    assert len(exclusions) == 1
    exclusion = exclusions[0]
    assert exclusion["scope"] == "task"
    assert exclusion["domain"] == "retail"
    assert exclusion["split"] == "test"
    assert exclusion["task_id"] == "bad"
    assert exclusion["stage"] == "target_snapshot"
    assert exclusion["reason"] == "target_snapshot_failed"
    assert exclusion["exception_type"] == "InvalidReferenceTargetError"
    assert exclusion["reference_action_id"] == "bad-write"
    assert exclusion["reference_action_name"] == "exchange_delivered_order_items"
    assert exclusion["excluded_trajectory_ids"] == ["sim-bad"]


def test_support_report_surfaces_excluded_tasks() -> None:
    exclusion = {
        "scope": "task",
        "domain": "retail",
        "split": "test",
        "task_id": "64",
        "stage": "target_snapshot",
        "reason": "target_snapshot_failed",
        "reference_action_id": "64_6",
        "reference_action_name": "exchange_delivered_order_items",
        "reference_action_requestor": "assistant",
        "reference_action_arguments": {"order_id": "#W7464385"},
        "error": "Non-delivered order cannot be exchanged",
        "excluded_trajectory_count": 1,
        "excluded_trajectory_ids": ["sim-64"],
    }

    report = _support_report(
        [],
        exclusions=[exclusion],
        completeness={"retail": _complete_audit()},
        domains=("retail",),
        splits=("train", "test"),
    )

    assert report["exclusion_summary"] == {
        "issue_count": 1,
        "task_count": 1,
        "trajectory_count": 1,
        "by_stage": {"target_snapshot": 1},
        "by_reason": {"target_snapshot_failed": 1},
    }
    assert report["exclusions"] == [exclusion]
    assert report["audit_completed"] is True
    assert report["domains"]["retail"]["test"]["excluded_task_ids"] == ["64"]


def test_request_validation_identifies_only_incomplete_trajectory() -> None:
    consequences = [
        {"decision_id": "good:2", "simulation_id": "good"},
        {"decision_id": "bad:2", "simulation_id": "bad"},
    ]
    requests = [
        {
            "decision_id": decision_id,
            "simulation_id": simulation_id,
            "sample_id": f"{decision_id}:{moment}",
            "moment": moment,
        }
        for decision_id, simulation_id, moments in (
            ("good:2", "good", ("before", "action", "result")),
            ("bad:2", "bad", ("before", "action")),
        )
        for moment in moments
    ]

    errors = _request_validation_errors(consequences, requests)

    assert set(errors) == {"bad"}
    assert errors["bad"][0]["observed_moments"] == ["before", "action"]


def test_domain_consequences_records_unexpected_task_error_and_continues(
    monkeypatch,
) -> None:
    task = SimpleNamespace(id="task")
    results = SimpleNamespace(
        tasks=[task],
        simulations=[SimpleNamespace(id="sim", task_id="task", trial=0)],
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
        lambda *args, **kwargs: _complete_audit(),
    )

    def fail_target(*args):
        raise ValueError("unexpected code error")

    monkeypatch.setattr(
        "trace_to_micro.runner.local_consequence.target_snapshot",
        fail_target,
    )

    rows, exclusions, _ = _domain_consequences(config, "retail")

    assert rows == []
    assert exclusions[0]["reason"] == "target_snapshot_failed"
    assert exclusions[0]["exception_type"] == "ValueError"
    assert exclusions[0]["error"] == "unexpected code error"
    assert "fail_target" in exclusions[0]["traceback"]


def test_domain_consequences_records_replay_error_and_keeps_other_trajectory(
    monkeypatch,
) -> None:
    task = SimpleNamespace(id="task")
    simulations = [
        SimpleNamespace(id="bad-sim", task_id="task", trial=0),
        SimpleNamespace(id="good-sim", task_id="task", trial=1),
    ]
    results = SimpleNamespace(tasks=[task], simulations=simulations)
    config = SimpleNamespace(
        train_split="train",
        test_split="test",
        expected_num_trials=2,
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
        lambda *args, **kwargs: _complete_audit(),
    )
    monkeypatch.setattr(
        "trace_to_micro.runner.local_consequence.target_snapshot",
        lambda *args: {"assistant": {}, "user": None},
    )

    def replay(**kwargs):
        if kwargs["simulation"].id == "bad-sim":
            raise ValueError("missing tool result")
        return [{"simulation_id": "good-sim"}]

    monkeypatch.setattr(
        "trace_to_micro.runner.local_consequence.replay_local_consequences",
        replay,
    )

    rows, exclusions, _ = _domain_consequences(config, "retail")

    assert rows == [{"simulation_id": "good-sim"}]
    assert exclusions[0]["simulation_id"] == "bad-sim"
    assert exclusions[0]["stage"] == "trajectory_replay"
    assert exclusions[0]["error"] == "missing tool result"
