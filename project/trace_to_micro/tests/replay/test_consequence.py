from types import SimpleNamespace

import pytest

from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from trace_to_micro.replay.consequence import (
    InvalidReferenceTargetError,
    compile_abstract_consequence,
    replay_local_consequences,
    target_snapshot,
)


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
        return name in {"failed_write", "increment"}

    def get_response(self, call: ToolCall) -> ToolMessage:
        if call.name == "failed_write":
            return ToolMessage(
                id=call.id,
                role="tool",
                requestor="assistant",
                content="write rejected",
                error=True,
            )
        assert call.name == "increment"
        self.tools.db.count += 1
        return ToolMessage(
            id=call.id,
            role="tool",
            requestor="assistant",
            content="1",
            error=False,
        )


def test_compile_abstract_consequence_uses_only_observed_local_effects() -> None:
    row = {
        "decision_id": "sim-1:2",
        "simulation_id": "sim-1",
        "domain": "retail",
        "split": "train",
        "task_id": "task-1",
        "trial": 0,
        "tool_name": "cancel_pending_order",
        "tool_success": True,
        "state_changed": True,
        "goal_progress": False,
        "target_distance_before": 5,
        "target_distance_after": 6,
        "changes": [
            {
                "path": "assistant.orders.order-1.status",
                "before": "pending",
                "after": "cancelled",
                "before_present": True,
                "after_present": True,
            },
            {
                "path": "assistant.orders.order-1.payment_history.1.transaction_type",
                "before": None,
                "after": "refund",
                "before_present": False,
                "after_present": True,
            },
        ],
    }

    consequence = compile_abstract_consequence(
        row,
        result_content=(
            '{"order_id":"order-1","status":"cancelled","payment_history":[]}'
        ),
    )

    assert consequence["execution"] == "success"
    assert consequence["operations"] == ["create", "update"]
    assert consequence["entity_types"] == ["order"]
    assert consequence["field_families"] == ["payment", "status"]
    assert consequence["targets"]["status.cancelled"] is True
    assert consequence["targets"]["transaction.refund"] is True
    assert consequence["output"] == {
        "kind": "object",
        "entity_types": ["order"],
        "field_families": ["payment", "status"],
    }
    assert consequence["targets"]["output.kind.object"] is True
    assert consequence["targets"]["output.entity.order"] is True
    assert "goal_progress" not in consequence
    assert all("goal" not in target for target in consequence["targets"])


def test_target_snapshot_skips_stale_read_only_reference_action(
    monkeypatch,
) -> None:
    environment = _Environment()
    monkeypatch.setattr(
        "trace_to_micro.replay.consequence.build_environment",
        lambda domain: environment,
    )
    actions = [
        SimpleNamespace(
            action_id="stale-read",
            requestor="assistant",
            name="missing_product_read",
            arguments={"product_id": "not-in-db"},
        ),
        SimpleNamespace(
            action_id="gold-write",
            requestor="assistant",
            name="increment",
            arguments={},
        ),
    ]
    task = SimpleNamespace(
        id="task-with-stale-read",
        initial_state=None,
        evaluation_criteria=SimpleNamespace(actions=actions),
    )

    snapshot = target_snapshot("retail", task)

    assert snapshot["assistant"] == {"count": 1}


def test_target_snapshot_exposes_failed_mutating_reference_action(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "trace_to_micro.replay.consequence.build_environment",
        lambda domain: _Environment(),
    )
    action = SimpleNamespace(
        action_id="failed-gold-write",
        requestor="assistant",
        name="failed_write",
        arguments={},
    )
    task = SimpleNamespace(
        id="task-with-bad-write",
        initial_state=None,
        evaluation_criteria=SimpleNamespace(actions=[action]),
    )

    with pytest.raises(
        InvalidReferenceTargetError,
        match="Mutating reference action failed for retail/task-with-bad-write",
    ) as raised:
        target_snapshot("retail", task)

    assert raised.value.action_id == "failed-gold-write"
    assert raised.value.action_name == "failed_write"
    assert raised.value.error == "write rejected"


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
