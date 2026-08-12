import pytest

from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.data_model.tasks import RewardType
from trace_to_micro.analysis.logged_trace import (
    _component_success,
    _language_success,
    audit_results_completeness,
    decision_indices,
    stratified_indices,
    tool_decision_records,
)


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


def test_component_success_uses_official_reward_breakdown() -> None:
    reward = RewardInfo(
        reward=0.0,
        reward_basis=[RewardType.DB, RewardType.COMMUNICATE],
        reward_breakdown={RewardType.DB: 1.0, RewardType.COMMUNICATE: 0.0},
    )

    assert _component_success(reward, RewardType.DB) is True
    assert _component_success(reward, RewardType.COMMUNICATE) is False
    assert _component_success(reward, RewardType.ACTION) is None


def test_language_success_uses_official_domain_language_component() -> None:
    airline_reward = RewardInfo(
        reward=1.0,
        reward_basis=[RewardType.DB, RewardType.COMMUNICATE],
        reward_breakdown={RewardType.DB: 1.0, RewardType.COMMUNICATE: 1.0},
    )
    retail_reward = RewardInfo(
        reward=0.0,
        reward_basis=[RewardType.DB, RewardType.NL_ASSERTION],
        reward_breakdown={RewardType.DB: 1.0, RewardType.NL_ASSERTION: 0.0},
    )
    state_only_reward = RewardInfo(
        reward=1.0,
        reward_basis=[RewardType.DB],
        reward_breakdown={RewardType.DB: 1.0},
    )

    assert _language_success(airline_reward) is True
    assert _language_success(retail_reward) is False
    assert _language_success(state_only_reward) is None


def test_completeness_rejects_results_missing_configured_tasks(
    monkeypatch, tmp_path
) -> None:
    metadata = type(
        "Metadata",
        (),
        {
            "tasks": [type("Task", (), {"id": "task-1"})()],
            "info": type(
                "Info",
                (),
                {
                    "num_trials": 1,
                    "agent_info": type(
                        "AgentInfo", (), {"llm": "agent", "llm_args": {}}
                    )(),
                    "user_info": type(
                        "UserInfo", (), {"llm": "user", "llm_args": {}}
                    )(),
                },
            )(),
        },
    )()
    simulation = _simulation()
    simulation.reward_info = RewardInfo(reward=1.0)
    monkeypatch.setattr(
        "trace_to_micro.analysis.logged_trace.Results.load_metadata", lambda _: metadata
    )
    monkeypatch.setattr(
        "trace_to_micro.analysis.logged_trace.Results.iter_simulations",
        lambda _: iter([simulation]),
    )

    with pytest.raises(ValueError, match="Incomplete trajectory results"):
        audit_results_completeness(
            tmp_path / "results.json",
            expected_task_ids={"task-1", "task-2"},
            expected_num_trials=1,
        )


def test_completeness_can_report_without_raising(monkeypatch, tmp_path) -> None:
    metadata = type(
        "Metadata",
        (),
        {
            "tasks": [type("Task", (), {"id": "task-1"})()],
            "info": type(
                "Info",
                (),
                {
                    "num_trials": 1,
                    "agent_info": type(
                        "AgentInfo", (), {"llm": "agent", "llm_args": {}}
                    )(),
                    "user_info": type(
                        "UserInfo", (), {"llm": "user", "llm_args": {}}
                    )(),
                },
            )(),
        },
    )()
    simulation = _simulation()
    simulation.reward_info = RewardInfo(reward=1.0)
    monkeypatch.setattr(
        "trace_to_micro.analysis.logged_trace.Results.load_metadata", lambda _: metadata
    )
    monkeypatch.setattr(
        "trace_to_micro.analysis.logged_trace.Results.iter_simulations",
        lambda _: iter([simulation]),
    )

    report = audit_results_completeness(
        tmp_path / "results.json",
        expected_task_ids={"task-1", "task-2"},
        expected_num_trials=1,
        raise_on_incomplete=False,
    )

    assert report["complete"] is False
    assert report["missing_task_trials"] == [("task-2", 0)]


def test_tool_decision_records_quarantines_bad_trajectory_atomically(
    monkeypatch, tmp_path
) -> None:
    good = _simulation().model_copy(
        update={"id": "good", "reward_info": RewardInfo(reward=1.0)}
    )
    bad_messages = [
        message
        for message in _simulation().messages
        if not isinstance(message, ToolMessage)
    ]
    bad = _simulation().model_copy(
        update={
            "id": "bad",
            "messages": bad_messages,
            "reward_info": RewardInfo(reward=0.0),
        }
    )
    environment = type(
        "Environment",
        (),
        {
            "get_policy": lambda self: "policy",
            "get_tools": lambda self: [],
        },
    )()
    monkeypatch.setattr(
        "trace_to_micro.analysis.logged_trace.load_task_splits",
        lambda _: {"train": ["task-1"], "test": []},
    )
    monkeypatch.setattr(
        "trace_to_micro.analysis.logged_trace.build_environment",
        lambda _: environment,
    )
    monkeypatch.setattr(
        "trace_to_micro.analysis.logged_trace.Results.iter_simulations",
        lambda _: iter([bad, good]),
    )
    errors = []

    rows = tool_decision_records(
        tmp_path / "results.json",
        domain="retail",
        task_set="retail",
        train_split="train",
        test_split="test",
        max_per_trajectory=None,
        on_simulation_error=lambda simulation, error: errors.append(
            (simulation.id, str(error))
        ),
    )

    assert {row["simulation_id"] for row in rows} == {"good"}
    assert len(rows) == 3
    assert errors[0][0] == "bad"
    assert "Tool-result mismatch" in errors[0][1]


def test_completeness_rejects_results_from_another_model(monkeypatch, tmp_path) -> None:
    metadata = type(
        "Metadata",
        (),
        {
            "tasks": [type("Task", (), {"id": "task-1"})()],
            "info": type(
                "Info",
                (),
                {
                    "num_trials": 1,
                    "agent_info": type(
                        "AgentInfo", (), {"llm": "old-agent", "llm_args": {}}
                    )(),
                    "user_info": type(
                        "UserInfo", (), {"llm": "new-user", "llm_args": {}}
                    )(),
                },
            )(),
        },
    )()
    monkeypatch.setattr(
        "trace_to_micro.analysis.logged_trace.Results.load_metadata", lambda _: metadata
    )
    monkeypatch.setattr(
        "trace_to_micro.analysis.logged_trace.Results.iter_simulations",
        lambda _: iter(
            [_simulation().model_copy(update={"reward_info": RewardInfo(reward=1.0)})]
        ),
    )

    with pytest.raises(ValueError, match="agent_model_mismatch.*True"):
        audit_results_completeness(
            tmp_path / "results.json",
            expected_task_ids={"task-1"},
            expected_num_trials=1,
            expected_agent_model="new-agent",
            expected_user_model="new-user",
        )


def test_completeness_rejects_missing_official_reward(monkeypatch, tmp_path) -> None:
    metadata = type(
        "Metadata",
        (),
        {
            "tasks": [type("Task", (), {"id": "task-1"})()],
            "info": type(
                "Info",
                (),
                {
                    "num_trials": 1,
                    "agent_info": type(
                        "AgentInfo", (), {"llm": "agent", "llm_args": {}}
                    )(),
                    "user_info": type(
                        "UserInfo", (), {"llm": "user", "llm_args": {}}
                    )(),
                },
            )(),
        },
    )()
    monkeypatch.setattr(
        "trace_to_micro.analysis.logged_trace.Results.load_metadata", lambda _: metadata
    )
    monkeypatch.setattr(
        "trace_to_micro.analysis.logged_trace.Results.iter_simulations",
        lambda _: iter([_simulation()]),
    )

    with pytest.raises(ValueError, match="missing_reward_simulation_ids"):
        audit_results_completeness(
            tmp_path / "results.json",
            expected_task_ids={"task-1"},
            expected_num_trials=1,
        )
