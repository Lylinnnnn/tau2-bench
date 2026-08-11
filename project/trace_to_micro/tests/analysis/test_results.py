from types import SimpleNamespace

from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from trace_to_micro.analysis.results import build_results_audit


def test_cross_role_call_is_behavior_error_not_structural_corruption(
    monkeypatch, tmp_path
) -> None:
    simulation = SimulationRun(
        id="sim-1",
        task_id="2",
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.TOO_MANY_ERRORS,
        messages=[
            UserMessage(role="user", content="help"),
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="check_network_status",
                        arguments={},
                        requestor="assistant",
                    )
                ],
            ),
            ToolMessage(
                id="call-1",
                role="tool",
                requestor="assistant",
                content="Error: Tool not found",
                error=True,
            ),
        ],
        trial=0,
        seed=300,
    )
    metadata = SimpleNamespace(
        tasks=[SimpleNamespace(id="2", user_tools=["check_network_status"])]
    )
    environment = SimpleNamespace(
        get_tools=lambda: [SimpleNamespace(name="lookup_customer")],
        get_user_tools=lambda include=None: [
            SimpleNamespace(name="check_network_status")
        ],
    )
    monkeypatch.setattr(
        "trace_to_micro.analysis.results.audit_results_completeness",
        lambda *args, **kwargs: {"complete": True},
    )
    monkeypatch.setattr(
        "trace_to_micro.analysis.results.Results.load_metadata", lambda _: metadata
    )
    monkeypatch.setattr(
        "trace_to_micro.analysis.results.Results.iter_simulations",
        lambda _: iter([simulation]),
    )
    monkeypatch.setattr(
        "trace_to_micro.analysis.results.build_environment", lambda _: environment
    )

    report = build_results_audit(
        tmp_path / "results.json",
        domain="telecom",
        expected_task_ids={"2"},
        expected_agent_model="agent",
        expected_user_model="user",
    )

    assert report["data_integrity"]["valid_for_analysis"] is True
    assert report["tool_behavior"]["invalid_call_counts"] == {
        "assistant:cross_role_tool": 1
    }
    assert report["tool_behavior"]["tool_error_count"] == 1
