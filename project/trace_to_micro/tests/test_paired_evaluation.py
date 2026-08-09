from tau2.data_model.tasks import Action, EvaluationCriteria, Task, UserScenario
from trace_to_micro.evaluation import build_paired_report, score_branch


def _task() -> Task:
    return Task(
        id="task",
        user_scenario=UserScenario(instructions="fix data"),
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(
                    action_id="1",
                    requestor="user",
                    name="toggle_data",
                    arguments={},
                )
            ]
        ),
    )


def _branch(name: str, changed: bool) -> dict:
    return {
        "assistant_action": {"kind": "text", "content": "enable data"},
        "user_continuation": {
            "kind": "tool",
            "calls": [{"requestor": "user", "name": name, "arguments": {}}],
        },
        "tool_error": False,
        "mutation_noop": not changed,
        "changes": (
            [{"path": "user.device.data_enabled", "before": False, "after": True}]
            if changed
            else []
        ),
    }


def test_score_branch_checks_dual_control_effect_and_gold_action() -> None:
    actual = _branch("toggle_data", changed=True)
    predicted = _branch("toggle_data", changed=True)

    metrics = score_branch(task=_task(), actual=actual, predicted=predicted)

    assert metrics["macro_tool_arguments_match"] is True
    assert metrics["stateful_effect_match"] is True
    assert metrics["gold_macro_action_match"] is True


def test_report_computes_paired_delta() -> None:
    base = {
        "snapshot_id": "s1",
        "position_bucket": "late",
        "context_length_bucket": "long",
        "prediction_usage": {"prompt_tokens": 100},
        "metrics": {
            "assistant_kind_match": False,
            "assistant_exact_match": False,
            "macro_tool_name_match": False,
            "macro_tool_arguments_match": False,
            "effect_match": False,
            "stateful_effect_match": False,
            "gold_macro_action_match": False,
            "tool_error": True,
            "mutation_noop": False,
        },
    }
    rows = [
        {**base, "variant": "long_raw"},
        {
            **base,
            "variant": "structured_state",
            "metrics": {
                **base["metrics"],
                "effect_match": True,
                "stateful_effect_match": True,
            },
        },
    ]

    report = build_paired_report(rows)

    delta = report["paired_comparisons"]["structured_state_vs_long_raw"]
    assert delta["metrics"]["stateful_effect_match"]["improved"] == 1
