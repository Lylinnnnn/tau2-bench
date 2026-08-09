from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.simulation import (
    RewardInfo,
    SimulationRun,
    TerminationReason,
)
from trace_to_micro.trajectory_analysis import (
    build_cross_split_trajectory_report,
    build_trajectory_report,
)


def _simulation(task_id: str, reward: float, decisions: int) -> SimulationRun:
    messages = [AssistantMessage(role="assistant", content="Hello")]
    for index in range(decisions):
        messages.extend(
            [
                UserMessage(role="user", content=f"request {index}"),
                AssistantMessage(role="assistant", content=f"response {index}"),
            ]
        )
    return SimulationRun(
        id=f"sim-{task_id}",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=messages,
        reward_info=RewardInfo(reward=reward),
        trial=0,
        seed=300,
    )


def test_trajectory_report_separates_outcome_horizon_and_transition_quality() -> None:
    snapshots = [
        {
            "position_bucket": "late",
            "context_length_bucket": "long",
            "prefix_message_count": 12,
            "prefix_char_count": 500,
        }
    ]
    transitions = [
        {
            "split": "train",
            "simulation_id": "sim-task-long",
            "step_index": 0,
            "actor": "assistant",
            "action_name": "lookup",
            "declared_mutating": False,
            "changes": [],
            "replay_error": True,
        },
        {
            "split": "train",
            "simulation_id": "sim-task-long",
            "step_index": 1,
            "actor": "assistant",
            "action_name": "fix",
            "declared_mutating": True,
            "changes": [{"path": "status", "before": "bad", "after": "ok"}],
            "replay_error": False,
        },
    ]

    report = build_trajectory_report(
        split="train",
        simulations=[
            _simulation("task-short", reward=1.0, decisions=2),
            _simulation("task-long", reward=0.0, decisions=7),
        ],
        snapshots=snapshots,
        transition_rows=transitions,
    )

    assert report["outcomes"]["pass_1"] == 0.5
    assert (
        report["trajectory_length"]["pass_1_by_horizon_bucket"]["long_7_plus"]["pass_1"]
        == 0.0
    )
    assert report["observed_transitions"]["recovery"]["observed_recovery_rate"] == 1.0


def test_cross_split_trajectory_report_keeps_observational_gap_separate() -> None:
    train = {
        "outcomes": {"pass_1": 0.5, "mean_reward": 0.5},
        "trajectory_length": {"mean_agent_decision_count": 4.0},
    }
    test = {
        "outcomes": {"pass_1": 0.25, "mean_reward": 0.25},
        "trajectory_length": {"mean_agent_decision_count": 7.0},
    }

    report = build_cross_split_trajectory_report(
        {"train": train, "test": test}, train_split="train"
    )

    assert report["descriptive_gaps"]["test_minus_train"]["pass_1"] == -0.25
    assert (
        report["descriptive_gaps"]["test_minus_train"]["mean_agent_decision_count"]
        == 3.0
    )
