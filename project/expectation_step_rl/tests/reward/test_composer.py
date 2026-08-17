import pytest

from expectation_step_rl.reward.composer import (
    GateRewardComposer,
    WeightedSumRewardComposer,
)
from expectation_step_rl.reward.interface import RewardSignal


def test_weighted_sum_preserves_raw_signals_and_diagnostics() -> None:
    composer = WeightedSumRewardComposer(
        {"expectation": 0.75, "official": 0.25}, mode="hybrid_weighted"
    )

    decision = composer.compose(
        {
            "expectation": RewardSignal(
                "expectation", 2.0, {"expectation_outcome": "scored"}
            ),
            "official": RewardSignal("official", 1.0, {"official_success": 1.0}),
        }
    )

    assert decision.reward == pytest.approx(1.75)
    assert decision.signals == {"expectation": 2.0, "official": 1.0}
    assert decision.diagnostics["reward_signal_expectation"] == 2.0
    assert decision.diagnostics["reward_signal_official"] == 1.0
    assert decision.diagnostics["reward_mode"] == "hybrid_weighted"


def test_gate_fusion_rejects_candidate_without_averaging() -> None:
    composer = GateRewardComposer(
        primary_signal="expectation",
        gate_signal="official",
        gate_threshold=0.0,
        rejected_reward=-5.0,
        mode="official_gate",
    )

    accepted = composer.compose(
        {
            "expectation": RewardSignal("expectation", 1.25),
            "official": RewardSignal("official", 1.0),
        }
    )
    rejected = composer.compose(
        {
            "expectation": RewardSignal("expectation", 4.0),
            "official": RewardSignal("official", 0.0),
        }
    )

    assert accepted.reward == 1.25
    assert accepted.diagnostics["reward_gate_passed"] == 1.0
    assert rejected.reward == -5.0
    assert rejected.diagnostics["reward_gate_passed"] == 0.0
