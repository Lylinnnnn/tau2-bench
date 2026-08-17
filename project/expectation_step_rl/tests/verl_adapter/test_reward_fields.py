import math

from expectation_step_rl.reward.interface import RewardDecision
from expectation_step_rl.verl_adapter.reward_fields import build_agent_extra_fields


def test_reward_fields_never_duplicate_unsanitized_diagnostics() -> None:
    diagnostics = {
        "expectation_outcome": "invalid_action",
        "tool_name": None,
        "tool_error": None,
        "tool_result": None,
        "contextual_min_k_deviation": None,
        "calibrated_z": None,
        "calibration_key": None,
        "calibration_count": None,
        "calibration_level": None,
        "suffix_token_count": None,
        "min_k_token_count": None,
        "scorer_base_url": None,
    }
    decision = RewardDecision(
        reward=-5.0,
        signals={"expectation": -5.0},
        diagnostics={
            **diagnostics,
            "reward_signal_expectation": -5.0,
            "reward_mode": "expectation_only",
            "final_reward": -5.0,
        },
    )

    fields = build_agent_extra_fields(decision)

    assert set(fields) == {"reward_extra_info", "turn_scores", "tool_rewards"}
    assert not (set(diagnostics) & set(fields))
    reward_extra = fields["reward_extra_info"]
    assert all(value is not None for value in reward_extra.values())
    assert math.isnan(reward_extra["calibrated_z"])
    assert reward_extra["tool_name"] == ""
    assert reward_extra["tool_error"] == 0.0
    assert reward_extra["reward_signal_expectation"] == -5.0
    assert reward_extra["reward_mode"] == "expectation_only"
    assert fields["turn_scores"] == [-5.0]
