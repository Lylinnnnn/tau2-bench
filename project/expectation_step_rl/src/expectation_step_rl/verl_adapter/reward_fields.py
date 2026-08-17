"""Build collision-free reward metadata for verl AgentLoop outputs."""

from __future__ import annotations

from typing import Any

from expectation_step_rl.reward.interface import RewardDecision

NUMERIC_FIELDS = (
    "contextual_min_k_deviation",
    "calibrated_z",
    "calibration_count",
    "suffix_token_count",
    "min_k_token_count",
    "reward_signal_expectation",
    "final_reward",
)
TEXT_FIELDS = (
    "expectation_outcome",
    "tool_name",
    "tool_result",
    "calibration_key",
    "calibration_level",
    "scorer_base_url",
    "reward_mode",
)


def build_agent_extra_fields(decision: RewardDecision) -> dict[str, Any]:
    """Return verl metadata without duplicating reward keys at the top level."""

    reward_extra = dict(decision.diagnostics)
    for key in NUMERIC_FIELDS:
        if reward_extra.get(key) is None:
            reward_extra[key] = float("nan")
    for key in TEXT_FIELDS:
        if reward_extra.get(key) is None:
            reward_extra[key] = ""
    reward_extra["tool_error"] = float(bool(reward_extra.get("tool_error")))
    return {
        "reward_extra_info": reward_extra,
        "turn_scores": [decision.reward],
        "tool_rewards": [decision.reward],
    }
