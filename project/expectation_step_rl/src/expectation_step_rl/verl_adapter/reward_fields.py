"""Build collision-free reward metadata for verl AgentLoop outputs."""

from __future__ import annotations

from typing import Any

from expectation_step_rl.expectation.calibration import RewardResult

NUMERIC_FIELDS = (
    "contextual_min_k_deviation",
    "calibrated_z",
    "calibration_count",
    "suffix_token_count",
    "min_k_token_count",
)
TEXT_FIELDS = (
    "expectation_outcome",
    "tool_name",
    "tool_result",
    "calibration_key",
    "calibration_level",
    "scorer_base_url",
)


def build_agent_extra_fields(
    diagnostics: dict[str, Any], reward: RewardResult
) -> dict[str, Any]:
    """Return verl metadata without duplicating reward keys at the top level."""

    reward_extra = dict(diagnostics)
    for key in NUMERIC_FIELDS:
        if reward_extra[key] is None:
            reward_extra[key] = float("nan")
    for key in TEXT_FIELDS:
        if reward_extra[key] is None:
            reward_extra[key] = ""
    reward_extra["tool_error"] = float(bool(reward_extra["tool_error"]))
    return {
        "reward_extra_info": reward_extra,
        "turn_scores": [reward.reward],
        "tool_rewards": [reward.reward],
    }
