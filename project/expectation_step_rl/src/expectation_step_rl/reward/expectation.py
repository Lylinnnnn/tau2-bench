"""Expectation-deviation reward provider used by the current experiment."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from expectation_step_rl.expectation.calibration import TrainCalibration
from expectation_step_rl.expectation.scoring import VLLMExpectationScorer
from expectation_step_rl.expectation.structure import result_structure_key
from expectation_step_rl.reward.interface import CandidateTransition, RewardSignal


class ExpectationDeviationRewardProvider:
    """Score whether a realized tool result violates the model's expectation."""

    name = "expectation"

    def __init__(
        self,
        *,
        scorer_base_urls: list[str],
        scorer_api_key: str,
        scorer_model: str,
        calibration_path: Path,
        scorer_timeout_seconds: float,
        min_k_fraction: float,
        reward_clip: float,
        invalid_action_penalty: float,
        tool_error_penalty: float,
        unsupported_tool_penalty: float,
    ) -> None:
        self.scorer = VLLMExpectationScorer(
            base_urls=scorer_base_urls,
            api_key=scorer_api_key,
            model=scorer_model,
            min_k_fraction=min_k_fraction,
            timeout_seconds=scorer_timeout_seconds,
        )
        self.calibration = TrainCalibration.load(
            calibration_path,
            clip=reward_clip,
            invalid_action_penalty=invalid_action_penalty,
            tool_error_penalty=tool_error_penalty,
            unsupported_tool_penalty=unsupported_tool_penalty,
        )

    @staticmethod
    def _action_message(transition: CandidateTransition) -> dict[str, Any]:
        action = transition.action
        if action is None:
            raise ValueError("A valid transition requires an action")
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": action.call_id,
                    "name": action.name,
                    "type": "function",
                    "function": {
                        "name": action.name,
                        "arguments": action.arguments_json,
                    },
                }
            ],
        }

    @staticmethod
    def _invalid_diagnostics(outcome: str) -> dict[str, Any]:
        return {
            "expectation_outcome": outcome,
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

    async def evaluate(self, transition: CandidateTransition) -> RewardSignal:
        """Measure and Train-calibrate one realized tool result."""

        action = transition.action
        if action is None:
            result = self.calibration.score(
                domain=transition.domain,
                tool_name=None,
                raw_score=None,
                action_valid=False,
            )
            return RewardSignal(
                self.name,
                result.reward,
                self._invalid_diagnostics(
                    transition.invalid_reason or "invalid_action"
                ),
            )

        structure_key = result_structure_key(transition.tool_result or "")
        measurement = None
        if not transition.tool_error and self.calibration.supports(
            transition.domain, action.name
        ):
            result_message = {
                "role": "tool",
                "tool_call_id": action.call_id,
                "content": transition.tool_result,
            }
            measurement = await asyncio.to_thread(
                self.scorer.measure,
                raw_prompt=transition.raw_prompt,
                action_message=self._action_message(transition),
                result_message=result_message,
                tools=transition.tools,
                routing_key=action.call_id,
            )
        result = self.calibration.score(
            domain=transition.domain,
            tool_name=action.name,
            raw_score=(
                measurement.contextual_min_k_deviation
                if measurement is not None
                else None
            ),
            structure_key=structure_key,
            tool_error=bool(transition.tool_error),
        )
        return RewardSignal(
            self.name,
            result.reward,
            {
                "expectation_outcome": result.outcome,
                "tool_name": action.name,
                "tool_error": transition.tool_error,
                "tool_result": transition.tool_result,
                "contextual_min_k_deviation": result.raw_score,
                "calibrated_z": result.z_score,
                "calibration_key": result.calibration_key,
                "calibration_count": result.calibration_count,
                "calibration_level": result.calibration_level,
                "suffix_token_count": (
                    measurement.suffix_token_count if measurement is not None else None
                ),
                "min_k_token_count": (
                    measurement.min_k_token_count if measurement is not None else None
                ),
                "scorer_base_url": (
                    measurement.scorer_base_url if measurement is not None else None
                ),
            },
        )
