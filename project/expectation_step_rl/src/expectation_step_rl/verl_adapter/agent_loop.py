"""One-action verl loop rewarded by realized expectation deviation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

from expectation_step_rl.expectation.calibration import RewardResult, TrainCalibration
from expectation_step_rl.expectation.scoring import VLLMExpectationScorer
from expectation_step_rl.tau2_adapter.execution import execute_one_tool_call


class Tau2ExpectationStepAgentLoop(AgentLoopBase):
    """Generate one candidate tool action, execute it, and score its result."""

    def __init__(
        self,
        *args,
        scorer_base_url: str,
        scorer_api_key: str,
        scorer_model: str,
        calibration_path: str,
        min_k_fraction: float = 0.1,
        reward_clip: float = 5.0,
        invalid_action_penalty: float = -5.0,
        tool_error_penalty: float = -4.0,
        unsupported_tool_penalty: float = -3.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.tool_parser = ToolParser.get_tool_parser("hermes", self.tokenizer)
        self.scorer = VLLMExpectationScorer(
            base_url=scorer_base_url,
            api_key=scorer_api_key,
            model=scorer_model,
            min_k_fraction=min_k_fraction,
        )
        self.calibration = TrainCalibration.load(
            Path(calibration_path),
            clip=reward_clip,
            invalid_action_penalty=invalid_action_penalty,
            tool_error_penalty=tool_error_penalty,
            unsupported_tool_penalty=unsupported_tool_penalty,
        )

    @staticmethod
    def _action_message(call_id: str, name: str, arguments: str) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "name": name,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }

    @staticmethod
    def _invalid_extra(outcome: str) -> dict[str, Any]:
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
        }

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        tools = json.loads(str(kwargs["tool_schemas_json"]))
        prompt_ids = await self.apply_chat_template(raw_prompt, tools=tools)
        metrics: dict[str, float | int] = {}
        with simple_timer("generate_sequences", metrics):
            generated: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
        metrics["num_preempted"] = (
            generated.num_preempted if generated.num_preempted is not None else -1
        )
        response_ids = generated.token_ids[: self.response_length]
        response_logprobs = (
            generated.log_probs[: self.response_length] if generated.log_probs else None
        )
        _, calls = await self.tool_parser.extract_tool_calls(response_ids)

        if len(calls) != 1:
            reward = self.calibration.score(
                domain=str(kwargs["domain"]),
                tool_name=None,
                raw_score=None,
                action_valid=False,
            )
            extra = self._invalid_extra(
                "no_tool_call" if not calls else "multiple_tool_calls"
            )
        else:
            call = calls[0]
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                reward = self.calibration.score(
                    domain=str(kwargs["domain"]),
                    tool_name=None,
                    raw_score=None,
                    action_valid=False,
                )
                extra = self._invalid_extra("invalid_tool_arguments_json")
            else:
                if not isinstance(arguments, dict):
                    reward = self.calibration.score(
                        domain=str(kwargs["domain"]),
                        tool_name=None,
                        raw_score=None,
                        action_valid=False,
                    )
                    extra = self._invalid_extra("tool_arguments_are_not_an_object")
                    return self._output(
                        prompt_ids=prompt_ids,
                        response_ids=response_ids,
                        response_logprobs=response_logprobs,
                        metrics=metrics,
                        reward=reward,
                        extra=extra,
                    )
                call_id = f"chatcmpl-tool-{uuid4().hex}"
                execution = await asyncio.to_thread(
                    execute_one_tool_call,
                    domain=str(kwargs["domain"]),
                    task_id=str(kwargs["task_id"]),
                    replay_prefix_json=str(kwargs["replay_prefix_json"]),
                    tool_name=call.name,
                    arguments=arguments,
                    call_id=call_id,
                )
                measurement = None
                if not execution.error and self.calibration.supports(
                    str(kwargs["domain"]), call.name
                ):
                    action_message = self._action_message(
                        call_id, call.name, call.arguments
                    )
                    result_message = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": execution.content,
                    }
                    measurement = await asyncio.to_thread(
                        self.scorer.measure,
                        raw_prompt=raw_prompt,
                        action_message=action_message,
                        result_message=result_message,
                        tools=tools,
                    )
                reward = self.calibration.score(
                    domain=str(kwargs["domain"]),
                    tool_name=call.name,
                    raw_score=(
                        measurement.contextual_min_k_deviation
                        if measurement is not None
                        else None
                    ),
                    tool_error=execution.error,
                )
                extra = {
                    "expectation_outcome": reward.outcome,
                    "tool_name": call.name,
                    "tool_error": execution.error,
                    "tool_result": execution.content,
                    "contextual_min_k_deviation": reward.raw_score,
                    "calibrated_z": reward.z_score,
                    "calibration_key": reward.calibration_key,
                    "calibration_count": reward.calibration_count,
                    "calibration_level": reward.calibration_level,
                    "suffix_token_count": (
                        measurement.suffix_token_count
                        if measurement is not None
                        else None
                    ),
                    "min_k_token_count": (
                        measurement.min_k_token_count
                        if measurement is not None
                        else None
                    ),
                }

        return self._output(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_logprobs=response_logprobs,
            metrics=metrics,
            reward=reward,
            extra=extra,
        )

    @staticmethod
    def _output(
        *,
        prompt_ids: list[int],
        response_ids: list[int],
        response_logprobs: list[float] | None,
        metrics: dict[str, float | int],
        reward: RewardResult,
        extra: dict[str, Any],
    ) -> AgentLoopOutput:
        reward_extra = dict(extra)
        for key in (
            "contextual_min_k_deviation",
            "calibrated_z",
            "calibration_count",
            "suffix_token_count",
            "min_k_token_count",
        ):
            if reward_extra[key] is None:
                reward_extra[key] = float("nan")
        for key in (
            "expectation_outcome",
            "tool_name",
            "tool_result",
            "calibration_key",
            "calibration_level",
        ):
            if reward_extra[key] is None:
                reward_extra[key] = ""
        reward_extra["tool_error"] = float(bool(reward_extra["tool_error"]))
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            reward_score=reward.reward,
            num_turns=2,
            metrics=AgentLoopMetrics.model_validate(metrics),
            extra_fields={
                **extra,
                "reward_extra_info": reward_extra,
                "turn_scores": [reward.reward],
                "tool_rewards": [reward.reward],
            },
        )
