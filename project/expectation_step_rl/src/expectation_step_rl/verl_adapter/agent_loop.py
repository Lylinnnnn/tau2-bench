"""One-action verl loop rewarded by realized expectation deviation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
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

from expectation_step_rl.reward.composer import WeightedSumRewardComposer
from expectation_step_rl.reward.expectation import (
    ExpectationDeviationRewardProvider,
)
from expectation_step_rl.reward.interface import (
    CandidateAction,
    CandidateTransition,
    RewardDecision,
)
from expectation_step_rl.reward.pipeline import RewardPipeline
from expectation_step_rl.tau2_adapter.execution import execute_one_tool_call
from expectation_step_rl.verl_adapter.reward_fields import build_agent_extra_fields


class Tau2ExpectationStepAgentLoop(AgentLoopBase):
    """Generate one candidate tool action, execute it, and score its result."""

    def __init__(
        self,
        *args,
        scorer_base_urls: str,
        scorer_api_key: str,
        scorer_model: str,
        calibration_path: str,
        scorer_timeout_seconds: float = 180.0,
        min_k_fraction: float = 0.1,
        reward_clip: float = 5.0,
        invalid_action_penalty: float = -5.0,
        tool_error_penalty: float = -4.0,
        unsupported_tool_penalty: float = -3.0,
        expectation_reward_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.tool_parser = ToolParser.get_tool_parser("hermes", self.tokenizer)
        expectation_provider = ExpectationDeviationRewardProvider(
            scorer_base_urls=[value.strip() for value in scorer_base_urls.split(",")],
            scorer_api_key=scorer_api_key,
            scorer_model=scorer_model,
            calibration_path=Path(calibration_path),
            scorer_timeout_seconds=float(scorer_timeout_seconds),
            min_k_fraction=min_k_fraction,
            reward_clip=reward_clip,
            invalid_action_penalty=invalid_action_penalty,
            tool_error_penalty=tool_error_penalty,
            unsupported_tool_penalty=unsupported_tool_penalty,
        )
        self.reward_pipeline = RewardPipeline(
            [expectation_provider],
            WeightedSumRewardComposer(
                {expectation_provider.name: expectation_reward_weight},
                mode="expectation_only",
            ),
        )

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

        transition = CandidateTransition(
            domain=str(kwargs["domain"]),
            task_id=str(kwargs["task_id"]),
            raw_prompt=raw_prompt,
            tools=tools,
            replay_prefix_json=str(kwargs["replay_prefix_json"]),
        )
        if len(calls) != 1:
            transition = replace(
                transition,
                invalid_reason=("no_tool_call" if not calls else "multiple_tool_calls"),
            )
        else:
            call = calls[0]
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                transition = replace(
                    transition,
                    invalid_reason="invalid_tool_arguments_json",
                )
            else:
                if not isinstance(arguments, dict):
                    transition = replace(
                        transition,
                        invalid_reason="tool_arguments_are_not_an_object",
                    )
                else:
                    call_id = f"chatcmpl-tool-{uuid4().hex}"
                    execution = await asyncio.to_thread(
                        execute_one_tool_call,
                        domain=transition.domain,
                        task_id=transition.task_id,
                        replay_prefix_json=transition.replay_prefix_json,
                        tool_name=call.name,
                        arguments=arguments,
                        call_id=call_id,
                    )
                    transition = CandidateTransition(
                        domain=transition.domain,
                        task_id=transition.task_id,
                        raw_prompt=transition.raw_prompt,
                        tools=transition.tools,
                        replay_prefix_json=transition.replay_prefix_json,
                        action=CandidateAction(
                            call_id=call_id,
                            name=call.name,
                            arguments=arguments,
                            arguments_json=call.arguments,
                        ),
                        tool_result=execution.content,
                        tool_error=execution.error,
                    )

        decision = await self.reward_pipeline.evaluate(transition)

        return self._output(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_logprobs=response_logprobs,
            metrics=metrics,
            decision=decision,
        )

    @staticmethod
    def _output(
        *,
        prompt_ids: list[int],
        response_ids: list[int],
        response_logprobs: list[float] | None,
        metrics: dict[str, float | int],
        decision: RewardDecision,
    ) -> AgentLoopOutput:
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            reward_score=decision.reward,
            num_turns=2,
            metrics=AgentLoopMetrics.model_validate(metrics),
            extra_fields=build_agent_extra_fields(decision),
        )
