import asyncio

from expectation_step_rl.reward.composer import WeightedSumRewardComposer
from expectation_step_rl.reward.interface import (
    CandidateTransition,
    RewardSignal,
)
from expectation_step_rl.reward.pipeline import RewardPipeline


class FixedProvider:
    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self.value = value

    async def evaluate(self, transition: CandidateTransition) -> RewardSignal:
        return RewardSignal(
            self.name, self.value, {f"{self.name}_task": transition.task_id}
        )


def test_pipeline_keeps_signal_generation_separate_from_fusion() -> None:
    pipeline = RewardPipeline(
        [FixedProvider("expectation", 2.0), FixedProvider("official", 1.0)],
        WeightedSumRewardComposer({"expectation": 0.5, "official": 2.0}),
    )
    transition = CandidateTransition(
        domain="retail",
        task_id="2",
        raw_prompt=[],
        tools=[],
        replay_prefix_json="[]",
    )

    decision = asyncio.run(pipeline.evaluate(transition))

    assert decision.reward == 3.0
    assert decision.diagnostics["expectation_task"] == "2"
    assert decision.diagnostics["official_task"] == "2"
