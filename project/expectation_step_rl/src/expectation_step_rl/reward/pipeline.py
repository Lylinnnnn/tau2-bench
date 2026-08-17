"""Orchestrate reward providers and a separately configured composer."""

from __future__ import annotations

from expectation_step_rl.reward.interface import (
    CandidateTransition,
    RewardComposer,
    RewardDecision,
    RewardProvider,
)


class RewardPipeline:
    """Evaluate independent signals before composing the final reward."""

    def __init__(
        self, providers: list[RewardProvider], composer: RewardComposer
    ) -> None:
        names = [provider.name for provider in providers]
        if len(names) != len(set(names)):
            raise ValueError(f"Reward provider names must be unique: {names}")
        self.providers = providers
        self.composer = composer

    async def evaluate(self, transition: CandidateTransition) -> RewardDecision:
        """Compute and compose all configured signals for one transition."""

        signals = {
            provider.name: await provider.evaluate(transition)
            for provider in self.providers
        }
        return self.composer.compose(signals)
