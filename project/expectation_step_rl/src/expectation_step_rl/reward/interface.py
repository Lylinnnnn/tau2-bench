"""Data contracts shared by reward providers and composers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CandidateAction:
    """One tool action sampled by the current policy."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    arguments_json: str


@dataclass(frozen=True)
class CandidateTransition:
    """The information available after executing one candidate action."""

    domain: str
    task_id: str
    raw_prompt: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    replay_prefix_json: str
    action: CandidateAction | None = None
    tool_result: str | None = None
    tool_error: bool | None = None
    invalid_reason: str | None = None


@dataclass(frozen=True)
class RewardSignal:
    """One named scalar signal plus provider-specific diagnostics."""

    name: str
    value: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardDecision:
    """The final scalar consumed by GRPO and its auditable components."""

    reward: float
    signals: dict[str, float]
    diagnostics: dict[str, Any]


class RewardProvider(Protocol):
    """Compute one reward signal from a realized candidate transition."""

    name: str

    async def evaluate(self, transition: CandidateTransition) -> RewardSignal:
        """Return this provider's signal for one candidate."""


class RewardComposer(Protocol):
    """Fuse named signals into the scalar required by GRPO."""

    def compose(self, signals: dict[str, RewardSignal]) -> RewardDecision:
        """Combine all configured signals without executing the environment."""
