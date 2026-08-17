"""Reward fusion policies independent of signal computation."""

from __future__ import annotations

from dataclasses import dataclass

from expectation_step_rl.reward.interface import RewardDecision, RewardSignal


def _diagnostics(
    *, mode: str, signals: dict[str, RewardSignal], reward: float
) -> dict[str, object]:
    diagnostics: dict[str, object] = {}
    for signal in signals.values():
        diagnostics.update(signal.diagnostics)
        diagnostics[f"reward_signal_{signal.name}"] = signal.value
    diagnostics["reward_mode"] = mode
    diagnostics["final_reward"] = reward
    return diagnostics


@dataclass(frozen=True)
class WeightedSumRewardComposer:
    """Fuse signals by an explicit weighted sum."""

    weights: dict[str, float]
    mode: str = "weighted_sum"

    def compose(self, signals: dict[str, RewardSignal]) -> RewardDecision:
        """Return the weighted scalar while preserving each raw signal."""

        if set(signals) != set(self.weights):
            raise ValueError(
                "Reward signal names do not match composer weights: "
                f"signals={sorted(signals)}, weights={sorted(self.weights)}"
            )
        reward = sum(
            self.weights[name] * signal.value for name, signal in signals.items()
        )
        values = {name: signal.value for name, signal in signals.items()}
        return RewardDecision(
            reward=float(reward),
            signals=values,
            diagnostics=_diagnostics(mode=self.mode, signals=signals, reward=reward),
        )


@dataclass(frozen=True)
class GateRewardComposer:
    """Use one signal as a hard validity gate for another signal."""

    primary_signal: str
    gate_signal: str
    gate_threshold: float
    rejected_reward: float
    mode: str = "gate"

    def compose(self, signals: dict[str, RewardSignal]) -> RewardDecision:
        """Reject candidates that fail the gate instead of averaging signals."""

        expected = {self.primary_signal, self.gate_signal}
        if set(signals) != expected:
            raise ValueError(
                "Gate composer requires exactly its primary and gate signals: "
                f"signals={sorted(signals)}, expected={sorted(expected)}"
            )
        gate_passed = signals[self.gate_signal].value > self.gate_threshold
        reward = (
            signals[self.primary_signal].value if gate_passed else self.rejected_reward
        )
        values = {name: signal.value for name, signal in signals.items()}
        diagnostics = _diagnostics(mode=self.mode, signals=signals, reward=reward)
        diagnostics["reward_gate_passed"] = float(gate_passed)
        return RewardDecision(float(reward), values, diagnostics)
