import pytest

from expectation_step_rl.expectation.scoring import contextual_min_k


def test_contextual_min_k_uses_largest_deviations() -> None:
    full = [-1.0, -4.0, -2.0, -5.0]
    action_only = [-1.0, -1.0, -2.0, -1.0]

    assert contextual_min_k(full, action_only, fraction=0.5) == pytest.approx(3.5)


def test_contextual_min_k_rejects_unaligned_vectors() -> None:
    with pytest.raises(ValueError, match="aligned"):
        contextual_min_k([-1.0], [-1.0, -2.0], fraction=0.1)
