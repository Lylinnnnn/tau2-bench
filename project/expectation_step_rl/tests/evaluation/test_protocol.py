from expectation_step_rl.evaluation.protocol import baseline_id, protocol_digest


def _protocol(seed: int) -> dict:
    return {
        "base_model": {"served_name": "qwen3-32b"},
        "domains": ["airline", "retail"],
        "split": "test",
        "num_trials": 1,
        "seed": seed,
    }


def test_baseline_identity_is_stable_and_protocol_sensitive() -> None:
    protocol = _protocol(seed=300)

    assert protocol_digest(protocol) == protocol_digest(
        dict(reversed(protocol.items()))
    )
    assert baseline_id(protocol).endswith(protocol_digest(protocol)[:12])
    assert baseline_id(protocol) != baseline_id(_protocol(seed=301))
