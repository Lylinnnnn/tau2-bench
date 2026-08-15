from expectation_step_rl.expectation.calibration import (
    TrainCalibration,
    fit_training_calibration,
)


def _calibration() -> TrainCalibration:
    return TrainCalibration(
        {
            "source": "official Train clean tool results",
            "groups": {
                "retail|get_order_details|order": {
                    "count": 20,
                    "statistics": {
                        "contextual_min_k_deviation": {"mean": 1.0, "std": 0.5}
                    },
                }
            },
            "tool_fallbacks": {},
            "structure_fallbacks": {},
            "domain_fallbacks": {
                "retail": {
                    "count": 100,
                    "statistics": {
                        "contextual_min_k_deviation": {"mean": 2.0, "std": 1.0}
                    },
                }
            },
        },
        clip=5.0,
        invalid_action_penalty=-5.0,
        tool_error_penalty=-4.0,
        unsupported_tool_penalty=-3.0,
    )


def test_larger_deviation_receives_lower_reward() -> None:
    calibration = _calibration()

    expected = calibration.score(
        domain="retail",
        tool_name="get_order_details",
        structure_key="order",
        raw_score=1.0,
    )
    anomalous = calibration.score(
        domain="retail",
        tool_name="get_order_details",
        structure_key="order",
        raw_score=2.0,
    )

    assert expected.reward == 0.0
    assert anomalous.reward == -2.0
    assert anomalous.reward < expected.reward


def test_candidate_failures_have_explicit_outcomes() -> None:
    calibration = _calibration()

    invalid = calibration.score(
        domain="retail",
        tool_name=None,
        raw_score=None,
        action_valid=False,
    )
    error = calibration.score(
        domain="retail",
        tool_name="get_order_details",
        raw_score=None,
        tool_error=True,
    )
    unsupported = calibration.score(
        domain="airline", tool_name="unknown", raw_score=None
    )

    assert (invalid.reward, invalid.outcome) == (-5.0, "invalid_action")
    assert (error.reward, error.outcome) == (-4.0, "tool_error")
    assert (unsupported.reward, unsupported.outcome) == (
        -3.0,
        "unsupported_calibration_group",
    )


def test_rare_valid_tool_uses_train_domain_fallback() -> None:
    result = _calibration().score(
        domain="retail", tool_name="rare_valid_tool", raw_score=3.0
    )

    assert result.reward == -1.0
    assert result.calibration_level == "domain_fallback"
    assert result.outcome == "scored_domain_fallback"


def test_calibration_prefers_tool_then_structure_before_domain() -> None:
    calibration = TrainCalibration(
        {
            "source": "official Train logged clean tool results only",
            "groups": {},
            "tool_fallbacks": {
                "retail|lookup": {
                    "count": 10,
                    "statistics": {
                        "contextual_min_k_deviation": {"mean": 1.0, "std": 1.0}
                    },
                }
            },
            "structure_fallbacks": {
                "retail|scalar": {
                    "count": 20,
                    "statistics": {
                        "contextual_min_k_deviation": {"mean": 2.0, "std": 1.0}
                    },
                }
            },
            "domain_fallbacks": {
                "retail": {
                    "count": 100,
                    "statistics": {
                        "contextual_min_k_deviation": {"mean": 3.0, "std": 1.0}
                    },
                }
            },
        },
        clip=5.0,
        invalid_action_penalty=-5.0,
        tool_error_penalty=-4.0,
        unsupported_tool_penalty=-3.0,
    )

    tool = calibration.score(
        domain="retail", tool_name="lookup", structure_key="scalar", raw_score=2.0
    )
    structure = calibration.score(
        domain="retail", tool_name="rare", structure_key="scalar", raw_score=2.0
    )

    assert (tool.reward, tool.calibration_level) == (-1.0, "tool_fallback")
    assert (structure.reward, structure.calibration_level) == (
        0.0,
        "structure_fallback",
    )


def test_fit_calibration_ignores_test_scores() -> None:
    rows = [
        {
            "split": "train",
            "track": "logged_clean_result",
            "severity": 0,
            "domain": "retail",
            "tool_name": "lookup",
            "structure_key": "scalar",
            "contextual_min_k_deviation": value,
        }
        for value in (1.0, 3.0)
    ]
    rows.append(
        {
            "split": "test",
            "track": "min_k_controlled",
            "severity": 3,
            "domain": "retail",
            "tool_name": "lookup",
            "contextual_min_k_deviation": 1000.0,
        }
    )

    calibration = fit_training_calibration(rows, minimum_tool_count=2)

    statistics = calibration["groups"]["retail|lookup|scalar"]["statistics"]
    assert statistics["contextual_min_k_deviation"]["mean"] == 2.0
    assert calibration["domain_fallbacks"]["retail"]["count"] == 2
