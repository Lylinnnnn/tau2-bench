import pytest

from trace_to_micro.evaluation.deviation_metrics import (
    build_contextual_min_k_report,
    build_deviation_report,
    calibrated_anomaly_score,
    contextual_min_k_statistics,
    fit_clean_calibration,
)


def test_contextual_min_k_uses_worst_aligned_token_ratios() -> None:
    statistics = contextual_min_k_statistics(
        {
            "suffix_token_ids": [1, 2, 3, 4],
            "token_logprobs": [-0.1, -0.2, -3.0, -0.4],
        },
        {
            "suffix_token_ids": [1, 2, 3, 4],
            "token_logprobs": [-0.2, -0.3, -0.5, -0.5],
        },
        fraction=0.25,
    )

    assert statistics == pytest.approx(
        {
            "suffix_token_count": 4,
            "min_k_token_count": 1,
            "full_mean_surprisal": 0.925,
            "full_min_k_surprisal": 3.0,
            "mean_contextual_deviation": 0.55,
            "contextual_min_k_deviation": 2.5,
        }
    )


def test_calibration_prefers_structure_then_falls_back_to_tool() -> None:
    train = []
    for index, score in enumerate((-0.4, -0.5, -0.6)):
        train.append(
            {
                "domain": "retail",
                "context_variant": "full",
                "tool_name": "lookup",
                "structure_key": "common",
                "mean_logprob": score,
            }
        )
    train.extend(
        {
            "domain": "retail",
            "context_variant": "full",
            "tool_name": "lookup",
            "structure_key": f"rare-{index}",
            "mean_logprob": score,
        }
        for index, score in enumerate((-0.7, -0.8))
    )
    calibration = fit_clean_calibration(train, minimum_count=3)

    structure_score = calibrated_anomaly_score(
        {**train[0], "mean_logprob": -0.9}, calibration
    )
    fallback_score = calibrated_anomaly_score(
        {**train[-1], "mean_logprob": -0.9}, calibration
    )

    assert structure_score[1:] == ("structure", 3)
    assert fallback_score[1:] == ("tool", 5)
    assert structure_score[0] > 0


def test_deviation_report_measures_anomaly_order_and_calibration() -> None:
    rows = []
    for domain in ("airline", "retail"):
        for task_index in range(6):
            for context in ("full", "action_only", "shuffled"):
                rows.append(
                    {
                        "track": "calibration_clean",
                        "query_id": f"train-{domain}-{task_index}",
                        "domain": domain,
                        "task_id": f"train-{task_index}",
                        "tool_name": "lookup",
                        "structure_key": "schema",
                        "context_variant": context,
                        "severity": 0,
                        "mean_logprob": -0.4 - task_index * 0.01,
                    }
                )
                for severity, score in enumerate((-0.4, -0.7, -1.0, -1.3)):
                    rows.append(
                        {
                            "track": "controlled_anomaly",
                            "query_id": f"test-{domain}-{task_index}",
                            "domain": domain,
                            "task_id": f"test-{task_index}",
                            "tool_name": "lookup",
                            "structure_key": "schema",
                            "context_variant": context,
                            "severity": severity,
                            "mean_logprob": score,
                        }
                    )

    report, measurements, calibration = build_deviation_report(
        rows,
        domains=("airline", "retail"),
        context_variants=("full", "action_only", "shuffled"),
        calibration_minimum_count=5,
        bootstrap_samples=100,
        random_seed=300,
    )

    condition = report["domains"]["retail"]["conditions"]["full"]
    assert condition["raw_surprisal_auroc"] == 1.0
    assert condition["paired_clean_margin_auroc"] == 1.0
    assert condition["calibrated_surprisal_auroc"] == 1.0
    assert condition["strictly_monotonic_query_rate"] == 1.0
    assert condition["severity_order_accuracy"] == 1.0
    assert condition["mean_logprob_by_severity"] == pytest.approx(
        {"0": -0.4, "1": -0.7, "2": -1.0, "3": -1.3}
    )
    assert len(measurements) == 2 * 6 * 3 * 4
    assert calibration["structure"]
    assert report["domains"]["retail"]["calibration_level_counts"] == {"structure": 24}


def test_contextual_min_k_report_uses_train_only_tool_calibration() -> None:
    rows = []
    fields = (
        "full_mean_surprisal",
        "full_min_k_surprisal",
        "mean_contextual_deviation",
        "contextual_min_k_deviation",
    )
    for domain in ("airline", "retail"):
        for index in range(6):
            rows.append(
                {
                    "track": "min_k_calibration",
                    "query_id": f"train-{domain}-{index}",
                    "domain": domain,
                    "task_id": f"train-{index}",
                    "tool_name": "lookup",
                    "severity": 0,
                    **{field: 0.1 * index for field in fields},
                }
            )
        for task_index in range(2):
            for severity in range(4):
                rows.append(
                    {
                        "track": "min_k_controlled",
                        "query_id": f"test-{domain}-{task_index}",
                        "domain": domain,
                        "task_id": f"test-{task_index}",
                        "tool_name": "lookup",
                        "severity": severity,
                        **{field: float(severity) for field in fields},
                    }
                )

    report, measurements, calibration = build_contextual_min_k_report(
        rows,
        domains=("airline", "retail"),
        calibration_minimum_count=5,
        min_k_fraction=0.1,
        sigma_threshold=3.0,
        bootstrap_samples=20,
        random_seed=300,
    )

    retail = report["domains"]["retail"]
    assert retail["primary"]["raw"]["auroc"] == 1.0
    assert retail["primary"]["train_calibrated"]["auroc"] == 1.0
    assert retail["primary"]["paired_ordering"]["strictly_monotonic_query_rate"] == 1.0
    assert retail["calibrated_row_count"] == 8
    assert calibration["group_by"] == ["domain", "tool_name"]
    assert len(measurements) == 16
