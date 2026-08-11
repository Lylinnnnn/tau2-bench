from trace_to_micro.clean.report import build_hybrid_context_report


def test_hybrid_report_counts_retries_repairs_and_eligibility() -> None:
    rows = [
        {
            "context": {
                "training_eligible": True,
                "verified_context": {
                    "interaction_phase": "error_recovery",
                    "quality_flags": [],
                },
                "action_contract": {"repairs": ["corrected_executor_to_user"]},
            },
            "builder_metadata": {
                "status": "passed",
                "retry_count": 1,
                "attempts": [
                    {"validation_errors": [{"code": "REPEATS_INVALID_TOOL"}]},
                    {"validation_errors": []},
                ],
            },
        }
    ]

    report = build_hybrid_context_report(rows)

    assert report["training_eligible_rate"] == 1.0
    assert report["retry_count_histogram"] == {"1": 1}
    assert report["validation_error_counts"] == {"REPEATS_INVALID_TOOL": 1}
    assert report["action_contract_repair_counts"] == {"corrected_executor_to_user": 1}
