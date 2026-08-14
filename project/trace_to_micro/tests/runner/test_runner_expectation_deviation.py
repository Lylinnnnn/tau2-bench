from types import SimpleNamespace

from trace_to_micro.runner.expectation_deviation import (
    _expected_min_k_row_count,
    _min_k_work,
)


def test_min_k_work_scores_only_controlled_test_and_matching_train_tools() -> None:
    config = SimpleNamespace(train_split="train")
    records = {
        "train-match": {
            "decision_id": "train-match",
            "domain": "retail",
            "split": "train",
            "tool_name": "lookup",
        },
        "train-other": {
            "decision_id": "train-other",
            "domain": "retail",
            "split": "train",
            "tool_name": "unused",
        },
        "test-match": {
            "decision_id": "test-match",
            "domain": "retail",
            "split": "test",
            "tool_name": "lookup",
        },
    }
    queries = [
        {
            "query_id": "test-match",
            "domain": "retail",
            "tool_name": "lookup",
            "variants": [{"severity": value} for value in range(4)],
        }
    ]

    work = _min_k_work(config, records, queries)

    assert [item["kind"] for item in work] == ["calibration", "controlled"]
    assert work[0]["value"]["decision_id"] == "train-match"
    assert [_expected_min_k_row_count(item) for item in work] == [1, 4]
