import json
from pathlib import Path

from expectation_step_rl.expectation import calibration_data
from expectation_step_rl.expectation.calibration_data import _score_record
from expectation_step_rl.expectation.scoring import ExpectationMeasurement


class _Scorer:
    routed: list[str] = []

    def __init__(self, **kwargs) -> None:
        pass

    def measure(self, **kwargs) -> ExpectationMeasurement:
        self.routed.append(kwargs["routing_key"])
        assert kwargs["raw_prompt"][-1]["role"] == "user"
        return ExpectationMeasurement(
            contextual_min_k_deviation=0.25,
            suffix_token_count=4,
            min_k_token_count=1,
            full_mean_logprob=-0.5,
            action_only_mean_logprob=-0.25,
            anonymized_identifier_counts={},
            scorer_base_url="http://127.0.0.1:8002/v1",
        )


def _record(decision_id: str) -> dict:
    call_id = f"call-{decision_id}"
    return {
        "decision_id": decision_id,
        "domain": "retail",
        "split": "train",
        "tool_name": "find_user_id_by_name_zip",
        "tool_call_id": call_id,
        "result_content": "alice_smith_1234",
        "tools": [],
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "I am Alice Smith"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": call_id, "name": "find_user_id_by_name_zip"}],
            },
        ],
    }


def test_score_record_uses_only_logged_train_action_and_result() -> None:
    record = _record("decision-1")

    row = _score_record(record, _Scorer())

    assert row["track"] == "logged_clean_result"
    assert row["split"] == "train"
    assert row["contextual_min_k_deviation"] == 0.25
    assert row["scorer_base_url"].endswith("8002/v1")


def test_training_calibration_resumes_only_missing_records(
    tmp_path: Path, monkeypatch
) -> None:
    records_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "scores.jsonl"
    records = [_record("decision-1"), _record("decision-2")]
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    existing = {
        "decision_id": "decision-1",
        "domain": "retail",
        "split": "train",
        "tool_name": "find_user_id_by_name_zip",
        "structure_key": "scalar",
        "track": "logged_clean_result",
        "severity": 0,
        "contextual_min_k_deviation": 0.5,
    }
    output_path.write_text(json.dumps(existing) + "\n")
    _Scorer.routed = []
    monkeypatch.setattr(calibration_data, "VLLMExpectationScorer", _Scorer)

    rows = calibration_data.score_training_records(
        records_path=records_path,
        output_path=output_path,
        base_urls=["http://127.0.0.1:8000/v1"],
        api_key="EMPTY",
        model="qwen3-32b",
        workers_per_server=1,
    )

    assert _Scorer.routed == ["decision-2"]
    assert [row["decision_id"] for row in rows] == ["decision-1", "decision-2"]
    assert len(output_path.read_text().splitlines()) == 2
