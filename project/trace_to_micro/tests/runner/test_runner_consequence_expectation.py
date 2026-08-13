import json
from pathlib import Path
from types import SimpleNamespace

from trace_to_micro.runner.consequence_expectation import (
    run_consequence_expectation_evaluation,
)
from trace_to_micro.utils.io import read_jsonl


def test_runner_compiles_factual_result_and_writes_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    requests = [
        {
            "sample_id": f"sim:2:{moment}",
            "request_fingerprint": f"fingerprint-{moment}",
            "decision_id": "sim:2",
            "moment": moment,
            "messages": (
                [{"role": "tool", "content": '{"order_id":"#1"}'}]
                if moment == "result"
                else [{"role": "assistant", "content": "context"}]
            ),
        }
        for moment in ("before", "action", "result")
    ]
    activations = [
        {**request, "activations": {"47": "encoded"}} for request in requests
    ]
    consequence = {"decision_id": "sim:2", "changes": []}
    for name, rows in (
        ("activation_requests.jsonl", requests),
        ("activations.jsonl", activations),
        ("local_consequences.jsonl", [consequence]),
    ):
        (output_dir / name).write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    config = SimpleNamespace(
        output_dir=output_dir,
        domains=("airline", "retail"),
        train_split="train",
        test_split="test",
        primary_layer_id=47,
        expectation_regularization=1.0,
        expectation_minimum_train_class_count=2,
        bootstrap_samples=10,
        random_seed=300,
    )
    monkeypatch.setattr(
        "trace_to_micro.runner.consequence_expectation.LocalConsequenceConfig.load",
        lambda _: config,
    )
    seen = {}

    def compile_row(row, *, result_content):
        seen["content"] = result_content
        return {"decision_id": row["decision_id"], "targets": {}}

    monkeypatch.setattr(
        "trace_to_micro.runner.consequence_expectation.compile_abstract_consequence",
        compile_row,
    )
    monkeypatch.setattr(
        "trace_to_micro.runner.consequence_expectation."
        "build_consequence_expectation_report",
        lambda *args, **kwargs: ({"complete": True}, [{"decision_id": "sim:2"}]),
    )

    paths = run_consequence_expectation_evaluation(tmp_path / "config.toml")

    assert seen["content"] == '{"order_id":"#1"}'
    assert paths["report"].is_file()
    assert read_jsonl(paths["abstract_consequences"]) == [
        {"decision_id": "sim:2", "targets": {}}
    ]
