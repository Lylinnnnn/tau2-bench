"""Run target-free consequence expectation evaluation over saved artifacts."""

from pathlib import Path

from trace_to_micro.config import LocalConsequenceConfig
from trace_to_micro.evaluation.consequence_expectation import (
    build_consequence_expectation_report,
)
from trace_to_micro.replay.consequence import compile_abstract_consequence
from trace_to_micro.utils.io import read_jsonl, write_json, write_jsonl


def run_consequence_expectation_evaluation(
    config_path: Path,
) -> dict[str, Path]:
    """Fit and evaluate the target-free expected-consequence probe."""

    config = LocalConsequenceConfig.load(config_path)
    requests = read_jsonl(config.output_dir / "activation_requests.jsonl")
    activations = read_jsonl(config.output_dir / "activations.jsonl")
    expected = {(row["sample_id"], row["request_fingerprint"]) for row in requests}
    observed = {
        (row["sample_id"], row.get("request_fingerprint")) for row in activations
    }
    if len(activations) != len(requests) or observed != expected:
        raise ValueError(
            "Local consequence activation extraction is incomplete or stale"
        )
    source_consequences = read_jsonl(config.output_dir / "local_consequences.jsonl")
    result_content_by_decision = {}
    for request in requests:
        if request["moment"] != "result":
            continue
        messages = request["messages"]
        if not messages or messages[-1].get("role") != "tool":
            raise ValueError(
                f"Result request does not end in a tool message: {request['sample_id']}"
            )
        result_content_by_decision[request["decision_id"]] = messages[-1]["content"]
    if len(result_content_by_decision) * 3 != len(requests):
        raise ValueError("Each decision must have one factual result request")
    abstract_consequences = [
        compile_abstract_consequence(
            row,
            result_content=result_content_by_decision[row["decision_id"]],
        )
        for row in source_consequences
    ]
    report, predictions = build_consequence_expectation_report(
        activations,
        abstract_consequences,
        domains=config.domains,
        train_split=config.train_split,
        test_split=config.test_split,
        layer_id=config.primary_layer_id,
        regularization=config.expectation_regularization,
        minimum_train_class_count=config.expectation_minimum_train_class_count,
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed,
    )
    consequences_path = config.output_dir / "abstract_consequences.jsonl"
    predictions_path = config.output_dir / "consequence_expectation_predictions.jsonl"
    report_path = config.output_dir / "consequence_expectation_report.json"
    write_jsonl(consequences_path, abstract_consequences)
    write_jsonl(predictions_path, predictions)
    write_json(report_path, report)
    return {
        "abstract_consequences": consequences_path,
        "predictions": predictions_path,
        "report": report_path,
    }
