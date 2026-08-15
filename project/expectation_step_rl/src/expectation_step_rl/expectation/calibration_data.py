"""Score every logged clean Train result used for reward calibration."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from expectation_step_rl.expectation.scoring import VLLMExpectationScorer
from expectation_step_rl.expectation.structure import result_structure_key


def _read_train_records(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    selected = [row for row in rows if row.get("split") == "train"]
    if not selected:
        raise ValueError("No official Train records were found")
    return sorted(selected, key=lambda row: row["decision_id"])


def _score_record(
    record: dict[str, Any], scorer: VLLMExpectationScorer
) -> dict[str, Any]:
    messages = record["messages"]
    action_message = messages[-1]
    calls = action_message.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["id"] != record["tool_call_id"]:
        raise ValueError(f"Malformed logged action: {record['decision_id']}")
    measurement = scorer.measure(
        raw_prompt=messages[:-1],
        action_message=action_message,
        result_message={
            "role": "tool",
            "tool_call_id": record["tool_call_id"],
            "content": record["result_content"],
        },
        tools=record["tools"],
        routing_key=record["decision_id"],
    )
    return {
        "decision_id": record["decision_id"],
        "domain": record["domain"],
        "split": "train",
        "tool_name": record["tool_name"],
        "structure_key": result_structure_key(record["result_content"]),
        "track": "logged_clean_result",
        "severity": 0,
        "contextual_min_k_deviation": measurement.contextual_min_k_deviation,
        "suffix_token_count": measurement.suffix_token_count,
        "min_k_token_count": measurement.min_k_token_count,
        "scorer_base_url": measurement.scorer_base_url,
    }


def _read_existing_scores(
    path: Path, expected_ids: set[str]
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            decision_id = str(row["decision_id"])
            if decision_id not in expected_ids:
                raise ValueError(f"Unexpected calibration decision: {decision_id}")
            if decision_id in rows:
                raise ValueError(f"Duplicate calibration decision: {decision_id}")
            rows[decision_id] = row
    return rows


def score_training_records(
    *,
    records_path: Path,
    output_path: Path,
    base_urls: list[str],
    api_key: str,
    model: str,
    workers_per_server: int,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Score clean Train results concurrently across frozen scorer replicas."""

    records = _read_train_records(records_path)
    expected_ids = {str(row["decision_id"]) for row in records}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_by_id = (
        {} if overwrite else _read_existing_scores(output_path, expected_ids)
    )
    pending = [row for row in records if str(row["decision_id"]) not in completed_by_id]
    if completed_by_id:
        print(
            f"Resuming Train calibration: {len(completed_by_id)}/{len(records)} "
            "already scored"
        )
    if overwrite or not output_path.exists():
        output_path.write_text("")
    scorer = VLLMExpectationScorer(
        base_urls=base_urls,
        api_key=api_key,
        model=model,
        min_k_fraction=0.1,
    )
    with ThreadPoolExecutor(
        max_workers=len(base_urls) * workers_per_server
    ) as executor:
        futures = {executor.submit(_score_record, row, scorer): row for row in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            completed_by_id[str(row["decision_id"])] = row
            with output_path.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            total = len(completed_by_id)
            if index % 25 == 0 or total == len(records):
                print(f"Scored Train calibration results: {total}/{len(records)}")
    if set(completed_by_id) != expected_ids:
        raise RuntimeError(
            f"Calibration is incomplete: {len(completed_by_id)}/{len(records)}"
        )
    completed = sorted(completed_by_id.values(), key=lambda row: row["decision_id"])
    with output_path.open("w") as handle:
        for row in completed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-urls", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3-32b")
    parser.add_argument("--workers-per-server", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows = score_training_records(
        records_path=args.records,
        output_path=args.output,
        base_urls=[value.strip() for value in args.base_urls.split(",")],
        api_key=args.api_key,
        model=args.model,
        workers_per_server=args.workers_per_server,
        overwrite=args.overwrite,
    )
    print(json.dumps({"output": str(args.output), "train_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
