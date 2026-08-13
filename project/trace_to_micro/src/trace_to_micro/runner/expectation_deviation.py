"""Run identity-preserving rematching and controlled anomaly scoring."""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from tau2.runner import load_task_splits
from trace_to_micro.analysis.logged_trace import (
    audit_results_completeness,
    tool_decision_records,
)
from trace_to_micro.config import ExpectationDeviationConfig
from trace_to_micro.evaluation.deviation_metrics import (
    build_consistent_rematch_report,
    build_deviation_report,
)
from trace_to_micro.evaluation.expectation_deviation import (
    build_controlled_anomaly_queries,
)
from trace_to_micro.evaluation.expectation_matching import (
    build_matching_queries,
    mask_identifiers,
    matching_record,
)
from trace_to_micro.runner.deviation_scoring import (
    anonymize_scoring_bundle,
    score_calibration_record,
    score_controlled_query,
    score_rematch_query,
)
from trace_to_micro.runner.expectation_matching import messages_for_context
from trace_to_micro.utils.io import (
    append_jsonl_batch,
    read_jsonl,
    write_json,
    write_jsonl,
)


def _paths(config: ExpectationDeviationConfig) -> dict[str, Path]:
    return {
        "records": config.output_dir / "expectation_deviation_records.jsonl",
        "queries": config.output_dir / "controlled_anomaly_queries.jsonl",
        "rematch_queries": config.output_dir / "consistent_rematch_queries.jsonl",
        "audit": config.output_dir / "expectation_deviation_audit.json",
        "scores": config.output_dir / "expectation_deviation_scores.jsonl",
        "measurements": config.output_dir / "expectation_deviation_measurements.jsonl",
        "calibration": config.output_dir / "expectation_deviation_calibration.json",
        "report": config.output_dir / "expectation_deviation_report.json",
        "rematch_report": config.output_dir / "consistent_rematch_report.json",
    }


def _trajectory_exclusion(
    *, domain: str, split: str, simulation, error: Exception
) -> dict[str, Any]:
    return {
        "scope": "trajectory",
        "domain": domain,
        "split": split,
        "task_id": str(simulation.task_id),
        "simulation_id": simulation.id,
        "trial": simulation.trial,
        "stage": "logged_result_extraction",
        "reason": "trajectory_extraction_failed",
        "exception_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(traceback.format_exception(error)),
    }


def build_expectation_deviation_dataset(config_path: Path) -> dict[str, Path]:
    """Extract Train/Test clean results and compile controlled Test anomalies."""

    config = ExpectationDeviationConfig.load(config_path)
    paths = _paths(config)
    records = []
    trajectory_exclusions = []
    completeness = {}
    source_counts = {}
    for domain in config.domains:
        split_map = load_task_splits(domain)
        if split_map is None:
            raise ValueError(f"Domain {domain!r} has no Train/Test split")
        split_by_task = {
            str(task_id): split
            for split in (config.train_split, config.test_split)
            for task_id in split_map[split]
        }
        results_path = config.results_path(domain)
        completeness[domain] = audit_results_completeness(
            results_path,
            expected_task_ids=set(split_by_task),
            expected_num_trials=config.expected_num_trials,
            expected_agent_model=config.expected_agent_model,
            expected_user_model=config.expected_user_model,
            raise_on_incomplete=False,
        )
        if (
            completeness[domain]["agent_model_mismatch"]
            or completeness[domain]["user_model_mismatch"]
        ):
            raise ValueError(f"{domain} trajectories come from a different model")
        invalid_ids = set(
            completeness[domain]["empty_simulation_ids"]
            + completeness[domain]["missing_reward_simulation_ids"]
            + completeness[domain]["infrastructure_error_simulation_ids"]
        )

        def record_error(simulation, error: Exception) -> None:
            trajectory_exclusions.append(
                _trajectory_exclusion(
                    domain=domain,
                    split=split_by_task.get(str(simulation.task_id), "outside_split"),
                    simulation=simulation,
                    error=error,
                )
            )

        rows = tool_decision_records(
            results_path,
            domain=domain,
            task_set=domain,
            train_split=config.train_split,
            test_split=config.test_split,
            max_per_trajectory=None,
            on_simulation_error=record_error,
        )
        domain_records = []
        for row in rows:
            if (
                row["moment"] != "action"
                or row["split"] not in {config.train_split, config.test_split}
                or row["simulation_id"] in invalid_ids
                or not row["tool_success"]
                or len(row["messages"][-1].get("tool_calls") or []) != 1
                or len(row["tool_result_contents"]) != 1
            ):
                continue
            domain_records.append(matching_record(row))
        records.extend(domain_records)
        source_counts[domain] = {
            split: {
                "successful_single_call_decisions": sum(
                    row["split"] == split for row in domain_records
                ),
                "task_count": len(
                    {row["task_id"] for row in domain_records if row["split"] == split}
                ),
                "tools": dict(
                    Counter(
                        row["tool_name"]
                        for row in domain_records
                        if row["split"] == split
                    )
                ),
            }
            for split in (config.train_split, config.test_split)
        }
    duplicate_ids = [
        decision_id
        for decision_id, count in Counter(row["decision_id"] for row in records).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"Duplicate deviation decision IDs: {duplicate_ids}")
    queries, anomaly_exclusions = build_controlled_anomaly_queries(
        records,
        train_split=config.train_split,
        test_split=config.test_split,
        severity_levels=config.severity_levels,
        max_length_delta_ratio=config.max_length_delta_ratio,
        random_seed=config.random_seed,
    )
    if not queries:
        raise ValueError("Controlled anomaly construction produced no query")
    rematch_queries, rematch_exclusions = build_matching_queries(
        [row for row in records if row["split"] == config.test_split],
        candidate_count=config.rematch_candidate_count,
        minimum_candidate_count=config.rematch_candidate_count,
        random_seed=config.random_seed,
    )
    records_by_id = {row["decision_id"]: row for row in records}
    legacy_collapsed = Counter()
    for query in rematch_queries:
        source = records_by_id[query["query_id"]]
        raw_contents = [
            records_by_id[decision_id]["result_content"]
            for decision_id in query["candidate_decision_ids"]
        ]
        if len(set(mask_identifiers(content) for content in raw_contents)) < len(
            raw_contents
        ):
            legacy_collapsed[query["tool_name"]] += 1
        for context in config.context_variants:
            _, _, _, contents, _ = anonymize_scoring_bundle(
                messages_for_context(query, records_by_id, context),
                source["tools"],
                source["tool_call_id"],
                raw_contents,
            )
            if len(set(contents)) != len(contents):
                raise ValueError(
                    "Consistent anonymization collapsed candidates for "
                    f"{query['query_id']} under {context}"
                )
    write_jsonl(paths["records"], records)
    write_jsonl(paths["queries"], queries)
    write_jsonl(paths["rematch_queries"], rematch_queries)
    write_json(
        paths["audit"],
        {
            "completed": True,
            "data_source": "saved behavior-policy trajectories",
            "new_rollout_used": False,
            "official_goal_used": False,
            "reference_action_used": False,
            "train_split_use": "clean likelihood calibration and donor values only",
            "test_split_use": "controlled anomaly and consistent rematch evaluation",
            "source_completeness": completeness,
            "source_counts": source_counts,
            "controlled_query_count": len(queries),
            "controlled_task_count": len(
                {(row["domain"], row["task_id"]) for row in queries}
            ),
            "controlled_tool_support": dict(
                Counter(row["tool_name"] for row in queries)
            ),
            "rematch_query_count": len(rematch_queries),
            "legacy_mask_collapsed_rematch_query_count": sum(legacy_collapsed.values()),
            "legacy_mask_collapsed_rematch_tools": dict(legacy_collapsed),
            "consistent_anonymization_collapsed_query_count": 0,
            "trajectory_exclusions": trajectory_exclusions,
            "anomaly_exclusions": anomaly_exclusions,
            "rematch_exclusions": rematch_exclusions,
        },
    )
    return {
        name: paths[name] for name in ("records", "queries", "rematch_queries", "audit")
    }


def _load_inputs(
    config: ExpectationDeviationConfig,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = _paths(config)
    records = read_jsonl(paths["records"])
    queries = read_jsonl(paths["queries"])
    rematch = read_jsonl(paths["rematch_queries"])
    if not records or not queries or not rematch:
        raise ValueError("Run expectation-deviation-prepare before scoring")
    record_by_id = {row["decision_id"]: row for row in records}
    if len(record_by_id) != len(records):
        raise ValueError("Deviation records contain duplicate IDs")
    for query in [*queries, *rematch]:
        for decision_id, expected in query["record_fingerprints"].items():
            if decision_id not in record_by_id:
                raise ValueError(
                    f"Prepared query references missing record {decision_id}"
                )
            observed = hashlib.sha256(
                json.dumps(
                    record_by_id[decision_id],
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if observed != expected:
                raise ValueError(
                    f"Prepared record changed after query construction: {decision_id}"
                )
    return record_by_id, queries, rematch


def _assigned_work(
    config: ExpectationDeviationConfig,
    records: dict[str, dict[str, Any]],
    queries: list[dict[str, Any]],
    rematch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    work = [{"kind": "controlled", "value": row} for row in queries]
    evaluated_tools = {(row["domain"], row["tool_name"]) for row in queries}
    train = [
        row
        for row in records.values()
        if row["split"] == config.train_split
        and (row["domain"], row["tool_name"]) in evaluated_tools
        and any(
            other["domain"] == row["domain"]
            and other["tool_name"] == row["tool_name"]
            and other["task_id"] != row["task_id"]
            for other in records.values()
            if other["split"] == config.train_split
        )
    ]
    by_tool = {}
    for record in train:
        pool = sorted(
            (
                row
                for row in train
                if row["domain"] == record["domain"]
                and row["tool_name"] == record["tool_name"]
                and row["task_id"] != record["task_id"]
            ),
            key=lambda row: row["decision_id"],
        )
        by_tool[record["decision_id"]] = pool[0]["decision_id"]
    work.extend(
        {
            "kind": "calibration",
            "value": row,
            "shuffled_id": by_tool[row["decision_id"]],
        }
        for row in train
    )
    work.extend({"kind": "rematch", "value": row} for row in rematch)
    return sorted(
        work,
        key=lambda row: (
            row["kind"],
            row["value"].get("query_id", row["value"].get("decision_id")),
        ),
    )


def _work_id(item: dict[str, Any]) -> str:
    return f"{item['kind']}:{item['value'].get('query_id', item['value'].get('decision_id'))}"


def _work_fingerprint(item: dict[str, Any], config: ExpectationDeviationConfig) -> str:
    payload = {
        "work": item,
        "model": config.scoring_model,
        "context_variants": config.context_variants,
        "severity_levels": config.severity_levels,
        "scoring": "consistent anonymization and mean tool-result logprob",
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _expected_work_row_count(
    item: dict[str, Any], config: ExpectationDeviationConfig
) -> int:
    multiplier = (
        len(config.severity_levels) + 1
        if item["kind"] == "controlled"
        else config.rematch_candidate_count
        if item["kind"] == "rematch"
        else 1
    )
    return multiplier * len(config.context_variants)


def run_expectation_deviation_score_shard(
    config_path: Path,
    *,
    shard_index: int,
    num_shards: int,
    base_url: str,
) -> Path:
    """Score one resumable shard of Train calibration and Test deviations."""

    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard index or shard count")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be non-empty")
    config = ExpectationDeviationConfig.load(config_path)
    records, queries, rematch = _load_inputs(config)
    assigned = [
        item
        for index, item in enumerate(_assigned_work(config, records, queries, rematch))
        if index % num_shards == shard_index
    ]
    output = config.score_shard_path(shard_index, num_shards)
    existing = read_jsonl(output)
    counts = Counter(row["work_id"] for row in existing)
    expected_counts = {
        _work_id(item): _expected_work_row_count(item, config) for item in assigned
    }
    if set(counts) - set(expected_counts):
        raise ValueError("Score shard contains work assigned to another shard")
    fingerprints = {
        _work_id(item): _work_fingerprint(item, config) for item in assigned
    }
    stale = {
        row["work_id"]
        for row in existing
        if row.get("work_fingerprint") != fingerprints[row["work_id"]]
    }
    if stale:
        raise ValueError(f"Score shard contains stale work groups: {sorted(stale)}")
    partial = {
        work_id: count
        for work_id, count in counts.items()
        if count != expected_counts[work_id]
    }
    if partial:
        existing = [row for row in existing if row["work_id"] not in partial]
        write_jsonl(output, existing)
        counts = Counter(row["work_id"] for row in existing)
    completed = set(counts)
    for item in assigned:
        work_id = _work_id(item)
        if work_id in completed:
            continue
        if item["kind"] == "controlled":
            rows = score_controlled_query(
                item["value"],
                records,
                config=config,
                base_url=base_url,
                api_key=api_key,
            )
        elif item["kind"] == "calibration":
            rows = score_calibration_record(
                item["value"],
                item["shuffled_id"],
                records,
                config=config,
                base_url=base_url,
                api_key=api_key,
            )
        else:
            rows = score_rematch_query(
                item["value"],
                records,
                config=config,
                base_url=base_url,
                api_key=api_key,
            )
        completed_rows = [
            {
                "work_id": work_id,
                "work_fingerprint": fingerprints[work_id],
                **row,
            }
            for row in rows
        ]
        append_jsonl_batch(output, completed_rows)
        existing.extend(completed_rows)
    return output


def merge_expectation_deviation_scores(config_path: Path, *, num_shards: int) -> Path:
    """Merge complete deviation work groups from all shards."""

    config = ExpectationDeviationConfig.load(config_path)
    records, queries, rematch = _load_inputs(config)
    expected = {
        _work_id(item) for item in _assigned_work(config, records, queries, rematch)
    }
    rows = []
    for shard_index in range(num_shards):
        path = config.score_shard_path(shard_index, num_shards)
        if not path.is_file():
            raise ValueError(f"Missing score shard {path}")
        rows.extend(read_jsonl(path))
    observed = {row["work_id"] for row in rows}
    if observed != expected:
        raise ValueError("Merged deviation shards do not cover every work item")
    expected_fingerprints = {
        _work_id(item): _work_fingerprint(item, config)
        for item in _assigned_work(config, records, queries, rematch)
    }
    stale = {
        row["work_id"]
        for row in rows
        if row.get("work_fingerprint") != expected_fingerprints[row["work_id"]]
    }
    if stale:
        raise ValueError(
            f"Merged deviation shards contain stale groups: {sorted(stale)}"
        )
    counts = Counter(row["work_id"] for row in rows)
    expected_counts = {}
    for item in _assigned_work(config, records, queries, rematch):
        expected_counts[_work_id(item)] = _expected_work_row_count(item, config)
    bad = {
        work_id: count
        for work_id, count in counts.items()
        if count != expected_counts[work_id]
    }
    if bad:
        raise ValueError(f"Merged deviation work groups are incomplete: {bad}")
    output = _paths(config)["scores"]
    write_jsonl(
        output,
        sorted(
            rows,
            key=lambda row: (
                row["work_id"],
                row.get("context_variant", ""),
                row.get("severity", -1),
                row.get("candidate_decision_id", ""),
            ),
        ),
    )
    return output


def run_expectation_deviation_evaluation(config_path: Path) -> dict[str, Path]:
    """Fit Train calibration and report Test anomaly and rematch results."""

    config = ExpectationDeviationConfig.load(config_path)
    paths = _paths(config)
    rows = read_jsonl(paths["scores"])
    if not rows:
        raise ValueError("Merge expectation-deviation scores before evaluation")
    report, measurements, calibration = build_deviation_report(
        rows,
        domains=config.domains,
        context_variants=config.context_variants,
        calibration_minimum_count=config.calibration_minimum_count,
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed,
    )
    rematch = build_consistent_rematch_report(
        [row for row in rows if row["track"] == "consistent_rematch"]
    )
    write_jsonl(paths["measurements"], measurements)
    write_json(paths["calibration"], calibration)
    write_json(paths["report"], report)
    write_json(paths["rematch_report"], rematch)
    return {
        name: paths[name]
        for name in ("measurements", "calibration", "report", "rematch_report")
    }


def run_expectation_deviation_smoke(
    config_path: Path, *, base_url: str
) -> dict[str, Path]:
    """Score one controlled query and one rematch query end to end."""

    config = ExpectationDeviationConfig.load(config_path)
    records, queries, rematch = _load_inputs(config)
    controlled = next(
        (
            row
            for row in queries
            if row["domain"] == config.smoke_domain
            and row["task_id"] == config.smoke_task_id
        ),
        next(row for row in queries if row["domain"] == config.smoke_domain),
    )
    rematch_query = next(
        (
            row
            for row in rematch
            if row["domain"] == controlled["domain"]
            and row["task_id"] == controlled["task_id"]
        ),
        next(row for row in rematch if row["domain"] == config.smoke_domain),
    )
    train_record = next(
        row
        for row in records.values()
        if row["domain"] == config.smoke_domain
        and row["split"] == config.train_split
        and row["tool_name"] == controlled["tool_name"]
        and row["result_structure"]
        == records[controlled["source_decision_id"]]["result_structure"]
        and any(
            other["domain"] == row["domain"]
            and other["split"] == config.train_split
            and other["tool_name"] == row["tool_name"]
            and other["result_structure"] == row["result_structure"]
            and other["task_id"] != row["task_id"]
            for other in records.values()
        )
    )
    train_shuffled = next(
        row["decision_id"]
        for row in records.values()
        if row["domain"] == train_record["domain"]
        and row["split"] == config.train_split
        and row["tool_name"] == train_record["tool_name"]
        and row["result_structure"] == train_record["result_structure"]
        and row["task_id"] != train_record["task_id"]
    )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be non-empty")
    controlled_rows = score_controlled_query(
        controlled, records, config=config, base_url=base_url, api_key=api_key
    )
    rematch_rows = score_rematch_query(
        rematch_query, records, config=config, base_url=base_url, api_key=api_key
    )
    calibration_rows = score_calibration_record(
        train_record,
        train_shuffled,
        records,
        config=config,
        base_url=base_url,
        api_key=api_key,
    )
    output_dir = config.smoke_output_dir()
    scores_path = output_dir / "expectation_deviation_scores.jsonl"
    report_path = output_dir / "smoke_report.json"
    rows = (
        [
            {"work_id": f"controlled:{controlled['query_id']}", **row}
            for row in controlled_rows
        ]
        + [
            {"work_id": f"rematch:{rematch_query['query_id']}", **row}
            for row in rematch_rows
        ]
        + [
            {"work_id": f"calibration:{train_record['decision_id']}", **row}
            for row in calibration_rows
        ]
    )
    write_jsonl(scores_path, rows)
    write_json(
        report_path,
        {
            "completed": True,
            "controlled_query": controlled,
            "rematch_query": rematch_query,
            "calibration_decision_id": train_record["decision_id"],
            "score_row_count": len(rows),
            "expected_score_row_count": len(config.context_variants)
            * (len(config.severity_levels) + 1 + config.rematch_candidate_count + 1),
            "calibration_matches_controlled_tool_and_structure": True,
            "controlled_strictly_monotonic": {
                context: all(
                    left["mean_logprob"] > right["mean_logprob"]
                    for left, right in zip(
                        sorted(
                            (
                                row
                                for row in controlled_rows
                                if row["context_variant"] == context
                            ),
                            key=lambda row: row["severity"],
                        ),
                        sorted(
                            (
                                row
                                for row in controlled_rows
                                if row["context_variant"] == context
                            ),
                            key=lambda row: row["severity"],
                        )[1:],
                    )
                )
                for context in config.context_variants
            },
        },
    )
    return {"scores": scores_path, "report": report_path}
