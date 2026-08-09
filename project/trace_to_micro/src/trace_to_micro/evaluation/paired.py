"""Scoring and aggregation for paired context-format probes."""

import json
from collections import defaultdict
from typing import Any

from tau2.data_model.tasks import Task
from trace_to_micro.state_diff import canonicalize_value

RATE_METRICS = (
    "assistant_kind_match",
    "assistant_exact_match",
    "macro_tool_name_match",
    "macro_tool_arguments_match",
    "effect_match",
    "stateful_effect_match",
    "gold_macro_action_match",
    "tool_error",
    "mutation_noop",
)
LOWER_IS_BETTER = {"tool_error", "mutation_noop"}


def _macro_calls(branch: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for action in (branch["assistant_action"], branch["user_continuation"]):
        if action is not None and action["kind"] == "tool":
            calls.extend(action["calls"])
    return calls


def _call_names(calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(call["requestor"], call["name"]) for call in calls]


def _call_signatures(calls: list[dict[str, Any]]) -> list[str]:
    return [
        json.dumps(
            {
                "requestor": call["requestor"],
                "name": call["name"],
                "arguments": canonicalize_value(call["arguments"]),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        for call in calls
    ]


def _gold_signatures(task: Task) -> set[str]:
    if task.evaluation_criteria is None:
        return set()
    return {
        json.dumps(
            {
                "requestor": action.requestor,
                "name": action.name,
                "arguments": canonicalize_value(action.arguments),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        for action in (task.evaluation_criteria.actions or [])
    }


def score_branch(
    *,
    task: Task,
    actual: dict[str, Any],
    predicted: dict[str, Any],
) -> dict[str, bool | None]:
    """Compare a counterfactual macro step with the logged macro step."""

    actual_assistant = actual["assistant_action"]
    predicted_assistant = predicted["assistant_action"]
    actual_calls = _macro_calls(actual)
    predicted_calls = _macro_calls(predicted)
    gold = _gold_signatures(task)
    actual_changes = actual["changes"]
    predicted_changes = predicted["changes"]
    return {
        "assistant_kind_match": (
            predicted_assistant["kind"] == actual_assistant["kind"]
        ),
        "assistant_exact_match": predicted_assistant == actual_assistant,
        "macro_tool_name_match": (
            _call_names(predicted_calls) == _call_names(actual_calls)
            if actual_calls
            else None
        ),
        "macro_tool_arguments_match": (
            _call_signatures(predicted_calls) == _call_signatures(actual_calls)
            if actual_calls
            else None
        ),
        "effect_match": predicted_changes == actual_changes,
        "stateful_effect_match": (
            predicted_changes == actual_changes if actual_changes else None
        ),
        "gold_macro_action_match": (
            any(signature in gold for signature in _call_signatures(predicted_calls))
            if gold and predicted_calls
            else None
        ),
        "tool_error": predicted["tool_error"],
        "mutation_noop": predicted["mutation_noop"],
    }


def _rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(rows)}
    for metric in RATE_METRICS:
        values = [row["metrics"][metric] for row in rows]
        observed = [value for value in values if value is not None]
        result[metric] = {
            "count": len(observed),
            "rate": (
                sum(bool(value) for value in observed) / len(observed)
                if observed
                else None
            ),
        }
    prompt_tokens = [
        row["prediction_usage"]["prompt_tokens"]
        for row in rows
        if row.get("prediction_usage")
        and row["prediction_usage"].get("prompt_tokens") is not None
    ]
    result["mean_prompt_tokens"] = (
        sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None
    )
    return result


def _paired_delta(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    baseline: str,
    alternative: str,
) -> dict[str, Any]:
    snapshot_ids = sorted(
        snapshot_id
        for snapshot_id, variant in rows_by_key
        if variant == baseline and (snapshot_id, alternative) in rows_by_key
    )
    result: dict[str, Any] = {"paired_count": len(snapshot_ids), "metrics": {}}
    for metric in RATE_METRICS:
        pairs = [
            (
                rows_by_key[(snapshot_id, baseline)]["metrics"][metric],
                rows_by_key[(snapshot_id, alternative)]["metrics"][metric],
            )
            for snapshot_id in snapshot_ids
        ]
        pairs = [
            (left, right)
            for left, right in pairs
            if left is not None and right is not None
        ]
        if metric in LOWER_IS_BETTER:
            improved = sum(left and not right for left, right in pairs)
            regressed = sum(not left and right for left, right in pairs)
        else:
            improved = sum(not left and right for left, right in pairs)
            regressed = sum(left and not right for left, right in pairs)
        result["metrics"][metric] = {
            "count": len(pairs),
            "alternative_minus_baseline": (
                sum(int(right) - int(left) for left, right in pairs) / len(pairs)
                if pairs
                else None
            ),
            "improved": improved,
            "regressed": regressed,
        }
    return result


def build_paired_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate variants, trajectory positions, and paired deltas."""

    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_variant_position: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_variant_context: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rows_by_key = {}
    for row in rows:
        variant = row["variant"]
        by_variant[variant].append(row)
        by_variant_position[(variant, row["position_bucket"])].append(row)
        by_variant_context[(variant, row["context_length_bucket"])].append(row)
        rows_by_key[(row["snapshot_id"], variant)] = row
    comparisons = {}
    for baseline, alternative in (
        ("long_raw", "structured_state"),
        ("structured_state", "clean_subtask"),
        ("long_raw", "clean_subtask"),
    ):
        comparison = _paired_delta(rows_by_key, baseline, alternative)
        comparison["by_position"] = {
            position: _paired_delta(
                {
                    key: row
                    for key, row in rows_by_key.items()
                    if row["position_bucket"] == position
                },
                baseline,
                alternative,
            )
            for position in ("early", "middle", "late")
        }
        comparison["by_context_length"] = {
            context_length: _paired_delta(
                {
                    key: row
                    for key, row in rows_by_key.items()
                    if row["context_length_bucket"] == context_length
                },
                baseline,
                alternative,
            )
            for context_length in ("short", "medium", "long")
        }
        comparisons[f"{alternative}_vs_{baseline}"] = comparison
    return {
        "prediction_count": len(rows),
        "unique_snapshots": len({row["snapshot_id"] for row in rows}),
        "by_variant": {
            variant: _rates(variant_rows)
            for variant, variant_rows in sorted(by_variant.items())
        },
        "by_variant_and_position": {
            f"{variant}:{position}": _rates(group)
            for (variant, position), group in sorted(by_variant_position.items())
        },
        "by_variant_and_context_length": {
            f"{variant}:{context_length}": _rates(group)
            for (variant, context_length), group in sorted(by_variant_context.items())
        },
        "paired_comparisons": comparisons,
        "metric_scope": {
            "assistant_exact_match": "agreement with logged behavior, not correctness",
            "gold_macro_action_match": "evaluation-only membership in task reference actions",
            "effect_match": "exact one-macro-step state delta match",
            "stateful_effect_match": "effect match only where logged macro step changed state",
        },
    }
