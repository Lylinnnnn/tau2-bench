"""Score clean and perturbed tool results under identity-consistent contexts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from trace_to_micro.config import ExpectationDeviationConfig
from trace_to_micro.evaluation.deviation_metrics import contextual_min_k_statistics
from trace_to_micro.evaluation.expectation_deviation import consistently_anonymize
from trace_to_micro.runner.expectation_matching import messages_for_context
from trace_to_micro.runtime.activations import score_chat_suffix


def anonymize_scoring_bundle(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_call_id: str,
    contents: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, list[str], dict[str, int]]:
    """Apply one identity mapping to a prefix and every candidate result."""

    parsed_contents = []
    content_is_json = []
    for content in contents:
        try:
            parsed_contents.append(json.loads(content))
            content_is_json.append(True)
        except json.JSONDecodeError:
            parsed_contents.append(content)
            content_is_json.append(False)
    values, counts = consistently_anonymize(
        [messages, tools, {"tool_call_id": tool_call_id}, *parsed_contents]
    )
    masked_messages, masked_tools, masked_link, *masked_contents = values
    serialized = [
        json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        if is_json
        else value
        for value, is_json in zip(masked_contents, content_is_json)
    ]
    return (
        masked_messages,
        masked_tools,
        masked_link["tool_call_id"],
        serialized,
        counts,
    )


def anonymize_context_scoring_bundle(
    messages_by_context: dict[str, list[dict[str, Any]]],
    tools: list[dict[str, Any]],
    tool_call_id: str,
    contents: list[str],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    str,
    list[str],
    dict[str, int],
]:
    """Apply one identity mapping across contexts and candidate results."""

    parsed_contents = []
    content_is_json = []
    for content in contents:
        try:
            parsed_contents.append(json.loads(content))
            content_is_json.append(True)
        except json.JSONDecodeError:
            parsed_contents.append(content)
            content_is_json.append(False)
    context_names = list(messages_by_context)
    values, counts = consistently_anonymize(
        [
            *(messages_by_context[name] for name in context_names),
            tools,
            {"tool_call_id": tool_call_id},
            *parsed_contents,
        ]
    )
    context_count = len(context_names)
    masked_contexts = {name: values[index] for index, name in enumerate(context_names)}
    masked_tools = values[context_count]
    masked_link = values[context_count + 1]
    masked_contents = values[context_count + 2 :]
    serialized = [
        json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        if is_json
        else value
        for value, is_json in zip(masked_contents, content_is_json)
    ]
    return (
        masked_contexts,
        masked_tools,
        masked_link["tool_call_id"],
        serialized,
        counts,
    )


def _score_fingerprint(
    *, query: dict[str, Any], model: str, context_variants: tuple[str, ...]
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "query": query,
                "model": model,
                "context_variants": context_variants,
                "scoring": "consistent anonymization and mean tool-result logprob",
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _contextual_min_k_fingerprint(
    *, query: dict[str, Any], model: str, fraction: float
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "query": query,
                "model": model,
                "contexts": ["full", "action_only"],
                "min_k_fraction": fraction,
                "scoring": "contextual token likelihood-ratio upper tail",
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _score_contextual_pair(
    *,
    messages_by_context: dict[str, list[dict[str, Any]]],
    content: str,
    tool_call_id: str,
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
    config: ExpectationDeviationConfig,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    scores = {}
    for context in ("full", "action_only"):
        scores[context] = score_chat_suffix(
            prefix_messages=messages_by_context[context],
            suffix_message={
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call_id,
            },
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
            base_url=base_url,
            api_key=api_key,
            model=config.scoring_model,
            include_token_details=True,
        )
    statistics = contextual_min_k_statistics(
        scores["full"], scores["action_only"], fraction=config.min_k_fraction
    )
    return {
        "full_prefix_token_count": scores["full"]["prefix_token_count"],
        "action_only_prefix_token_count": scores["action_only"]["prefix_token_count"],
        "full_sum_logprob": scores["full"]["sum_logprob"],
        "action_only_sum_logprob": scores["action_only"]["sum_logprob"],
        "full_mean_logprob": scores["full"]["mean_logprob"],
        "action_only_mean_logprob": scores["action_only"]["mean_logprob"],
        "min_k_fraction": config.min_k_fraction,
        **statistics,
    }


def score_contextual_min_k_query(
    query: dict[str, Any],
    records: dict[str, dict[str, Any]],
    *,
    config: ExpectationDeviationConfig,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    """Score one controlled Test query with a contextual token-tail probe."""

    source = records[query["source_decision_id"]]
    raw_contexts = {
        context: messages_for_context(query, records, context)
        for context in ("full", "action_only")
    }
    contents = [variant["result_content"] for variant in query["variants"]]
    contexts, tools, tool_call_id, anonymized, counts = (
        anonymize_context_scoring_bundle(
            raw_contexts, source["tools"], source["tool_call_id"], contents
        )
    )
    fingerprint = _contextual_min_k_fingerprint(
        query=query, model=config.scoring_model, fraction=config.min_k_fraction
    )
    rows = []
    for variant, content in zip(query["variants"], anonymized):
        rows.append(
            {
                "track": "min_k_controlled",
                "query_id": query["query_id"],
                "query_fingerprint": query["query_fingerprint"],
                "score_fingerprint": fingerprint,
                "domain": query["domain"],
                "split": query["split"],
                "task_id": query["task_id"],
                "simulation_id": query["simulation_id"],
                "trial": query["trial"],
                "tool_name": query["tool_name"],
                "severity": variant["severity"],
                "edit_count": len(variant["edits"]),
                "edited_paths": [row["path_string"] for row in variant["edits"]],
                "character_length_delta": variant["character_length_delta"],
                "anonymized_identifier_count": sum(counts.values()),
                **_score_contextual_pair(
                    messages_by_context=contexts,
                    content=content,
                    tool_call_id=tool_call_id,
                    tools=tools,
                    chat_template_kwargs=source["chat_template_kwargs"],
                    config=config,
                    base_url=base_url,
                    api_key=api_key,
                ),
            }
        )
    return rows


def score_contextual_min_k_calibration(
    record: dict[str, Any],
    records: dict[str, dict[str, Any]],
    *,
    config: ExpectationDeviationConfig,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    """Score one official Train clean return for Min-K calibration."""

    query = {"query_id": record["decision_id"]}
    raw_contexts = {
        context: messages_for_context(query, records, context)
        for context in ("full", "action_only")
    }
    contexts, tools, tool_call_id, contents, counts = anonymize_context_scoring_bundle(
        raw_contexts,
        record["tools"],
        record["tool_call_id"],
        [record["result_content"]],
    )
    return {
        "track": "min_k_calibration",
        "query_id": record["decision_id"],
        "domain": record["domain"],
        "split": record["split"],
        "task_id": record["task_id"],
        "simulation_id": record["simulation_id"],
        "trial": record["trial"],
        "tool_name": record["tool_name"],
        "severity": 0,
        "edit_count": 0,
        "edited_paths": [],
        "character_length_delta": 0,
        "anonymized_identifier_count": sum(counts.values()),
        **_score_contextual_pair(
            messages_by_context=contexts,
            content=contents[0],
            tool_call_id=tool_call_id,
            tools=tools,
            chat_template_kwargs=record["chat_template_kwargs"],
            config=config,
            base_url=base_url,
            api_key=api_key,
        ),
    }


def score_controlled_query(
    query: dict[str, Any],
    records: dict[str, dict[str, Any]],
    *,
    config: ExpectationDeviationConfig,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    """Score one clean result and its cumulative anomaly variants."""

    source = records[query["source_decision_id"]]
    rows = []
    fingerprint = _score_fingerprint(
        query=query,
        model=config.scoring_model,
        context_variants=config.context_variants,
    )
    for context in config.context_variants:
        raw_messages = messages_for_context(query, records, context)
        contents = [variant["result_content"] for variant in query["variants"]]
        messages, tools, tool_call_id, anonymized, counts = anonymize_scoring_bundle(
            raw_messages, source["tools"], source["tool_call_id"], contents
        )
        for variant, content in zip(query["variants"], anonymized):
            score = score_chat_suffix(
                prefix_messages=messages,
                suffix_message={
                    "role": "tool",
                    "content": content,
                    "tool_call_id": tool_call_id,
                },
                tools=tools,
                chat_template_kwargs=source["chat_template_kwargs"],
                base_url=base_url,
                api_key=api_key,
                model=config.scoring_model,
            )
            rows.append(
                {
                    "track": "controlled_anomaly",
                    "query_id": query["query_id"],
                    "query_fingerprint": query["query_fingerprint"],
                    "score_fingerprint": fingerprint,
                    "domain": query["domain"],
                    "split": query["split"],
                    "task_id": query["task_id"],
                    "simulation_id": query["simulation_id"],
                    "trial": query["trial"],
                    "tool_name": query["tool_name"],
                    "structure_key": query["structure_key"],
                    "context_variant": context,
                    "severity": variant["severity"],
                    "edit_count": len(variant["edits"]),
                    "edited_paths": [row["path_string"] for row in variant["edits"]],
                    "character_length_delta": variant["character_length_delta"],
                    "anonymized_identifier_count": sum(counts.values()),
                    **score,
                }
            )
    return rows


def score_calibration_record(
    record: dict[str, Any],
    shuffled_id: str,
    records: dict[str, dict[str, Any]],
    *,
    config: ExpectationDeviationConfig,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    """Score one Train-clean result for offline calibration."""

    query = {
        "query_id": record["decision_id"],
        "shuffled_source_decision_id": shuffled_id,
    }
    rows = []
    for context in config.context_variants:
        messages, tools, tool_call_id, contents, counts = anonymize_scoring_bundle(
            messages_for_context(query, records, context),
            record["tools"],
            record["tool_call_id"],
            [record["result_content"]],
        )
        score = score_chat_suffix(
            prefix_messages=messages,
            suffix_message={
                "role": "tool",
                "content": contents[0],
                "tool_call_id": tool_call_id,
            },
            tools=tools,
            chat_template_kwargs=record["chat_template_kwargs"],
            base_url=base_url,
            api_key=api_key,
            model=config.scoring_model,
        )
        rows.append(
            {
                "track": "calibration_clean",
                "query_id": record["decision_id"],
                "domain": record["domain"],
                "split": record["split"],
                "task_id": record["task_id"],
                "simulation_id": record["simulation_id"],
                "trial": record["trial"],
                "tool_name": record["tool_name"],
                "structure_key": hashlib.sha256(
                    json.dumps(
                        record["result_structure"], separators=(",", ":")
                    ).encode()
                ).hexdigest()[:16],
                "context_variant": context,
                "severity": 0,
                "edit_count": 0,
                "edited_paths": [],
                "character_length_delta": 0,
                "anonymized_identifier_count": sum(counts.values()),
                **score,
            }
        )
    return rows


def score_rematch_query(
    query: dict[str, Any],
    records: dict[str, dict[str, Any]],
    *,
    config: ExpectationDeviationConfig,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    """Score same-tool factual candidates without collapsing their identities."""

    source = records[query["query_id"]]
    rows = []
    candidates = [records[value] for value in query["candidate_decision_ids"]]
    for context in config.context_variants:
        messages, tools, tool_call_id, contents, counts = anonymize_scoring_bundle(
            messages_for_context(query, records, context),
            source["tools"],
            source["tool_call_id"],
            [row["result_content"] for row in candidates],
        )
        if len(set(contents)) != len(contents):
            raise ValueError(
                f"Consistent anonymization collapsed rematch candidates: {query['query_id']}"
            )
        for candidate, content in zip(candidates, contents):
            score = score_chat_suffix(
                prefix_messages=messages,
                suffix_message={
                    "role": "tool",
                    "content": content,
                    "tool_call_id": tool_call_id,
                },
                tools=tools,
                chat_template_kwargs=source["chat_template_kwargs"],
                base_url=base_url,
                api_key=api_key,
                model=config.scoring_model,
            )
            rows.append(
                {
                    "track": "consistent_rematch",
                    "query_id": query["query_id"],
                    "domain": query["domain"],
                    "split": query["split"],
                    "task_id": query["task_id"],
                    "simulation_id": query["simulation_id"],
                    "trial": query["trial"],
                    "tool_name": query["tool_name"],
                    "context_variant": context,
                    "candidate_decision_id": candidate["decision_id"],
                    "is_positive": candidate["decision_id"]
                    == query["positive_decision_id"],
                    "anonymized_identifier_count": sum(counts.values()),
                    **score,
                }
            )
    return rows
