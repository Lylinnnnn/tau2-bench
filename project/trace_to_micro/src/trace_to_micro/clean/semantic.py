"""LLM semantic brief generation with deterministic validation and bounded retry."""

import json
from typing import Any, Callable

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import extract_json_from_llm_response, generate
from trace_to_micro.clean.contract import compile_action_contract
from trace_to_micro.clean.evidence import build_verified_context
from trace_to_micro.clean.validation import (
    SEMANTIC_FIELDS,
    validate_semantic_brief,
)
from trace_to_micro.clean.validation import (
    regenerate_fields as select_regenerate_fields,
)

MAX_SEMANTIC_RETRIES = 2
MAX_SEMANTIC_ATTEMPTS = 1 + MAX_SEMANTIC_RETRIES
RESPONSE_EXCERPT_CHARS = 1_000

SEMANTIC_INSTRUCTION = """
Create a concise semantic brief from the verified evidence below. The evidence
contains only information available before the next agent decision.

Return exactly one JSON object with these fields:
- user_goal: string
- active_issue: string
- situation_summary: string
- proposed_subgoal: one immediate next subtask as a string
- proposed_action: {"name": string or null, "executor": "assistant", "user", or "none"}
- success_evidence: an observable completion condition as a string
- evidence_ids: a non-empty list of evidence IDs supporting the brief

Do not invent entity identifiers, facts, or tool names. Prefer the latest
evidence over stale statements. Do not repeat a completed action. A user-owned
device action may be a valid subgoal, but its executor must be "user". Use a
null action and executor "none" when the next step should only be a message.
""".strip()


def _response_metadata(response: AssistantMessage) -> dict[str, Any]:
    raw_data = response.raw_data or {}
    choices = raw_data.get("choices") or []
    return {
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "usage": response.usage,
        "generation_time_seconds": response.generation_time_seconds,
        "response_chars": len(response.content or ""),
    }


def _parse_object(response: AssistantMessage) -> dict[str, Any]:
    if response.content is None:
        raise TypeError("Semantic builder returned no text content")
    value = json.loads(extract_json_from_llm_response(response.content))
    if not isinstance(value, dict):
        raise TypeError("Semantic builder response must be a JSON object")
    return value


def _build_request(
    *,
    verified_context: dict[str, Any],
    prior_brief: dict[str, Any] | None,
    errors: list[dict[str, Any]],
    regenerate_fields: list[str],
) -> UserMessage:
    if prior_brief is None:
        instruction = "Return the complete semantic brief."
    else:
        instruction = (
            "Return only one JSON object containing these fields to regenerate: "
            f"{regenerate_fields}. Preserve all other fields from the prior brief.\n"
            f"<prior_brief>{json.dumps(prior_brief, ensure_ascii=False)}</prior_brief>\n"
            f"<validation_errors>{json.dumps(errors, ensure_ascii=False)}</validation_errors>"
        )
    return UserMessage(
        role="user",
        content=(
            f"{instruction}\n<verified_context>"
            f"{json.dumps(verified_context, ensure_ascii=False)}"
            "</verified_context>"
        ),
    )


def _fallback_brief(verified_context: dict[str, Any]) -> dict[str, Any]:
    latest = next(
        (
            row
            for row in reversed(verified_context["recent_dialogue"])
            if row["role"] == "user"
        ),
        None,
    )
    summary = latest["content"] if latest else "No user message is available."
    return {
        "user_goal": summary,
        "active_issue": "Unable to determine safely from verified evidence.",
        "situation_summary": summary,
        "proposed_subgoal": "Respond conservatively to the latest user request.",
        "proposed_action": {"name": None, "executor": "none"},
        "success_evidence": "The user confirms the next required information or outcome.",
        "evidence_ids": [latest["evidence_id"]] if latest else [],
    }


def build_hybrid_clean_context(
    *,
    model: str,
    llm_args: dict[str, Any],
    prefix: list[Message],
    agent_tools: list[Tool],
    user_tools: list[Tool],
    generate_fn: Callable[..., AssistantMessage | UserMessage] = generate,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build verified evidence, a gated brief, and an action contract."""

    verified = build_verified_context(
        prefix=prefix,
        agent_tools=agent_tools,
        user_tools=user_tools,
    )
    prior_brief = None
    errors: list[dict[str, Any]] = []
    regenerate_fields = list(SEMANTIC_FIELDS)
    attempts = []
    final_brief = None

    for attempt in range(1, MAX_SEMANTIC_ATTEMPTS + 1):
        request = _build_request(
            verified_context=verified,
            prior_brief=prior_brief,
            errors=errors,
            regenerate_fields=regenerate_fields,
        )
        response = generate_fn(
            model=model,
            messages=[
                SystemMessage(role="system", content=SEMANTIC_INSTRUCTION),
                request,
            ],
            call_name="trace_to_micro_semantic_brief",
            **llm_args,
        )
        if not isinstance(response, AssistantMessage):
            raise TypeError("Semantic builder must return an AssistantMessage")
        metadata = {"attempt": attempt, **_response_metadata(response)}
        try:
            patch = _parse_object(response)
            if prior_brief is None:
                candidate = patch
            else:
                if set(patch) == set(SEMANTIC_FIELDS):
                    candidate = patch
                elif set(patch) == set(regenerate_fields):
                    candidate = {**prior_brief, **patch}
                else:
                    raise ValueError(
                        f"Retry must return exactly {regenerate_fields}; got {tuple(patch)}"
                    )
            errors = validate_semantic_brief(candidate, verified)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            errors = [
                {
                    "code": "INVALID_JSON_OR_SCHEMA",
                    "field": "semantic_brief",
                    "message": str(error),
                }
            ]
            candidate = prior_brief
        if errors:
            content = response.content or ""
            metadata.update(
                {
                    "response_head": content[:RESPONSE_EXCERPT_CHARS],
                    "response_tail": content[-RESPONSE_EXCERPT_CHARS:],
                }
            )
        attempts.append({**metadata, "validation_errors": errors})
        if not errors:
            final_brief = candidate
            break
        prior_brief = candidate
        regenerate_fields = select_regenerate_fields(errors)

    status = "passed"
    if final_brief is None:
        final_brief = _fallback_brief(verified)
        status = "rule_fallback"
    contract = compile_action_contract(
        semantic_brief=final_brief,
        verified_context=verified,
        agent_tools=agent_tools,
    )
    training_eligible = status == "passed" and not verified["quality_flags"]
    context = {
        "schema_version": "hybrid_clean_v1",
        "verified_context": verified,
        "semantic_brief": final_brief,
        "action_contract": contract,
        "training_eligible": training_eligible,
    }
    return context, {
        "status": status,
        "attempt_count": len(attempts),
        "retry_count": max(0, len(attempts) - 1),
        "attempts": attempts,
    }
