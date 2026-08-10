"""Deterministic validation rules for LLM-generated Semantic Briefs."""

import re
from typing import Any

from trace_to_micro.clean.evidence import evidence_ids

SEMANTIC_FIELDS = (
    "user_goal",
    "active_issue",
    "situation_summary",
    "proposed_subgoal",
    "proposed_action",
    "success_evidence",
    "evidence_ids",
)
ACTION_FIELDS = ("name", "executor")
OBSERVABLE_PATTERN = re.compile(
    r"\b(?:user|tool|status|state|result|response|reports?|confirms?|returns?|shows?)\b",
    re.I,
)
ENTITY_PATTERN = re.compile(r"\b(?:[CBLPD]\d+|\d{3}-\d{3}-\d{4})\b")


def _validate_shape(brief: dict[str, Any]) -> list[dict[str, Any]]:
    errors = []
    if set(brief) != set(SEMANTIC_FIELDS):
        return [
            {
                "code": "INVALID_SCHEMA",
                "field": "semantic_brief",
                "message": f"Expected exactly {SEMANTIC_FIELDS}",
            }
        ]
    for field in SEMANTIC_FIELDS[:4] + ("success_evidence",):
        if not isinstance(brief[field], str) or not brief[field].strip():
            errors.append(
                {
                    "code": "INVALID_FIELD_TYPE",
                    "field": field,
                    "message": "Expected a non-empty string",
                }
            )
    action = brief["proposed_action"]
    if not isinstance(action, dict) or set(action) != set(ACTION_FIELDS):
        errors.append(
            {
                "code": "INVALID_ACTION_SCHEMA",
                "field": "proposed_action",
                "message": f"Expected exactly {ACTION_FIELDS}",
            }
        )
    elif action["executor"] not in {"assistant", "user", "none"} or not (
        action["name"] is None or isinstance(action["name"], str)
    ):
        errors.append(
            {
                "code": "INVALID_ACTION_VALUE",
                "field": "proposed_action",
                "message": "Invalid action name or executor",
            }
        )
    if not isinstance(brief["evidence_ids"], list) or not all(
        isinstance(item, str) for item in brief["evidence_ids"]
    ):
        errors.append(
            {
                "code": "INVALID_FIELD_TYPE",
                "field": "evidence_ids",
                "message": "Expected a list of strings",
            }
        )
    return errors


def _known_entity_values(verified_context: dict[str, Any]) -> set[str]:
    return {
        row["value"]
        for records in verified_context["entities"].values()
        for row in records
    }


def validate_semantic_brief(
    brief: dict[str, Any], verified_context: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return deterministic semantic validation errors for one brief."""

    errors = _validate_shape(brief)
    if errors:
        return errors
    available_evidence = evidence_ids(verified_context)
    unknown_evidence = sorted(set(brief["evidence_ids"]) - available_evidence)
    if not brief["evidence_ids"] or unknown_evidence:
        errors.append(
            {
                "code": "UNKNOWN_EVIDENCE",
                "field": "evidence_ids",
                "message": f"Unknown evidence IDs: {unknown_evidence}",
            }
        )

    text = " ".join(
        brief[field] for field in SEMANTIC_FIELDS[:4] + ("success_evidence",)
    )
    unknown_entities = sorted(
        set(ENTITY_PATTERN.findall(text)) - _known_entity_values(verified_context)
    )
    if unknown_entities:
        errors.append(
            {
                "code": "UNGROUNDED_ENTITY",
                "field": "semantic_brief",
                "message": f"Entities lack evidence: {unknown_entities}",
            }
        )

    action = brief["proposed_action"]
    action_name = action["name"]
    known_tools = set(verified_context["tool_ownership"]["assistant"]) | set(
        verified_context["tool_ownership"]["user"]
    )
    if action_name is not None and action_name not in known_tools:
        errors.append(
            {
                "code": "UNKNOWN_ACTION",
                "field": "proposed_action",
                "message": f"Unknown tool: {action_name}",
            }
        )
    if (action_name is None) != (action["executor"] == "none"):
        errors.append(
            {
                "code": "ACTION_EXECUTOR_MISMATCH",
                "field": "proposed_action",
                "message": "A null action must use executor 'none' and vice versa",
            }
        )

    terminal = verified_context["terminal_state"]
    if terminal["transfer_completed"] and (
        action_name is not None
        or (
            "transfer" in brief["proposed_subgoal"].lower()
            and "confirm" not in brief["proposed_subgoal"].lower()
        )
    ):
        errors.append(
            {
                "code": "SUBGOAL_ALREADY_COMPLETED",
                "field": "proposed_subgoal",
                "message": "Transfer already succeeded; only confirm it to the user",
            }
        )
    if terminal["resolution_reported"] and action_name is not None:
        errors.append(
            {
                "code": "STALE_ACTIVE_ISSUE",
                "field": "active_issue",
                "message": "The latest user message reports resolution",
            }
        )

    invalid_events = {
        event["name"]
        for event in verified_context["tool_events"]
        if event["status"] == "error" and not event["executor_valid"]
    }
    if action_name in invalid_events:
        errors.append(
            {
                "code": "REPEATS_INVALID_TOOL",
                "field": "proposed_action",
                "message": f"{action_name} already failed because the executor is invalid",
            }
        )
    if not OBSERVABLE_PATTERN.search(brief["success_evidence"]):
        errors.append(
            {
                "code": "UNOBSERVABLE_SUCCESS_CONDITION",
                "field": "success_evidence",
                "message": "Completion must be observable in a report, tool result, or state",
            }
        )
    return errors


def regenerate_fields(errors: list[dict[str, Any]]) -> list[str]:
    """Expand invalid fields to the coupled fields that must be regenerated."""

    fields = {error["field"] for error in errors}
    if "semantic_brief" in fields:
        return list(SEMANTIC_FIELDS)
    if "active_issue" in fields:
        fields.update(
            {
                "proposed_subgoal",
                "proposed_action",
                "success_evidence",
                "evidence_ids",
            }
        )
    if fields & {"proposed_subgoal", "proposed_action"}:
        fields.update(
            {"proposed_subgoal", "proposed_action", "success_evidence", "evidence_ids"}
        )
    return [field for field in SEMANTIC_FIELDS if field in fields]
