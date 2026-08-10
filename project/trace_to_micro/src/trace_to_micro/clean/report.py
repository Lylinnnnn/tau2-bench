"""Aggregate diagnostics emitted by the hybrid clean builder."""

from collections import Counter
from typing import Any


def build_hybrid_context_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize retry, fallback, repair, and quality-gate behavior."""

    status = Counter()
    retries = Counter()
    validation_errors = Counter()
    repairs = Counter()
    phases = Counter()
    quality_flags = Counter()
    eligible = 0
    for row in rows:
        context = row["context"]
        metadata = row["builder_metadata"]
        status[metadata["status"]] += 1
        retries[str(metadata["retry_count"])] += 1
        phases[context["verified_context"]["interaction_phase"]] += 1
        repairs.update(context["action_contract"]["repairs"])
        quality_flags.update(context["verified_context"]["quality_flags"])
        eligible += int(context["training_eligible"])
        for attempt in metadata["attempts"]:
            validation_errors.update(
                error["code"] for error in attempt["validation_errors"]
            )
    count = len(rows)
    return {
        "context_count": count,
        "training_eligible_count": eligible,
        "training_eligible_rate": eligible / count if count else None,
        "status_counts": dict(sorted(status.items())),
        "retry_count_histogram": dict(sorted(retries.items())),
        "validation_error_counts": dict(sorted(validation_errors.items())),
        "action_contract_repair_counts": dict(sorted(repairs.items())),
        "interaction_phase_counts": dict(sorted(phases.items())),
        "quality_flag_counts": dict(sorted(quality_flags.items())),
        "max_semantic_retries": 2,
    }
