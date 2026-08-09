"""Cross-composition support and exact-effect consistency metrics."""

import math
from collections import Counter, defaultdict
from typing import Any

from trace_to_micro.models import TransitionEvent


def _select_events(
    events: list[TransitionEvent], state_changing_only: bool
) -> list[TransitionEvent]:
    if state_changing_only:
        return [event for event in events if event.changed_state]
    return events


def _distinct_task_support(
    events: list[TransitionEvent], key_fn
) -> dict[Any, set[str]]:
    support: dict[Any, set[str]] = defaultdict(set)
    for event in events:
        support[key_fn(event)].add(event.task_id)
    return support


def _effect_consistency(events: list[TransitionEvent]) -> dict[str, Any]:
    per_action: dict[str, list[TransitionEvent]] = defaultdict(list)
    for event in events:
        per_action[event.action_key()].append(event)

    total_observations = 0
    dominant_observations = 0
    weighted_entropy = 0.0
    ambiguous_actions = []
    for action_key, action_events in sorted(per_action.items()):
        effect_tasks: dict[Any, set[str]] = defaultdict(set)
        for event in action_events:
            effect_tasks[event.effect_key()].add(event.task_id)
        counts = [len(task_ids) for task_ids in effect_tasks.values()]
        total = sum(counts)
        probabilities = [count / total for count in counts]
        entropy = -sum(
            probability * math.log2(probability) for probability in probabilities
        )
        total_observations += total
        dominant_observations += max(counts)
        weighted_entropy += total * entropy
        if len(counts) > 1:
            ambiguous_actions.append(
                {
                    "action_key": action_key,
                    "distinct_effects": len(counts),
                    "task_support": total,
                    "dominant_effect_rate": max(counts) / total,
                    "entropy_bits": entropy,
                }
            )

    ambiguous_actions.sort(key=lambda item: (-item["task_support"], item["action_key"]))
    return {
        "action_count": len(per_action),
        "dominant_effect_rate": (
            dominant_observations / total_observations if total_observations else 0.0
        ),
        "conditional_effect_entropy_bits": (
            weighted_entropy / total_observations if total_observations else 0.0
        ),
        "ambiguous_actions": ambiguous_actions,
    }


def _coverage(
    test_events: list[TransitionEvent],
    train_support: dict[Any, set[str]],
    key_fn,
    threshold: int,
    *,
    leave_one_task_out: bool,
) -> dict[str, Any]:
    support_counts = []
    for event in test_events:
        task_ids = train_support.get(key_fn(event), set())
        if leave_one_task_out:
            task_ids = task_ids - {event.task_id}
        support_counts.append(len(task_ids))
    covered = sum(count >= threshold for count in support_counts)
    return {
        "covered": covered,
        "total": len(test_events),
        "rate": covered / len(test_events) if test_events else 0.0,
        "support_histogram": {
            str(key): value for key, value in sorted(Counter(support_counts).items())
        },
    }


def build_support_report(
    train_events: list[TransitionEvent],
    test_events: list[TransitionEvent],
    thresholds: tuple[int, ...] = (1, 3, 5),
    state_changing_only: bool = True,
    leave_one_task_out: bool = False,
) -> dict[str, Any]:
    """Measure action/effect reuse from train to held-out task compositions."""

    selected_train = _select_events(train_events, state_changing_only)
    selected_test = _select_events(test_events, state_changing_only)
    action_support = _distinct_task_support(
        selected_train, lambda event: event.action_key()
    )
    exact_effect_support = _distinct_task_support(
        selected_train, lambda event: (event.action_key(), event.effect_key())
    )
    declaration_mismatches = {
        "declared_mutating_without_change": sum(
            event.declared_mutating and not event.changed_state
            for event in train_events + test_events
        ),
        "declared_non_mutating_with_change": sum(
            not event.declared_mutating and event.changed_state
            for event in train_events + test_events
        ),
    }

    coverage = {}
    for threshold in thresholds:
        coverage[str(threshold)] = {
            "action": _coverage(
                selected_test,
                action_support,
                lambda event: event.action_key(),
                threshold,
                leave_one_task_out=leave_one_task_out,
            ),
            "exact_effect": _coverage(
                selected_test,
                exact_effect_support,
                lambda event: (event.action_key(), event.effect_key()),
                threshold,
                leave_one_task_out=leave_one_task_out,
            ),
        }

    return {
        "state_changing_only": state_changing_only,
        "leave_one_task_out": leave_one_task_out,
        "train_event_count": len(selected_train),
        "test_event_count": len(selected_test),
        "train_task_count": len({event.task_id for event in selected_train}),
        "test_task_count": len({event.task_id for event in selected_test}),
        "train_actor_counts": dict(
            sorted(Counter(event.actor for event in selected_train).items())
        ),
        "test_actor_counts": dict(
            sorted(Counter(event.actor for event in selected_test).items())
        ),
        "coverage": coverage,
        "train_effect_consistency": _effect_consistency(selected_train),
        "declaration_mismatches": declaration_mismatches,
    }
