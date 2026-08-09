"""Task-ID parsing and compositional split audits."""

import re
from collections import Counter
from typing import Any

from tau2.data_model.tasks import Task
from trace_to_micro.models import TaskIdentity


TASK_ID_PATTERN = re.compile(
    r"^\[(?P<intent>[A-Za-z_]+)\]"
    r"(?P<atoms>.+)"
    r"\[PERSONA:(?P<persona>[^\]]+)\]$"
)


def parse_task_id(task_id: str) -> TaskIdentity:
    """Parse intent, atomic issues, and persona from a Telecom task ID."""

    match = TASK_ID_PATTERN.fullmatch(task_id)
    if match is None:
        raise ValueError(f"Invalid Telecom task ID: {task_id}")
    persona = match.group("persona")
    return TaskIdentity(
        intent=match.group("intent"),
        atoms=tuple(match.group("atoms").split("|")),
        persona=None if persona == "None" else persona,
    )


def _task_counts(task_ids: list[str]) -> dict[str, Any]:
    identities = [parse_task_id(task_id) for task_id in task_ids]
    atoms = Counter(atom for identity in identities for atom in identity.atoms)
    personas = Counter(
        identity.persona if identity.persona is not None else "None"
        for identity in identities
    )
    issue_counts = Counter(len(identity.atoms) for identity in identities)
    intents = Counter(identity.intent for identity in identities)
    return {
        "size": len(task_ids),
        "atom_counts": dict(sorted(atoms.items())),
        "persona_counts": dict(sorted(personas.items())),
        "issue_count_histogram": {
            str(key): value for key, value in sorted(issue_counts.items())
        },
        "intent_counts": dict(sorted(intents.items())),
    }


def _initialized_users(tasks: list[Task]) -> list[dict[str, Any]]:
    users: set[tuple[str, str]] = set()
    for task in tasks:
        initial_state = task.initial_state
        if initial_state is None or initial_state.initialization_actions is None:
            continue
        for action in initial_state.initialization_actions:
            if action.func_name == "set_user_info":
                users.add(
                    (
                        action.arguments["name"],
                        action.arguments["phone_number"],
                    )
                )
    return [
        {"name": name, "phone_number": phone_number}
        for name, phone_number in sorted(users)
    ]


def build_task_inventory(
    tasks: list[Task],
    split_map: dict[str, list[str]],
    train_split: str,
    test_split: str,
) -> dict[str, Any]:
    """Summarize the LOCO split and the benchmark's user/text diversity."""

    train_ids = split_map[train_split]
    test_ids = split_map[test_split]
    known_task_ids = {task.id for task in tasks}
    requested_ids = set(train_ids) | set(test_ids)
    missing_ids = sorted(requested_ids - known_task_ids)
    if missing_ids:
        raise ValueError(f"Split contains unknown task IDs: {missing_ids}")

    train_summary = _task_counts(train_ids)
    test_summary = _task_counts(test_ids)
    train_atoms = set(train_summary["atom_counts"])
    test_atoms = set(test_summary["atom_counts"])

    return {
        "split_sizes": {
            name: len(task_ids) for name, task_ids in sorted(split_map.items())
        },
        "train_split": train_split,
        "test_split": test_split,
        "exact_task_overlap": sorted(set(train_ids) & set(test_ids)),
        "shared_atoms": sorted(train_atoms & test_atoms),
        "train_only_atoms": sorted(train_atoms - test_atoms),
        "test_only_atoms": sorted(test_atoms - train_atoms),
        "splits": {
            train_split: train_summary,
            test_split: test_summary,
        },
        "unique_initialized_users": _initialized_users(tasks),
        "unique_ticket_count": len(
            {task.ticket for task in tasks if task.ticket is not None}
        ),
    }
