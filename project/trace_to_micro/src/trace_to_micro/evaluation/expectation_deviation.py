"""Consistent anonymization and controlled expected-result deviations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

from trace_to_micro.evaluation.expectation_matching import result_structure

_IDENTIFIER = re.compile(
    r"(?P<CALL>chatcmpl-tool-[A-Za-z0-9]+)"
    r"|(?P<EMAIL>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    r"|(?P<HASH>#[A-Za-z0-9_-]+)"
    r"|(?P<USER>\b[A-Za-z]+(?:_[A-Za-z]+)+_\d+\b)"
    r"|(?P<CODE>\b(?=[A-Z0-9]{5,}\b)(?=[A-Z0-9]*[A-Z])"
    r"(?=[A-Z0-9]*\d)[A-Z0-9]+\b)"
    r"|(?P<UPPER>\b[A-Z]{5,8}\b)"
    r"|(?P<NUMBER>\b\d{5,}\b)"
)


def _walk_strings(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def _identifier_map(values: list[Any]) -> dict[str, str]:
    typed = {}
    for value in values:
        for text in _walk_strings(value):
            for match in _IDENTIFIER.finditer(text):
                typed[match.group()] = str(match.lastgroup)
    counters: Counter[str] = Counter()
    mapping = {}
    for raw, kind in sorted(typed.items(), key=lambda item: (item[1], item[0])):
        counters[kind] += 1
        mapping[raw] = f"<{kind}_{counters[kind]:03d}>"
    return mapping


def _replace_identifiers(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            _IDENTIFIER.sub(lambda match: mapping[match.group()], str(key)): (
                _replace_identifiers(item, mapping)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_identifiers(item, mapping) for item in value]
    if isinstance(value, str):
        return _IDENTIFIER.sub(lambda match: mapping[match.group()], value)
    return value


def consistently_anonymize(values: list[Any]) -> tuple[list[Any], dict[str, int]]:
    """Anonymize a bundle while preserving equal and unequal entity identities."""

    mapping = _identifier_map(values)
    transformed = [_replace_identifiers(value, mapping) for value in values]
    kinds = Counter(alias.split("_", 1)[0][1:] for alias in mapping.values())
    return transformed, dict(sorted(kinds.items()))


def _scalar_leaves(value: Any, path: tuple[Any, ...] = ()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _scalar_leaves(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _scalar_leaves(item, (*path, index))
    elif value is not None:
        yield path, value


def _set_path(value: Any, path: tuple[Any, ...], replacement: Any) -> None:
    target = value
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement


def _path_string(path: tuple[Any, ...]) -> str:
    output = "$"
    for part in path:
        output += f"[{part}]" if isinstance(part, int) else f".{part}"
    return output


def _grounded(value: Any, serialized_messages: str) -> bool:
    if isinstance(value, str):
        needle = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        needle = str(value)
    else:
        return False
    return len(needle) >= 3 and needle.lower() in serialized_messages.lower()


def _structure_key(content: str) -> str:
    paths = result_structure(content)
    return hashlib.sha256(
        json.dumps(paths, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _record_fingerprint(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            record, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _candidate_edits(
    source: dict[str, Any], donors: list[dict[str, Any]], *, random_seed: int
) -> list[dict[str, Any]]:
    try:
        source_value = json.loads(source["result_content"])
    except json.JSONDecodeError:
        return []
    if not isinstance(source_value, (dict, list)):
        return []
    donor_values = []
    for donor in donors:
        try:
            value = json.loads(donor["result_content"])
        except json.JSONDecodeError:
            continue
        donor_values.append((donor, dict(_scalar_leaves(value))))
    messages = json.dumps(source["messages"], ensure_ascii=False, sort_keys=True)
    edits = []
    for path, old_value in _scalar_leaves(source_value):
        if not _grounded(old_value, messages):
            continue
        choices = []
        for donor, leaves in donor_values:
            new_value = leaves.get(path)
            if (
                new_value is None
                or type(new_value) is not type(old_value)
                or new_value == old_value
                or _grounded(new_value, messages)
            ):
                continue
            length_gap = abs(
                len(json.dumps(new_value, ensure_ascii=False))
                - len(json.dumps(old_value, ensure_ascii=False))
            )
            tie = hashlib.sha256(
                (
                    f"{random_seed}:{source['decision_id']}:{path}:"
                    f"{donor['decision_id']}"
                ).encode()
            ).hexdigest()
            choices.append((length_gap, tie, donor, new_value))
        if not choices:
            continue
        length_gap, tie, donor, new_value = min(choices)
        edits.append(
            {
                "path": list(path),
                "path_string": _path_string(path),
                "old_value": old_value,
                "new_value": new_value,
                "donor_decision_id": donor["decision_id"],
                "donor_task_id": donor["task_id"],
                "length_gap": length_gap,
                "tie": tie,
            }
        )
    edits.sort(key=lambda row: (row["length_gap"], row["tie"], row["path_string"]))
    return edits


def build_controlled_anomaly_queries(
    records: list[dict[str, Any]],
    *,
    train_split: str,
    test_split: str,
    severity_levels: tuple[int, ...],
    max_length_delta_ratio: float,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct cumulative, same-schema corruptions using Train donor values."""

    if not severity_levels or severity_levels != tuple(
        range(1, max(severity_levels) + 1)
    ):
        raise ValueError("Severity levels must be consecutive and start at one")
    train_by_tool: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    test_by_tool: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record["domain"], record["tool_name"])
        if record["split"] == train_split:
            train_by_tool[key].append(record)
        elif record["split"] == test_split:
            test_by_tool[key].append(record)
    queries = []
    exclusions = []
    maximum = max(severity_levels)
    for source in sorted(
        (row for row in records if row["split"] == test_split),
        key=lambda row: row["decision_id"],
    ):
        key = (source["domain"], source["tool_name"])
        edits = _candidate_edits(source, train_by_tool[key], random_seed=random_seed)
        if len(edits) < maximum:
            exclusions.append(
                {
                    "decision_id": source["decision_id"],
                    "domain": source["domain"],
                    "task_id": source["task_id"],
                    "tool_name": source["tool_name"],
                    "reason": "insufficient_grounded_same_path_train_donors",
                    "available_edit_count": len(edits),
                }
            )
            continue
        clean_value = json.loads(source["result_content"])
        clean_length = len(source["result_content"])
        tolerance = max(8, math.ceil(clean_length * max_length_delta_ratio))
        variants = [
            {
                "severity": 0,
                "result_content": source["result_content"],
                "edits": [],
                "character_length": clean_length,
                "character_length_delta": 0,
            }
        ]
        corrupted = deepcopy(clean_value)
        selected_edits = []
        valid = True
        for severity in severity_levels:
            edit = edits[severity - 1]
            _set_path(corrupted, tuple(edit["path"]), edit["new_value"])
            selected_edits.append(
                {key: value for key, value in edit.items() if key != "tie"}
            )
            content = json.dumps(corrupted, ensure_ascii=False, separators=(", ", ": "))
            delta = len(content) - clean_length
            if abs(delta) > tolerance:
                valid = False
                break
            if result_structure(content) != source["result_structure"]:
                raise ValueError("Controlled corruption changed the result schema")
            variants.append(
                {
                    "severity": severity,
                    "result_content": content,
                    "edits": deepcopy(selected_edits),
                    "character_length": len(content),
                    "character_length_delta": delta,
                }
            )
        if not valid:
            exclusions.append(
                {
                    "decision_id": source["decision_id"],
                    "domain": source["domain"],
                    "task_id": source["task_id"],
                    "tool_name": source["tool_name"],
                    "reason": "controlled_corruption_exceeded_length_tolerance",
                    "length_tolerance": tolerance,
                }
            )
            continue
        shuffled = sorted(
            (row for row in test_by_tool[key] if row["task_id"] != source["task_id"]),
            key=lambda row: row["decision_id"],
        )
        if not shuffled:
            exclusions.append(
                {
                    "decision_id": source["decision_id"],
                    "domain": source["domain"],
                    "task_id": source["task_id"],
                    "tool_name": source["tool_name"],
                    "reason": "no_cross_task_shuffled_context",
                }
            )
            continue
        offset = int(
            hashlib.sha256(
                f"{random_seed}:{source['decision_id']}:shuffled".encode()
            ).hexdigest(),
            16,
        ) % len(shuffled)
        query = {
            "query_id": source["decision_id"],
            "domain": source["domain"],
            "split": source["split"],
            "task_id": source["task_id"],
            "simulation_id": source["simulation_id"],
            "trial": source["trial"],
            "tool_name": source["tool_name"],
            "structure_key": _structure_key(source["result_content"]),
            "source_decision_id": source["decision_id"],
            "shuffled_source_decision_id": shuffled[offset]["decision_id"],
            "variants": variants,
        }
        query["record_fingerprints"] = {
            source["decision_id"]: _record_fingerprint(source),
            shuffled[offset]["decision_id"]: _record_fingerprint(shuffled[offset]),
        }
        query["query_fingerprint"] = hashlib.sha256(
            json.dumps(
                query, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest()
        queries.append(query)
    return queries, exclusions
