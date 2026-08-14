"""Consistent identifier anonymization for expectation scoring."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

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
    """Mask a bundle while preserving equality relations between identifiers."""

    mapping = _identifier_map(values)
    transformed = [_replace_identifiers(value, mapping) for value in values]
    kinds = Counter(alias.split("_", 1)[0][1:] for alias in mapping.values())
    return transformed, dict(sorted(kinds.items()))
