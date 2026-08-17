"""Train-only calibration and continuous expectation-deviation rewards."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PRIMARY_SCORE = "contextual_min_k_deviation"


@dataclass(frozen=True)
class CalibrationResult:
    """One calibrated expectation score and its source statistics."""

    reward: float
    raw_score: float | None
    z_score: float | None
    calibration_key: str | None
    calibration_count: int | None
    calibration_level: str | None
    outcome: str


class TrainCalibration:
    """Read fixed domain/tool statistics fitted only on official Train data."""

    def __init__(
        self,
        calibration: dict[str, Any],
        *,
        clip: float,
        invalid_action_penalty: float,
        tool_error_penalty: float,
        unsupported_tool_penalty: float,
    ) -> None:
        source = str(calibration.get("source", ""))
        if "Train" not in source:
            raise ValueError("Calibration provenance must explicitly identify Train")
        if clip <= 0:
            raise ValueError("Reward clip must be positive")
        self.groups = calibration["groups"]
        self.tool_fallbacks = calibration.get("tool_fallbacks", {})
        self.structure_fallbacks = calibration.get("structure_fallbacks", {})
        self.domain_fallbacks = calibration.get("domain_fallbacks", {})
        self.clip = clip
        self.invalid_action_penalty = invalid_action_penalty
        self.tool_error_penalty = tool_error_penalty
        self.unsupported_tool_penalty = unsupported_tool_penalty

    @classmethod
    def load(cls, path: Path, **kwargs: float) -> "TrainCalibration":
        """Load a checked-in Train-only calibration file."""

        return cls(json.loads(path.read_text()), **kwargs)

    def score(
        self,
        *,
        domain: str,
        tool_name: str | None,
        raw_score: float | None,
        structure_key: str | None = None,
        action_valid: bool = True,
        tool_error: bool = False,
    ) -> CalibrationResult:
        """Map contextual deviation to reward; larger deviation is worse."""

        if not action_valid or tool_name is None:
            return CalibrationResult(
                self.invalid_action_penalty,
                raw_score,
                None,
                None,
                None,
                None,
                "invalid_action",
            )
        if tool_error:
            return CalibrationResult(
                self.tool_error_penalty,
                raw_score,
                None,
                None,
                None,
                None,
                "tool_error",
            )
        key = f"{domain}|{tool_name}|{structure_key}"
        group = self.groups.get(key)
        if group is None:
            tool_key = f"{domain}|{tool_name}"
            group = self.tool_fallbacks.get(tool_key)
            if group is not None:
                key = tool_key
                level = "tool_fallback"
            else:
                structure_key_with_domain = f"{domain}|{structure_key}"
                group = self.structure_fallbacks.get(structure_key_with_domain)
                if group is not None:
                    key = f"{domain}|*|{structure_key}"
                    level = "structure_fallback"
                else:
                    group = self.domain_fallbacks.get(domain)
                    if group is None:
                        return CalibrationResult(
                            self.unsupported_tool_penalty,
                            raw_score,
                            None,
                            key,
                            None,
                            None,
                            "unsupported_calibration_group",
                        )
                    key = f"{domain}|*|*"
                    level = "domain_fallback"
        else:
            level = "tool_structure"
        if raw_score is None:
            raise ValueError("A successful tool call requires a measured score")
        statistics = group["statistics"][PRIMARY_SCORE]
        z_score = (raw_score - statistics["mean"]) / statistics["std"]
        clipped = max(-self.clip, min(self.clip, z_score))
        return CalibrationResult(
            reward=-float(clipped),
            raw_score=float(raw_score),
            z_score=float(z_score),
            calibration_key=key,
            calibration_count=int(group["count"]),
            calibration_level=level,
            outcome=("scored" if level == "tool_structure" else f"scored_{level}"),
        )

    def supports(self, domain: str, tool_name: str) -> bool:
        """Whether Train contains a usable calibration group for this tool."""

        return (
            f"{domain}|{tool_name}" in self.tool_fallbacks
            or domain in self.domain_fallbacks
        )


def _statistics(values: list[float]) -> dict[str, Any]:
    scores = np.asarray(values, dtype=float)
    std = float(np.std(scores, ddof=1))
    if std <= 0:
        raise ValueError("Calibration group has zero variance")
    return {
        "count": len(values),
        "statistics": {PRIMARY_SCORE: {"mean": float(np.mean(scores)), "std": std}},
    }


def fit_training_calibration(
    rows: list[dict[str, Any]], *, minimum_tool_count: int
) -> dict[str, Any]:
    """Fit tool groups and domain fallbacks from clean official Train rows."""

    if minimum_tool_count < 2:
        raise ValueError("minimum_tool_count must be at least two")
    selected = [
        row
        for row in rows
        if row.get("split") == "train"
        and row.get("track") == "logged_clean_result"
        and int(row.get("severity", 0)) == 0
    ]
    if not selected:
        raise ValueError("No logged clean Train rows were found")
    by_tool_structure: dict[str, list[float]] = defaultdict(list)
    by_tool: dict[str, list[float]] = defaultdict(list)
    by_structure: dict[str, list[float]] = defaultdict(list)
    by_domain: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        score = float(row[PRIMARY_SCORE])
        domain = str(row["domain"])
        tool = str(row["tool_name"])
        structure = str(row["structure_key"])
        by_tool_structure[f"{domain}|{tool}|{structure}"].append(score)
        by_tool[f"{domain}|{tool}"].append(score)
        by_structure[f"{domain}|{structure}"].append(score)
        by_domain[domain].append(score)
    groups = {
        key: _statistics(values)
        for key, values in sorted(by_tool_structure.items())
        if len(values) >= minimum_tool_count
    }
    return {
        "source": "official Train logged clean tool results only",
        "source_filter": {
            "split": "train",
            "track": "logged_clean_result",
            "severity": 0,
        },
        "primary_score": PRIMARY_SCORE,
        "minimum_tool_count": minimum_tool_count,
        "fallback_policy": "tool, then result structure, then domain; all fitted from Train",
        "scored_train_records": len(selected),
        "groups": groups,
        "tool_fallbacks": {
            key: _statistics(values)
            for key, values in sorted(by_tool.items())
            if len(values) >= minimum_tool_count
        },
        "structure_fallbacks": {
            key: _statistics(values)
            for key, values in sorted(by_structure.items())
            if len(values) >= minimum_tool_count
        },
        "domain_fallbacks": {
            domain: _statistics(values) for domain, values in sorted(by_domain.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-tool-count", type=int, default=5)
    args = parser.parse_args()
    with args.scores.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    calibration = fit_training_calibration(
        rows, minimum_tool_count=args.minimum_tool_count
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tool_groups": len(calibration["groups"]),
                "domain_fallbacks": len(calibration["domain_fallbacks"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
