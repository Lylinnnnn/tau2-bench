"""Canonical definition and identity of one full-task evaluation protocol."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

PROTOCOL_SCHEMA_VERSION = 1


def generation_parameters() -> dict[str, Any]:
    """Return model-generation settings shared by baseline and checkpoints."""

    return {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "max_tokens": 4096,
        "enable_thinking": True,
    }


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    found = False
    for path in paths:
        if not path.is_file():
            continue
        found = True
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    if not found:
        raise ValueError("No model metadata files were found for fingerprinting")
    return digest.hexdigest()


def model_metadata_fingerprint(model_path: Path) -> str:
    """Fingerprint small model metadata without rereading all 62 GB of weights."""

    names = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    return _sha256_files([model_path / name for name in names])


def _git_tree(repo_root: Path, revision_path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", f"HEAD:{revision_path}"],
        text=True,
    ).strip()


def build_evaluation_protocol(
    *,
    repo_root: Path,
    source_model_path: Path,
    base_model_name: str,
    domains: list[str],
    split: str,
    num_trials: int,
    seed: int,
    max_tasks_per_domain: int,
) -> dict[str, Any]:
    """Build everything that must match before a baseline may be reused."""

    benchmark_trees = {"tau2_source": _git_tree(repo_root, "src/tau2")}
    for domain in domains:
        benchmark_trees[f"domain_data/{domain}"] = _git_tree(
            repo_root, f"data/tau2/domains/{domain}"
        )
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "base_model": {
            "served_name": base_model_name,
            "source_path": str(source_model_path),
            "metadata_sha256": model_metadata_fingerprint(source_model_path),
        },
        "benchmark_trees": benchmark_trees,
        "domains": domains,
        "split": split,
        "num_trials": num_trials,
        "seed": seed,
        "max_tasks_per_domain": max_tasks_per_domain,
        "agent_generation": generation_parameters(),
        "user_generation": generation_parameters(),
        "runner": {
            "max_steps": 200,
            "max_errors": 10,
            "max_retries": 0,
            "hallucination_retries": 0,
            "enforce_communication_protocol": False,
        },
        "metric": "tau2.metrics.agent_metrics.compute_metrics",
    }


def protocol_digest(protocol: dict[str, Any]) -> str:
    """Return a stable digest for exact protocol equality."""

    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def baseline_id(protocol: dict[str, Any]) -> str:
    """Return a readable canonical-baseline identifier."""

    model = str(protocol["base_model"]["served_name"]).replace("_", "-")
    domains = "-".join(protocol["domains"])
    return (
        f"{model}_{domains}_{protocol['split']}_t{protocol['num_trials']}"
        f"_s{protocol['seed']}_{protocol_digest(protocol)[:12]}"
    )
