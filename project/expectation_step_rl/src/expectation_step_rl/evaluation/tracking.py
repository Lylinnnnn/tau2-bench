"""Versioned manifests and a derived registry for evaluation runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from expectation_step_rl.evaluation.checkpoints import (
    AdapterCheckpoint,
    discover_checkpoints,
    parse_steps,
)
from expectation_step_rl.evaluation.protocol import (
    baseline_id,
    build_evaluation_protocol,
    protocol_digest,
)

MANIFEST_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_record(checkpoint: AdapterCheckpoint) -> dict[str, Any]:
    config_path = checkpoint.path / "adapter_config.json"
    weights_path = checkpoint.path / "adapter_model.safetensors"
    return {
        "step": checkpoint.step,
        "model_key": checkpoint.key,
        "served_model_name": checkpoint.served_model_name,
        "source_adapter_path": str(checkpoint.path),
        "adapter_config_sha256": _file_sha256(config_path),
        "adapter_weights_bytes": weights_path.stat().st_size,
    }


def prepare_run_manifest(
    *,
    repo_root: Path,
    evaluation_root: Path,
    run_tag: str,
    checkpoint_root: Path,
    checkpoint_steps: str | None,
    source_model_path: Path,
    base_model_name: str,
    domains: list[str],
    split: str,
    num_trials: int,
    seed: int,
    max_tasks_per_domain: int,
    include_baseline: bool,
    local_staging_enabled: bool,
) -> dict[str, Any]:
    """Create or validate the immutable identity of an evaluation run."""

    checkpoints = discover_checkpoints(
        checkpoint_root,
        selected_steps=parse_steps(checkpoint_steps),
    )
    protocol = build_evaluation_protocol(
        repo_root=repo_root,
        source_model_path=source_model_path,
        base_model_name=base_model_name,
        domains=domains,
        split=split,
        num_trials=num_trials,
        seed=seed,
        max_tasks_per_domain=max_tasks_per_domain,
    )
    baseline = baseline_id(protocol)
    expected_model_keys = [checkpoint.key for checkpoint in checkpoints]
    if include_baseline:
        expected_model_keys.insert(0, "base")
    identity = {
        "run_tag": run_tag,
        "checkpoint_root": str(checkpoint_root),
        "checkpoints": [_checkpoint_record(value) for value in checkpoints],
        "include_baseline": include_baseline,
        "baseline_id": baseline,
        "protocol_sha256": protocol_digest(protocol),
        "protocol": protocol,
        "expected_model_keys": expected_model_keys,
    }
    run_root = evaluation_root / run_tag
    manifest_path = run_root / "experiment_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing_identity = {key: existing[key] for key in identity}
        if existing_identity != identity:
            raise ValueError(
                f"RUN_TAG already belongs to a different evaluation: {manifest_path}"
            )
        return existing
    commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        **identity,
        "training_run_id": checkpoint_root.name,
        "evaluation_git_commit": commit,
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "status": "initialized",
        "local_staging_enabled": local_staging_enabled,
        "artifacts": {
            "run_root": str(run_root),
            "official_metrics": str(run_root / "official_metrics.json"),
        },
    }
    _write_json(manifest_path, manifest)
    rebuild_registry(evaluation_root)
    return manifest


def update_run_status(
    run_root: Path, *, status: str, error: str | None = None
) -> dict[str, Any]:
    """Update mutable run status without changing its experiment identity."""

    manifest_path = run_root / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = status
    manifest["updated_at_utc"] = _utc_now()
    if error is None:
        manifest.pop("error", None)
    else:
        manifest["error"] = error
    _write_json(manifest_path, manifest)
    rebuild_registry(run_root.parent)
    return manifest


def rebuild_registry(evaluation_root: Path) -> dict[str, Any]:
    """Regenerate a compact index from per-run source-of-truth manifests."""

    runs = []
    for manifest_path in sorted(evaluation_root.glob("*/experiment_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        metrics_path = manifest_path.parent / "official_metrics.json"
        row = {
            "run_tag": manifest["run_tag"],
            "training_run_id": manifest["training_run_id"],
            "status": manifest["status"],
            "created_at_utc": manifest["created_at_utc"],
            "evaluation_git_commit": manifest["evaluation_git_commit"],
            "checkpoint_steps": [value["step"] for value in manifest["checkpoints"]],
            "baseline_id": manifest["baseline_id"],
            "protocol_sha256": manifest["protocol_sha256"],
            "run_root": str(manifest_path.parent),
            "official_metrics_ready": metrics_path.is_file(),
        }
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            row["domain_macro_summaries"] = metrics.get("domain_macro_summaries", {})
        runs.append(row)
    registry = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "runs": runs,
    }
    _write_json(evaluation_root / "experiment_registry.json", registry)
    return registry
