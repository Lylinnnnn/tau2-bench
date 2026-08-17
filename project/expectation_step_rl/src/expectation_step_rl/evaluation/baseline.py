"""Publish and reuse one canonical base-model result per evaluation protocol."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from expectation_step_rl.evaluation.audit import audit_results
from tau2.data_model.simulation import Results
from tau2.runner import load_tasks


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_ids(protocol: dict[str, Any], domain: str) -> list[str]:
    task_ids = [str(task.id) for task in load_tasks(domain, protocol["split"])]
    maximum = int(protocol["max_tasks_per_domain"])
    return task_ids[:maximum] if maximum else task_ids


def _baseline_dir(baseline_root: Path, run_manifest: dict[str, Any]) -> Path:
    return baseline_root / run_manifest["baseline_id"]


def baseline_is_ready(baseline_root: Path, run_manifest: dict[str, Any]) -> bool:
    """Whether all domains have an exact-protocol canonical baseline."""

    root = _baseline_dir(baseline_root, run_manifest)
    manifest_path = root / "baseline_manifest.json"
    if not manifest_path.is_file():
        return False
    baseline_manifest = json.loads(manifest_path.read_text())
    if baseline_manifest["protocol_sha256"] != run_manifest["protocol_sha256"]:
        raise ValueError(f"Canonical baseline protocol mismatch: {manifest_path}")
    split = run_manifest["protocol"]["split"]
    return all(
        (root / "trajectories" / domain / split / "results.json").is_file()
        for domain in run_manifest["protocol"]["domains"]
    )


def publish_baseline(*, run_root: Path, baseline_root: Path, domain: str) -> Path:
    """Promote a newly generated base result into the canonical baseline store."""

    run_manifest = json.loads((run_root / "experiment_manifest.json").read_text())
    protocol = run_manifest["protocol"]
    split = protocol["split"]
    source = run_root / "trajectories" / "base" / domain / split / "results.json"
    results = Results.load(source)
    audit_results(
        results,
        expected_task_ids=_task_ids(protocol, domain),
        expected_num_trials=int(protocol["num_trials"]),
        expected_agent_model=f"openai/{protocol['base_model']['served_name']}",
        expected_user_model=f"openai/{protocol['base_model']['served_name']}",
    )
    root = _baseline_dir(baseline_root, run_manifest)
    destination = root / "trajectories" / domain / split / "results.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)

    manifest_path = root / "baseline_manifest.json"
    if manifest_path.exists():
        baseline_manifest = json.loads(manifest_path.read_text())
        if baseline_manifest["protocol_sha256"] != run_manifest["protocol_sha256"]:
            raise ValueError(f"Canonical baseline identity collision: {manifest_path}")
    else:
        baseline_manifest = {
            "schema_version": 1,
            "baseline_id": run_manifest["baseline_id"],
            "protocol_sha256": run_manifest["protocol_sha256"],
            "protocol": protocol,
            "created_at_utc": _utc_now(),
            "source_run_tag": run_manifest["run_tag"],
            "domains": {},
        }
    baseline_manifest["domains"][domain] = {
        "results_path": str(destination),
        "results_sha256": _sha256(destination),
    }
    baseline_manifest["updated_at_utc"] = _utc_now()
    _write_json(manifest_path, baseline_manifest)

    run_audit_path = (
        run_root / "trajectories" / "base" / domain / split / "inference_audit.json"
    )
    run_audit = json.loads(run_audit_path.read_text())
    run_audit.update(
        {
            "baseline_id": run_manifest["baseline_id"],
            "baseline_reused": False,
            "canonical_results_path": str(destination),
        }
    )
    _write_json(run_audit_path, run_audit)
    return destination


def materialize_baseline(*, run_root: Path, baseline_root: Path, domain: str) -> Path:
    """Audit and copy a canonical baseline into the current result matrix."""

    run_manifest = json.loads((run_root / "experiment_manifest.json").read_text())
    if not baseline_is_ready(baseline_root, run_manifest):
        raise ValueError(
            f"Canonical baseline is incomplete: {run_manifest['baseline_id']}"
        )
    protocol = run_manifest["protocol"]
    split = protocol["split"]
    canonical = (
        _baseline_dir(baseline_root, run_manifest)
        / "trajectories"
        / domain
        / split
        / "results.json"
    )
    results = Results.load(canonical)
    audit = audit_results(
        results,
        expected_task_ids=_task_ids(protocol, domain),
        expected_num_trials=int(protocol["num_trials"]),
        expected_agent_model=f"openai/{protocol['base_model']['served_name']}",
        expected_user_model=f"openai/{protocol['base_model']['served_name']}",
    )
    output_dir = run_root / "trajectories" / "base" / domain / split
    destination = output_dir / "results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(canonical, destination)
    audit.update(
        {
            "domain": domain,
            "split": split,
            "full_split": int(protocol["max_tasks_per_domain"]) == 0,
            "model_key": "base",
            "adapter_path": None,
            "results_path": str(destination),
            "expected_model_keys": run_manifest["expected_model_keys"],
            "expected_domains": protocol["domains"],
            "shards": [],
            "baseline_id": run_manifest["baseline_id"],
            "baseline_reused": True,
            "canonical_results_path": str(canonical),
        }
    )
    _write_json(output_dir / "inference_audit.json", audit)
    return destination
