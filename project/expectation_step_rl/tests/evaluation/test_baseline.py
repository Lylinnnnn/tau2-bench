import json
from pathlib import Path

from expectation_step_rl.evaluation.baseline import baseline_is_ready


def test_canonical_baseline_requires_exact_protocol_and_all_domains(
    tmp_path: Path,
) -> None:
    run_manifest = {
        "baseline_id": "baseline-a",
        "protocol_sha256": "protocol-a",
        "protocol": {
            "domains": ["airline", "retail"],
            "split": "test",
        },
    }
    root = tmp_path / "baseline-a"
    root.mkdir()
    (root / "baseline_manifest.json").write_text(
        json.dumps({"protocol_sha256": "protocol-a"})
    )
    airline = root / "trajectories" / "airline" / "test" / "results.json"
    airline.parent.mkdir(parents=True)
    airline.write_text("{}")

    assert baseline_is_ready(tmp_path, run_manifest) is False

    retail = root / "trajectories" / "retail" / "test" / "results.json"
    retail.parent.mkdir(parents=True)
    retail.write_text("{}")
    assert baseline_is_ready(tmp_path, run_manifest) is True
