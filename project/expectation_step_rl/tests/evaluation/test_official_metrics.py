import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from expectation_step_rl.evaluation import official_metrics
from tau2.data_model.simulation import TerminationReason


def test_summarize_results_uses_tau2_pass_one(monkeypatch: pytest.MonkeyPatch) -> None:
    tau2_metrics = SimpleNamespace(
        infra_error_count=0,
        total_tasks=2,
        total_simulations=2,
        avg_reward=0.5,
        pass_hat_ks={1: 0.5},
    )
    monkeypatch.setattr(
        official_metrics,
        "compute_metrics",
        lambda results: tau2_metrics,
    )
    results = SimpleNamespace(
        info=SimpleNamespace(num_trials=1),
        simulations=[
            SimpleNamespace(termination_reason=TerminationReason.AGENT_STOP),
            SimpleNamespace(termination_reason=TerminationReason.USER_STOP),
        ],
    )

    report = official_metrics.summarize_results(results)

    assert report["implementation"] == "tau2.metrics.agent_metrics.compute_metrics"
    assert report["pass^1"] == 0.5
    assert report["termination_reasons"] == {"agent_stop": 1, "user_stop": 1}


def test_summarize_results_rejects_infrastructure_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        official_metrics,
        "compute_metrics",
        lambda results: SimpleNamespace(infra_error_count=1),
    )

    with pytest.raises(ValueError, match="infrastructure errors"):
        official_metrics.summarize_results(SimpleNamespace())


def test_build_report_rejects_an_incomplete_model_domain_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_path = (
        tmp_path / "trajectories" / "base" / "airline" / "test" / "inference_audit.json"
    )
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(
        json.dumps(
            {
                "complete": True,
                "expected_model_keys": ["base", "step_10"],
                "expected_domains": ["airline", "retail"],
                "results_path": str(tmp_path / "results.json"),
                "model_key": "base",
                "domain": "airline",
                "split": "test",
                "full_split": True,
                "adapter_path": None,
            }
        )
    )
    monkeypatch.setattr(official_metrics.Results, "load", lambda path: object())
    monkeypatch.setattr(
        official_metrics,
        "summarize_results",
        lambda results: {"pass^1": 0.0},
    )

    with pytest.raises(ValueError, match="matrix incomplete"):
        official_metrics.build_report(tmp_path)
