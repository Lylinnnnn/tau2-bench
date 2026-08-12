from pathlib import Path

import pytest

from trace_to_micro.runner.trajectory import remove_existing_run


def test_remove_existing_run_unlinks_only_exact_results_file(tmp_path: Path) -> None:
    results = tmp_path / "data" / "simulations" / "run" / "results.json"
    results.parent.mkdir(parents=True)
    results.write_text("{}", encoding="utf-8")

    assert remove_existing_run(results) is True
    assert not results.exists()
    assert remove_existing_run(results) is False


def test_remove_existing_run_rejects_unexpected_path(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        remove_existing_run(results)
