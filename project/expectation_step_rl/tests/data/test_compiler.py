import json
from pathlib import Path

from expectation_step_rl.data.compiler import compile_dataset


def test_compile_dataset_physically_separates_splits(
    tmp_path: Path, monkeypatch
) -> None:
    rows = [
        {
            "split": split,
            "domain": "retail",
            "decision_id": f"sim:{split}",
        }
        for split in ("train", "test")
    ]

    def fake_compile_domain_rows(*, domain, results_path):
        assert domain == "retail"
        assert results_path == Path("results.json")
        return rows, {"train_decisions": 1, "test_decisions": 1}

    monkeypatch.setattr(
        "expectation_step_rl.data.compiler.compile_domain_rows",
        fake_compile_domain_rows,
    )

    report = compile_dataset(
        inputs={"retail": Path("results.json")},
        output_dir=tmp_path,
        overwrite=False,
    )

    train = [
        json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()
    ]
    test = [
        json.loads(line) for line in (tmp_path / "test.jsonl").read_text().splitlines()
    ]
    assert [row["split"] for row in train] == ["train"]
    assert [row["split"] for row in test] == ["test"]
    assert report["contains_official_reference_actions"] is False
    assert report["contains_official_final_rewards"] is False
