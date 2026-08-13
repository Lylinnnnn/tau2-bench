from pathlib import Path

from trace_to_micro.utils.io import append_jsonl_batch, read_jsonl


def test_append_jsonl_batch_preserves_complete_group_order(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"

    append_jsonl_batch(path, [{"row": 1}, {"row": 2}])
    append_jsonl_batch(path, [{"row": 3}])

    assert read_jsonl(path) == [{"row": 1}, {"row": 2}, {"row": 3}]
