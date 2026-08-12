from pathlib import Path

from trace_to_micro.config import SuccessDirectionConfig
from trace_to_micro.runner.success_direction import (
    activation_requests_for_shard,
    task_ids_for_shard,
)


def test_task_shards_balance_combined_domains_without_overlap(monkeypatch) -> None:
    config = SuccessDirectionConfig.load(
        Path(__file__).parents[2] / "configs/qwen3_32b_success_direction.toml"
    )
    splits = {
        "airline": {"base": ["a0", "a1", "a2"]},
        "retail": {"base": ["r0", "r1"]},
    }
    monkeypatch.setattr(
        "trace_to_micro.runner.success_direction.load_task_splits",
        lambda domain: splits[domain],
    )

    shard_zero = {
        domain: task_ids_for_shard(config, domain, 0, 2) for domain in config.domains
    }
    shard_one = {
        domain: task_ids_for_shard(config, domain, 1, 2) for domain in config.domains
    }

    assert shard_zero == {"airline": ["a0", "a2"], "retail": ["r1"]}
    assert shard_one == {"airline": ["a1"], "retail": ["r0"]}
    assert {
        (domain, task_id)
        for shard in (shard_zero, shard_one)
        for domain, task_ids in shard.items()
        for task_id in task_ids
    } == {
        ("airline", "a0"),
        ("airline", "a1"),
        ("airline", "a2"),
        ("retail", "r0"),
        ("retail", "r1"),
    }


def test_activation_shards_keep_all_moments_of_one_decision_together() -> None:
    requests = [
        {"decision_id": decision_id, "moment": moment}
        for decision_id in ("d0", "d1", "d2")
        for moment in ("before", "action", "result")
    ]

    shards = [
        activation_requests_for_shard(requests, shard_index, 2)
        for shard_index in range(2)
    ]

    assert [{row["decision_id"] for row in shard} for shard in shards] == [
        {"d0", "d2"},
        {"d1"},
    ]
    assert all(
        {row["moment"] for row in shard if row["decision_id"] == decision_id}
        == {"before", "action", "result"}
        for shard in shards
        for decision_id in {row["decision_id"] for row in shard}
    )
