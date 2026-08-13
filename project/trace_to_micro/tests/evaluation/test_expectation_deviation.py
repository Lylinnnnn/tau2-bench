import json

from trace_to_micro.evaluation.expectation_deviation import (
    build_controlled_anomaly_queries,
    consistently_anonymize,
)
from trace_to_micro.evaluation.expectation_matching import result_structure


def _record(
    decision_id: str,
    *,
    split: str,
    task_id: str,
    name: str,
    city: str,
    date: str,
) -> dict:
    content = json.dumps(
        {"name": name, "city": city, "date": date},
        separators=(", ", ": "),
    )
    return {
        "decision_id": decision_id,
        "simulation_id": f"sim-{decision_id}",
        "domain": "retail",
        "split": split,
        "task_id": task_id,
        "trial": 0,
        "tool_name": "lookup",
        "messages": [
            {"role": "system", "content": "policy"},
            {
                "role": "user",
                "content": f"Find {name} in {city} on {date}",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": f"call-{decision_id}", "name": "lookup"}],
            },
        ],
        "result_content": content,
        "result_structure": result_structure(content),
    }


def test_consistent_anonymization_preserves_identity_relations() -> None:
    values, counts = consistently_anonymize(
        [
            {"user": "alice_smith_1234", "order": "#W123456"},
            {
                "same_user": "alice_smith_1234",
                "other_user": "bob_jones_9876",
                "order": "#W123456",
            },
        ]
    )

    assert values[0]["user"] == values[1]["same_user"]
    assert values[0]["user"] != values[1]["other_user"]
    assert values[0]["order"] == values[1]["order"]
    assert "alice_smith_1234" not in json.dumps(values)
    assert counts == {"HASH": 1, "USER": 2}


def test_controlled_anomalies_are_cumulative_train_donor_edits() -> None:
    source = _record(
        "test-source",
        split="test",
        task_id="test-1",
        name="Alice",
        city="Boston",
        date="2026-08-13",
    )
    donors = [
        _record(
            f"train-{index}",
            split="train",
            task_id=f"train-{index}",
            name=name,
            city=city,
            date=date,
        )
        for index, (name, city, date) in enumerate(
            [
                ("Brenda", "Denver", "2026-08-14"),
                ("Carlos", "Seattle", "2026-08-15"),
            ]
        )
    ]
    other_test = _record(
        "test-other",
        split="test",
        task_id="test-2",
        name="Diana",
        city="Austin",
        date="2026-08-16",
    )

    queries, exclusions = build_controlled_anomaly_queries(
        [source, *donors, other_test],
        train_split="train",
        test_split="test",
        severity_levels=(1, 2, 3),
        max_length_delta_ratio=0.5,
        random_seed=300,
    )

    query = next(row for row in queries if row["query_id"] == "test-source")
    assert [row["severity"] for row in query["variants"]] == [0, 1, 2, 3]
    assert [len(row["edits"]) for row in query["variants"]] == [0, 1, 2, 3]
    assert query["variants"][1]["edits"] == query["variants"][2]["edits"][:1]
    assert query["variants"][2]["edits"] == query["variants"][3]["edits"][:2]
    assert len({row["path_string"] for row in query["variants"][3]["edits"]}) == 3
    assert all(
        result_structure(row["result_content"]) == source["result_structure"]
        for row in query["variants"]
    )
    assert all(
        edit["donor_task_id"].startswith("train-")
        for row in query["variants"][1:]
        for edit in row["edits"]
    )
    assert {row["query_id"] for row in queries} == {"test-source", "test-other"}
    assert exclusions == []
