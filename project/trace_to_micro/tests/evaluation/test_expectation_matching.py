from trace_to_micro.evaluation.expectation_matching import (
    build_expectation_matching_report,
    build_matching_queries,
    mask_identifiers,
    result_structure,
)


def _matching_record(index: int, task_id: str, content: str) -> dict:
    return {
        "decision_id": f"decision-{index}",
        "simulation_id": f"simulation-{task_id}",
        "domain": "retail",
        "split": "test",
        "task_id": task_id,
        "trial": 0,
        "tool_name": "get_order_details",
        "result_content": content,
        "result_structure": result_structure(content),
    }


def test_matching_queries_use_same_tool_cross_task_hard_negatives() -> None:
    records = [
        _matching_record(
            index,
            str(index),
            f'{{"order_id":"#W{index}","status":"pending","items":[{index}]}}',
        )
        for index in range(5)
    ]
    records.append(
        {
            **_matching_record(9, "9", '{"user_id":"other_9"}'),
            "tool_name": "get_user_details",
        }
    )

    queries, exclusions = build_matching_queries(
        records,
        candidate_count=4,
        minimum_candidate_count=4,
        random_seed=300,
    )

    query = next(row for row in queries if row["query_id"] == "decision-0")
    assert query["candidate_decision_ids"][0] == "decision-0"
    assert len(query["candidate_decision_ids"]) == 4
    assert all(value != "decision-9" for value in query["candidate_decision_ids"])
    assert query["shuffled_source_decision_id"] != "decision-0"
    assert any(row["decision_id"] == "decision-9" for row in exclusions)


def test_identifier_masking_preserves_structure_but_removes_values() -> None:
    value = {
        "email": "alice@example.com",
        "order_id": "#W123456",
        "item_id": "1234567890",
        "status": "pending",
    }

    masked = mask_identifiers(value)

    assert masked == {
        "email": "<IDENTIFIER>",
        "order_id": "<IDENTIFIER>",
        "item_id": "<IDENTIFIER>",
        "status": "pending",
    }


def test_expectation_matching_report_requires_context_beyond_shuffled() -> None:
    rows = []
    for domain in ("airline", "retail"):
        for query_index in range(8):
            for context in ("full", "action_only", "shuffled"):
                for content in ("raw", "identifier_masked"):
                    positive_score = 0.9 if context == "full" else 0.4
                    if context == "action_only":
                        positive_score = 0.6
                    rows.extend(
                        [
                            {
                                "query_id": f"{domain}-{query_index}",
                                "domain": domain,
                                "task_id": f"task-{query_index}",
                                "tool_name": "lookup",
                                "context_variant": context,
                                "content_variant": content,
                                "is_positive": True,
                                "mean_logprob": positive_score,
                            },
                            {
                                "query_id": f"{domain}-{query_index}",
                                "domain": domain,
                                "task_id": f"task-{query_index}",
                                "tool_name": "lookup",
                                "context_variant": context,
                                "content_variant": content,
                                "is_positive": False,
                                "mean_logprob": 0.5,
                            },
                        ]
                    )

    report, measurements = build_expectation_matching_report(
        rows,
        domains=("airline", "retail"),
        evaluation_split="test",
        context_variants=("full", "action_only", "shuffled"),
        content_variants=("raw", "identifier_masked"),
        bootstrap_samples=100,
        random_seed=300,
    )

    assert report["primary_test"]["evidence_status"] == (
        "reliable_contextual_expectation_signal"
    )
    assert (
        report["domains"]["retail"]["conditions"]["full:raw"]["pairwise_accuracy"]
        == 1.0
    )
    assert len(measurements) == 2 * 8 * 3 * 2
