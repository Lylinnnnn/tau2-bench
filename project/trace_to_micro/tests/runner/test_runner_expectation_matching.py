from types import SimpleNamespace

from trace_to_micro.runner import expectation_matching


def _record(decision_id: str) -> dict:
    return {
        "decision_id": decision_id,
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "history"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "name": "lookup"}],
            },
        ],
    }


def test_context_controls_keep_current_action() -> None:
    records = {
        "current": _record("current"),
        "other": {
            **_record("other"),
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "other history"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "other-call", "name": "lookup"}],
                },
            ],
        },
    }
    query = {"query_id": "current", "shuffled_source_decision_id": "other"}

    action_only = expectation_matching._messages_for_context(
        query, records, "action_only"
    )
    shuffled = expectation_matching._messages_for_context(query, records, "shuffled")

    assert [row["role"] for row in action_only] == ["system", "assistant"]
    assert shuffled[1]["content"] == "other history"
    assert shuffled[-1]["tool_calls"][0]["id"] == "call-1"


def test_score_matching_query_emits_every_registered_condition(monkeypatch) -> None:
    records = {
        "current": {
            **_record("current"),
            "domain": "retail",
            "split": "test",
            "task_id": "1",
            "simulation_id": "sim-1",
            "trial": 0,
            "tool_name": "lookup",
            "tool_call_id": "call-1",
            "result_content": "current result",
            "tools": [],
            "chat_template_kwargs": {"enable_thinking": True},
        },
        "negative": {
            **_record("negative"),
            "domain": "retail",
            "split": "test",
            "task_id": "2",
            "simulation_id": "sim-2",
            "trial": 0,
            "tool_name": "lookup",
            "tool_call_id": "call-2",
            "result_content": "negative result",
            "tools": [],
            "chat_template_kwargs": {"enable_thinking": True},
        },
        "other": {
            **_record("other"),
            "domain": "retail",
            "split": "test",
            "task_id": "3",
            "simulation_id": "sim-3",
            "trial": 0,
            "tool_name": "lookup",
            "tool_call_id": "call-3",
            "result_content": "other result",
            "tools": [],
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }
    query = {
        "query_id": "current",
        "query_fingerprint": "fingerprint",
        "domain": "retail",
        "split": "test",
        "task_id": "1",
        "simulation_id": "sim-1",
        "trial": 0,
        "tool_name": "lookup",
        "positive_decision_id": "current",
        "candidate_decision_ids": ["current", "negative"],
        "shuffled_source_decision_id": "other",
    }
    config = SimpleNamespace(
        scoring_model="qwen3-32b",
        context_variants=("full", "action_only", "shuffled"),
        content_variants=("raw", "identifier_masked"),
    )
    seen = []

    def fake_score(**kwargs):
        seen.append(kwargs)
        return {
            "prefix_token_count": 10,
            "suffix_token_count": 2,
            "sum_logprob": -1.0,
            "mean_logprob": -0.5,
        }

    monkeypatch.setattr(expectation_matching, "score_chat_suffix", fake_score)

    rows = expectation_matching._score_matching_query(
        query,
        records,
        config=config,
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
    )

    assert len(rows) == 2 * 3 * 2
    assert sum(row["is_positive"] for row in rows) == 3 * 2
    assert {row["context_variant"] for row in rows} == {
        "full",
        "action_only",
        "shuffled",
    }
    assert all(call["suffix_message"]["tool_call_id"] == "call-1" for call in seen)
