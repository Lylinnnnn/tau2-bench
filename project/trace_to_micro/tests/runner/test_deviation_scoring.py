import json
from types import SimpleNamespace

from trace_to_micro.runner import deviation_scoring


def _record(decision_id: str, task_id: str, user_id: str) -> dict:
    return {
        "decision_id": decision_id,
        "domain": "retail",
        "split": "test",
        "task_id": task_id,
        "simulation_id": f"sim-{task_id}",
        "trial": 0,
        "tool_name": "get_user_details",
        "tool_call_id": f"chatcmpl-tool-{decision_id}1234",
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": f"My user ID is {user_id}"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"chatcmpl-tool-{decision_id}1234",
                        "name": "get_user_details",
                        "arguments": {"user_id": user_id},
                    }
                ],
            },
        ],
        "result_content": json.dumps({"user_id": user_id}),
        "result_structure": [["user_id", "str"]],
        "tools": [],
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_rematch_scoring_keeps_distinct_anonymous_candidates(monkeypatch) -> None:
    records = {
        "current": _record("current", "1", "alice_smith_1234"),
        "negative": _record("negative", "2", "bob_jones_5678"),
        "other": _record("other", "3", "carol_white_9012"),
    }
    query = {
        "query_id": "current",
        "domain": "retail",
        "split": "test",
        "task_id": "1",
        "simulation_id": "sim-1",
        "trial": 0,
        "tool_name": "get_user_details",
        "positive_decision_id": "current",
        "candidate_decision_ids": ["current", "negative"],
        "shuffled_source_decision_id": "other",
    }
    config = SimpleNamespace(
        scoring_model="qwen3-32b",
        context_variants=("full", "action_only", "shuffled"),
    )
    calls = []

    def fake_score(**kwargs):
        calls.append(kwargs)
        return {
            "prefix_token_count": 10,
            "suffix_token_count": 2,
            "sum_logprob": -1.0,
            "mean_logprob": -0.5,
        }

    monkeypatch.setattr(deviation_scoring, "score_chat_suffix", fake_score)

    rows = deviation_scoring.score_rematch_query(
        query,
        records,
        config=config,
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
    )

    assert len(rows) == 2 * 3
    assert all(
        len({call["suffix_message"]["content"] for call in calls[offset : offset + 2]})
        == 2
        for offset in (0, 2, 4)
    )
    full_prefix = json.dumps(calls[0]["prefix_messages"])
    positive_content = calls[0]["suffix_message"]["content"]
    negative_content = calls[1]["suffix_message"]["content"]
    assert "<USER_001>" in full_prefix
    assert "<USER_001>" in positive_content
    assert "<USER_002>" in negative_content
    assert calls[0]["suffix_message"]["tool_call_id"] == "<CALL_001>"
    assert "<CALL_001>" in full_prefix
