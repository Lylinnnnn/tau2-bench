import json
from pathlib import Path

import pytest

from trace_to_micro import server_preflight


def _write_config(path: Path) -> None:
    path.write_text(
        """
[trajectory]
agent_llm = "openai/agent-model"
user_llm = "openai/user-model"

[probe]
agent_llm = "openai/agent-model"
user_llm = "openai/user-model"
context_builder_llm = "openai/builder-model"
""".strip()
    )


def test_configured_model_ids_come_from_experiment_config(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.toml"
    _write_config(config_path)

    model_ids, agent_model, builder_model = server_preflight.configured_model_ids(
        config_path
    )

    assert model_ids == {"agent-model", "user-model", "builder-model"}
    assert agent_model == "agent-model"
    assert builder_model == "builder-model"


def test_wait_for_models_retries_transient_failure(monkeypatch) -> None:
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            raise server_preflight.ServerPreflightError("starting")
        return {"data": [{"id": "agent-model"}]}

    monkeypatch.setattr(server_preflight, "request_json", fake_request)
    monkeypatch.setattr(server_preflight.time, "sleep", lambda _: None)

    result = server_preflight.wait_for_models(
        "http://127.0.0.1:8000/v1/",
        api_key="local-vllm",
        wait_seconds=10,
        poll_seconds=1,
    )

    assert result == {"data": [{"id": "agent-model"}]}
    assert [call[0] for call in calls] == [
        "http://127.0.0.1:8000/v1/models",
        "http://127.0.0.1:8000/v1/models",
    ]


def test_require_models_reports_missing_ids() -> None:
    with pytest.raises(server_preflight.ServerPreflightError, match="builder-model"):
        server_preflight.require_models(
            {"data": [{"id": "agent-model"}]},
            {"agent-model", "builder-model"},
        )


def test_tool_and_json_checks_validate_response_shapes(monkeypatch) -> None:
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [{"function": {"name": "health_check"}}]
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
        ]
    )
    monkeypatch.setattr(
        server_preflight,
        "request_json",
        lambda *args, **kwargs: next(responses),
    )

    server_preflight.check_tool_call(
        "http://127.0.0.1:8000/v1",
        api_key="local-vllm",
        model="agent-model",
    )
    server_preflight.check_json_mode(
        "http://127.0.0.1:8000/v1",
        api_key="local-vllm",
        model="builder-model",
    )
