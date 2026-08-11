"""Preflight checks for the local OpenAI-compatible model server."""

from __future__ import annotations

import argparse
import json
import os
import time
import tomllib
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ServerPreflightError(RuntimeError):
    """Raised when the configured model endpoint is not ready."""


def configured_model_ids(config_path: Path) -> tuple[set[str], str, str]:
    """Return served IDs plus the agent and context-builder IDs from TOML."""

    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    model_names = {
        config["trajectory"]["agent_llm"],
        config["trajectory"]["user_llm"],
        config["probe"]["agent_llm"],
        config["probe"]["user_llm"],
        config["probe"]["context_builder_llm"],
    }

    def served_id(model: str) -> str:
        provider, separator, model_id = model.partition("/")
        if separator != "/" or provider != "openai" or not model_id:
            raise ServerPreflightError(
                f"Expected an openai/<served-model-id> model, got {model!r}"
            )
        return model_id

    return (
        {served_id(model) for model in model_names},
        served_id(config["trajectory"]["agent_llm"]),
        served_id(config["probe"]["context_builder_llm"]),
    )


def configured_request_overrides(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract only model-generation fields relevant to direct HTTP probes."""

    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    def convert(llm_args: dict[str, Any]) -> dict[str, Any]:
        overrides = {
            key: llm_args[key]
            for key in ("temperature", "top_p", "seed")
            if key in llm_args
        }
        overrides.update(llm_args.get("extra_body", {}))
        return overrides

    return (
        convert(config["trajectory"].get("agent_llm_args", {})),
        convert(config["probe"].get("context_builder_llm_args", {})),
    )


def request_json(
    url: str,
    *,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Send one authenticated JSON request and decode its response."""

    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ServerPreflightError(
            f"{url} returned HTTP {error.code}: {detail}"
        ) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ServerPreflightError(f"{url} is not ready: {error}") from error


def wait_for_models(
    base_url: str,
    *,
    api_key: str,
    wait_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    """Poll the models endpoint until it responds or the deadline passes."""

    deadline = time.monotonic() + wait_seconds
    models_url = f"{base_url.rstrip('/')}/models"
    last_error: Exception | None = None
    while True:
        try:
            return request_json(
                models_url,
                api_key=api_key,
                timeout_seconds=min(30.0, max(poll_seconds, 1.0)),
            )
        except ServerPreflightError as error:
            last_error = error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ServerPreflightError(
                    f"Model server did not become ready within {wait_seconds:g}s. "
                    f"Last error: {last_error}"
                ) from error
            time.sleep(min(poll_seconds, remaining))


def require_models(payload: dict[str, Any], expected_ids: set[str]) -> None:
    """Require every model referenced by the experiment configuration."""

    available_ids = {
        item.get("id") for item in payload.get("data", []) if isinstance(item, dict)
    }
    missing = expected_ids - available_ids
    if missing:
        raise ServerPreflightError(
            f"Missing served model IDs {sorted(missing)}; available IDs are "
            f"{sorted(model_id for model_id in available_ids if model_id)}"
        )


def check_tool_call(
    base_url: str,
    *,
    api_key: str,
    model: str,
    request_overrides: dict[str, Any] | None = None,
) -> None:
    """Verify the agent model can emit OpenAI-format tool calls."""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Call the health_check tool with status ready.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "health_check",
                    "description": "Report model-server readiness.",
                    "parameters": {
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                        "required": ["status"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 0,
        **(request_overrides or {}),
        "max_tokens": 1_024,
    }
    response = request_json(
        f"{base_url.rstrip('/')}/chat/completions",
        api_key=api_key,
        payload=payload,
        timeout_seconds=180.0,
    )
    try:
        tool_calls = response["choices"][0]["message"]["tool_calls"]
        function_name = tool_calls[0]["function"]["name"]
    except (KeyError, IndexError, TypeError) as error:
        raise ServerPreflightError(
            f"Agent model returned no valid tool call: {response}"
        ) from error
    if function_name != "health_check":
        raise ServerPreflightError(
            f"Expected health_check tool call, received {function_name!r}"
        )


def check_json_mode(
    base_url: str,
    *,
    api_key: str,
    model: str,
    request_overrides: dict[str, Any] | None = None,
) -> None:
    """Verify the context-builder model supports JSON response mode."""

    response = request_json(
        f"{base_url.rstrip('/')}/chat/completions",
        api_key=api_key,
        payload={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": 'Return exactly one JSON object: {"status":"ready"}',
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            **(request_overrides or {}),
            "max_tokens": 256,
        },
        timeout_seconds=180.0,
    )
    try:
        content = response["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise ServerPreflightError(
            f"Context-builder model returned invalid JSON content: {response}"
        ) from error
    if not isinstance(parsed, dict):
        raise ServerPreflightError("JSON-mode response was not an object")


def run_preflight(
    *,
    config_path: Path,
    base_url: str,
    api_key: str,
    wait_seconds: float,
    poll_seconds: float,
) -> None:
    """Wait for the endpoint and validate the experiment's required features."""

    expected_ids, agent_model, context_builder_model = configured_model_ids(config_path)
    agent_overrides, builder_overrides = configured_request_overrides(config_path)
    models = wait_for_models(
        base_url,
        api_key=api_key,
        wait_seconds=wait_seconds,
        poll_seconds=poll_seconds,
    )
    require_models(models, expected_ids)
    check_tool_call(
        base_url,
        api_key=api_key,
        model=agent_model,
        request_overrides=agent_overrides,
    )
    check_json_mode(
        base_url,
        api_key=api_key,
        model=context_builder_model,
        request_overrides=builder_overrides,
    )


def main(argv: list[str] | None = None) -> None:
    """Run the server preflight from the command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--wait-seconds", type=float, default=600.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ServerPreflightError("OPENAI_API_KEY must be non-empty")
    run_preflight(
        config_path=args.config,
        base_url=args.base_url,
        api_key=api_key,
        wait_seconds=args.wait_seconds,
        poll_seconds=args.poll_seconds,
    )
    print("Model-server preflight passed: models, tool calls, and JSON mode are ready.")


if __name__ == "__main__":
    main()
