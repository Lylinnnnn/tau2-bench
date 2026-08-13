"""Client and compact serialization for exported vLLM hidden states."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

from trace_to_micro.runtime.server_preflight import request_json
from trace_to_micro.utils.io import encode_float16_vector


def activation_request_fingerprint(
    request: dict[str, Any], *, model: str, layer_ids: tuple[int, ...]
) -> str:
    """Fingerprint one factual request together with its export configuration."""

    payload = {
        "request_record": request,
        "hidden_model": model,
        "hidden_layer_ids": layer_ids,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _api_root(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value[:-3] if value.endswith("/v1") else value


def _selected_token_logprob(entry: dict[str, Any], token_id: int) -> float:
    value = entry.get(str(token_id), entry.get(token_id))
    if value is None:
        raise ValueError(f"Prompt logprobs omitted selected token {token_id}: {entry}")
    if not isinstance(value, dict) or "logprob" not in value:
        raise ValueError(f"Invalid prompt-logprob entry for token {token_id}: {value}")
    return float(value["logprob"])


def score_chat_suffix(
    *,
    prefix_messages: list[dict[str, Any]],
    suffix_message: dict[str, Any],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    """Score a factual chat suffix under an already selected action."""

    tokenizer_url = f"{_api_root(base_url)}/tokenize"
    common = {
        "model": model,
        "tools": tools,
        "add_generation_prompt": False,
        "chat_template_kwargs": chat_template_kwargs,
    }
    prefix = request_json(
        tokenizer_url,
        api_key=api_key,
        payload={**common, "messages": prefix_messages},
        timeout_seconds=180.0,
    )["tokens"]
    complete = request_json(
        tokenizer_url,
        api_key=api_key,
        payload={**common, "messages": [*prefix_messages, suffix_message]},
        timeout_seconds=180.0,
    )["tokens"]
    if complete[: len(prefix)] != prefix:
        raise ValueError(
            "Chat template is not prefix-stable for the tool-result suffix"
        )
    content = suffix_message.get("content")
    if not isinstance(content, str) or content.startswith("\0"):
        raise ValueError(
            "Tool-result content must be text that does not start with NUL"
        )
    sentinel = request_json(
        tokenizer_url,
        api_key=api_key,
        payload={
            **common,
            "messages": [*prefix_messages, {**suffix_message, "content": "\0"}],
        },
        timeout_seconds=180.0,
    )["tokens"]
    if sentinel[: len(prefix)] != prefix:
        raise ValueError("Chat template changed the action prefix for a tool result")
    content_start = next(
        (
            position
            for position, (actual, alternate) in enumerate(zip(complete, sentinel))
            if actual != alternate
        ),
        min(len(complete), len(sentinel)),
    )
    if content_start < len(prefix) or content_start == len(complete):
        raise ValueError("Could not isolate the tool-result content boundary")
    suffix_token_ids = complete[content_start:]
    if not suffix_token_ids:
        raise ValueError("Tool-result suffix produced no tokens")
    response = request_json(
        f"{base_url.rstrip('/')}/completions",
        api_key=api_key,
        payload={
            "model": model,
            "prompt": complete,
            "temperature": 0,
            "max_tokens": 1,
            "prompt_logprobs": 1,
            "return_token_ids": True,
        },
        timeout_seconds=180.0,
    )
    hidden_path = None
    if (response.get("kv_transfer_params") or {}).get("hidden_states_path"):
        hidden_path = _hidden_state_path(response)
    try:
        choice = response["choices"][0]
        response_token_ids = choice["prompt_token_ids"]
        prompt_logprobs = choice["prompt_logprobs"]
        if response_token_ids != complete:
            raise ValueError("vLLM scored token IDs differ from the tokenized chat")
        selected = []
        for position in range(content_start, len(complete)):
            entry = prompt_logprobs[position]
            if not isinstance(entry, dict):
                raise ValueError(f"Missing prompt logprob at position {position}")
            selected.append(_selected_token_logprob(entry, complete[position]))
        return {
            "action_prefix_token_count": len(prefix),
            "prefix_token_count": content_start,
            "suffix_token_count": len(selected),
            "sum_logprob": float(sum(selected)),
            "mean_logprob": float(sum(selected) / len(selected)),
        }
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"vLLM omitted prompt likelihoods: {response}") from error
    finally:
        if hidden_path is not None:
            _cleanup_hidden_state_file(hidden_path)


def _load_tensor(path: Path, name: str) -> np.ndarray:
    lock_path = Path(f"{path}.lock")
    with lock_path.open() as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_SH)
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
            spec = header[name]
            start, end = spec["data_offsets"]
            handle.seek(8 + header_size + start)
            data = handle.read(end - start)
    dtype = spec["dtype"]
    if dtype == "F32":
        value = np.frombuffer(data, dtype="<f4")
    elif dtype == "F16":
        value = np.frombuffer(data, dtype="<f2").astype(np.float32)
    elif dtype == "BF16":
        bits = np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16
        value = bits.view(np.float32)
    elif dtype == "I64":
        value = np.frombuffer(data, dtype="<i8")
    elif dtype == "I32":
        value = np.frombuffer(data, dtype="<i4")
    else:
        raise ValueError(f"Unsupported safetensors dtype {dtype!r}")
    return value.reshape(spec["shape"])


def load_last_token_vectors(
    path: Path, layer_ids: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """Read selected-layer vectors for the last exported prompt token."""

    hidden = _load_tensor(path, "hidden_states")
    if hidden.ndim != 3 or hidden.shape[1] != len(layer_ids):
        raise ValueError(
            f"Expected [tokens,{len(layer_ids)},hidden], got {hidden.shape}"
        )
    return {
        layer_id: hidden[-1, index].astype(np.float32, copy=False)
        for index, layer_id in enumerate(layer_ids)
    }


def load_hidden_state_export(
    path: Path,
    layer_ids: tuple[int, ...],
    expected_token_ids: list[int],
) -> dict[int, np.ndarray]:
    """Load an export and prove that it belongs to the submitted prompt."""

    exported_token_ids = _load_tensor(path, "token_ids")
    if exported_token_ids.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional token_ids, got {exported_token_ids.shape}"
        )
    expected = np.asarray(expected_token_ids, dtype=np.int64)
    if not np.array_equal(exported_token_ids.astype(np.int64), expected):
        raise ValueError("Exported token_ids do not match the submitted prompt")
    return load_last_token_vectors(path, layer_ids)


def _hidden_state_path(response: dict[str, Any]) -> Path:
    params = response.get("kv_transfer_params") or {}
    value = params.get("hidden_states_path")
    if not value:
        raise ValueError(f"vLLM response omitted hidden_states_path: {response}")
    return Path(value)


def _cleanup_hidden_state_file(path: Path) -> None:
    path.unlink()
    Path(f"{path}.lock").unlink()


def extract_request_activation(
    request: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    layer_ids: tuple[int, ...],
) -> dict[str, Any]:
    """Render one factual chat prefix and export its selected hidden states."""

    add_generation_prompt = request["moment"] != "action"
    rendered = request_json(
        f"{_api_root(base_url)}/tokenize",
        api_key=api_key,
        payload={
            "model": model,
            "messages": request["messages"],
            "tools": request["tools"],
            "add_generation_prompt": add_generation_prompt,
            "chat_template_kwargs": request["chat_template_kwargs"],
        },
        timeout_seconds=180.0,
    )
    token_ids = rendered["tokens"]
    if not token_ids:
        raise ValueError("Tokenizer returned an empty prompt")
    response = request_json(
        f"{base_url.rstrip('/')}/completions",
        api_key=api_key,
        payload={
            "model": model,
            "prompt": token_ids,
            "temperature": 0,
            "max_tokens": 1,
        },
        timeout_seconds=180.0,
    )
    hidden_path = _hidden_state_path(response)
    try:
        vectors = load_hidden_state_export(hidden_path, layer_ids, token_ids)
    finally:
        _cleanup_hidden_state_file(hidden_path)
    return {
        key: value
        for key, value in request.items()
        if key not in {"messages", "tools", "chat_template_kwargs"}
    } | {
        "prompt_tokens": len(token_ids),
        "last_token_id": token_ids[-1],
        "prompt_token_sha256": hashlib.sha256(
            np.asarray(token_ids, dtype="<i8").tobytes()
        ).hexdigest(),
        "vector_encoding": "base64_float16",
        "activations": {
            str(layer_id): encode_float16_vector(vector)
            for layer_id, vector in vectors.items()
        },
    }


def main(argv: list[str] | None = None) -> None:
    """Verify that an endpoint exports the configured hidden-state layers."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer-ids", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be non-empty")
    response = request_json(
        f"{args.base_url.rstrip('/')}/completions",
        api_key=api_key,
        payload={
            "model": args.model,
            "prompt": [1, 2],
            "temperature": 0,
            "max_tokens": 1,
        },
        timeout_seconds=180.0,
    )
    path = _hidden_state_path(response)
    try:
        vectors = load_last_token_vectors(path, tuple(args.layer_ids))
    finally:
        _cleanup_hidden_state_file(path)
    print(
        "Hidden-state export ready: "
        + ", ".join(
            f"layer {layer_id}={vector.shape}" for layer_id, vector in vectors.items()
        )
    )


if __name__ == "__main__":
    main()
