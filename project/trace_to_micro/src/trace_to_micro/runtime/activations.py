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


def _api_root(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value[:-3] if value.endswith("/v1") else value


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
