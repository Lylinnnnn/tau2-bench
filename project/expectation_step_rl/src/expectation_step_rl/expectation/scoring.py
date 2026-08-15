"""Score a realized tool result under the model's selected action."""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np

from expectation_step_rl.expectation.anonymization import consistently_anonymize


def _post_json(
    url: str, *, api_key: str, payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _api_root(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value[:-3] if value.endswith("/v1") else value


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    for position, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return position
    return min(len(left), len(right))


def _selected_logprob(entry: dict[str, Any], token_id: int) -> float:
    value = entry.get(str(token_id), entry.get(token_id))
    if not isinstance(value, dict) or "logprob" not in value:
        raise ValueError(f"Selected token {token_id} is absent from prompt_logprobs")
    return float(value["logprob"])


def contextual_min_k(
    full_logprobs: list[float],
    action_only_logprobs: list[float],
    *,
    fraction: float,
) -> float:
    """Average the largest action-only minus full-context token deviations."""

    if not 0 < fraction <= 1:
        raise ValueError("Min-K fraction must be in (0, 1]")
    if not full_logprobs or len(full_logprobs) != len(action_only_logprobs):
        raise ValueError("Likelihood vectors must be non-empty and aligned")
    deviation = np.asarray(action_only_logprobs) - np.asarray(full_logprobs)
    count = max(1, math.ceil(len(deviation) * fraction))
    return float(np.mean(np.sort(deviation)[-count:]))


def scorer_url_for_key(base_urls: list[str], routing_key: str) -> str:
    """Deterministically keep all requests for one candidate on one replica."""

    if not base_urls:
        raise ValueError("At least one scorer base URL is required")
    digest = hashlib.sha256(routing_key.encode()).digest()
    return base_urls[int.from_bytes(digest[:8], byteorder="big") % len(base_urls)]


@dataclass(frozen=True)
class ExpectationMeasurement:
    """Result likelihoods in full and action-only contexts."""

    contextual_min_k_deviation: float
    suffix_token_count: int
    min_k_token_count: int
    full_mean_logprob: float
    action_only_mean_logprob: float
    anonymized_identifier_counts: dict[str, int]
    scorer_base_url: str


class VLLMExpectationScorer:
    """Frozen OpenAI-compatible vLLM client for result prompt likelihoods."""

    def __init__(
        self,
        *,
        base_urls: list[str],
        api_key: str,
        model: str,
        min_k_fraction: float,
        timeout_seconds: float = 180.0,
    ) -> None:
        if not base_urls:
            raise ValueError("At least one scorer base URL is required")
        self.base_urls = [value.rstrip("/") for value in base_urls]
        self.api_key = api_key
        self.model = model
        self.min_k_fraction = min_k_fraction
        self.timeout_seconds = timeout_seconds

    def _score_suffix(
        self,
        *,
        base_url: str,
        prefix_messages: list[dict[str, Any]],
        result_message: dict[str, Any],
        tools: list[dict[str, Any]],
    ) -> tuple[list[int], list[float]]:
        common = {
            "model": self.model,
            "tools": tools,
            "add_generation_prompt": False,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        tokenize_url = f"{_api_root(base_url)}/tokenize"
        complete = _post_json(
            tokenize_url,
            api_key=self.api_key,
            payload={**common, "messages": [*prefix_messages, result_message]},
            timeout=self.timeout_seconds,
        )["tokens"]
        sentinel = _post_json(
            tokenize_url,
            api_key=self.api_key,
            payload={
                **common,
                "messages": [*prefix_messages, {**result_message, "content": "\0"}],
            },
            timeout=self.timeout_seconds,
        )["tokens"]
        content_start = _common_prefix_length(complete, sentinel)
        if content_start == 0 or content_start == len(complete):
            raise ValueError("Could not isolate the tool-result token boundary")
        response = _post_json(
            f"{base_url}/completions",
            api_key=self.api_key,
            payload={
                "model": self.model,
                "prompt": complete,
                "temperature": 0,
                "max_tokens": 1,
                "prompt_logprobs": 1,
                "return_token_ids": True,
            },
            timeout=self.timeout_seconds,
        )
        choice = response["choices"][0]
        if choice["prompt_token_ids"] != complete:
            raise ValueError("Scorer token IDs differ from /tokenize output")
        selected = [
            _selected_logprob(choice["prompt_logprobs"][position], complete[position])
            for position in range(content_start, len(complete))
        ]
        return complete[content_start:], selected

    def measure(
        self,
        *,
        raw_prompt: list[dict[str, Any]],
        action_message: dict[str, Any],
        result_message: dict[str, Any],
        tools: list[dict[str, Any]],
        routing_key: str,
    ) -> ExpectationMeasurement:
        """Measure context-specific surprise after consistent anonymization."""

        base_url = scorer_url_for_key(self.base_urls, routing_key)

        content = result_message.get("content")
        if not isinstance(content, str):
            raise ValueError("Tool-result content must be text")
        try:
            parsed_content = json.loads(content)
            content_is_json = True
        except json.JSONDecodeError:
            parsed_content = content
            content_is_json = False
        bundle, identifier_counts = consistently_anonymize(
            [
                raw_prompt,
                action_message,
                tools,
                {"tool_call_id": result_message["tool_call_id"]},
                parsed_content,
            ]
        )
        clean_prompt, clean_action, clean_tools, clean_link, clean_content = bundle
        clean_result = {
            "role": "tool",
            "tool_call_id": clean_link["tool_call_id"],
            "content": (
                json.dumps(clean_content, ensure_ascii=False, separators=(", ", ": "))
                if content_is_json
                else clean_content
            ),
        }
        full_ids, full = self._score_suffix(
            base_url=base_url,
            prefix_messages=[*clean_prompt, clean_action],
            result_message=clean_result,
            tools=clean_tools,
        )
        action_ids, action_only = self._score_suffix(
            base_url=base_url,
            prefix_messages=[clean_prompt[0], clean_action],
            result_message=clean_result,
            tools=clean_tools,
        )
        if full_ids != action_ids:
            raise ValueError("Full and action-only result token IDs are not aligned")
        count = max(1, math.ceil(len(full) * self.min_k_fraction))
        return ExpectationMeasurement(
            contextual_min_k_deviation=contextual_min_k(
                full, action_only, fraction=self.min_k_fraction
            ),
            suffix_token_count=len(full),
            min_k_token_count=count,
            full_mean_logprob=float(np.mean(full)),
            action_only_mean_logprob=float(np.mean(action_only)),
            anonymized_identifier_counts=identifier_counts,
            scorer_base_url=base_url,
        )
