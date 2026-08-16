"""Fail-fast checks for data, calibration, framework pin, and scorer API."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


def _get_json(url: str, api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=30.0) as response:
        return json.loads(response.read())


def _post_json(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30.0) as response:
        return json.loads(response.read())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_dataset(train_path: Path, test_path: Path) -> dict[str, int]:
    """Prove Train/Test decision IDs do not overlap and required fields exist."""

    train = _read_jsonl(train_path)
    test = _read_jsonl(test_path)
    if not train or not test:
        raise ValueError("Train and Test datasets must both be non-empty")
    required = {
        "prompt",
        "agent_name",
        "domain",
        "split",
        "task_id",
        "decision_id",
        "replay_prefix_json",
        "tool_schemas_json",
    }
    forbidden = {
        "reward",
        "reward_info",
        "reference_action",
        "logged_action",
        "target_state",
        "evaluation_criteria",
    }
    for expected_split, rows in (("train", train), ("test", test)):
        for row in rows:
            missing = required - row.keys()
            if missing:
                raise ValueError(f"Dataset row omitted fields: {sorted(missing)}")
            leaked = forbidden & row.keys()
            if leaked:
                raise ValueError(
                    f"Dataset row contains forbidden supervision: {sorted(leaked)}"
                )
            if row["split"] != expected_split:
                raise ValueError(f"{expected_split} file contains {row['split']!r} row")
    overlap = {row["decision_id"] for row in train} & {
        row["decision_id"] for row in test
    }
    if overlap:
        raise ValueError(f"Train/Test decision leakage: {sorted(overlap)[:5]}")
    return {"train_rows": len(train), "test_rows": len(test)}


def check_calibration(path: Path) -> int:
    """Require explicit Train provenance and non-empty domain/tool groups."""

    value = json.loads(path.read_text())
    if "Train" not in str(value.get("source", "")):
        raise ValueError("Calibration does not declare official Train provenance")
    groups = value.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Calibration has no usable groups")
    fallbacks = value.get("domain_fallbacks")
    if not isinstance(fallbacks, dict) or not fallbacks:
        raise ValueError("Calibration has no Train-fitted domain fallbacks")
    if value.get("source_filter", {}).get("track") != "logged_clean_result":
        raise ValueError("Calibration does not cover logged clean Train results")
    if int(value.get("scored_train_records", 0)) <= 0:
        raise ValueError("Calibration reports no scored Train records")
    return len(groups)


def check_submodule(project_root: Path) -> str:
    """Compare the checked-out verl submodule with FRAMEWORK.lock."""

    lock = dict(
        line.split("=", 1)
        for line in (project_root / "FRAMEWORK.lock").read_text().splitlines()
        if line and not line.startswith("#")
    )
    submodule = project_root / "third_party" / "verl"
    commit = subprocess.run(
        ["git", "-C", str(submodule), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != lock["verl_commit"]:
        raise ValueError(
            f"verl checkout {commit} differs from lock {lock['verl_commit']}"
        )
    importlib.metadata.version("verl")
    from expectation_step_rl.verl_adapter.agent_loop import (  # noqa: F401
        Tau2ExpectationStepAgentLoop,
    )

    return commit


def check_training_packages(project_root: Path) -> dict[str, str]:
    """Require the exact GPU package versions used by the pinned verl release."""

    lock = dict(
        line.split("=", 1)
        for line in (project_root / "FRAMEWORK.lock").read_text().splitlines()
        if line and not line.startswith("#")
    )
    requirements = {
        "vllm": lock["training_vllm"],
        "torch": lock["training_torch"],
        "flash-attn": lock["training_flash_attn"],
        "flashinfer-python": lock["training_flashinfer"],
        "transformers": lock["training_transformers"],
        "huggingface-hub": lock["training_huggingface_hub"],
        "tokenizers": lock["training_tokenizers"],
        "swanlab": lock["training_swanlab"],
    }
    installed = {
        package: importlib.metadata.version(package) for package in requirements
    }
    mismatches = {
        package: {"expected": requirements[package], "installed": version}
        for package, version in installed.items()
        if version != requirements[package]
    }
    if mismatches:
        raise ValueError(f"Training package versions differ from lock: {mismatches}")

    import flash_attn

    if not callable(flash_attn.flash_attn_func):
        raise RuntimeError("flash_attn.flash_attn_func is unavailable")
    return installed


def check_scorer(base_url: str, api_key: str, expected_model: str) -> str:
    """Exercise the exact tokenize and prompt-logprob protocol used by reward."""

    normalized = base_url.rstrip("/")
    models = _get_json(f"{normalized}/models", api_key)["data"]
    identifiers = [str(model["id"]) for model in models]
    if expected_model not in identifiers:
        raise ValueError(
            f"Scorer model {expected_model!r} is absent; available={identifiers}"
        )
    api_root = normalized[:-3] if normalized.endswith("/v1") else normalized
    tokens = _post_json(
        f"{api_root}/tokenize",
        api_key,
        {
            "model": expected_model,
            "messages": [{"role": "user", "content": "protocol check"}],
            "add_generation_prompt": False,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )["tokens"]
    scored = _post_json(
        f"{normalized}/completions",
        api_key,
        {
            "model": expected_model,
            "prompt": tokens,
            "temperature": 0,
            "max_tokens": 1,
            "prompt_logprobs": 1,
            "return_token_ids": True,
        },
    )["choices"][0]
    if scored.get("prompt_token_ids") != tokens:
        raise ValueError("Scorer does not preserve submitted prompt token IDs")
    prompt_logprobs = scored.get("prompt_logprobs")
    if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) != len(tokens):
        raise ValueError("Scorer does not return aligned prompt_logprobs")
    if not any(isinstance(entry, dict) and entry for entry in prompt_logprobs):
        raise ValueError("Scorer prompt_logprobs contain no selected-token entries")
    return expected_model


def check_parallelism(
    *,
    training_gpus: int,
    rollout_tensor_parallel_size: int,
    train_batch_size: int,
    rollout_n: int,
    ppo_mini_batch_size: int,
    ppo_micro_batch_size_per_gpu: int,
    agent_loop_workers: int,
) -> dict[str, int]:
    """Reject batch and worker layouts that verl cannot divide evenly."""

    values = {
        "training_gpus": training_gpus,
        "rollout_tensor_parallel_size": rollout_tensor_parallel_size,
        "train_batch_size": train_batch_size,
        "rollout_n": rollout_n,
        "ppo_mini_batch_size": ppo_mini_batch_size,
        "ppo_micro_batch_size_per_gpu": ppo_micro_batch_size_per_gpu,
        "agent_loop_workers": agent_loop_workers,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"Parallelism values must be positive: {values}")
    if training_gpus % rollout_tensor_parallel_size:
        raise ValueError(
            "training_gpus must be divisible by rollout_tensor_parallel_size"
        )
    if ppo_mini_batch_size > train_batch_size:
        raise ValueError("ppo_mini_batch_size cannot exceed train_batch_size")

    sampled_sequences = train_batch_size * rollout_n
    if sampled_sequences % training_gpus:
        raise ValueError(
            "train_batch_size * rollout_n must be divisible by training_gpus"
        )
    if sampled_sequences < agent_loop_workers or sampled_sequences % agent_loop_workers:
        raise ValueError(
            "train_batch_size * rollout_n must be at least and divisible by "
            "agent_loop_workers"
        )

    normalized_ppo_batch = ppo_mini_batch_size * rollout_n
    if normalized_ppo_batch % training_gpus:
        raise ValueError(
            "ppo_mini_batch_size * rollout_n must be divisible by training_gpus"
        )
    normalized_ppo_batch //= training_gpus
    if normalized_ppo_batch % ppo_micro_batch_size_per_gpu:
        raise ValueError("per-GPU PPO mini batch must be divisible by PPO micro batch")

    return {
        "sampled_sequences_per_step": sampled_sequences,
        "rollout_replicas": training_gpus // rollout_tensor_parallel_size,
        "sequences_per_agent_loop_worker": sampled_sequences // agent_loop_workers,
        "ppo_samples_per_training_gpu": normalized_ppo_batch,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--scorer-base-urls", required=True)
    parser.add_argument("--scorer-api-key", default="EMPTY")
    parser.add_argument("--scorer-model", required=True)
    parser.add_argument("--training-gpus", type=int, required=True)
    parser.add_argument("--rollout-tensor-parallel-size", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--rollout-n", type=int, required=True)
    parser.add_argument("--ppo-mini-batch-size", type=int, required=True)
    parser.add_argument("--ppo-micro-batch-size-per-gpu", type=int, required=True)
    parser.add_argument("--agent-loop-workers", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = {
        **check_dataset(args.train_data, args.test_data),
        "calibration_groups": check_calibration(args.calibration),
        "training_packages": check_training_packages(args.project_root),
        "verl_commit": check_submodule(args.project_root),
        "scorer_models": [
            check_scorer(base_url, args.scorer_api_key, args.scorer_model)
            for base_url in args.scorer_base_urls.split(",")
        ],
        "parallelism": check_parallelism(
            training_gpus=args.training_gpus,
            rollout_tensor_parallel_size=args.rollout_tensor_parallel_size,
            train_batch_size=args.train_batch_size,
            rollout_n=args.rollout_n,
            ppo_mini_batch_size=args.ppo_mini_batch_size,
            ppo_micro_batch_size_per_gpu=args.ppo_micro_batch_size_per_gpu,
            agent_loop_workers=args.agent_loop_workers,
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
