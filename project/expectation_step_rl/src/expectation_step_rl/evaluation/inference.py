"""Run and merge complete tau2 trajectories for frozen LoRA checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from expectation_step_rl.evaluation.audit import audit_results
from expectation_step_rl.evaluation.protocol import generation_parameters
from tau2.data_model.simulation import Results, TextRunConfig
from tau2.runner import load_tasks, run_domain


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def task_ids_for_shard(
    task_ids: list[str], shard_index: int, num_shards: int
) -> list[str]:
    """Assign an ordered task list round-robin without overlap."""

    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard layout")
    return task_ids[shard_index::num_shards]


def selected_task_ids(domain: str, split: str, max_tasks: int) -> list[str]:
    """Load the official ordered split, optionally limited for a smoke run."""

    task_ids = [str(task.id) for task in load_tasks(domain, split)]
    if max_tasks < 0:
        raise ValueError("max_tasks must be non-negative")
    return task_ids[:max_tasks] if max_tasks else task_ids


def result_name(
    *,
    run_tag: str,
    model_key: str,
    domain: str,
    split: str,
    num_trials: int,
    seed: int,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> str:
    """Build a stable tau2 simulation directory name."""

    name = (
        f"expectation_step_rl_eval_{run_tag}_{model_key}_{domain}_{split}"
        f"_t{num_trials}_s{seed}"
    )
    if shard_index is not None and num_shards is not None:
        name += f"_shard_{shard_index:02d}_of_{num_shards:02d}"
    return name


def simulation_results_path(repo_root: Path, save_name: str) -> Path:
    return repo_root / "data" / "simulations" / save_name / "results.json"


def _llm_args(api_base: str) -> dict[str, Any]:
    generation = generation_parameters()
    return {
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "max_tokens": generation["max_tokens"],
        "api_base": api_base,
        "api_key": "EMPTY",
        "timeout": 180.0,
        "num_retries": 0,
        "extra_body": {
            "top_k": generation["top_k"],
            "min_p": generation["min_p"],
            "chat_template_kwargs": {"enable_thinking": generation["enable_thinking"]},
        },
    }


def run_shard(args: argparse.Namespace) -> None:
    """Generate or resume one disjoint official task shard."""

    all_task_ids = selected_task_ids(args.domain, args.split, args.max_tasks)
    task_ids = task_ids_for_shard(all_task_ids, args.shard_index, args.num_shards)
    if not task_ids:
        print(json.dumps({"skipped_empty_shard": args.shard_index}))
        return
    save_name = result_name(
        run_tag=args.run_tag,
        model_key=args.model_key,
        domain=args.domain,
        split=args.split,
        num_trials=args.num_trials,
        seed=args.seed,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    config = TextRunConfig(
        domain=args.domain,
        task_set_name=args.domain,
        task_split_name=args.split,
        task_ids=task_ids,
        agent="llm_agent",
        user="user_simulator",
        llm_agent=f"openai/{args.agent_model}",
        llm_user=f"openai/{args.base_model}",
        llm_args_agent=_llm_args(args.api_base),
        llm_args_user=_llm_args(args.api_base),
        num_trials=args.num_trials,
        max_steps=200,
        max_errors=10,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        timeout=args.task_timeout,
        save_to=save_name,
        auto_resume=True,
        max_retries=0,
        hallucination_retries=0,
        verbose_logs=False,
        enforce_communication_protocol=False,
    )
    results = run_domain(config)
    audit = audit_results(
        results,
        expected_task_ids=task_ids,
        expected_num_trials=args.num_trials,
        expected_agent_model=f"openai/{args.agent_model}",
        expected_user_model=f"openai/{args.base_model}",
    )
    print(json.dumps(audit, ensure_ascii=False))


def merge_shards(args: argparse.Namespace) -> None:
    """Strictly merge complete disjoint shards into one canonical result."""

    expected_task_ids = selected_task_ids(args.domain, args.split, args.max_tasks)
    shard_results = []
    shard_audits = []
    for shard_index in range(args.num_shards):
        task_ids = task_ids_for_shard(expected_task_ids, shard_index, args.num_shards)
        if not task_ids:
            continue
        save_name = result_name(
            run_tag=args.run_tag,
            model_key=args.model_key,
            domain=args.domain,
            split=args.split,
            num_trials=args.num_trials,
            seed=args.seed,
            shard_index=shard_index,
            num_shards=args.num_shards,
        )
        path = simulation_results_path(args.repo_root, save_name)
        result = Results.load(path)
        shard_audits.append(
            {
                "shard_index": shard_index,
                "path": str(path),
                **audit_results(
                    result,
                    expected_task_ids=task_ids,
                    expected_num_trials=args.num_trials,
                    expected_agent_model=f"openai/{args.agent_model}",
                    expected_user_model=f"openai/{args.base_model}",
                ),
            }
        )
        shard_results.append(result)
    task_order = {
        task_id: position for position, task_id in enumerate(expected_task_ids)
    }
    simulations = [
        simulation
        for shard_result in shard_results
        for simulation in shard_result.simulations
    ]
    simulations.sort(
        key=lambda simulation: (
            task_order[str(simulation.task_id)],
            simulation.trial,
        )
    )
    info = shard_results[0].info.model_copy(deep=True)
    for model_info in (info.agent_info, info.user_info):
        llm_args = dict(model_info.llm_args or {})
        llm_args["api_base"] = "evaluation endpoint; see inference_audit.json"
        model_info.llm_args = llm_args
    tasks_by_id = {str(task.id): task for task in load_tasks(args.domain, args.split)}
    merged = Results(
        info=info,
        tasks=[tasks_by_id[task_id] for task_id in expected_task_ids],
        simulations=simulations,
    )
    output_dir = (
        args.evaluation_root
        / args.run_tag
        / "trajectories"
        / args.model_key
        / args.domain
        / args.split
    )
    results_path = output_dir / "results.json"
    merged.save(results_path)
    audit = audit_results(
        merged,
        expected_task_ids=expected_task_ids,
        expected_num_trials=args.num_trials,
        expected_agent_model=f"openai/{args.agent_model}",
        expected_user_model=f"openai/{args.base_model}",
    )
    audit.update(
        {
            "domain": args.domain,
            "split": args.split,
            "full_split": args.max_tasks == 0,
            "model_key": args.model_key,
            "adapter_path": args.adapter_path,
            "results_path": str(results_path),
            "expected_model_keys": args.expected_model_keys.split(","),
            "expected_domains": args.expected_domains.split(","),
            "shards": shard_audits,
        }
    )
    _write_json(output_dir / "inference_audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
