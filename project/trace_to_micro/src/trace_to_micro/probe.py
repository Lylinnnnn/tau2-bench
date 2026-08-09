"""Resumable paired model inference over logged decision snapshots."""

from pathlib import Path
from typing import Any

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import Results
from tau2.runner import build_environment
from tau2.utils.llm_utils import generate
from trace_to_micro.branching import execute_macro_step
from trace_to_micro.config import ModelExperimentConfig
from trace_to_micro.context_builder import (
    ContextBuildError,
    build_structured_context,
    make_probe_messages,
)
from trace_to_micro.evaluation import score_branch
from trace_to_micro.io import append_jsonl, read_jsonl


def _load_source(results_path: Path):
    metadata = Results.load_metadata(results_path)
    simulations = {
        simulation.id: simulation
        for simulation in Results.iter_simulations(results_path)
    }
    tasks = {str(task.id): task for task in metadata.tasks}
    return simulations, tasks


def run_paired_probe(
    config: ModelExperimentConfig,
    snapshots: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Run/resume builder and three paired next-action conditions."""

    trajectory = config.trajectory
    probe = config.probe
    contexts_path = output_dir / "structured_contexts.jsonl"
    context_failures_path = output_dir / "context_failures.jsonl"
    predictions_path = output_dir / "paired_predictions.jsonl"
    contexts = {row["snapshot_id"]: row for row in read_jsonl(contexts_path)}
    completed = {
        (row["snapshot_id"], row["variant"]) for row in read_jsonl(predictions_path)
    }
    simulations, tasks = _load_source(probe.results_path)
    seed_args = {"seed": probe.seed}
    failed_contexts = []

    for snapshot in snapshots:
        simulation = simulations[snapshot["simulation_id"]]
        task = tasks[str(snapshot["task_id"])]
        messages = simulation.get_messages()
        target_index = snapshot["message_index"]
        target = messages[target_index]
        if not isinstance(target, AssistantMessage):
            raise TypeError(
                f"Snapshot target is not assistant: {snapshot['snapshot_id']}"
            )
        prefix = messages[:target_index]

        environment = build_environment(trajectory.domain)
        tools = environment.get_tools()
        agent = LLMAgent(
            tools=tools,
            domain_policy=environment.get_policy(),
            llm=probe.agent_llm,
            llm_args=probe.agent_llm_args,
        )
        actual_branch = execute_macro_step(
            domain=trajectory.domain,
            task=task,
            prefix=prefix,
            assistant_message=target,
            user_name=trajectory.user,
            user_llm=probe.user_llm,
            user_llm_args=probe.user_llm_args,
            seed=simulation.seed or probe.seed,
            logged_messages=messages,
            logged_target_index=target_index,
        )

        needs_context = any(
            variant != "long_raw"
            and (snapshot["snapshot_id"], variant) not in completed
            for variant in probe.variants
        )
        context_row = contexts.get(snapshot["snapshot_id"])
        if needs_context and context_row is None:
            try:
                context, builder_metadata = build_structured_context(
                    model=probe.context_builder_llm,
                    llm_args={**probe.context_builder_llm_args, **seed_args},
                    policy=environment.get_policy(),
                    tools=tools,
                    prefix=prefix,
                )
            except ContextBuildError as error:
                failure = {
                    "snapshot_id": snapshot["snapshot_id"],
                    "split": snapshot["split"],
                    "simulation_id": snapshot["simulation_id"],
                    "task_id": snapshot["task_id"],
                    "diagnostics": error.diagnostics,
                }
                append_jsonl(context_failures_path, failure)
                failed_contexts.append(failure)
            else:
                context_row = {
                    "snapshot_id": snapshot["snapshot_id"],
                    "context": context,
                    "builder_metadata": builder_metadata,
                    "source_scope": (
                        "agent-visible prefix only; no target or benchmark labels"
                    ),
                }
                append_jsonl(contexts_path, context_row)
                contexts[snapshot["snapshot_id"]] = context_row

        for variant in probe.variants:
            key = (snapshot["snapshot_id"], variant)
            if key in completed:
                continue
            if variant != "long_raw" and context_row is None:
                continue
            model_messages = make_probe_messages(
                variant=variant,
                agent_system_prompt=agent.system_prompt,
                prefix=prefix,
                structured_context=(
                    context_row["context"] if context_row is not None else None
                ),
            )
            prediction = generate(
                model=probe.agent_llm,
                tools=tools,
                messages=model_messages,
                call_name=f"trace_to_micro_{variant}",
                **{**probe.agent_llm_args, **seed_args},
            )
            if not isinstance(prediction, AssistantMessage):
                raise TypeError("Agent probe must return an AssistantMessage")
            predicted_branch = execute_macro_step(
                domain=trajectory.domain,
                task=task,
                prefix=prefix,
                assistant_message=prediction,
                user_name=trajectory.user,
                user_llm=probe.user_llm,
                user_llm_args=probe.user_llm_args,
                seed=simulation.seed or probe.seed,
            )
            row = {
                **snapshot,
                "variant": variant,
                "agent_model": probe.agent_llm,
                "user_model": probe.user_llm,
                "context_builder_model": probe.context_builder_llm,
                "prediction_usage": prediction.usage,
                "prediction_generation_seconds": prediction.generation_time_seconds,
                "actual_branch": actual_branch,
                "predicted_branch": predicted_branch,
                "same_pre_state": (
                    actual_branch["pre_state_hash"]
                    == predicted_branch["pre_state_hash"]
                ),
                "metrics": score_branch(
                    task=task,
                    actual=actual_branch,
                    predicted=predicted_branch,
                ),
            }
            append_jsonl(predictions_path, row)
            completed.add(key)

    if failed_contexts:
        raise ValueError(
            f"Context construction failed for {len(failed_contexts)} snapshots; "
            f"diagnostics were written to {context_failures_path}"
        )
    return {
        "contexts": contexts_path,
        "context_failures": context_failures_path,
        "predictions": predictions_path,
    }
