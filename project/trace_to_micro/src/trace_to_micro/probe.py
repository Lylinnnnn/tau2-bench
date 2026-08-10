"""Resumable paired model inference over logged decision snapshots."""

from pathlib import Path
from typing import Any

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import Results
from tau2.runner import build_environment
from tau2.utils.llm_utils import generate
from trace_to_micro.branching import execute_macro_step
from trace_to_micro.clean import (
    attach_source_audit,
    audit_source_action,
    build_hybrid_clean_context,
)
from trace_to_micro.config import ModelExperimentConfig
from trace_to_micro.context_builder import (
    ContextBuildError,
    build_structured_context,
    make_probe_messages,
)
from trace_to_micro.evaluation import score_branch
from trace_to_micro.io import append_jsonl, read_jsonl, write_jsonl


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
    """Run/resume legacy and hybrid paired next-action conditions."""

    trajectory = config.trajectory
    probe = config.probe
    contexts_path = output_dir / "structured_contexts.jsonl"
    context_failures_path = output_dir / "context_failures.jsonl"
    hybrid_contexts_path = output_dir / "hybrid_contexts.jsonl"
    predictions_path = output_dir / "paired_predictions.jsonl"
    contexts = {row["snapshot_id"]: row for row in read_jsonl(contexts_path)}
    hybrid_contexts = {
        row["snapshot_id"]: row for row in read_jsonl(hybrid_contexts_path)
    }
    prediction_rows = read_jsonl(predictions_path)
    predictions_by_key = {
        (row["snapshot_id"], row["variant"]): row for row in prediction_rows
    }
    completed = set(predictions_by_key)
    simulations, tasks = _load_source(probe.results_path)
    seed_args = {"seed": probe.seed}
    failed_contexts = []
    upgraded_predictions = False

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
        user_tools = environment.get_user_tools()
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
        source_action_audit = audit_source_action(
            target=target,
            actual_branch=actual_branch,
            agent_tools=tools,
            user_tools=user_tools,
        )
        for key, existing_row in predictions_by_key.items():
            if key[0] != snapshot["snapshot_id"]:
                continue
            hybrid_row = hybrid_contexts.get(snapshot["snapshot_id"])
            context_eligible = (
                hybrid_row["context"]["training_eligible"]
                if key[1] in {"hybrid_clean", "hybrid_clean_scoped_tools"}
                and hybrid_row is not None
                else True
            )
            upgraded_predictions = (
                attach_source_audit(
                    existing_row,
                    audit=source_action_audit,
                    context_eligible=context_eligible,
                )
                or upgraded_predictions
            )

        needs_context = any(
            variant in {"structured_state", "clean_subtask"}
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

        needs_hybrid_context = any(
            variant in {"hybrid_clean", "hybrid_clean_scoped_tools"}
            and (snapshot["snapshot_id"], variant) not in completed
            for variant in probe.variants
        )
        hybrid_row = hybrid_contexts.get(snapshot["snapshot_id"])
        if needs_hybrid_context and hybrid_row is None:
            hybrid_context, hybrid_metadata = build_hybrid_clean_context(
                model=probe.context_builder_llm,
                llm_args={**probe.context_builder_llm_args, **seed_args},
                prefix=prefix,
                agent_tools=tools,
                user_tools=user_tools,
            )
            hybrid_row = {
                "snapshot_id": snapshot["snapshot_id"],
                "context": hybrid_context,
                "builder_metadata": hybrid_metadata,
                "source_scope": (
                    "agent-visible prefix and tool ownership only; no target, "
                    "reward, benchmark labels, or reference actions"
                ),
            }
            append_jsonl(hybrid_contexts_path, hybrid_row)
            hybrid_contexts[snapshot["snapshot_id"]] = hybrid_row

        for variant in probe.variants:
            key = (snapshot["snapshot_id"], variant)
            if key in completed:
                continue
            if variant in {"structured_state", "clean_subtask"} and context_row is None:
                continue
            if (
                variant in {"hybrid_clean", "hybrid_clean_scoped_tools"}
                and hybrid_row is None
            ):
                continue
            selected_context = (
                hybrid_row["context"]
                if variant in {"hybrid_clean", "hybrid_clean_scoped_tools"}
                else context_row["context"]
                if context_row is not None
                else None
            )
            model_messages = make_probe_messages(
                variant=variant,
                agent_system_prompt=agent.system_prompt,
                prefix=prefix,
                structured_context=selected_context,
            )
            prediction_tools = tools
            if variant == "hybrid_clean_scoped_tools":
                allowed = set(
                    hybrid_row["context"]["action_contract"]["allowed_agent_tools"]
                )
                prediction_tools = [tool for tool in tools if tool.name in allowed]
            prediction = generate(
                model=probe.agent_llm,
                tools=prediction_tools,
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
                "source_action_audit": source_action_audit,
                "training_eligible": (
                    source_action_audit["valid"]
                    and (
                        selected_context.get("training_eligible", True)
                        if selected_context is not None
                        else True
                    )
                ),
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
            prediction_rows.append(row)
            predictions_by_key[key] = row
            completed.add(key)

    if failed_contexts:
        raise ValueError(
            f"Context construction failed for {len(failed_contexts)} snapshots; "
            f"diagnostics were written to {context_failures_path}"
        )
    if upgraded_predictions:
        write_jsonl(predictions_path, prediction_rows)
    return {
        "contexts": contexts_path,
        "hybrid_contexts": hybrid_contexts_path,
        "context_failures": context_failures_path,
        "predictions": predictions_path,
    }
