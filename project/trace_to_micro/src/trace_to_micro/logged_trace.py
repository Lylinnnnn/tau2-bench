"""Completeness audit and decision snapshot extraction from real model traces."""

from collections import Counter
from pathlib import Path
from typing import Any

from tau2.agent.base_agent import is_valid_agent_history_message
from tau2.data_model.message import AssistantMessage, Message, ToolMessage, UserMessage
from tau2.data_model.simulation import Results, SimulationRun
from tau2.runner import load_task_splits
from trace_to_micro.message_utils import action_record, prompt_tokens, transcript_json


def audit_results_completeness(
    path: Path,
    *,
    expected_task_ids: set[str] | None = None,
    expected_num_trials: int | None = None,
) -> dict[str, Any]:
    """Validate that every configured task/trial has one non-empty trajectory."""

    metadata = Results.load_metadata(path)
    simulations = list(Results.iter_simulations(path))
    metadata_task_ids = {str(task.id) for task in metadata.tasks}
    required_task_ids = expected_task_ids or metadata_task_ids
    required_num_trials = expected_num_trials or metadata.info.num_trials
    expected = {
        (task_id, trial)
        for task_id in required_task_ids
        for trial in range(required_num_trials)
    }
    observed = [(str(sim.task_id), sim.trial) for sim in simulations]
    counts = Counter(observed)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    empty = sorted(sim.id for sim in simulations if not sim.get_messages())
    missing_metadata_tasks = sorted(required_task_ids - metadata_task_ids)
    unexpected_metadata_tasks = sorted(metadata_task_ids - required_task_ids)
    num_trials_mismatch = metadata.info.num_trials != required_num_trials
    report = {
        "results_path": str(path),
        "behavior_agent_model": metadata.info.agent_info.llm,
        "behavior_user_model": metadata.info.user_info.llm,
        "behavior_agent_args": metadata.info.agent_info.llm_args,
        "behavior_user_args": metadata.info.user_info.llm_args,
        "task_count": len(metadata.tasks),
        "num_trials": metadata.info.num_trials,
        "required_task_count": len(required_task_ids),
        "required_num_trials": required_num_trials,
        "expected_simulations": len(expected),
        "observed_simulations": len(simulations),
        "missing_task_trials": missing,
        "duplicate_task_trials": duplicates,
        "unexpected_task_trials": unexpected,
        "empty_simulation_ids": empty,
        "missing_metadata_tasks": missing_metadata_tasks,
        "unexpected_metadata_tasks": unexpected_metadata_tasks,
        "num_trials_mismatch": num_trials_mismatch,
        "termination_reasons": dict(
            sorted(Counter(str(sim.termination_reason) for sim in simulations).items())
        ),
        "complete": not (
            missing
            or duplicates
            or unexpected
            or empty
            or missing_metadata_tasks
            or unexpected_metadata_tasks
            or num_trials_mismatch
        ),
    }
    if not report["complete"]:
        raise ValueError(f"Incomplete trajectory results: {report}")
    return report


def is_agent_decision(messages: list[Message], index: int) -> bool:
    """Whether an assistant message was generated in response to user/environment."""

    if index == 0 or not isinstance(messages[index], AssistantMessage):
        return False
    previous = messages[index - 1]
    return (isinstance(previous, UserMessage) and not previous.is_tool_call()) or (
        isinstance(previous, ToolMessage) and previous.requestor == "assistant"
    )


def decision_indices(simulation: SimulationRun) -> list[int]:
    """Return all model-generated assistant decision indices."""

    messages = simulation.get_messages()
    return [
        index for index in range(len(messages)) if is_agent_decision(messages, index)
    ]


def stratified_indices(indices: list[int], limit: int) -> list[int]:
    """Select deterministic early/middle/late positions without random sampling."""

    if limit <= 0:
        raise ValueError("max_snapshots_per_task must be positive")
    if len(indices) <= limit:
        return indices
    if limit == 1:
        return [indices[-1]]
    positions = [round(i * (len(indices) - 1) / (limit - 1)) for i in range(limit)]
    return [indices[position] for position in positions]


def _position_bucket(position: int, total: int) -> str:
    if total == 1 or position / (total - 1) < 1 / 3:
        return "early"
    if position / (total - 1) < 2 / 3:
        return "middle"
    return "late"


def _assign_context_buckets(snapshots: list[dict[str, Any]]) -> None:
    ordered = sorted(
        snapshots,
        key=lambda row: (
            row["behavior_prompt_tokens"]
            if row["behavior_prompt_tokens"] is not None
            else row["prefix_char_count"],
            row["snapshot_id"],
        ),
    )
    labels = ("short", "medium", "long")
    for rank, row in enumerate(ordered):
        bucket_index = min(2, rank * 3 // len(ordered))
        row["context_length_bucket"] = labels[bucket_index]


def extract_decision_snapshots(
    results_path: Path,
    *,
    task_set: str,
    split: str,
    max_snapshots_per_task: int,
) -> list[dict[str, Any]]:
    """Extract compact pointers to paired decision states, not copied prefixes."""

    split_map = load_task_splits(task_set)
    if split_map is None or split not in split_map:
        raise ValueError(f"Unknown split {split!r} for task set {task_set!r}")
    allowed = set(split_map[split])
    snapshots: list[dict[str, Any]] = []
    for simulation in Results.iter_simulations(results_path):
        if simulation.task_id not in allowed:
            continue
        messages = simulation.get_messages()
        all_indices = decision_indices(simulation)
        selected = set(stratified_indices(all_indices, max_snapshots_per_task))
        for decision_position, message_index in enumerate(all_indices):
            if message_index not in selected:
                continue
            target = messages[message_index]
            assert isinstance(target, AssistantMessage)
            agent_prefix = [
                message
                for message in messages[:message_index]
                if is_valid_agent_history_message(message)
            ]
            snapshots.append(
                {
                    "snapshot_id": f"{simulation.id}:{message_index}",
                    "split": split,
                    "simulation_id": simulation.id,
                    "task_id": simulation.task_id,
                    "trial": simulation.trial,
                    "seed": simulation.seed,
                    "message_index": message_index,
                    "decision_position": decision_position,
                    "decision_count": len(all_indices),
                    "position_bucket": _position_bucket(
                        decision_position, len(all_indices)
                    ),
                    "prefix_message_count": len(agent_prefix),
                    "prefix_char_count": len(transcript_json(agent_prefix)),
                    "behavior_prompt_tokens": prompt_tokens(target),
                    "logged_action": action_record(target),
                    "reward": (
                        simulation.reward_info.reward
                        if simulation.reward_info is not None
                        else None
                    ),
                    "termination_reason": str(simulation.termination_reason),
                }
            )
    _assign_context_buckets(snapshots)
    return snapshots
