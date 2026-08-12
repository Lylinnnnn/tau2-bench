"""Structural and tool-ownership audit for generated τ² result files."""

from collections import Counter
from pathlib import Path
from typing import Any

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import compute_metrics, is_successful
from tau2.runner import build_environment
from trace_to_micro.analysis.logged_trace import (
    audit_results_completeness,
    decision_indices,
)


def build_official_metrics(results_path: Path) -> dict[str, Any]:
    """Compute τ²'s own aggregate reward and pass^k metrics."""

    results = Results.load(results_path)
    missing_rewards = [sim.id for sim in results.simulations if sim.reward_info is None]
    if missing_rewards:
        raise ValueError(f"Official metrics require rewards: {missing_rewards}")
    metrics = compute_metrics(results)
    if 1 not in metrics.pass_hat_ks:
        raise ValueError("Official metrics did not produce pass^1")
    return {
        "implementation": "tau2.metrics.agent_metrics.compute_metrics",
        "task_aggregation": "per-task pass^k, then mean across tasks",
        "configured_num_trials": results.info.num_trials,
        "total_tasks": metrics.total_tasks,
        "total_simulations": metrics.total_simulations,
        "infrastructure_errors_excluded_by_official_metric": (
            metrics.infra_error_count
        ),
        "average_reward": metrics.avg_reward,
        "pass^1": metrics.pass_hat_ks[1],
        "pass^k": {str(k): value for k, value in sorted(metrics.pass_hat_ks.items())},
    }


def _finish_reason(message: AssistantMessage | UserMessage) -> str | None:
    choices = (message.raw_data or {}).get("choices") or []
    return choices[0].get("finish_reason") if choices else None


def build_results_audit(
    results_path: Path,
    *,
    domain: str,
    expected_task_ids: set[str],
    expected_agent_model: str,
    expected_user_model: str,
    expected_num_trials: int = 1,
) -> dict[str, Any]:
    """Validate extraction structure while separating model errors from bad data."""

    completeness = audit_results_completeness(
        results_path,
        expected_task_ids=expected_task_ids,
        expected_num_trials=expected_num_trials,
        expected_agent_model=expected_agent_model,
        expected_user_model=expected_user_model,
    )
    metadata = Results.load_metadata(results_path)
    tasks = {str(task.id): task for task in metadata.tasks}
    environment = build_environment(domain)
    agent_tools = {tool.name for tool in environment.get_tools()}
    all_user_tools = {tool.name for tool in environment.get_user_tools()}

    call_counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    invalid_calls = []
    missing_tool_responses = []
    orphan_tool_responses = []
    duplicate_tool_responses = []
    simulation_summaries = []

    for simulation in Results.iter_simulations(results_path):
        task = tasks[str(simulation.task_id)]
        task_user_tools = {
            tool.name for tool in environment.get_user_tools(include=task.user_tools)
        }
        messages = simulation.get_messages()
        calls = {}
        response_counts: Counter[tuple[str, str]] = Counter()
        tool_error_count = 0

        for message_index, message in enumerate(messages):
            if isinstance(message, (AssistantMessage, UserMessage)):
                actor = message.role
                finish_reason = _finish_reason(message)
                finish_reasons[f"{actor}:{finish_reason or 'missing'}"] += 1
                allowed_tools = agent_tools if actor == "assistant" else task_user_tools
                opposite_tools = all_user_tools if actor == "assistant" else agent_tools
                for call in message.tool_calls or []:
                    key = (call.id, call.requestor)
                    calls[key] = {
                        "message_index": message_index,
                        "name": call.name,
                        "message_actor": actor,
                    }
                    call_counts[f"{actor}:total"] += 1
                    if call.requestor != actor:
                        invalid_kind = "requestor_mismatch"
                    elif call.name in allowed_tools:
                        call_counts[f"{actor}:allowed"] += 1
                        continue
                    elif call.name in opposite_tools:
                        invalid_kind = "cross_role_tool"
                    else:
                        invalid_kind = "unknown_tool"
                    invalid_counts[f"{actor}:{invalid_kind}"] += 1
                    invalid_calls.append(
                        {
                            "simulation_id": simulation.id,
                            "task_id": str(simulation.task_id),
                            "message_index": message_index,
                            "actor": actor,
                            "tool_name": call.name,
                            "kind": invalid_kind,
                        }
                    )
            elif isinstance(message, ToolMessage):
                response_counts[(message.id, message.requestor)] += 1
                tool_error_count += message.error

        for key, call in calls.items():
            count = response_counts[key]
            if count == 0:
                missing_tool_responses.append(
                    {
                        "simulation_id": simulation.id,
                        "task_id": str(simulation.task_id),
                        "tool_call_id": key[0],
                        "requestor": key[1],
                        **call,
                    }
                )
            elif count > 1:
                duplicate_tool_responses.append(
                    {
                        "simulation_id": simulation.id,
                        "task_id": str(simulation.task_id),
                        "tool_call_id": key[0],
                        "requestor": key[1],
                        "response_count": count,
                    }
                )
        for key, count in response_counts.items():
            if key not in calls:
                orphan_tool_responses.append(
                    {
                        "simulation_id": simulation.id,
                        "task_id": str(simulation.task_id),
                        "tool_call_id": key[0],
                        "requestor": key[1],
                        "response_count": count,
                    }
                )

        reward = (
            simulation.reward_info.reward
            if simulation.reward_info is not None
            else None
        )
        simulation_summaries.append(
            {
                "simulation_id": simulation.id,
                "task_id": str(simulation.task_id),
                "trial": simulation.trial,
                "reward": reward,
                "success": is_successful(reward) if reward is not None else None,
                "termination_reason": str(simulation.termination_reason),
                "message_count": len(messages),
                "agent_decision_count": len(decision_indices(simulation)),
                "tool_error_count": tool_error_count,
                "reward_breakdown": (
                    simulation.reward_info.reward_breakdown
                    if simulation.reward_info is not None
                    else None
                ),
            }
        )

    linkage_valid = not (
        missing_tool_responses or orphan_tool_responses or duplicate_tool_responses
    )
    rewards = [
        row["reward"] for row in simulation_summaries if row["reward"] is not None
    ]
    return {
        "results_path": str(results_path),
        "domain": domain,
        "data_integrity": {
            "complete_and_model_matched": completeness["complete"],
            "tool_call_response_linkage_valid": linkage_valid,
            "valid_for_analysis": completeness["complete"] and linkage_valid,
            "missing_tool_responses": missing_tool_responses,
            "orphan_tool_responses": orphan_tool_responses,
            "duplicate_tool_responses": duplicate_tool_responses,
            "interpretation": (
                "Invalid or cross-role actions are retained behavior-policy outcomes; "
                "they do not by themselves make the result file structurally invalid."
            ),
        },
        "completeness": completeness,
        "outcomes": {
            "mean_reward": sum(rewards) / len(rewards) if rewards else None,
            "successful": sum(row["success"] is True for row in simulation_summaries),
            "reward_count": len(rewards),
            "termination_reasons": dict(
                sorted(
                    Counter(
                        row["termination_reason"] for row in simulation_summaries
                    ).items()
                )
            ),
        },
        "tool_behavior": {
            "call_counts": dict(sorted(call_counts.items())),
            "invalid_call_counts": dict(sorted(invalid_counts.items())),
            "invalid_calls": invalid_calls,
            "tool_error_count": sum(
                row["tool_error_count"] for row in simulation_summaries
            ),
        },
        "generation": {
            "finish_reasons": dict(sorted(finish_reasons.items())),
        },
        "simulations": simulation_summaries,
    }
