"""One-macro-step counterfactual execution from an identical replayed state."""

from dataclasses import asdict
from typing import Any

from tau2.data_model.message import AssistantMessage, Message, UserMessage
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.runner import build_environment, build_user
from tau2.user.user_simulator_base import is_valid_user_history_message
from trace_to_micro.message_utils import action_record
from trace_to_micro.state_diff import (
    diff_snapshots,
    snapshot_environment,
    snapshot_hash,
)


def replay_prefix_environment(
    domain: str, task: Task, prefix: list[Message]
) -> Environment:
    """Reconstruct the factual environment immediately before a decision."""

    environment = build_environment(domain)
    initial_state = task.initial_state
    environment.set_state(
        initialization_data=(
            initial_state.initialization_data if initial_state is not None else None
        ),
        initialization_actions=(
            initial_state.initialization_actions if initial_state is not None else None
        ),
        message_history=prefix,
        strict=False,
    )
    return environment


def _execute_calls(
    environment: Environment, message: AssistantMessage | UserMessage
) -> tuple[list[dict[str, Any]], bool, bool]:
    responses = []
    has_error = False
    declared_mutating = False
    for call in message.tool_calls or []:
        declared_mutating = declared_mutating or environment._is_mutating_tool(
            call.name
        )
        response = environment.get_response(call)
        has_error = has_error or response.error
        responses.append(
            {
                "tool_call": {
                    "requestor": call.requestor,
                    "name": call.name,
                    "arguments": call.arguments,
                },
                "content": response.content,
                "error": response.error,
            }
        )
    return responses, has_error, declared_mutating


def _next_logged_user(messages: list[Message], target_index: int) -> UserMessage | None:
    next_index = target_index + 1
    if next_index < len(messages) and isinstance(messages[next_index], UserMessage):
        return messages[next_index]
    return None


def execute_macro_step(
    *,
    domain: str,
    task: Task,
    prefix: list[Message],
    assistant_message: AssistantMessage,
    user_name: str,
    user_llm: str,
    user_llm_args: dict[str, Any],
    seed: int,
    logged_messages: list[Message] | None = None,
    logged_target_index: int | None = None,
) -> dict[str, Any]:
    """Execute an assistant action and, for text, one immediate user response."""

    environment = replay_prefix_environment(domain, task, prefix)
    before = snapshot_environment(environment)
    pre_state_hash = snapshot_hash(before)
    tool_responses: list[dict[str, Any]] = []
    has_error = False
    declared_mutating = False
    user_message: UserMessage | None = None

    if assistant_message.tool_calls:
        responses, error, mutating = _execute_calls(environment, assistant_message)
        tool_responses.extend(responses)
        has_error = has_error or error
        declared_mutating = declared_mutating or mutating
    elif logged_messages is not None and logged_target_index is not None:
        user_message = _next_logged_user(logged_messages, logged_target_index)
    else:
        user = build_user(
            user_name,
            environment,
            task,
            llm=user_llm,
            llm_args=user_llm_args,
        )
        user.set_seed(seed)
        user_history = [
            message for message in prefix if is_valid_user_history_message(message)
        ]
        user_state = user.get_init_state(message_history=user_history)
        user_message, _ = user.generate_next_message(assistant_message, user_state)

    if user_message is not None and user_message.tool_calls:
        responses, error, mutating = _execute_calls(environment, user_message)
        tool_responses.extend(responses)
        has_error = has_error or error
        declared_mutating = declared_mutating or mutating

    changes = diff_snapshots(before, snapshot_environment(environment))
    return {
        "pre_state_hash": pre_state_hash,
        "assistant_action": action_record(assistant_message),
        "user_continuation": (
            action_record(user_message) if user_message is not None else None
        ),
        "tool_responses": tool_responses,
        "tool_error": has_error,
        "declared_mutating": declared_mutating,
        "mutation_noop": declared_mutating and not changes and not has_error,
        "changes": [asdict(change) for change in changes],
    }
