"""Label-free structured contexts built only from the visible logged prefix."""

import json
from typing import Any, Callable

from tau2.agent.base_agent import is_valid_agent_history_message
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import extract_json_from_llm_response, generate
from trace_to_micro.message_utils import transcript_json

CONTEXT_FIELDS = (
    "user_goal",
    "observed_facts",
    "completed_steps",
    "tool_observations",
    "open_questions",
    "local_subgoal",
    "success_condition",
)

MAX_CONTEXT_BUILD_ATTEMPTS = 2
RESPONSE_EXCERPT_CHARS = 2_000

BUILDER_INSTRUCTION = """
You create a compact decision context for a customer-service agent. Use only
facts visible in the transcript. Do not use benchmark labels, hidden scenario
instructions, reference actions, or future messages. Do not invent facts.
Return exactly one JSON object with these keys:
- user_goal: string
- observed_facts: list of strings
- completed_steps: list of strings
- tool_observations: list of strings
- open_questions: list of strings
- local_subgoal: one immediate, outcome-oriented next subtask as a string
- success_condition: observable evidence that the local subtask is complete
Keep every list to at most 8 short items and every string concise. Do not copy
the policy, tool schemas, or transcript into the output.
The local subgoal must not name a tool or prescribe its arguments unless that
tool call was already explicitly requested in the visible transcript.
""".strip()


class ContextBuildError(ValueError):
    """Raised after compact-context JSON remains invalid after one retry."""

    def __init__(self, diagnostics: list[dict[str, Any]]):
        self.diagnostics = diagnostics
        last = diagnostics[-1]
        super().__init__(
            "Context builder returned invalid JSON after "
            f"{len(diagnostics)} attempts: {last['error']} "
            f"(finish_reason={last['finish_reason']!r}, "
            f"response_chars={last['response_chars']})"
        )


def _response_metadata(response: AssistantMessage) -> dict[str, Any]:
    raw_data = response.raw_data or {}
    choices = raw_data.get("choices") or []
    finish_reason = choices[0].get("finish_reason") if choices else None
    content = response.content or ""
    return {
        "finish_reason": finish_reason,
        "usage": response.usage,
        "generation_time_seconds": response.generation_time_seconds,
        "response_chars": len(content),
    }


def _parse_context_response(response: AssistantMessage) -> dict[str, Any]:
    if response.content is None:
        raise TypeError("Context builder returned no text content")
    parsed = json.loads(extract_json_from_llm_response(response.content))
    if not isinstance(parsed, dict):
        raise TypeError("Context builder response must be a JSON object")
    return _validate_context(parsed)


def _validate_context(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != set(CONTEXT_FIELDS):
        raise ValueError(
            f"Context builder keys must be {CONTEXT_FIELDS}; got {tuple(value)}"
        )
    for field in ("user_goal", "local_subgoal", "success_condition"):
        if not isinstance(value[field], str):
            raise TypeError(f"Context field {field!r} must be a string")
    for field in (
        "observed_facts",
        "completed_steps",
        "tool_observations",
        "open_questions",
    ):
        if not isinstance(value[field], list) or not all(
            isinstance(item, str) for item in value[field]
        ):
            raise TypeError(f"Context field {field!r} must be a list of strings")
    return value


def build_structured_context(
    *,
    model: str,
    llm_args: dict[str, Any],
    policy: str,
    tools: list[Tool],
    prefix: list[Message],
    generate_fn: Callable[..., AssistantMessage | UserMessage] = generate,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call a context-builder model without exposing target/future messages."""

    visible_prefix = [
        message for message in prefix if is_valid_agent_history_message(message)
    ]
    tool_schemas = [tool.openai_schema for tool in tools]
    system = SystemMessage(
        role="system",
        content=(
            f"{BUILDER_INSTRUCTION}\n\n<policy>\n{policy}\n</policy>\n"
            f"<available_tools>\n{json.dumps(tool_schemas, ensure_ascii=False)}\n"
            "</available_tools>"
        ),
    )
    transcript = transcript_json(visible_prefix)
    diagnostics = []
    for attempt in range(1, MAX_CONTEXT_BUILD_ATTEMPTS + 1):
        retry_instruction = ""
        if attempt > 1:
            retry_instruction = (
                "\nYour previous response was not valid, complete JSON. Return a "
                "shorter object and close every string, list, and brace."
            )
        request = UserMessage(
            role="user",
            content=(
                "Build the structured context from this agent-visible prefix:\n"
                f"{transcript}{retry_instruction}"
            ),
        )
        response = generate_fn(
            model=model,
            messages=[system, request],
            call_name="trace_to_micro_context_builder",
            **llm_args,
        )
        if not isinstance(response, AssistantMessage):
            raise TypeError("Context builder must return an AssistantMessage")
        metadata = _response_metadata(response)
        try:
            context = _parse_context_response(response)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            content = response.content or ""
            diagnostics.append(
                {
                    "attempt": attempt,
                    **metadata,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "response_head": content[:RESPONSE_EXCERPT_CHARS],
                    "response_tail": content[-RESPONSE_EXCERPT_CHARS:],
                }
            )
            continue
        return context, {"attempt": attempt, **metadata}
    raise ContextBuildError(diagnostics)


def make_probe_messages(
    *,
    variant: str,
    agent_system_prompt: str,
    prefix: list[Message],
    structured_context: dict[str, Any] | None,
) -> list[Message]:
    """Construct one of the paired agent inputs."""

    system = SystemMessage(role="system", content=agent_system_prompt)
    if variant == "long_raw":
        visible = [
            message for message in prefix if is_valid_agent_history_message(message)
        ]
        return [system, *visible]
    if structured_context is None:
        raise ValueError(f"Variant {variant!r} requires a structured context")
    if variant in {"hybrid_clean", "hybrid_clean_scoped_tools"}:
        instruction = (
            "Continue from the verified context. Follow the action contract exactly. "
            "Do not invent identifiers or call a user-owned action as an agent tool."
        )
        return [
            system,
            UserMessage(
                role="user",
                content=(
                    f"{instruction}\n<context>\n"
                    f"{json.dumps(structured_context, ensure_ascii=False)}"
                    "\n</context>"
                ),
            ),
        ]
    if variant == "structured_state":
        context = {
            key: value
            for key, value in structured_context.items()
            if key not in {"local_subgoal", "success_condition"}
        }
        instruction = "Continue from this compact observed conversation state."
    elif variant == "clean_subtask":
        context = structured_context
        instruction = (
            "Complete only the stated local subgoal, then obtain its observable "
            "success evidence."
        )
    else:
        raise ValueError(f"Unknown context variant: {variant}")
    return [
        system,
        UserMessage(
            role="user",
            content=f"{instruction}\n<context>\n{json.dumps(context, ensure_ascii=False)}\n</context>",
        ),
    ]
