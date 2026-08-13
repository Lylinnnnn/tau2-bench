"""Compile observed tool decisions into deterministic local consequences."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.runner import build_environment
from trace_to_micro.replay.state import (
    diff_snapshots,
    snapshot_distance,
    snapshot_environment,
)

FIELD_FAMILIES = {
    "address": {
        "address",
        "address1",
        "address2",
        "city",
        "country",
        "state",
        "zip",
    },
    "items": {
        "exchange_items",
        "exchange_new_items",
        "item_id",
        "items",
        "new_item_ids",
        "return_items",
    },
    "itinerary": {
        "baggages",
        "cabin",
        "date",
        "destination",
        "flight_number",
        "flights",
        "origin",
        "passengers",
    },
    "payment": {
        "amount",
        "balance",
        "exchange_payment_method_id",
        "exchange_price_difference",
        "payment_history",
        "payment_method_id",
        "price",
        "return_payment_method_id",
        "transaction_type",
    },
    "status": {"cancel_reason", "status"},
}


def _category_key(prefix: str, value: str) -> str:
    normalized = "_".join(value.lower().replace("(", " ").replace(")", " ").split())
    return f"{prefix}.{normalized}"


def _entity_type(path: str) -> str | None:
    parts = path.split(".")
    if len(parts) < 2:
        return None
    collection = parts[1]
    aliases = {
        "flights": "flight",
        "orders": "order",
        "reservations": "reservation",
        "users": "user",
    }
    return aliases.get(collection, collection.removesuffix("s"))


def _field_families(path: str) -> set[str]:
    parts = set(path.split("."))
    return {
        family for family, field_names in FIELD_FAMILIES.items() if parts & field_names
    }


def _output_facts(content: str | None) -> dict[str, Any]:
    if content is None:
        return {
            "kind": None,
            "entity_types": [],
            "field_families": [],
        }
    value = _content_value(content)
    if isinstance(value, dict):
        kind = "object"
    elif isinstance(value, list):
        kind = "array"
    elif isinstance(value, bool):
        kind = "boolean"
    elif isinstance(value, (int, float)):
        kind = "number"
    elif value is None:
        kind = "null"
    else:
        kind = "string"

    keys = set()

    def collect_keys(item: Any) -> None:
        if isinstance(item, dict):
            keys.update(str(key) for key in item)
            for child in item.values():
                collect_keys(child)
        elif isinstance(item, list):
            for child in item:
                collect_keys(child)

    collect_keys(value)
    entity_markers = {
        "order": {"order_id", "orders"},
        "reservation": {"reservation_id", "reservations"},
        "user": {"user_id", "users"},
        "flight": {"flight_number", "flights"},
    }
    return {
        "kind": kind,
        "entity_types": sorted(
            entity for entity, markers in entity_markers.items() if keys & markers
        ),
        "field_families": sorted(
            family
            for family, field_names in FIELD_FAMILIES.items()
            if keys & field_names
        ),
    }


def compile_abstract_consequence(
    row: dict[str, Any], *, result_content: str | None = None
) -> dict[str, Any]:
    """Compile a tool result and state diff into a target-free consequence.

    The record deliberately excludes distance to the task's official final state.
    It describes only what actually happened immediately after the logged action,
    so it can supervise an expectation probe without assuming one golden path.
    """

    changes = row["changes"]
    operations = set()
    entity_types = set()
    field_families = set()
    status_values = set()
    transaction_types = set()
    for change in changes:
        if not change["before_present"] and change["after_present"]:
            operations.add("create")
        elif change["before_present"] and not change["after_present"]:
            operations.add("delete")
        else:
            operations.add("update")
        entity_type = _entity_type(change["path"])
        if entity_type is not None:
            entity_types.add(entity_type)
        field_families.update(_field_families(change["path"]))
        field_name = change["path"].rsplit(".", maxsplit=1)[-1]
        if field_name == "status" and change["after"] is not None:
            status_values.add(str(change["after"]))
        if field_name == "transaction_type" and change["after"] is not None:
            transaction_types.add(str(change["after"]))

    output = _output_facts(result_content)

    targets = {
        "execution.success": bool(row["tool_success"]),
        "effect.changed": bool(row["state_changed"]),
        **{
            f"operation.{operation}": operation in operations
            for operation in ("create", "update", "delete")
        },
        **{
            f"entity.{entity_type}": entity_type in entity_types
            for entity_type in ("order", "reservation", "user", "flight")
        },
        **{
            f"field.{family}": family in field_families
            for family in sorted(FIELD_FAMILIES)
        },
        **{_category_key("status", value): True for value in status_values},
        **{_category_key("transaction", value): True for value in transaction_types},
        **{
            f"output.kind.{kind}": output["kind"] == kind
            for kind in ("object", "array", "string", "number", "boolean", "null")
        },
        **{
            f"output.entity.{entity_type}": (entity_type in output["entity_types"])
            for entity_type in ("order", "reservation", "user", "flight")
        },
        **{
            f"output.field.{family}": family in output["field_families"]
            for family in sorted(FIELD_FAMILIES)
        },
    }
    return {
        "schema_version": 1,
        "decision_id": row["decision_id"],
        "simulation_id": row["simulation_id"],
        "domain": row["domain"],
        "split": row["split"],
        "task_id": row["task_id"],
        "trial": row["trial"],
        "tool_name": row["tool_name"],
        "execution": "success" if row["tool_success"] else "error",
        "state_changed": bool(row["state_changed"]),
        "changed_leaf_count": len(changes),
        "operations": sorted(operations),
        "entity_types": sorted(entity_types),
        "field_families": sorted(field_families),
        "status_values": sorted(status_values),
        "transaction_types": sorted(transaction_types),
        "output": output,
        "targets": targets,
    }


class InvalidReferenceTargetError(RuntimeError):
    """A task's reference write cannot produce a trustworthy DB target."""

    def __init__(
        self,
        *,
        domain: str,
        task_id: str,
        action: Any,
        error: str,
    ) -> None:
        self.domain = domain
        self.task_id = task_id
        self.action_id = action.action_id
        self.requestor = action.requestor
        self.action_name = action.name
        self.arguments = action.arguments
        self.error = error
        super().__init__(
            f"Mutating reference action failed for {domain}/{task_id}: "
            f"{action.name}: {error}"
        )


def _initialize(environment, task: Task) -> None:
    initial_state = task.initial_state
    message_history = (
        initial_state.message_history or [] if initial_state is not None else []
    )
    environment.set_state(
        initialization_data=(
            initial_state.initialization_data if initial_state is not None else None
        ),
        initialization_actions=(
            initial_state.initialization_actions if initial_state is not None else None
        ),
        message_history=message_history,
    )


def target_snapshot(domain: str, task: Task) -> dict[str, Any]:
    """Build the official DB target from state-mutating reference actions.

    The official evaluator compares final database hashes. Read-only reference
    actions cannot affect that target and may contain stale lookup arguments, so
    replaying them only introduces failures that are irrelevant to the target.
    Mutating reference actions remain strict because ignoring one would silently
    corrupt the local goal-progress label.
    """

    environment = build_environment(domain)
    _initialize(environment, task)
    actions = task.evaluation_criteria.actions if task.evaluation_criteria else []
    for action in actions or []:
        if not environment._is_mutating_tool(action.name):
            continue
        response = environment.get_response(
            ToolCall(
                id=action.action_id,
                requestor=action.requestor,
                name=action.name,
                arguments=action.arguments,
            )
        )
        if response.error:
            raise InvalidReferenceTargetError(
                domain=domain,
                task_id=str(task.id),
                action=action,
                error=response.content,
            )
    return snapshot_environment(environment)


def _tool_type(environment, requestor: str, tool_name: str) -> str:
    toolkit = environment.user_tools if requestor == "user" else environment.tools
    if toolkit is None or not toolkit.has_tool(tool_name):
        return "unknown"
    return toolkit.tool_type(tool_name).value


def _content_value(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def replay_local_consequences(
    *,
    simulation: SimulationRun,
    task: Task,
    split: str,
    domain: str,
    goal: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Replay one logged trace and label each assistant tool decision locally."""

    environment = build_environment(domain)
    _initialize(environment, task)
    if goal is None:
        goal = target_snapshot(domain, task)
    messages = simulation.get_messages()
    recorded_results = {
        message.id: message for message in messages if isinstance(message, ToolMessage)
    }
    rows = []
    assistant_position = 0
    for message_index, message in enumerate(messages):
        if not isinstance(message, (AssistantMessage, UserMessage)):
            continue
        calls = message.tool_calls or []
        if isinstance(message, AssistantMessage) and len(calls) > 1:
            raise ValueError(
                "Local consequence extraction requires one assistant call per "
                f"message, got {len(calls)} at {simulation.id}:{message_index}"
            )
        for call in calls:
            before = snapshot_environment(environment)
            distance_before = snapshot_distance(before, goal)
            declared_mutating = environment._is_mutating_tool(call.name)
            tool_type = _tool_type(environment, call.requestor, call.name)
            progress_eligible = declared_mutating and tool_type != "unknown"
            replayed = environment.get_response(call)
            recorded = recorded_results.get(call.id)
            if recorded is None:
                raise ValueError(
                    f"Missing recorded result for {simulation.id}:{message_index}:"
                    f"{call.id}"
                )
            if recorded.error != replayed.error:
                raise ValueError(
                    "Recorded/replayed error mismatch at "
                    f"{simulation.id}:{message_index}:{call.name}"
                )
            if _content_value(recorded.content) != _content_value(replayed.content):
                raise ValueError(
                    "Recorded/replayed tool result mismatch at "
                    f"{simulation.id}:{message_index}:{call.name}"
                )
            after = snapshot_environment(environment)
            distance_after = snapshot_distance(after, goal)
            changes = diff_snapshots(before, after)
            if isinstance(message, AssistantMessage):
                rows.append(
                    {
                        "decision_id": f"{simulation.id}:{message_index}",
                        "simulation_id": simulation.id,
                        "domain": domain,
                        "split": split,
                        "task_id": str(task.id),
                        "trial": simulation.trial,
                        "message_index": message_index,
                        "decision_position": assistant_position,
                        "tool_name": call.name,
                        "tool_type": tool_type,
                        "declared_mutating": declared_mutating,
                        "tool_success": not replayed.error,
                        "state_changed": bool(changes),
                        "goal_progress_eligible": progress_eligible,
                        "goal_progress": (
                            not replayed.error and distance_after < distance_before
                            if progress_eligible
                            else None
                        ),
                        "target_distance_before": distance_before,
                        "target_distance_after": distance_after,
                        "target_distance_reduction": (distance_before - distance_after),
                        "changes": [asdict(change) for change in changes],
                    }
                )
                assistant_position += 1
    return rows
