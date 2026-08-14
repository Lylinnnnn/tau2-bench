"""Compile real tau2 trajectories into one-step GRPO decision states."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from expectation_step_rl.tau2_adapter.execution import serialize_messages
from tau2.agent.base_agent import is_valid_agent_history_message
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolMessage
from tau2.data_model.simulation import Results
from tau2.runner import build_environment, load_task_splits
from tau2.utils.llm_utils import to_litellm_messages

AGENT_LOOP_NAME = "tau2_expectation_step"


def _is_tool_decision(messages, index: int) -> bool:
    message = messages[index]
    if not isinstance(message, AssistantMessage) or not message.tool_calls:
        return False
    if len(message.tool_calls) != 1 or index == 0:
        return False
    previous = messages[index - 1]
    return previous.role == "user" or (
        isinstance(previous, ToolMessage) and previous.requestor == "assistant"
    )


def compile_domain_rows(
    *, domain: str, results_path: Path
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Extract every single-tool assistant decision from official Train/Test tasks."""

    splits = load_task_splits(domain)
    if splits is None or "train" not in splits or "test" not in splits:
        raise ValueError(f"Domain {domain!r} has no official train/test split")
    split_by_task = {
        str(task_id): split for split in ("train", "test") for task_id in splits[split]
    }
    environment = build_environment(domain)
    system_prompt = SYSTEM_PROMPT.format(
        domain_policy=environment.get_policy(),
        agent_instruction=AGENT_INSTRUCTION,
    )
    system_message = SystemMessage(role="system", content=system_prompt)
    tools = [tool.openai_schema for tool in environment.get_tools()]
    rows = []
    counts: Counter[str] = Counter()
    for simulation in Results.iter_simulations(results_path):
        split = split_by_task.get(str(simulation.task_id))
        if split is None:
            counts["outside_official_train_test"] += 1
            continue
        messages = simulation.get_messages()
        for index in range(len(messages)):
            if not _is_tool_decision(messages, index):
                continue
            agent_prefix = [
                message
                for message in messages[:index]
                if is_valid_agent_history_message(message)
            ]
            raw_prompt = to_litellm_messages([system_message, *agent_prefix])
            decision_id = f"{simulation.id}:{index}"
            rows.append(
                {
                    "data_source": "tau2_expectation_step",
                    "prompt": raw_prompt,
                    "agent_name": AGENT_LOOP_NAME,
                    "domain": domain,
                    "split": split,
                    "task_id": str(simulation.task_id),
                    "simulation_id": simulation.id,
                    "decision_id": decision_id,
                    "message_index": index,
                    "replay_prefix_json": serialize_messages(messages[:index]),
                    "tool_schemas_json": json.dumps(
                        tools, ensure_ascii=False, separators=(",", ":")
                    ),
                    "extra_info": {"index": decision_id},
                }
            )
            counts[f"{split}_decisions"] += 1
    return rows, counts


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compile_dataset(
    *, inputs: dict[str, Path], output_dir: Path, overwrite: bool
) -> dict[str, Any]:
    """Compile domains and keep Train and Test artifacts physically separate."""

    targets = {
        "train": output_dir / "train.jsonl",
        "test": output_dir / "test.jsonl",
        "report": output_dir / "dataset_report.json",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output already exists: {existing}")
    all_rows = []
    domain_reports = {}
    for domain, results_path in sorted(inputs.items()):
        rows, counts = compile_domain_rows(domain=domain, results_path=results_path)
        all_rows.extend(rows)
        domain_reports[domain] = {
            "results_path": str(results_path),
            **dict(sorted(counts.items())),
        }
    split_rows = {
        split: sorted(
            (row for row in all_rows if row["split"] == split),
            key=lambda row: (row["domain"], row["decision_id"]),
        )
        for split in ("train", "test")
    }
    _write_jsonl(targets["train"], split_rows["train"])
    _write_jsonl(targets["test"], split_rows["test"])
    report = {
        "method": "all real single-tool decision states from complete trajectories",
        "official_split_policy": "Train for optimization; Test for held-out evaluation",
        "contains_official_reference_actions": False,
        "contains_official_final_rewards": False,
        "train_rows": len(split_rows["train"]),
        "test_rows": len(split_rows["test"]),
        "domains": domain_reports,
    }
    targets["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def _parse_input(value: str) -> tuple[str, Path]:
    domain, separator, path = value.partition("=")
    if not separator or not domain or not path:
        raise argparse.ArgumentTypeError("Expected DOMAIN=/path/to/results.json")
    return domain, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=_parse_input,
        help="Repeat DOMAIN=/path/to/results.json for Airline and Retail",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inputs = dict(args.input)
    if len(inputs) != len(args.input):
        raise ValueError("Each domain may be provided only once")
    report = compile_dataset(
        inputs=inputs,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
