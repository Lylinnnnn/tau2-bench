"""Load, merge, and run one saved LoRA adapter on a single GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def run_inference_smoke(
    *,
    model_path: Path,
    checkpoint_dir: Path,
    expected_step: int,
    output_path: Path,
) -> dict[str, Any]:
    """Prove that the adapter loads, merges in memory, and generates tokens."""

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Inference smoke requires exactly one visible CUDA GPU; set "
            "CUDA_VISIBLE_DEVICES to one physical GPU"
        )

    adapter_path = (
        checkpoint_dir / f"global_step_{expected_step}" / "actor" / "lora_adapter"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )
    adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
    adapter_model.eval()

    messages = [
        {
            "role": "user",
            "content": "Reply with exactly one lowercase word: ready",
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    generation_args = {
        **inputs,
        "do_sample": False,
        "max_new_tokens": 8,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        adapter_logits = adapter_model(**inputs).logits[:, -1, :].float()
        adapter_tokens = adapter_model.generate(**generation_args)
    adapter_new_tokens = adapter_tokens[:, inputs["input_ids"].shape[1] :]
    if adapter_new_tokens.numel() == 0:
        raise RuntimeError("Base model plus LoRA adapter generated no new tokens")

    adapter_next_token = int(adapter_logits.argmax(dim=-1).item())
    merged_model = adapter_model.merge_and_unload(safe_merge=True)
    merged_model.eval()
    with torch.inference_mode():
        merged_logits = merged_model(**inputs).logits[:, -1, :].float()
        merged_tokens = merged_model.generate(**generation_args)
    merged_new_tokens = merged_tokens[:, inputs["input_ids"].shape[1] :]
    if merged_new_tokens.numel() == 0:
        raise RuntimeError("In-memory merged model generated no new tokens")

    merged_next_token = int(merged_logits.argmax(dim=-1).item())
    if merged_next_token != adapter_next_token:
        raise RuntimeError(
            "LoRA and merged models disagree on the first greedy inference token: "
            f"adapter={adapter_next_token}, merged={merged_next_token}"
        )

    report = {
        "model_path": str(model_path),
        "adapter_path": str(adapter_path),
        "checkpoint_step": expected_step,
        "device_name": torch.cuda.get_device_name(0),
        "dtype": str(next(merged_model.parameters()).dtype),
        "adapter_next_token": adapter_next_token,
        "merged_next_token": merged_next_token,
        "adapter_generated_token_ids": adapter_new_tokens[0].tolist(),
        "merged_generated_token_ids": merged_new_tokens[0].tolist(),
        "adapter_text": tokenizer.decode(
            adapter_new_tokens[0], skip_special_tokens=True
        ),
        "merged_text": tokenizer.decode(merged_new_tokens[0], skip_special_tokens=True),
        "merge_saved_to_disk": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_inference_smoke(
                model_path=args.model_path,
                checkpoint_dir=args.checkpoint_dir,
                expected_step=args.expected_step,
                output_path=args.output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
