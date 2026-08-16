"""Verify that verl saved a usable LoRA adapter without heavy train state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify_checkpoint(checkpoint_dir: Path, expected_step: int) -> dict[str, int | str]:
    """Require an inference adapter and reject full FSDP train-state shards."""

    marker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    saved_step = int(marker.read_text().strip())
    if saved_step != expected_step:
        raise ValueError(
            f"Expected checkpoint step {expected_step}, found {saved_step}"
        )
    step_dir = checkpoint_dir / f"global_step_{expected_step}"
    dataloader = step_dir / "data.pt"
    if not dataloader.is_file() or dataloader.stat().st_size == 0:
        raise ValueError(f"Missing non-empty dataloader checkpoint: {dataloader}")
    actor_dir = step_dir / "actor"
    adapter_dir = actor_dir / "lora_adapter"
    adapter_model = adapter_dir / "adapter_model.safetensors"
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_model.is_file() or adapter_model.stat().st_size == 0:
        raise ValueError(f"Missing non-empty LoRA adapter weights: {adapter_model}")
    if not adapter_config.is_file() or adapter_config.stat().st_size == 0:
        raise ValueError(f"Missing non-empty LoRA adapter config: {adapter_config}")
    json.loads(adapter_config.read_text())

    heavy_patterns = (
        "model_world_size_*_rank_*.pt",
        "optim_world_size_*_rank_*.pt",
        "extra_state_world_size_*_rank_*.pt",
        "huggingface/model*.safetensors",
        "huggingface/pytorch_model*.bin",
    )
    heavy_files = [
        path for pattern in heavy_patterns for path in actor_dir.glob(pattern)
    ]
    if heavy_files:
        raise ValueError(
            "Inference-only checkpoint contains heavy training state: "
            f"{[str(path) for path in heavy_files]}"
        )

    actor_files = [
        path
        for path in actor_dir.rglob("*")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not actor_files:
        raise ValueError(f"No non-empty actor checkpoint files under: {actor_dir}")
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "step": saved_step,
        "adapter_path": str(adapter_dir),
        "adapter_bytes": adapter_model.stat().st_size + adapter_config.stat().st_size,
        "actor_file_count": len(actor_files),
        "actor_bytes": sum(path.stat().st_size for path in actor_files),
        "dataloader_bytes": dataloader.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(verify_checkpoint(args.checkpoint_dir, args.expected_step), indent=2)
    )


if __name__ == "__main__":
    main()
