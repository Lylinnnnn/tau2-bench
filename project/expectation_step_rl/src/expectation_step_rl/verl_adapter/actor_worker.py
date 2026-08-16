"""New-engine verl worker that also exports an inference-only LoRA adapter."""

from __future__ import annotations

import tempfile
import shutil
from pathlib import Path
from typing import Any

import torch.distributed as dist
from safetensors.torch import save_file
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker

def save_lora_adapter(
    *,
    engine: Any,
    output_dir: Path,
    layered_summon: bool,
) -> None:
    """Collect the trained adapter from FSDP and save it on rank zero.
    
    Saves to a local temp directory first, then copies to final destination
    to avoid I/O errors on OSS filesystems.
    """

    peft_model = getattr(engine.module, "_fsdp_wrapped_module", engine.module)
    peft_config = getattr(peft_model, "peft_config", {}).get("default")
    if peft_config is None:
        raise RuntimeError("LoRA training is enabled but no PEFT config was found")

    parameters, _ = engine.get_per_tensor_param(
        layered_summon=layered_summon,
        base_sync_done=True,
    )
    adapter_state = {
        name: tensor.detach().cpu().contiguous() for name, tensor in parameters
    }
    if not adapter_state or not any("lora_" in name for name in adapter_state):
        raise RuntimeError("FSDP returned no LoRA parameters for adapter export")

    error = None
    if dist.get_rank() == 0:
        try:
            # Save to local temp directory first to avoid OSS I/O errors
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                save_file(adapter_state, tmp_path / "adapter_model.safetensors")
                peft_config.save_pretrained(tmp_path)
                
                # Copy to final destination
                output_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tmp_path / "adapter_model.safetensors", output_dir / "adapter_model.safetensors")
                shutil.copy2(tmp_path / "adapter_config.json", output_dir / "adapter_config.json")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    status = [error]
    dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(f"LoRA adapter export failed: {status[0]}")
    dist.barrier()
    if dist.get_rank() == 0:
        print(
            "Saved inference-only LoRA adapter: "
            f"{output_dir} ({len(adapter_state)} tensors)"
        )