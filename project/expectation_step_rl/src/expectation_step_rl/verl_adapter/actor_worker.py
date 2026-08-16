"""New-engine verl worker that also exports an inference-only LoRA adapter."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch.distributed as dist
from safetensors.torch import save_file
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker


def _copy_file_contents(source: Path, destination: Path) -> None:
    """Copy bytes without filesystem metadata or fast-copy syscalls."""

    with source.open("rb") as source_handle, destination.open("wb") as output_handle:
        shutil.copyfileobj(source_handle, output_handle, length=16 * 1024 * 1024)


def save_lora_adapter(
    *,
    engine: Any,
    output_dir: Path,
    layered_summon: bool,
) -> None:
    """Collect the trained adapter from FSDP and save it on rank zero.

    Serialize locally, then stream file contents to the OSS mount because its
    FUSE implementation does not support safetensors' direct write operations.
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
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                save_file(adapter_state, tmp_path / "adapter_model.safetensors")
                peft_config.save_pretrained(tmp_path)

                output_dir.mkdir(parents=True, exist_ok=True)
                _copy_file_contents(
                    tmp_path / "adapter_model.safetensors",
                    output_dir / "adapter_model.safetensors",
                )
                _copy_file_contents(
                    tmp_path / "adapter_config.json",
                    output_dir / "adapter_config.json",
                )
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


class Tau2ActorRolloutRefWorker(ActorRolloutRefWorker):
    """Add adapter export missing from verl's new FSDP worker."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self,
        local_path,
        hdfs_path=None,
        global_step=0,
        max_ckpt_to_keep=None,
    ) -> None:
        super().save_checkpoint(
            local_path,
            hdfs_path,
            global_step,
            max_ckpt_to_keep,
        )
        save_lora_adapter(
            engine=self.actor.engine,
            output_dir=Path(local_path) / "lora_adapter",
            layered_summon=self.layered_summon,
        )
