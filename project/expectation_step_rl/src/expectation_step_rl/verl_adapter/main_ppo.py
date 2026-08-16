"""Launch the pinned verl trainer with project compatibility overrides."""

from __future__ import annotations

import ray
from verl.trainer import main_ppo as verl_main_ppo
from verl.trainer.ppo.ray_trainer import Role
from verl.workers.engine_workers import ActorRolloutRefWorker

from expectation_step_rl.verl_adapter.actor_worker import (
    Tau2ActorRolloutRefWorker,
)
from expectation_step_rl.verl_adapter.generation_dump import (
    install_generation_dump_patch,
)


class Tau2TaskRunner(verl_main_ppo.TaskRunner):
    """Install compatibility code inside the Ray task-runner process."""

    def add_actor_rollout_worker(self, config):
        actor_cls, worker_group_cls = super().add_actor_rollout_worker(config)
        if actor_cls is not ActorRolloutRefWorker:
            raise RuntimeError(
                "The inference-only adapter exporter requires verl's new worker"
            )
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        lora_path = config.actor_rollout_ref.model.get("lora_adapter_path")
        role = (
            Role.ActorRollout
            if lora_rank > 0 or lora_path is not None
            else Role.ActorRolloutRef
        )
        self.role_worker_mapping[role] = ray.remote(Tau2ActorRolloutRefWorker)
        return Tau2ActorRolloutRefWorker, worker_group_cls

    def run(self, config) -> None:
        install_generation_dump_patch()
        print("Using Tau2 new-engine worker with LoRA-only checkpoint export")
        super().run(config)


def main() -> None:
    """Delegate Hydra configuration and training to the pinned verl entrypoint."""

    verl_main_ppo.TaskRunner = Tau2TaskRunner
    verl_main_ppo.main()


if __name__ == "__main__":
    main()
