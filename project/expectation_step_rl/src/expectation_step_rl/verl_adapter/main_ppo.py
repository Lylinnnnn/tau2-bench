"""Launch the pinned verl trainer with project compatibility overrides."""

from __future__ import annotations

from verl.trainer import main_ppo as verl_main_ppo

from expectation_step_rl.verl_adapter.generation_dump import (
    install_generation_dump_patch,
)


class Tau2TaskRunner(verl_main_ppo.TaskRunner):
    """Install compatibility code inside the Ray task-runner process."""

    def run(self, config) -> None:
        install_generation_dump_patch()
        super().run(config)


def main() -> None:
    """Delegate Hydra configuration and training to the pinned verl entrypoint."""

    verl_main_ppo.TaskRunner = Tau2TaskRunner
    verl_main_ppo.main()


if __name__ == "__main__":
    main()
