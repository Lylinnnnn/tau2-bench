"""Launch complete behavior-policy trajectories through the τ² runner."""

from pathlib import Path

from tau2.data_model.simulation import Results, TextRunConfig
from tau2.runner import run_domain
from trace_to_micro.config import TrajectoryConfig


def build_run_config(
    config: TrajectoryConfig,
    *,
    num_tasks: int | None = None,
    save_to: str | None = None,
    auto_resume: bool = True,
) -> TextRunConfig:
    """Translate the experiment TOML into τ²'s canonical run config."""

    return TextRunConfig(
        domain=config.domain,
        task_set_name=config.task_set,
        task_split_name=config.task_split,
        agent=config.agent,
        user=config.user,
        llm_agent=config.agent_llm,
        llm_user=config.user_llm,
        llm_args_agent=config.agent_llm_args,
        llm_args_user=config.user_llm_args,
        num_trials=config.num_trials,
        num_tasks=num_tasks,
        max_steps=config.max_steps,
        max_errors=config.max_errors,
        max_concurrency=config.max_concurrency,
        seed=config.seed,
        timeout=config.timeout_seconds,
        save_to=save_to or config.save_to,
        auto_resume=auto_resume,
        auto_review=False,
        hallucination_retries=0,
        verbose_logs=False,
        enforce_communication_protocol=False,
    )


def run_complete_trajectories(
    config: TrajectoryConfig,
    *,
    num_tasks: int | None = None,
    save_to: str | None = None,
    auto_resume: bool = True,
) -> Results:
    """Generate or resume exactly one configured rollout per task/trial."""

    return run_domain(
        build_run_config(
            config,
            num_tasks=num_tasks,
            save_to=save_to,
            auto_resume=auto_resume,
        )
    )


def remove_existing_run(results_path: Path) -> bool:
    """Remove one exact τ² text result file so the next run cannot resume it."""

    if not results_path.exists():
        return False
    is_results_file = results_path.name == "results.json"
    is_simulation_run = results_path.parent.parent.name == "simulations"
    if not is_results_file or not is_simulation_run:
        raise ValueError(f"Refusing to overwrite unexpected path: {results_path}")
    results_path.unlink()
    return True
