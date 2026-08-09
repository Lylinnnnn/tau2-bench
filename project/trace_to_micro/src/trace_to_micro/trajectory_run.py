"""Launch complete behavior-policy trajectories through the τ² runner."""

from tau2.data_model.simulation import Results, TextRunConfig
from tau2.runner import run_domain
from trace_to_micro.config import TrajectoryConfig


def build_run_config(
    config: TrajectoryConfig,
    *,
    num_tasks: int | None = None,
    save_to: str | None = None,
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
        auto_resume=True,
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
) -> Results:
    """Generate or resume exactly one configured rollout per task/trial."""

    return run_domain(build_run_config(config, num_tasks=num_tasks, save_to=save_to))
