import numpy as np

from trace_to_micro.evaluation.consequence_metrics import (
    cluster_bootstrap_delta,
    head_metrics,
)


def test_family_balanced_metric_and_task_bootstrap_treat_families_equally() -> None:
    target_names = [
        "execution.success",
        "effect.changed",
        "output.kind.object",
    ]
    actual = np.asarray(
        [[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]],
        dtype=np.float64,
    )
    perfect = np.where(actual == 1, 0.9, 0.1)
    tied = np.full(actual.shape, 0.5)

    _, summary = head_metrics(actual, perfect, target_names)
    interval = cluster_bootstrap_delta(
        actual,
        perfect,
        tied,
        target_names,
        ["task-a", "task-a", "task-b", "task-b"],
        metric="auroc",
        samples=50,
        random_seed=300,
    )

    assert summary["all_heads"]["consequence_family_balanced_auroc"] == 1.0
    assert interval == [0.5, 0.5]
