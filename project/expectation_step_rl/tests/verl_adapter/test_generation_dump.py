import json
from types import SimpleNamespace

import numpy as np
import pytest

from expectation_step_rl.verl_adapter.generation_dump import dump_generations


def test_dump_generations_serializes_numpy_diagnostics(tmp_path) -> None:
    trainer = SimpleNamespace(global_steps=3)

    dump_generations(
        trainer,
        inputs=["prompt"],
        outputs=["answer"],
        gts=[np.int64(17)],
        scores=[np.float32(0.25)],
        reward_extra_infos_dict={
            "calibration_count": np.array([np.int64(8)]),
            "tool_error": np.array([np.bool_(False)]),
            "token_ids": np.array([[1, 2, 3]], dtype=np.int64),
        },
        dump_path=str(tmp_path),
    )

    row = json.loads((tmp_path / "3.jsonl").read_text())
    assert row == {
        "input": "prompt",
        "output": "answer",
        "gts": 17,
        "score": 0.25,
        "step": 3,
        "calibration_count": 8,
        "tool_error": False,
        "token_ids": [1, 2, 3],
    }


def test_dump_generations_rejects_unknown_objects(tmp_path) -> None:
    trainer = SimpleNamespace(global_steps=1)

    with pytest.raises(TypeError, match="is not JSON serializable"):
        dump_generations(
            trainer,
            inputs=["prompt"],
            outputs=["answer"],
            gts=[object()],
            scores=[0.0],
            reward_extra_infos_dict={},
            dump_path=str(tmp_path),
        )
