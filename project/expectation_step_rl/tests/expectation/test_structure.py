from expectation_step_rl.expectation.structure import (
    result_structure,
    result_structure_key,
)


def test_result_structure_ignores_values() -> None:
    left = '{"user_id":"alice_1","orders":[{"status":"pending"}]}'
    right = '{"user_id":"bob_2","orders":[{"status":"delivered"}]}'

    assert result_structure(left) == result_structure(right)
    assert result_structure_key(left) == result_structure_key(right)
