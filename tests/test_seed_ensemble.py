import numpy as np
import pytest

from scripts.evaluate_seed_ensemble import checked_stack, pearson


def test_pearson_handles_scale_and_constant_input() -> None:
    target = np.array([1.0, 2.0, 3.0])
    assert pearson(target, 2.0 * target + 7.0) == pytest.approx(1.0)
    assert pearson(target, np.ones_like(target)) == 0.0


def test_checked_stack_rejects_mismatched_prediction_shapes() -> None:
    with pytest.raises(ValueError, match="prediction shapes must match"):
        checked_stack(
            [np.zeros((4, 5), dtype=np.float32), np.zeros((3, 5), dtype=np.float32)],
            "validation",
        )


def test_checked_stack_preserves_candidate_axis() -> None:
    values = [np.full((4, 5), index, dtype=np.float32) for index in range(3)]
    stacked = checked_stack(values, "validation")
    assert stacked.shape == (3, 4, 5)
    np.testing.assert_array_equal(stacked[:, 0, 0], np.array([0.0, 1.0, 2.0]))
