import numpy as np
import pytest

from ecog_decoding.postprocessing import (
    fit_nonnegative_gain,
    project_nonnegative,
    smooth_nonnegative,
)


def test_projection_sets_only_negative_values_to_zero() -> None:
    prediction = np.array([[-0.2, 0.0, 0.3], [0.5, -1.0, 2.0]])
    projected = project_nonnegative(prediction)
    np.testing.assert_allclose(projected, [[0.0, 0.0, 0.3], [0.5, 0.0, 2.0]])


def test_projection_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        project_nonnegative(np.array([0.0, np.nan]))


def test_smooth_nonnegative_preserves_softplus_outputs_and_maps_linear_values() -> None:
    nonnegative = np.array([0.0, 0.2, 1.0], dtype=np.float64)
    np.testing.assert_allclose(
        smooth_nonnegative(nonnegative, already_nonnegative=True), nonnegative
    )
    mapped = smooth_nonnegative(np.array([-1.0, 0.0, 1.0]), beta=10.0)
    assert np.all(mapped > 0.0)
    assert mapped[0] < 1.0e-4
    assert mapped[2] == pytest.approx(1.0, abs=1.0e-4)


def test_fit_nonnegative_gain_is_origin_preserving_and_clamped() -> None:
    prediction = np.array([0.0, 1.0, 2.0])
    assert fit_nonnegative_gain(prediction, 2.5 * prediction) == pytest.approx(2.5)
    assert fit_nonnegative_gain(prediction, -prediction) == 0.0
