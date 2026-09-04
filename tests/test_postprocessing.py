import numpy as np
import pytest

from ecog_decoding.postprocessing import project_nonnegative


def test_projection_sets_only_negative_values_to_zero() -> None:
    prediction = np.array([[-0.2, 0.0, 0.3], [0.5, -1.0, 2.0]])
    projected = project_nonnegative(prediction)
    np.testing.assert_allclose(projected, [[0.0, 0.0, 0.3], [0.5, 0.0, 2.0]])


def test_projection_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        project_nonnegative(np.array([0.0, np.nan]))
