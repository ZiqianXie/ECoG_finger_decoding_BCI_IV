from __future__ import annotations

import numpy as np

from experiments.scripts.cross_validate_signed_lmp_ridge import binned_signed_mean


def test_binned_signed_mean_preserves_polarity() -> None:
    values = np.asarray(
        [[-2.0, 1.0], [-4.0, 3.0], [2.0, -1.0], [4.0, -3.0]],
        dtype=np.float32,
    )
    observed = binned_signed_mean(values, samples_per_bin=2)
    expected = np.asarray([[-3.0, 2.0], [3.0, -2.0]], dtype=np.float32)
    assert np.array_equal(observed, expected)
