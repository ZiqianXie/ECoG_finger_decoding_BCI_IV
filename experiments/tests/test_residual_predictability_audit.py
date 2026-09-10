import numpy as np

from audit_single_wavelet_residual_predictability import causal_ema


def test_causal_ema_resets_at_interval_boundaries() -> None:
    values = np.asarray([[1.0], [3.0], [100.0], [10.0], [14.0]], dtype=np.float32)

    observed = causal_ema(values, [[0, 2], [3, 5]], decay=0.5)

    np.testing.assert_allclose(observed[[0, 1], 0], [1.0, 2.0])
    np.testing.assert_allclose(observed[[3, 4], 0], [10.0, 12.0])
    assert observed[2, 0] == 0.0
