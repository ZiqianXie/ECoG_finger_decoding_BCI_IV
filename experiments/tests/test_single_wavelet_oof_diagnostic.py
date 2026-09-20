from __future__ import annotations

import numpy as np

from experiments.scripts.diagnose_single_wavelet_oof import (
    state_f1,
    strongest_windows,
    worst_rest_window,
)


def test_state_f1_matches_binary_counts() -> None:
    target = np.array([0.0, 0.2, 0.3, 0.0], dtype=np.float32)
    prediction = np.array([0.0, 0.2, 0.0, 0.2], dtype=np.float32)
    # One true positive, one false positive, and one false negative.
    assert state_f1(prediction, target, threshold=0.1) == 0.5


def test_window_selection_is_nonoverlapping_and_rest_constrained() -> None:
    target = np.zeros(40, dtype=np.float32)
    target[4:8] = 1.0
    target[28:32] = 0.8
    starts = strongest_windows(target, width=6, count=2)
    assert len(starts) == 2
    assert abs(starts[1] - starts[0]) >= 6

    prediction = np.zeros_like(target)
    prediction[15:21] = 2.0
    rest_start = worst_rest_window(prediction, target, width=6, threshold=0.1)
    assert rest_start == 15
