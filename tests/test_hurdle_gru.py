import numpy as np

from scripts.train_blocked_hurdle_gru import (
    contiguous_starts,
    morphology,
    select_lag,
    state_f1,
)


def test_contiguous_starts_do_not_cross_purged_gap() -> None:
    indices = np.r_[np.arange(0, 8), np.arange(12, 20)]
    starts = contiguous_starts(indices, sequence_steps=4, stride=2)
    assert set(starts) == {0, 2, 4, 12, 14, 16}


def test_state_f1_and_morphology_reward_exact_trace() -> None:
    target = np.array([0.0, 0.0, 0.2, 0.5, 0.0, 0.3], dtype=np.float32)
    assert state_f1(target, target, 0.1) == 1.0
    exact = morphology(target, target, 0.1)
    flat = morphology(np.zeros_like(target), target, 0.1)
    assert exact["score"] > flat["score"]


def test_lag_requires_minimum_validation_gain() -> None:
    target = np.sin(np.linspace(0, 4 * np.pi, 100)).astype(np.float32)
    delayed = np.r_[target[0], target[:-1]]
    lag, _ = select_lag(delayed, target, maximum=3, minimum_gain=0.001)
    assert lag == -1
    rejected, _ = select_lag(delayed, target, maximum=3, minimum_gain=1.0)
    assert rejected == 0
