import numpy as np

from scripts.prepare_split_safe_targets import split_safe_local_target


def test_validation_changes_do_not_alter_fit_target() -> None:
    time = np.linspace(0.0, 10.0, 250)
    raw = np.column_stack((0.1 * time + np.sin(time) ** 2, np.cos(time) ** 2))
    split = 150
    first, _, first_scale = split_safe_local_target(
        raw,
        split,
        sampling_rate_hz=25.0,
        window_seconds=4.0,
        quantile=0.1,
        smoothing_seconds=0.16,
    )
    changed = raw.copy()
    changed[split:] += 100.0
    second, _, second_scale = split_safe_local_target(
        changed,
        split,
        sampling_rate_hz=25.0,
        window_seconds=4.0,
        quantile=0.1,
        smoothing_seconds=0.16,
    )
    assert np.array_equal(first[:split], second[:split])
    assert np.array_equal(first_scale, second_scale)
