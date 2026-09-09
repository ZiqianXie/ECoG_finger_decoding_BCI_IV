import numpy as np
import cross_validate_single_wavelet as cv

from cross_validate_single_wavelet import (
    DenseSequenceSampler,
    FROZEN_UPDATES,
    UNFREEZE_AFTER,
    UNFROZEN_UPDATES,
    UniformGroupSampler,
    one_standard_error_selection,
    scoped_event_groups,
    suppress_weak_little_events,
    validation_metrics,
)


def schedules() -> list[str]:
    return (
        ["frozen_0"]
        + [f"frozen_{value}" for value in FROZEN_UPDATES]
        + [
            f"unfrozen_{UNFREEZE_AFTER}_{value}"
            for value in UNFROZEN_UPDATES
        ]
    )


def test_scoped_groups_stay_inside_training_intervals() -> None:
    target = np.zeros((240, 5), dtype=np.float32)
    target[30:45, 4] = 0.8
    target[80:95, 4] = 0.9
    target[105:115, 4] = 0.6
    target[150:165, 4] = 0.7
    target[205:220, 4] = 0.8
    groups, _, _ = scoped_event_groups(
        target,
        [[0, 120], [130, 240]],
        threshold=0.08,
        merge_gap_bins=3,
        minimum_event_bins=3,
        maximum_rest_group_bins=50,
    )
    assert groups[0][0] == 0
    assert groups[-1][1] == 240
    assert all(stop <= 120 or start >= 130 for start, stop in groups)


def test_little_event_decontamination_suppresses_only_dominated_events() -> None:
    target = np.zeros((80, 5), dtype=np.float32)
    target[10:20, 3] = 1.0
    target[11:19, 4] = 0.2
    target[40:50, 3] = 0.5
    target[40:50, 4] = 1.0

    cleaned = suppress_weak_little_events(target, [[0, 80]])

    assert np.max(cleaned[11:19, 4]) == 0.0
    np.testing.assert_allclose(cleaned[40:50, 4], target[40:50, 4])
    np.testing.assert_allclose(cleaned[:, :4], target[:, :4])


def test_little_event_decontamination_does_not_cross_interval_boundaries() -> None:
    target = np.zeros((30, 5), dtype=np.float32)
    target[11:15, 3] = 1.0
    target[11:15, 4] = 0.2
    target[15:21, 4] = 0.8

    cleaned = suppress_weak_little_events(target, [[0, 15], [15, 30]])

    np.testing.assert_allclose(cleaned[15:21, 4], target[15:21, 4])


def test_uniform_sampler_visits_each_group_before_repeating() -> None:
    sampler = UniformGroupSampler([[0, 20], [30, 50], [60, 80]], steps=10, seed=3)
    starts = sampler.sample(3)
    represented = {0 if value < 20 else 1 if value < 50 else 2 for value in starts}
    assert represented == {0, 1, 2}


def test_validation_metrics_are_ideal_for_exact_prediction() -> None:
    target = np.asarray([0.0, 0.0, 0.2, 0.7, 0.1, 0.0, 0.3, 0.8], dtype=np.float32)
    metrics = validation_metrics(
        target.copy(), target, target, [[0, 4], [4, 8]], 0.1, 0.05
    )
    assert metrics["event_macro_nmse"] == 0.0
    assert np.isclose(metrics["raw_pcc"], 1.0)
    assert np.isclose(metrics["amplitude_slope"], 1.0)
    assert np.isclose(metrics["velocity_pcc"], 1.0)
    assert np.isclose(metrics["movement_balanced_accuracy"], 1.0)


def test_one_standard_error_prefers_earlier_checkpoint() -> None:
    records = []
    for fold in range(5):
        metrics = {}
        for schedule in schedules():
            value = 1.0
            if schedule == "frozen_20":
                value = 0.954 + 0.02 * (fold % 2)
            if schedule == "frozen_40":
                value = 0.95 + 0.02 * (fold % 2)
            metrics[schedule] = {"event_macro_nmse": value, "raw_pcc": 0.5}
        records.append({"metrics": metrics})
    selected, _ = one_standard_error_selection(records)
    assert selected == "frozen_20"


def test_required_lstm_update_excludes_linear_baseline() -> None:
    records = []
    for _ in range(5):
        metrics = {}
        for schedule in schedules():
            metrics[schedule] = {
                "event_macro_nmse": 0.8 if schedule == "frozen_0" else 1.0,
                "raw_pcc": 0.6,
            }
        records.append({"metrics": metrics})
    selected, _ = one_standard_error_selection(
        records, require_lstm_update=True
    )
    assert selected == "frozen_20"


def test_schedule_order_supports_lstm_only_sweep(monkeypatch) -> None:
    monkeypatch.setattr(cv, "FROZEN_UPDATES", (5, 20, 80))
    monkeypatch.setattr(cv, "UNFROZEN_UPDATES", ())

    assert cv.schedule_order() == ["frozen_0", "frozen_5", "frozen_20", "frozen_80"]


def test_raw_pcc_selection_maximizes_the_requested_metric() -> None:
    records = []
    for _ in range(4):
        metrics = {}
        for schedule in schedules():
            pcc = 0.4
            if schedule == "frozen_20":
                pcc = 0.5
            if schedule == "frozen_40":
                pcc = 0.6
            metrics[schedule] = {"event_macro_nmse": 1.0, "raw_pcc": pcc}
        records.append({"metrics": metrics})
    selected, _ = one_standard_error_selection(
        records, selection_metric="raw_pcc", selection_rule="best"
    )
    assert selected == "frozen_40"


def test_dense_sampler_uses_only_strided_interval_starts() -> None:
    sampler = DenseSequenceSampler([[0, 10], [20, 30]], steps=4, stride=3, seed=7)
    observed = sampler.sample(6)
    assert set(observed.tolist()) == {0, 3, 6, 20, 23, 26}
