import numpy as np
import torch
import cross_validate_single_wavelet as cv
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

from cross_validate_single_wavelet import (
    DenseSequenceSampler,
    FROZEN_UPDATES,
    UNFREEZE_AFTER,
    UNFROZEN_UPDATES,
    UniformGroupSampler,
    one_standard_error_selection,
    cached_initialization_features,
    load_initialization,
    load_or_create_initialization,
    grouped_velocity_scale,
    masked_sequence_correlation_loss,
    movement_positive_weight,
    save_initialization,
    scoped_event_groups,
    sequence_correlation_loss,
    split_local_raw_trajectory_blend,
    suppress_weak_little_events,
    trajectory_mse_loss,
    validation_metrics,
)


def test_movement_positive_weight_balances_training_scope() -> None:
    target = torch.zeros(6, 5)
    target[[1, 4], 2] = 0.2

    weight = movement_positive_weight(target, np.arange(6), 2, 0.1)

    torch.testing.assert_close(weight, torch.tensor(2.0))


def test_movement_positive_weight_can_balance_all_fingers() -> None:
    target = torch.zeros(6, 5)
    target[[1, 4], 0] = 0.2
    target[[0, 2, 4], 1] = 0.2

    weight = movement_positive_weight(target, np.arange(6), None, 0.1)

    torch.testing.assert_close(
        weight, torch.tensor([2.0, 1.0, 6.0, 6.0, 6.0])
    )


def test_grouped_velocity_scale_excludes_interval_jumps() -> None:
    target = torch.tensor([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])

    scale = grouped_velocity_scale(target, [[0, 3], [3, 6]])

    torch.testing.assert_close(scale, torch.tensor(0.01))


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


def test_sequence_correlation_loss_rewards_shape_and_backpropagates() -> None:
    target = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    exact = target.clone().requires_grad_(True)
    reversed_values = torch.flip(target, dims=(1,))

    exact_loss = sequence_correlation_loss(exact, target)
    reversed_loss = sequence_correlation_loss(reversed_values, target)
    exact_loss.backward()

    assert torch.isclose(exact_loss, torch.tensor(0.0), atol=1.0e-6)
    assert reversed_loss > exact_loss
    assert exact.grad is not None
    assert torch.isfinite(exact.grad).all()


def test_masked_sequence_correlation_uses_only_movement_bins() -> None:
    target = torch.tensor([[50.0, 0.0, 1.0, 2.0, -50.0]])
    prediction = torch.tensor([[-50.0, 0.0, 1.0, 2.0, 50.0]], requires_grad=True)
    mask = torch.tensor([[False, True, True, True, False]])

    loss = masked_sequence_correlation_loss(prediction, target, mask)
    loss.backward()

    assert torch.isclose(loss, torch.tensor(0.0), atol=1.0e-6)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_masked_sequence_correlation_skips_rows_without_three_bins() -> None:
    prediction = torch.tensor([[0.0, 1.0], [2.0, 1.0]], requires_grad=True)
    target = prediction.detach().clone()
    mask = torch.ones_like(prediction, dtype=torch.bool)

    loss = masked_sequence_correlation_loss(prediction, target, mask)
    loss.backward()

    assert torch.isclose(loss, torch.tensor(0.0))
    assert prediction.grad is not None


def test_trajectory_mse_loss_can_emphasize_movement_bins() -> None:
    target = torch.tensor([[0.0, 0.2]])
    prediction = torch.tensor([[1.0, 1.2]])

    unweighted = trajectory_mse_loss(prediction, target, torch.tensor(1.0), 0.1, 1.0)
    weighted = trajectory_mse_loss(prediction, target, torch.tensor(1.0), 0.1, 3.0)

    torch.testing.assert_close(unweighted, torch.tensor(1.0))
    torch.testing.assert_close(weighted, torch.tensor(1.0))


def test_trajectory_mse_loss_changes_relative_bin_contribution() -> None:
    target = torch.tensor([[0.0, 0.2]])
    prediction = torch.tensor([[1.0, 0.2]])

    unweighted = trajectory_mse_loss(prediction, target, torch.tensor(1.0), 0.1, 1.0)
    movement_weighted = trajectory_mse_loss(
        prediction, target, torch.tensor(1.0), 0.1, 3.0
    )

    torch.testing.assert_close(unweighted, torch.tensor(0.5))
    torch.testing.assert_close(movement_weighted, torch.tensor(0.25))


def test_raw_trajectory_blend_fits_affine_map_on_training_rows_only() -> None:
    cleaned = torch.tensor([0.0, 1.0, 2.0, 100.0])
    raw = torch.tensor([1.0, 3.0, 5.0, 7.0])
    training_rows = torch.tensor([0, 1, 2])

    blended = split_local_raw_trajectory_blend(
        cleaned, raw, training_rows, blend=1.0
    )

    torch.testing.assert_close(blended, torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_initialization_cache_atomic_round_trip(tmp_path) -> None:
    initialization = {
        "selected_indices": np.asarray([1, 3], dtype=np.int64),
        "coefficients": np.asarray([0.2, -0.4], dtype=np.float32),
    }
    spatial = np.arange(6, dtype=np.float32).reshape(2, 3)
    target = np.asarray([[0.1], [0.2]], dtype=np.float32)
    split = {"fold": 1}
    audit = {"selected_features": 2}

    save_initialization(tmp_path, initialization, spatial, target, split, audit)
    loaded, loaded_spatial, loaded_target, loaded_split = load_initialization(
        tmp_path
    )

    assert set(loaded) == set(initialization)
    for name, values in initialization.items():
        np.testing.assert_array_equal(loaded[name], values)
    np.testing.assert_array_equal(loaded_spatial, spatial)
    np.testing.assert_array_equal(loaded_target, target)
    assert loaded_split["fold"] == 1
    assert loaded_split["initialization_audit"] == audit
    assert not list(tmp_path.glob(".*"))


def test_causal_cached_features_join_direct_candidates_and_current_sources() -> None:
    initialization = {
        "candidate_features": np.arange(12, dtype=np.float32).reshape(3, 4),
        "causal_features": np.arange(6, dtype=np.float32).reshape(3, 2),
    }

    observed = cached_initialization_features(initialization, "causal_candidate")

    np.testing.assert_array_equal(observed[:, :4], initialization["candidate_features"])
    np.testing.assert_array_equal(observed[:, 4:], initialization["causal_features"])


def test_selected_causal_cached_features_join_selected_and_current_sources() -> None:
    initialization = {
        "selected_features": np.arange(6, dtype=np.float32).reshape(3, 2),
        "selected_causal_features": np.arange(3, dtype=np.float32).reshape(3, 1),
    }

    observed = cached_initialization_features(initialization, "selected_causal")

    np.testing.assert_array_equal(observed[:, :2], initialization["selected_features"])
    np.testing.assert_array_equal(
        observed[:, 2:], initialization["selected_causal_features"]
    )


def test_initialization_cache_creation_is_serialized(tmp_path) -> None:
    calls = 0
    calls_lock = Lock()

    def create():
        nonlocal calls
        with calls_lock:
            calls += 1
        sleep(0.05)
        return (
            {"coefficients": np.asarray([0.3], dtype=np.float32)},
            np.asarray([[1.0]], dtype=np.float32),
            np.asarray([[0.2]], dtype=np.float32),
            {"fold": 0},
            {"selected_features": 1},
        )

    cache = tmp_path / "outer0" / "inner0"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: load_or_create_initialization(cache, create), range(2)
            )
        )

    assert calls == 1
    assert [result[3]["fold"] for result in results] == [0, 0]
    np.testing.assert_array_equal(
        results[0][0]["coefficients"], results[1][0]["coefficients"]
    )


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
