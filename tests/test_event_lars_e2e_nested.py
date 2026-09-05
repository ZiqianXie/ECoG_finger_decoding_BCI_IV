import numpy as np
import torch
from types import SimpleNamespace

from scripts.train_event_grouped_lars_e2e_nested import (
    batch_loss,
    event_grouped_cv_splits,
    fit_or_load_inner_lars,
    hurdle_validation_nll,
)
from scripts.train_event_grouped_lars_lstm import indices_from_intervals
from scripts.summarize_event_lars_lstm_cv import morphology_metrics


def test_event_grouped_subfolds_are_disjoint_and_complete() -> None:
    intervals = [[0, 9], [14, 29], [35, 46], [52, 61], [70, 83], [90, 98]]
    training_indices = indices_from_intervals(intervals)
    splits = event_grouped_cv_splits(intervals, training_indices, folds=3)
    validation_seen: list[int] = []
    for training, validation in splits:
        assert np.intersect1d(training, validation).size == 0
        assert np.union1d(training, validation).size == training_indices.size
        validation_seen.extend(validation.tolist())
    assert np.array_equal(np.sort(validation_seen), np.arange(training_indices.size))


def test_morphology_metrics_reward_matching_shape() -> None:
    target = np.array([0.0, 0.0, 0.2, 0.7, 0.2, 0.0, 0.0], dtype=np.float32)
    groups = [{"start": 0, "stop": target.size}]
    exact = morphology_metrics(target, target, groups)
    flat = morphology_metrics(np.zeros_like(target), target, groups)
    assert exact["cleaned_ccc"] > flat["cleaned_ccc"]
    assert exact["velocity_pcc"] > flat["velocity_pcc"]
    assert exact["movement_state_f1"] > flat["movement_state_f1"]
    assert exact["median_event_peak_ratio"] == 1.0


def test_hurdle_likelihood_backpropagates_through_both_heads() -> None:
    target = torch.tensor([[0.0, 0.2, 0.5]])
    state_logit = torch.zeros_like(target, requires_grad=True)
    amplitude = torch.full_like(target, 0.3, requires_grad=True)
    prediction = torch.sigmoid(state_logit) * amplitude
    loss = batch_loss(
        (prediction, state_logit, amplitude),
        target,
        SimpleNamespace(movement_threshold=0.08),
        torch.tensor(0.5),
        torch.tensor(1.0),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert state_logit.grad is not None
    assert amplitude.grad is not None


def test_null_lars_uses_training_only_ridge_fallback(tmp_path) -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(36, 8)).astype(np.float32)
    target = np.zeros(36, dtype=np.float32)
    intervals = [[0, 12], [12, 24], [24, 36]]

    saved = fit_or_load_inner_lars(
        features_all=features,
        target_all=target,
        training_intervals=intervals,
        cache=tmp_path / "selection.npz",
        max_features=8,
    )

    assert saved["selection_method"] == "ridge_fallback_after_null_lars"
    assert len(saved["selected_source"]) == 8


def test_hurdle_selection_likelihood_rewards_correct_components() -> None:
    target = np.array([0.0, 0.0, 0.4, 0.7], dtype=np.float32)
    good = hurdle_validation_nll(
        np.array([0.05, 0.10, 0.90, 0.95]),
        np.array([0.2, 0.1, 0.4, 0.7]),
        target,
        movement_threshold=0.08,
        amplitude_scale=0.7,
    )
    bad = hurdle_validation_nll(
        np.array([0.95, 0.90, 0.10, 0.05]),
        np.array([0.8, 0.8, 0.1, 0.1]),
        target,
        movement_threshold=0.08,
        amplitude_scale=0.7,
    )

    assert good < bad
