from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ecog_decoding.training import position_velocity_huber_loss
from scripts.train_event_grouped_lars_e2e_nested import (
    batch_loss,
    build_model,
    event_grouped_cv_splits,
    fit_or_load_inner_lars,
    hurdle_validation_nll,
)
from scripts.train_event_grouped_lars_lstm import indices_from_intervals
from scripts.summarize_event_lars_lstm_cv import morphology_metrics
from scripts.evaluate_cv_ensemble_final_validation import (
    frozen_oof_seed_inclusion,
    movement_groups,
    resolve_ensemble_spec,
    restore_model,
)
from scripts.train_exact_window_end_to_end import ExactWindowFingerDecoder


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


def test_nested_builder_supports_trainable_overcomplete_wavelet_tree() -> None:
    selected = np.array([0, 17, 324], dtype=np.int64)
    model = build_model(
        input_channels=2,
        ica=np.eye(2, dtype=np.float32),
        selected=selected,
        mean=np.zeros(selected.size, dtype=np.float32),
        scale=np.ones(selected.size, dtype=np.float32),
        coefficients=np.array([0.2, -0.1, 0.05], dtype=np.float32),
        intercept=0.01,
        hidden_size=3,
        near_zero_std=1.0e-3,
        output_activation="linear",
        device=torch.device("cpu"),
        frontend="overcomplete",
    )

    output = model(torch.randn(1, 2, 2, 1000))

    assert output.shape == (1, 2)
    assert model.frontend == "overcomplete"


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


def test_position_velocity_huber_rewards_matching_motion() -> None:
    target = torch.tensor([[0.0, 0.2, 0.7, 0.2]], dtype=torch.float32)
    exact, parts = position_velocity_huber_loss(
        target,
        target,
        level_scale=torch.tensor(0.7),
        velocity_scale=torch.tensor(0.5),
    )
    flat, _ = position_velocity_huber_loss(
        torch.zeros_like(target),
        target,
        level_scale=torch.tensor(0.7),
        velocity_scale=torch.tensor(0.5),
    )

    assert exact.item() == 0.0
    assert parts["velocity"].item() == 0.0
    assert flat > exact
    _, flat_parts = position_velocity_huber_loss(
        torch.zeros_like(target),
        target,
        level_scale=torch.tensor(0.7),
        velocity_scale=torch.tensor(0.5),
    )
    assert flat_parts["velocity"] > 0


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


def test_final_validation_event_groups_merge_overlapping_padding() -> None:
    target = np.zeros(30, dtype=np.float32)
    target[[5, 6, 12, 20]] = 0.5
    groups = movement_groups(target, threshold=0.08, padding=3)

    assert groups == [
        {"start": 2, "stop": 16},
        {"start": 17, "stop": 24},
    ]


def test_final_validation_restores_crossfold_checkpoint(tmp_path) -> None:
    model = ExactWindowFingerDecoder(
        input_channels=2,
        component_count=2,
        selected_indices=np.arange(3),
        feature_mean=np.zeros(3, dtype=np.float32),
        feature_scale=np.ones(3, dtype=np.float32),
        hidden_size=4,
        frontend="asymmetric",
        head_initialization="lars_linear_regime",
        output_activation="softplus",
    )
    checkpoint = tmp_path / "model.pt"
    summary = tmp_path / "summary.json"
    torch.save(
        {"model_state_dict": model.state_dict(), "feature_indices": np.arange(3)},
        checkpoint,
    )
    summary.write_text('{"configuration":{"output_activation":"softplus"}}')

    restored = restore_model(checkpoint, summary, input_channels=2, device=torch.device("cpu"))

    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_final_validation_freezes_seed_inclusion_from_oof(tmp_path) -> None:
    input_root = tmp_path / "models"
    fold_root = tmp_path / "folds"
    definition_root = fold_root / "sub1" / "thumb"
    definition_root.mkdir(parents=True)
    (definition_root / "folds.json").write_text('{"training_rows": 6}')
    cleaned = np.array([0.0, 0.2, 0.7, 0.1, 0.5, 0.0], dtype=np.float32)
    active = np.array([0.01, 0.18, 0.60, 0.08, 0.45, 0.02], dtype=np.float32)
    for fold, indices in enumerate((np.array([0, 1]), np.array([2, 3]), np.array([4, 5]))):
        for seed, prediction in ((0, active), (1, np.zeros_like(active))):
            root = input_root / "sub1" / "thumb" / f"fold{fold}" / f"seed{seed}"
            root.mkdir(parents=True)
            np.save(root / "validation_prediction.npy", prediction[indices])
            if seed == 0:
                np.save(root / "validation_indices.npy", indices)
                np.save(root / "validation_cleaned_target.npy", cleaned[indices])

    included, report = frozen_oof_seed_inclusion(
        input_root, fold_root, subject=1, finger_name="thumb", seeds=(0, 1)
    )

    assert included == [0]
    assert report["collapsed_seeds"] == [1]
    assert report["selection_partition"] == "training-partition out-of-fold predictions only"


def test_final_validation_resolves_per_finger_ensemble_override() -> None:
    mapping = {
        "default": {"input_root": "outputs/base", "seeds": [0, 1]},
        "subjects": {
            3: {
                "little": {"input_root": "outputs/long", "seeds": [1, 2]},
            }
        },
    }

    default = resolve_ensemble_spec(Path("unused"), (7,), mapping, 3, "ring")
    overridden = resolve_ensemble_spec(Path("unused"), (7,), mapping, 3, "little")

    assert default == (Path("outputs/base"), (0, 1))
    assert overridden == (Path("outputs/long"), (1, 2))
