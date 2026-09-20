from __future__ import annotations

import numpy as np
import torch

from experiments.scripts.cross_validate_bandpower_multitask_gru import (
    BandpowerMultitaskGRU,
    ridge_skip,
    sample_interval_batch,
    screen_features,
)


def test_interval_sampler_stays_inside_groups_and_masks_padding() -> None:
    indices, mask = sample_interval_batch(
        np.random.default_rng(4),
        [[10, 14]],
        batch_size=3,
        sequence_steps=6,
    )
    assert np.array_equal(indices[:, :4], np.asarray([[10, 11, 12, 13]] * 3))
    assert np.array_equal(indices[:, 4:], np.asarray([[13, 13]] * 3))
    assert np.array_equal(mask, np.asarray([[True] * 4 + [False] * 2] * 3))


def test_ridge_skip_and_zero_residual_model_match() -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(size=(30, 4)).astype(np.float32)
    targets = rng.normal(size=(30, 5)).astype(np.float32)
    coefficients = ridge_skip(features, targets, penalty=10.0)
    model = BandpowerMultitaskGRU(4, 6, coefficients)
    with torch.inference_mode():
        observed = model(torch.from_numpy(features[None])).squeeze(0).numpy()
    assert np.allclose(observed, features @ coefficients, atol=1.0e-6)


def test_feature_screen_uses_training_rows_only() -> None:
    features = np.zeros((12, 3), dtype=np.float32)
    target = np.arange(12, dtype=np.float32)
    features[:8, 1] = target[:8]
    features[8:, 2] = target[8:]
    selected = screen_features(features, target, np.arange(8), count=1)
    assert np.array_equal(selected, np.asarray([1]))
