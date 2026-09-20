from __future__ import annotations

import numpy as np

from experiments.scripts.cross_validate_electrode_bandpower_ridge import (
    binned_log_energy,
    contextual_features,
    screen_context_features,
)


def test_binned_log_energy_keeps_every_band_and_channel() -> None:
    bands = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    energy = binned_log_energy(bands, samples_per_bin=4)
    expected = []
    for band in bands:
        bins = band.reshape(2, 4, 3)
        expected.append(np.log1p(np.sqrt(np.sum(bins * bins, axis=1))))
    assert np.allclose(energy, np.concatenate(expected, axis=1))


def test_contextual_features_decode_flat_lag_feature_positions() -> None:
    energy = np.arange(80 * 2, dtype=np.float32).reshape(80, 2)
    rows = np.asarray((0, 1), dtype=np.int64)
    # With two base features and past=1, flat positions 0, 3, and 4 mean
    # (lag=-1, feature=0), (lag=0, feature=1), and (lag=+1, feature=0).
    selected = np.asarray((0, 3, 4), dtype=np.int64)
    observed = contextual_features(energy, rows, selected, past_bins=1, future_bins=1)
    expected = np.asarray(
        [
            (energy[23, 0], energy[24, 1], energy[25, 0]),
            (energy[24, 0], energy[25, 1], energy[26, 0]),
        ],
        dtype=np.float32,
    )
    assert np.array_equal(observed, expected)


def test_screen_context_features_finds_training_correlate() -> None:
    energy = np.zeros((80, 3), dtype=np.float32)
    training = np.arange(20, dtype=np.int64)
    target = np.linspace(-1.0, 1.0, training.size).astype(np.float32)
    energy[training + 24, 2] = target
    selected, correlations = screen_context_features(
        energy,
        target,
        training,
        past_bins=0,
        future_bins=0,
        top_features=1,
    )
    assert selected.tolist() == [2]
    assert correlations[0] > 0.999
