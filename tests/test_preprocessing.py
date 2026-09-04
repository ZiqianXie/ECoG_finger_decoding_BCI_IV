from __future__ import annotations

import numpy as np
from scipy import signal

from ecog_decoding.preprocessing import (
    _decode_states,
    apply_notches,
    apply_movement_corrector,
    asymmetric_baseline,
    downsample_glove,
    fit_movement_corrector,
    local_lower_envelope_baseline,
    paper_rope_baseline,
    preprocess_ecog,
)


def tone_amplitude(x: np.ndarray, fs: float, frequency: float) -> float:
    frequencies, power = signal.periodogram(x, fs=fs)
    return float(power[np.argmin(np.abs(frequencies - frequency))])


def test_notches_remove_line_harmonics_and_preserve_nearby_signal() -> None:
    fs = 1000.0
    time = np.arange(20_000) / fs
    x = (
        np.sin(2 * np.pi * 45 * time)
        + np.sin(2 * np.pi * 60 * time)
        + np.sin(2 * np.pi * 120 * time)
        + np.sin(2 * np.pi * 180 * time)
    )[:, None]
    filtered = apply_notches(x, fs=fs)[:, 0]
    for frequency in (60.0, 120.0, 180.0):
        assert tone_amplitude(filtered, fs, frequency) < 1e-4 * tone_amplitude(
            x[:, 0], fs, frequency
        )
    assert tone_amplitude(filtered, fs, 45.0) > 0.95 * tone_amplitude(x[:, 0], fs, 45.0)


def test_standardization_uses_only_training_fit_partition() -> None:
    rng = np.random.default_rng(3)
    train = rng.normal(size=(2_000, 62))
    train[1_000:] += 50.0
    test = rng.normal(size=(500, 62)) + 100.0
    result = preprocess_ecog(
        train,
        test,
        subject=1,
        notch_frequencies=(),
        normalization_fit_fraction=0.5,
    )
    assert result.train.shape == (2_000, 61)
    np.testing.assert_allclose(result.train[:1_000].mean(axis=0), 0.0, atol=2e-6)
    np.testing.assert_allclose(result.train[:1_000].std(axis=0), 1.0, atol=2e-6)
    assert float(result.test.mean()) > 90.0


def test_asymmetric_baseline_tracks_slow_drift() -> None:
    time = np.linspace(0.0, 20.0, 2_000)
    drift = 0.2 * np.sin(2 * np.pi * time / 20.0)
    movement = np.zeros_like(time)
    movement[400:500] = np.sin(np.linspace(0.0, np.pi, 100))
    movement[1_200:1_350] = 0.8 * np.sin(np.linspace(0.0, np.pi, 150))
    observed = drift + movement
    baseline = asymmetric_baseline(observed, smoothness=1e4)
    assert np.sqrt(np.mean((baseline - drift) ** 2)) < 0.08


def test_paper_rope_baseline_is_smooth_lower_envelope() -> None:
    time = np.linspace(0.0, 8.0, 400)
    drift = 0.15 * np.sin(2 * np.pi * time / 8.0)
    movement = np.zeros_like(time)
    movement[80:120] = np.sin(np.linspace(0.0, np.pi, 40))
    movement[260:310] = 0.7 * np.sin(np.linspace(0.0, np.pi, 50))
    observed = drift + movement
    baseline = paper_rope_baseline(observed, smoothness=1.0e5)
    assert np.max(baseline - observed) <= 1.0e-8
    assert np.sqrt(np.mean((baseline - drift) ** 2)) < 0.10


def test_local_lower_envelope_tracks_piecewise_drift_without_eating_pulses() -> None:
    sampling_rate = 25.0
    time = np.arange(750) / sampling_rate
    drift = np.where(
        time < 15.0,
        -0.2 + 0.025 * time,
        0.175 - 0.045 * (time - 15.0),
    )
    movement = np.zeros_like(time)
    pulse = np.sin(np.linspace(0.0, np.pi, 9))
    for start in (75, 155, 245, 420, 530, 650):
        movement[start : start + pulse.size] = pulse
    observed = drift + movement

    baseline = local_lower_envelope_baseline(
        observed,
        sampling_rate_hz=sampling_rate,
        window_seconds=1.5,
    )
    corrected = np.maximum(observed - baseline, 0.0)
    assert np.max(baseline - observed) <= 0.0
    assert np.sqrt(np.mean((baseline - drift) ** 2)) < 0.07
    assert np.mean(corrected[movement > 0.8]) > 0.75


def test_event_correction_learns_and_removes_little_to_ring_coupling() -> None:
    trajectories = np.zeros((2_000, 5))
    pulse = np.sin(np.linspace(0.0, np.pi, 25))
    for start in (100, 350, 600, 850, 1_100, 1_350):
        trajectories[start : start + 25, 4] = pulse
        trajectories[start : start + 25, 3] = 0.25 * pulse
    for start in (225, 475, 725, 975, 1_225, 1_475):
        trajectories[start : start + 25, 3] = pulse
        trajectories[start : start + 25, 4] = 0.10 * pulse
    drift = 0.03 * np.sin(np.linspace(0.0, 2 * np.pi, trajectories.shape[0]))
    trajectories += drift[:, None]

    model = fit_movement_corrector(
        trajectories,
        baseline_smoothness=1e5,
        activation_threshold=0.05,
    )
    result = apply_movement_corrector(trajectories, model)
    little_event = slice(600, 625)
    ring_event = slice(725, 750)
    assert np.max(result.intended[little_event, 4]) > 0.8
    assert np.max(result.intended[little_event, 3]) == 0.0
    assert np.max(result.intended[ring_event, 3]) > 0.8
    assert np.max(result.intended[ring_event, 4]) == 0.0
    assert model.coupling_matrix[3, 4] > 0.05


def test_decoder_does_not_turn_index_movement_into_little_movement() -> None:
    """Regression for the subject-1 preview around 4--5 seconds."""
    normalized = np.zeros((150, 5))
    pulse = 0.85 * np.sin(np.linspace(0.0, 4 * np.pi, 100)) ** 2
    normalized[25:125, 1] = pulse
    normalized[25:125, 4] = 0.12
    normalized[25:125, 3] = 0.05
    coupling = np.array(
        [
            [1.0, 0.056, 0.046, 0.037, 0.053],
            [0.071, 1.0, 0.172, 0.110, 0.190],
            [0.038, 0.169, 1.0, 0.074, 0.071],
            [0.036, 0.088, 0.155, 1.0, 0.271],
            [0.043, 0.070, 0.039, 0.040, 1.0],
        ]
    )
    states = _decode_states(
        normalized,
        coupling,
        activation_threshold=0.08,
        transition_penalty=0.05,
    )
    assert np.count_nonzero(states[25:125] == 1) > 70
    assert np.count_nonzero(states[25:125] == 4) == 0


def test_glove_downsample_preserves_native_grid() -> None:
    x = np.arange(400, dtype=float)[:, None]
    np.testing.assert_array_equal(downsample_glove(x)[:, 0], np.arange(0, 400, 40))
