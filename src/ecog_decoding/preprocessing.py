"""Signal and target preprocessing with explicit train/test boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import cvxpy as cp
from scipy import ndimage, signal, sparse
from scipy.sparse.linalg import spsolve

from .io import BAD_CHANNELS_ONE_BASED


@dataclass(frozen=True)
class EcogResult:
    train: np.ndarray
    test: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    retained_channels_one_based: tuple[int, ...]


@dataclass(frozen=True)
class BaselineResult:
    corrected: np.ndarray
    baseline: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class MovementCorrectionModel:
    """Training-derived parameters for temporally regularized finger demixing."""

    scale: np.ndarray
    coupling_matrix: np.ndarray
    sampling_rate_hz: float
    baseline_smoothness: float
    baseline_asymmetry: float
    baseline_iterations: int
    activation_threshold: float
    state_transition_penalty: float
    coupling_minimum_activation: float


@dataclass(frozen=True)
class MovementCorrectionResult:
    intended: np.ndarray
    detrended_multifinger: np.ndarray
    baseline: np.ndarray
    active_finger: np.ndarray
    events: tuple[tuple[int, int, int], ...]


def _as_time_by_feature(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"{name} must have shape (time, features); got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values")
    return x


def apply_notches(
    x: np.ndarray,
    fs: float,
    frequencies: Iterable[float] = (60.0, 120.0, 180.0),
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply zero-phase IIR notches to a continuous time-by-channel array."""
    out = _as_time_by_feature(x, "x").copy()
    nyquist = fs / 2.0
    for frequency in frequencies:
        if not 0.0 < frequency < nyquist:
            raise ValueError(f"notch frequency {frequency} is outside (0, {nyquist})")
        b, a = signal.iirnotch(float(frequency), quality_factor, fs=fs)
        out = signal.filtfilt(b, a, out, axis=0)
    return out


def preprocess_ecog(
    train: np.ndarray,
    test: np.ndarray,
    subject: int,
    fs: float = 1000.0,
    notch_frequencies: Iterable[float] = (60.0, 120.0, 180.0),
    notch_quality_factor: float = 30.0,
    normalization_fit_fraction: float = 2.0 / 3.0,
) -> EcogResult:
    """Remove bad channels, notch the continuous record, and standardize.

    The two public arrays are consecutive portions of one recording. They are
    joined for fixed zero-phase filtering to avoid creating an artificial edge
    at the competition split, then separated again. Mean and standard deviation
    are estimated only from the model-training portion of ``train``.
    """
    train = _as_time_by_feature(train, "train")
    test = _as_time_by_feature(test, "test")
    if train.shape[1] != test.shape[1]:
        raise ValueError("train and test channel counts differ")
    if subject not in BAD_CHANNELS_ONE_BASED:
        raise ValueError(f"unknown subject {subject}")
    if not 0.0 < normalization_fit_fraction <= 1.0:
        raise ValueError("normalization_fit_fraction must be in (0, 1]")

    bad_zero_based = {channel - 1 for channel in BAD_CHANNELS_ONE_BASED[subject]}
    keep = np.array(
        [i for i in range(train.shape[1]) if i not in bad_zero_based], dtype=int
    )
    retained = tuple(int(i + 1) for i in keep)
    joined = np.concatenate((train[:, keep], test[:, keep]), axis=0)
    joined = apply_notches(
        joined,
        fs=fs,
        frequencies=notch_frequencies,
        quality_factor=notch_quality_factor,
    )
    filtered_train = joined[: train.shape[0]]
    filtered_test = joined[train.shape[0] :]

    fit_stop = max(2, int(round(train.shape[0] * normalization_fit_fraction)))
    mean = filtered_train[:fit_stop].mean(axis=0)
    scale = filtered_train[:fit_stop].std(axis=0, ddof=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)

    return EcogResult(
        train=((filtered_train - mean) / scale).astype(np.float32),
        test=((filtered_test - mean) / scale).astype(np.float32),
        mean=mean,
        scale=scale,
        retained_channels_one_based=retained,
    )


def downsample_glove(
    trajectory: np.ndarray,
    source_rate_hz: int = 1000,
    target_rate_hz: int = 25,
) -> np.ndarray:
    """Return one sample per native glove interval without smoothing it twice."""
    trajectory = _as_time_by_feature(trajectory, "trajectory")
    if source_rate_hz % target_rate_hz:
        raise ValueError("source_rate_hz must be an integer multiple of target_rate_hz")
    return trajectory[:: source_rate_hz // target_rate_hz].copy()


def asymmetric_baseline(
    trajectory: np.ndarray,
    smoothness: float = 1.0e5,
    asymmetry: float = 1.0e-3,
    iterations: int = 20,
) -> np.ndarray:
    """Fit a robust smooth baseline beneath predominantly positive peaks.

    This is asymmetric least squares with a second-difference penalty. Unlike
    the old constrained optimizer, it is sparse, fast, and tolerant of local
    downward noise instead of forcing every sample to be an upper bound.
    """
    y = np.asarray(trajectory, dtype=np.float64)
    if y.ndim != 1 or y.size < 2 or not np.isfinite(y).all():
        raise ValueError("trajectory must be a finite one-dimensional array")
    if smoothness <= 0:
        raise ValueError("smoothness must be positive")
    if not 0.0 < asymmetry < 0.5:
        raise ValueError("asymmetry must be in (0, 0.5)")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    if y.size < 3:
        return np.full_like(y, np.min(y))
    difference = sparse.diags(
        (np.ones(y.size - 2), -2.0 * np.ones(y.size - 2), np.ones(y.size - 2)),
        (0, 1, 2),
        shape=(y.size - 2, y.size),
        format="csc",
    )
    penalty = smoothness * (difference.T @ difference)
    weights = np.ones(y.size)
    baseline = y.copy()
    for _ in range(iterations):
        system = sparse.spdiags(weights, 0, y.size, y.size) + penalty
        updated = spsolve(system, weights * y)
        new_weights = np.where(y > updated, asymmetry, 1.0 - asymmetry)
        baseline = updated
        if np.array_equal(new_weights, weights):
            break
        weights = new_weights
    return baseline


def paper_rope_baseline(
    trajectory: np.ndarray,
    smoothness: float = 1.0e5,
    max_iterations: int = 100_000,
    tolerance: float = 1.0e-5,
) -> np.ndarray:
    """Reproduce the constrained baseline objective in Xie et al. (2018).

    The published problem is

        minimize |y - b|_1 + lambda * sum_n (b[n + 1] - b[n])**2
        subject to b <= y.

    This uses SCS, matching the solver family reported in the paper.
    """
    y = np.asarray(trajectory, dtype=np.float64)
    if y.ndim != 1 or y.size < 2 or not np.isfinite(y).all():
        raise ValueError("trajectory must be a finite one-dimensional array")
    if smoothness <= 0:
        raise ValueError("smoothness must be positive")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive")

    baseline_variable = cp.Variable(y.size)
    objective = cp.Minimize(
        cp.norm1(y - baseline_variable)
        + float(smoothness) * cp.sum_squares(cp.diff(baseline_variable))
    )
    problem = cp.Problem(objective, [baseline_variable <= y])
    problem.solve(
        solver=cp.SCS,
        eps=float(tolerance),
        max_iters=int(max_iterations),
        verbose=False,
    )
    if problem.status == cp.OPTIMAL_INACCURATE:
        problem.solve(
            solver=cp.CLARABEL,
            max_iter=int(max_iterations),
            tol_gap_abs=float(tolerance),
            tol_feas=float(tolerance),
            verbose=False,
        )
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or baseline_variable.value is None:
        raise RuntimeError(f"paper baseline optimization failed: {problem.status}")
    return np.minimum(np.asarray(baseline_variable.value, dtype=np.float64), y)


def paper_baseline_correct(
    trajectories: np.ndarray,
    smoothness: float = 1.0e5,
    max_iterations: int = 100_000,
    tolerance: float = 1.0e-5,
) -> BaselineResult:
    """Apply the paper's constrained baseline independently to every finger."""
    trajectories = _as_time_by_feature(trajectories, "trajectories")
    baseline = np.column_stack(
        [
            paper_rope_baseline(
                trajectories[:, finger],
                smoothness=smoothness,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            for finger in range(trajectories.shape[1])
        ]
    )
    corrected = np.maximum(trajectories - baseline, 0.0)
    scale = np.quantile(corrected, 0.995, axis=0)
    scale_floor = max(float(np.max(scale)) * 0.01, np.finfo(np.float64).eps)
    return BaselineResult(
        corrected=corrected,
        baseline=baseline,
        scale=np.maximum(scale, scale_floor),
    )


def local_lower_envelope_baseline(
    trajectory: np.ndarray,
    sampling_rate_hz: float = 25.0,
    window_seconds: float = 1.5,
    quantile: float = 0.10,
    smoothing_seconds: float = 0.16,
) -> np.ndarray:
    """Estimate a locally adaptive lower envelope for glove drift.

    The paper's single global tension parameter can underfit a baseline whose
    slope changes during the recording.  This alternative takes a rolling low
    quantile over a window longer than an individual flexion and then smooths
    that envelope.  The final pointwise constraint keeps the estimate below
    the measured trajectory, so subtraction remains nonnegative.

    ``window_seconds`` is deliberately expressed in physical time.  That
    makes parameter comparisons meaningful if the glove grid changes.
    """
    y = np.asarray(trajectory, dtype=np.float64)
    if y.ndim != 1 or y.size < 2 or not np.isfinite(y).all():
        raise ValueError("trajectory must be a finite one-dimensional array")
    if sampling_rate_hz <= 0 or window_seconds <= 0:
        raise ValueError("sampling_rate_hz and window_seconds must be positive")
    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must be in (0, 0.5)")
    if smoothing_seconds < 0:
        raise ValueError("smoothing_seconds must be nonnegative")

    window_samples = max(3, int(round(window_seconds * sampling_rate_hz)))
    if window_samples % 2 == 0:
        window_samples += 1
    baseline = ndimage.percentile_filter(
        y,
        percentile=100.0 * quantile,
        size=window_samples,
        mode="nearest",
    )
    sigma = smoothing_seconds * sampling_rate_hz
    if sigma > 0:
        baseline = ndimage.gaussian_filter1d(
            baseline,
            sigma=sigma,
            mode="nearest",
        )
    return np.minimum(baseline, y)


def local_baseline_correct(
    trajectories: np.ndarray,
    sampling_rate_hz: float = 25.0,
    window_seconds: float = 1.5,
    quantile: float = 0.10,
    smoothing_seconds: float = 0.16,
) -> BaselineResult:
    """Subtract a fine-grained rolling lower envelope from every finger."""
    trajectories = _as_time_by_feature(trajectories, "trajectories")
    baseline = np.column_stack(
        [
            local_lower_envelope_baseline(
                trajectories[:, finger],
                sampling_rate_hz=sampling_rate_hz,
                window_seconds=window_seconds,
                quantile=quantile,
                smoothing_seconds=smoothing_seconds,
            )
            for finger in range(trajectories.shape[1])
        ]
    )
    corrected = np.maximum(trajectories - baseline, 0.0)
    scale = np.quantile(corrected, 0.995, axis=0)
    scale_floor = max(float(np.max(scale)) * 0.01, np.finfo(np.float64).eps)
    return BaselineResult(
        corrected=corrected,
        baseline=baseline,
        scale=np.maximum(scale, scale_floor),
    )


def baseline_correct(
    trajectories: np.ndarray,
    smoothness: float = 1.0e5,
    asymmetry: float = 1.0e-3,
    iterations: int = 20,
) -> BaselineResult:
    """Subtract a robust asymmetric baseline from every finger."""
    trajectories = _as_time_by_feature(trajectories, "trajectories")
    baseline = np.column_stack(
        [
            asymmetric_baseline(
                trajectories[:, finger],
                smoothness=smoothness,
                asymmetry=asymmetry,
                iterations=iterations,
            )
            for finger in range(trajectories.shape[1])
        ]
    )
    corrected = np.maximum(trajectories - baseline, 0.0)
    scale = np.quantile(corrected, 0.995, axis=0)
    # Do not amplify numerical baseline residue in an inactive/very weak digit.
    # A shared floor preserves relative amplitudes needed for event assignment.
    scale_floor = max(float(np.max(scale)) * 0.01, np.finfo(np.float64).eps)
    scale = np.maximum(scale, scale_floor)
    return BaselineResult(corrected=corrected, baseline=baseline, scale=scale)


def _estimate_coupling(
    normalized: np.ndarray,
    states: np.ndarray,
    finger_count: int,
    minimum_activation: float,
) -> np.ndarray:
    coupling = np.eye(finger_count, dtype=np.float64)
    for finger in range(finger_count):
        mask = (states == finger) & (normalized[:, finger] >= minimum_activation)
        selected = normalized[mask]
        if selected.shape[0] < 10:
            continue
        denominator = np.maximum(selected[:, [finger]], 1e-8)
        ratios = selected / denominator
        coupling[:, finger] = np.clip(np.median(ratios, axis=0), 0.0, 2.0)
        coupling[finger, finger] = 1.0
    return coupling


def _decode_states(
    normalized: np.ndarray,
    coupling: np.ndarray,
    activation_threshold: float,
    transition_penalty: float,
) -> np.ndarray:
    """Viterbi decode rest (-1) or one of five intended finger states."""
    if transition_penalty < 0:
        raise ValueError("transition_penalty must be nonnegative")
    smoothed = ndimage.gaussian_filter1d(normalized, sigma=1.0, axis=0, mode="nearest")
    sample_count, finger_count = smoothed.shape
    state_count = finger_count + 1
    emission = np.empty((sample_count, state_count), dtype=np.float64)
    emission[:, 0] = np.sum(smoothed * smoothed, axis=1)
    amplitudes = np.empty((sample_count, finger_count), dtype=np.float64)
    for finger in range(finger_count):
        pattern = coupling[:, finger]
        amplitude = np.maximum(smoothed @ pattern / max(float(pattern @ pattern), 1e-12), 0.0)
        amplitudes[:, finger] = amplitude
        residual = smoothed - amplitude[:, None] * pattern[None, :]
        emission[:, finger + 1] = (
            np.sum(residual * residual, axis=1) + activation_threshold**2
        )

    transition = np.full((state_count, state_count), 2.0 * transition_penalty)
    np.fill_diagonal(transition, 0.0)
    transition[0, 1:] = transition_penalty
    transition[1:, 0] = transition_penalty

    score = emission[0].copy()
    back_pointer = np.empty((sample_count, state_count), dtype=np.int8)
    back_pointer[0] = -1
    for time_index in range(1, sample_count):
        candidates = score[:, None] + transition
        back_pointer[time_index] = np.argmin(candidates, axis=0)
        score = emission[time_index] + np.min(candidates, axis=0)

    decoded = np.empty(sample_count, dtype=np.int16)
    decoded[-1] = int(np.argmin(score))
    for time_index in range(sample_count - 1, 0, -1):
        decoded[time_index - 1] = back_pointer[time_index, decoded[time_index]]
    return decoded - 1


def _states_to_events(states: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    starts = np.flatnonzero((states >= 0) & np.r_[True, states[1:] != states[:-1]])
    events: list[tuple[int, int, int]] = []
    for start in starts:
        changes = np.flatnonzero(states[start + 1 :] != states[start])
        stop = int(start + 1 + changes[0]) if changes.size else int(states.size)
        events.append((int(start), stop, int(states[start])))
    return tuple(events)


def fit_movement_corrector(
    trajectories: np.ndarray,
    sampling_rate_hz: float = 25.0,
    baseline_smoothness: float = 1.0e5,
    baseline_asymmetry: float = 0.05,
    baseline_iterations: int = 20,
    activation_threshold: float = 0.08,
    state_transition_penalty: float = 0.05,
    coupling_minimum_activation: float = 0.20,
    coupling_em_iterations: int = 4,
) -> MovementCorrectionModel:
    """Learn robust amplitude scales and cross-finger coupling from training labels."""
    corrected = baseline_correct(
        trajectories,
        smoothness=baseline_smoothness,
        asymmetry=baseline_asymmetry,
        iterations=baseline_iterations,
    )
    normalized = np.clip(corrected.corrected / corrected.scale, 0.0, 2.0)
    dominant = np.argmax(normalized, axis=1).astype(np.int16)
    dominant[np.max(normalized, axis=1) < activation_threshold] = -1
    coupling = _estimate_coupling(
        normalized,
        dominant,
        trajectories.shape[1],
        coupling_minimum_activation,
    )
    for _ in range(max(1, coupling_em_iterations)):
        states = _decode_states(
            normalized, coupling, activation_threshold, state_transition_penalty
        )
        coupling = _estimate_coupling(
            normalized,
            states,
            trajectories.shape[1],
            coupling_minimum_activation,
        )

    return MovementCorrectionModel(
        scale=corrected.scale,
        coupling_matrix=coupling,
        sampling_rate_hz=sampling_rate_hz,
        baseline_smoothness=baseline_smoothness,
        baseline_asymmetry=baseline_asymmetry,
        baseline_iterations=baseline_iterations,
        activation_threshold=activation_threshold,
        state_transition_penalty=state_transition_penalty,
        coupling_minimum_activation=coupling_minimum_activation,
    )


def apply_movement_corrector(
    trajectories: np.ndarray,
    model: MovementCorrectionModel,
) -> MovementCorrectionResult:
    """Infer the intended finger sequence and remove learned cross-finger coupling."""
    corrected = baseline_correct(
        trajectories,
        smoothness=model.baseline_smoothness,
        asymmetry=model.baseline_asymmetry,
        iterations=model.baseline_iterations,
    )
    normalized = np.clip(corrected.corrected / model.scale, 0.0, 2.0)
    states = _decode_states(
        normalized,
        model.coupling_matrix,
        model.activation_threshold,
        model.state_transition_penalty,
    )
    intended = np.zeros_like(normalized)
    for finger in range(normalized.shape[1]):
        mask = states == finger
        if not np.any(mask):
            continue
        pattern = model.coupling_matrix[:, finger]
        amplitude = normalized[mask] @ pattern / max(float(pattern @ pattern), 1e-12)
        intended[mask, finger] = np.clip(amplitude, 0.0, 1.0)

    return MovementCorrectionResult(
        intended=intended.astype(np.float32),
        detrended_multifinger=np.clip(normalized, 0.0, 1.0).astype(np.float32),
        baseline=corrected.baseline.astype(np.float32),
        active_finger=states,
        events=_states_to_events(states),
    )
