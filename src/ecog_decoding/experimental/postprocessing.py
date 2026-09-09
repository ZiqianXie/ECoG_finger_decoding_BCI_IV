"""Output-domain constraints for decoded finger trajectories."""

from __future__ import annotations

import numpy as np


def project_nonnegative(prediction: np.ndarray) -> np.ndarray:
    """Project baseline-corrected finger flexion onto its physical domain.

    The operation is deterministic and parameter-free.  Model outputs remain
    available separately when an unconstrained diagnostic is required.
    """
    values = np.asarray(prediction)
    if not np.isfinite(values).all():
        raise ValueError("prediction contains non-finite values")
    return np.maximum(values, 0.0)


def smooth_nonnegative(
    prediction: np.ndarray, *, already_nonnegative: bool = False, beta: float = 10.0
) -> np.ndarray:
    """Map a decoder output to the cleaned flexion domain smoothly."""
    values = np.asarray(prediction)
    if not np.isfinite(values).all():
        raise ValueError("prediction contains non-finite values")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if already_nonnegative:
        return np.maximum(values, 0.0)
    return np.logaddexp(0.0, beta * values) / beta


def fit_nonnegative_gain(prediction: np.ndarray, target: np.ndarray) -> float:
    """Fit a nonnegative, origin-preserving least-squares amplitude gain."""
    values = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if values.shape != truth.shape:
        raise ValueError("prediction and target must have the same shape")
    if not np.isfinite(values).all() or not np.isfinite(truth).all():
        raise ValueError("prediction and target must be finite")
    denominator = float(np.dot(values.ravel(), values.ravel()))
    if denominator <= 1.0e-12:
        return 1.0
    return max(0.0, float(np.dot(values.ravel(), truth.ravel())) / denominator)
