"""Spatial-filter initialization for the final ICA+CSP wavelet model."""

from __future__ import annotations

import numpy as np
from scipy import linalg


CSP_BANDS_HZ = (
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 30.0),
    (30.0, 55.0),
    (65.0, 95.0),
    (105.0, 145.0),
    (155.0, 195.0),
)

CSP_BAND_PROFILES = {
    "standard": CSP_BANDS_HZ,
    "low_extended": (
        (0.5, 4.0),
        (0.5, 5.0),
        (2.0, 6.0),
        *CSP_BANDS_HZ,
    ),
}


def regularized_covariance(
    values: np.ndarray, shrinkage: float = 0.05
) -> np.ndarray:
    """Return a trace-scaled shrinkage covariance matrix."""
    centered = np.asarray(values, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    isotropic = np.trace(covariance) / covariance.shape[0]
    return (1.0 - shrinkage) * covariance + shrinkage * isotropic * np.eye(
        covariance.shape[0]
    )


def fit_finger_csp_rows(
    filtered_bins: np.ndarray,
    target: np.ndarray,
    training: np.ndarray,
    finger_index: int,
    component_indices: tuple[int, ...],
    *,
    history_offset: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit selected rows from a finger-versus-rest CSP eigensystem."""
    if not component_indices:
        return np.empty((0, filtered_bins.shape[-1]), dtype=np.float32), {
            "active_bins": 0,
            "rest_bins": 0,
            "component_indices": [],
            "eigenvalues": [],
        }
    rest = np.max(np.nan_to_num(target, nan=np.inf), axis=1) < 0.05
    active = target[:, finger_index] > 0.20
    active_rows = training[active[training]]
    rest_rows = training[rest[training]]
    if active_rows.size < 2 or rest_rows.size < 2:
        raise RuntimeError("too few finger-movement or rest bins for CSP")
    active_values = filtered_bins[active_rows + history_offset].reshape(
        -1, filtered_bins.shape[-1]
    )
    rest_values = filtered_bins[rest_rows + history_offset].reshape(
        -1, filtered_bins.shape[-1]
    )
    active_covariance = regularized_covariance(active_values)
    rest_covariance = regularized_covariance(rest_values)
    eigenvalues, eigenvectors = linalg.eigh(
        active_covariance,
        active_covariance + rest_covariance,
        check_finite=False,
    )
    normalized_indices = [index % eigenvalues.size for index in component_indices]
    weights = eigenvectors[:, normalized_indices].T
    weights /= np.linalg.norm(weights, axis=1, keepdims=True).clip(min=1.0e-12)
    return weights.astype(np.float32), {
        "active_bins": int(active_rows.size),
        "rest_bins": int(rest_rows.size),
        "component_indices": list(component_indices),
        "eigenvalues": [float(eigenvalues[index]) for index in normalized_indices],
    }
