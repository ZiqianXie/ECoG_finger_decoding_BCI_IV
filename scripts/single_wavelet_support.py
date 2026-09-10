"""Shared utilities for the released frequency-compressed wavelet pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from scipy import linalg, signal

from ecog_decoding.models import WaveletPacketEnergy


SOURCE_RATE = 1000
TARGET_RATE = 25
SAMPLES_PER_BIN = 16
HISTORY = 25
OFFSET = HISTORY - 1
LITTLE = 4

def gray_frequency_order(levels: int) -> np.ndarray:
    """Map literal QMF path order to ascending physical frequency."""
    if levels < 1:
        raise ValueError("wavelet levels must be positive")
    indices = np.arange(2**levels, dtype=np.int64)
    return indices ^ (indices >> 1)


# Literal QMF paths are Gray-coded after a high-pass split.
FREQUENCY_ORDER = gray_frequency_order(3)


def frontend_frequency_order(frontend: WaveletPacketEnergy) -> np.ndarray:
    """Return ascending-frequency indices for every retained packet level."""
    level_counts = getattr(frontend, "output_band_level_counts", None)
    if level_counts is None:
        level_counts = (2**frontend.levels,)
    result = []
    offset = 0
    for count in level_counts:
        levels = int(np.log2(count))
        if 2**levels != count:
            raise ValueError("retained wavelet level must contain a power-of-two band count")
        result.append(offset + gray_frequency_order(levels))
        offset += count
    return np.concatenate(result)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    y = np.asarray(y, dtype=np.float64) - np.mean(y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 0 else 0.0


def resample_ecog(source: Path, cache: Path) -> np.ndarray:
    """Build the historical 400 Hz comparison cache when explicitly requested."""
    if cache.exists():
        try:
            return np.load(cache, mmap_mode="r")
        except (OSError, ValueError):
            pass
    cache.parent.mkdir(parents=True, exist_ok=True)
    ecog = np.load(source, mmap_mode="r")
    values = signal.resample_poly(
        np.asarray(ecog),
        up=2,
        down=5,
        axis=0,
        window=("kaiser", 8.6),
        padtype="line",
    ).astype(np.float32)
    temporary = cache.with_name(f".{cache.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, values, allow_pickle=False)
    os.replace(temporary, cache)
    return np.load(cache, mmap_mode="r")


@torch.inference_mode()
def extract_energy(
    ecog: np.ndarray,
    weights: np.ndarray,
    frontend: WaveletPacketEnergy,
    device: torch.device,
    component_chunk: int,
) -> np.ndarray:
    values = torch.from_numpy(np.asarray(ecog).T.copy()).unsqueeze(0).to(device)
    parts: list[np.ndarray] = []
    order = torch.as_tensor(
        frontend_frequency_order(frontend), dtype=torch.long, device=device
    )
    for begin in range(0, weights.shape[0], component_chunk):
        spatial = torch.nn.functional.conv1d(
            values,
            torch.as_tensor(
                weights[begin : begin + component_chunk],
                dtype=values.dtype,
                device=device,
            )[:, :, None],
        )
        energy = frontend(spatial)[0].index_select(1, order)
        parts.append(energy.permute(2, 0, 1).float().cpu().numpy())
    return np.concatenate(parts, axis=1)


def training_ecog_samples(
    ecog: np.ndarray, training_bins: np.ndarray, samples_per_bin: int
) -> np.ndarray:
    bins = np.asarray(training_bins, dtype=np.int64) + OFFSET
    sample_index = (
        bins[:, None] * samples_per_bin
        + np.arange(samples_per_bin, dtype=np.int64)[None]
    ).reshape(-1)
    return np.asarray(ecog[sample_index])


@torch.inference_mode()
def linear_wavelet_leaf_signals(
    ecog: np.ndarray,
    frontend: WaveletPacketEnergy,
    device: torch.device,
    physical_positions: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    """Return initialized tree leaves in ascending-frequency positions."""
    order = frontend_frequency_order(frontend)
    if not physical_positions or any(
        position < 0 or position >= order.size
        for position in physical_positions
    ):
        raise ValueError(
            f"physical leaf positions must be nonempty values from 0 to {order.size - 1}"
        )
    values = torch.from_numpy(np.asarray(ecog).T.copy()).to(device)[:, None]
    bands = values
    for layer in frontend.layers:
        bands = frontend._same_filter(bands, layer)
    indices = order[np.asarray(physical_positions, dtype=np.int64)]
    return tuple(
        bands[:, int(index)].T.float().cpu().numpy() for index in indices
    )


def linear_gamma_leaf_signals(
    ecog: np.ndarray,
    frontend: WaveletPacketEnergy,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the initialized 100--125 and 125--150 Hz tree leaves."""
    first, second = linear_wavelet_leaf_signals(ecog, frontend, device, (4, 5))
    return first, second


def linear_lower_high_gamma_leaf_signals(
    ecog: np.ndarray,
    frontend: WaveletPacketEnergy,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 50--75 and 75--100 Hz leaves used for one 50--100 Hz CSP row."""
    first, second = linear_wavelet_leaf_signals(ecog, frontend, device, (2, 3))
    return first, second


def regularized_covariance(
    values: np.ndarray, shrinkage: float = 0.05
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    covariance = values.T @ values / max(1, values.shape[0] - 1)
    isotropic = np.trace(covariance) / covariance.shape[0]
    return (1.0 - shrinkage) * covariance + shrinkage * isotropic * np.eye(
        covariance.shape[0]
    )


def finger_csp_bank(
    filtered_bins: np.ndarray,
    target: np.ndarray,
    training: np.ndarray,
    finger_index: int,
    component_indices: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit selected rows from one finger-versus-common-rest CSP eigensystem."""
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
        raise RuntimeError("too few movement or common-rest bins for CSP")
    active_values = filtered_bins[active_rows + OFFSET].reshape(
        -1, filtered_bins.shape[-1]
    )
    rest_values = filtered_bins[rest_rows + OFFSET].reshape(
        -1, filtered_bins.shape[-1]
    )
    active_covariance = regularized_covariance(active_values)
    rest_covariance = regularized_covariance(rest_values)
    eigenvalues, eigenvectors = linalg.eigh(
        active_covariance,
        active_covariance + rest_covariance,
        check_finite=False,
    )
    selected = [index % eigenvalues.size for index in component_indices]
    weights = eigenvectors[:, selected].T
    weights /= np.linalg.norm(weights, axis=1, keepdims=True).clip(min=1.0e-12)
    return weights.astype(np.float32), {
        "active_bins": int(active_rows.size),
        "rest_bins": int(rest_rows.size),
        "component_indices": list(component_indices),
        "eigenvalues": [float(eigenvalues[index]) for index in selected],
    }


def correlation_screen(x: np.ndarray, y: np.ndarray, count: int) -> np.ndarray:
    centered_x = x - x.mean(axis=0, keepdims=True)
    centered_y = y - y.mean()
    denominator = np.linalg.norm(centered_x, axis=0) * np.linalg.norm(centered_y)
    score = np.divide(
        np.abs(centered_x.T @ centered_y),
        denominator,
        out=np.zeros(centered_x.shape[1], dtype=np.float64),
        where=denominator > 0,
    )
    return np.argsort(score)[-min(count, score.size) :]


def csp_candidate_union(
    features: np.ndarray,
    target: np.ndarray,
    training: np.ndarray,
    per_bin: int,
    ica_per_bin: int,
    ica_prescreen: int,
    finger_index: int = LITTLE,
) -> tuple[np.ndarray, dict[str, int]]:
    position = np.arange(features.shape[1]) % per_bin
    ica_candidates = np.flatnonzero(position < ica_per_bin)
    csp_candidates = np.flatnonzero(position >= ica_per_bin)
    selected_local = correlation_screen(
        features[training][:, ica_candidates],
        target[training, finger_index],
        ica_prescreen,
    )
    selected_ica = ica_candidates[selected_local]
    return np.concatenate((selected_ica, csp_candidates)), {
        "ica_prescreen_candidates": int(selected_ica.size),
        "guaranteed_csp_candidates": int(csp_candidates.size),
    }
