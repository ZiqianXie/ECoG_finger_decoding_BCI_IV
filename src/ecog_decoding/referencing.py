"""Leakage-safe ECoG re-referencing transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def grid_neighbors(
    retained_channels_one_based: Sequence[int], columns: int = 8
) -> dict[int, tuple[int, ...]]:
    """Return physical four-neighborhoods for a row-major electrode grid."""
    if columns < 2:
        raise ValueError("columns must be at least 2")
    retained = {int(channel) for channel in retained_channels_one_based}
    if not retained or min(retained) < 1:
        raise ValueError("retained channel numbers must be positive")
    result: dict[int, tuple[int, ...]] = {}
    for channel in sorted(retained):
        zero = channel - 1
        row, column = divmod(zero, columns)
        candidates = []
        if column > 0:
            candidates.append(channel - 1)
        if column + 1 < columns:
            candidates.append(channel + 1)
        if row > 0:
            candidates.append(channel - columns)
        candidates.append(channel + columns)
        result[channel] = tuple(item for item in candidates if item in retained)
    return result


def rereference(
    values: np.ndarray,
    retained_channels_one_based: Sequence[int],
    method: str,
    grid_columns: int = 8,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply CAR, adjacent bipolar, or local-Laplacian referencing."""
    values = np.asarray(values, dtype=np.float64)
    channels = tuple(int(item) for item in retained_channels_one_based)
    if values.ndim != 2 or values.shape[1] != len(channels):
        raise ValueError("values and retained channel numbers must agree")
    if method == "car":
        return values - values.mean(axis=1, keepdims=True), {
            "output_labels": [f"car({channel})" for channel in channels]
        }

    neighbors = grid_neighbors(channels, columns=grid_columns)
    position = {channel: index for index, channel in enumerate(channels)}
    if method == "bipolar":
        pairs = []
        for channel in channels:
            for neighbor in neighbors[channel]:
                if neighbor > channel and (
                    neighbor == channel + 1 or neighbor == channel + grid_columns
                ):
                    pairs.append((channel, neighbor))
        if not pairs:
            raise ValueError("no adjacent channel pairs were found")
        transformed = np.column_stack(
            [values[:, position[left]] - values[:, position[right]] for left, right in pairs]
        )
        return transformed, {
            "bipolar_pairs_one_based": [list(pair) for pair in pairs],
            "output_labels": [f"{left}-{right}" for left, right in pairs],
        }
    if method == "laplacian":
        centers = [channel for channel in channels if len(neighbors[channel]) >= 2]
        if not centers:
            raise ValueError("no channels have enough grid neighbors")
        transformed = np.column_stack(
            [
                values[:, position[center]]
                - values[:, [position[item] for item in neighbors[center]]].mean(axis=1)
                for center in centers
            ]
        )
        return transformed, {
            "laplacian_centers_one_based": centers,
            "laplacian_neighbors_one_based": {
                str(center): list(neighbors[center]) for center in centers
            },
            "output_labels": [f"laplacian({center})" for center in centers],
        }
    raise ValueError("method must be 'car', 'bipolar', or 'laplacian'")


def fit_standardize(
    train: np.ndarray, test: np.ndarray, fit_stop: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize a derived reference using training-fit samples only."""
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("train and test must be compatible matrices")
    if not 2 <= fit_stop <= train.shape[0]:
        raise ValueError("fit_stop is outside the training array")
    mean = train[:fit_stop].mean(axis=0)
    scale = train[:fit_stop].std(axis=0)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    return (
        ((train - mean) / scale).astype(np.float32),
        ((test - mean) / scale).astype(np.float32),
        mean,
        scale,
    )
