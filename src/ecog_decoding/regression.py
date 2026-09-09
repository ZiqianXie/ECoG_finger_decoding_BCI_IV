"""Small regression helpers used by the final single-branch decoder."""

from __future__ import annotations

import numpy as np
import torch


def lagged(values: np.ndarray, history: int) -> np.ndarray:
    """Return contiguous flattened history windows without Python loops."""
    windows = np.lib.stride_tricks.sliding_window_view(values, history, axis=0)
    return np.ascontiguousarray(
        windows.transpose(0, 2, 1).reshape(windows.shape[0], -1)
    )


def ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Fit standardized ridge coefficients on the requested Torch device."""
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1.0e-6] = 1.0
    xt = torch.from_numpy((x - mean) / scale).to(device)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    y_mean = float(yt.mean())
    yt = yt - y_mean
    gram = xt.T @ xt / xt.shape[0]
    rhs = xt.T @ yt / xt.shape[0]
    gram.diagonal().add_(alpha)
    weight = torch.linalg.solve(gram, rhs)
    return mean, scale, y_mean, weight.cpu().numpy()
