#!/usr/bin/env python3
"""GPU-accelerated grouped ridge path selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TorchRidgeCVResult:
    coef_: np.ndarray
    intercept_: float
    alpha_: float
    alphas_: np.ndarray
    mean_validation_mse_: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features) @ self.coef_ + self.intercept_


def ridge_path_eigendecomposition(
    features: np.ndarray,
    target: np.ndarray,
    alphas: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit all penalties after one eigendecomposition of the standardized Gram matrix."""
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    penalties = torch.as_tensor(alphas, dtype=x.dtype, device=device)
    x_mean = x.mean(dim=0)
    x_scale = x.std(dim=0, correction=0).clamp_min(1.0e-7)
    y_mean = y.mean()
    standardized = (x - x_mean) / x_scale
    centered_y = y - y_mean
    gram = standardized.T @ standardized
    cross = standardized.T @ centered_y
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    projected = eigenvectors.T @ cross
    standardized_weights = (
        eigenvectors
        @ (projected[:, None] / (eigenvalues[:, None] + penalties[None]))
    ).T
    weights = standardized_weights / x_scale[None]
    intercepts = y_mean - weights @ x_mean
    return (
        weights.detach().cpu().numpy(),
        intercepts.detach().cpu().numpy(),
    )


def fit_torch_ridge_cv(
    features: np.ndarray,
    target: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    alphas: np.ndarray,
    device: torch.device,
) -> TorchRidgeCVResult:
    """Select a ridge penalty inside supplied folds and refit on all rows."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32)
    penalties = np.asarray(alphas, dtype=np.float32)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("features must be 2D and target must match its rows")
    if penalties.ndim != 1 or penalties.size == 0 or np.any(penalties <= 0):
        raise ValueError("alphas must be a nonempty positive one-dimensional array")
    fold_losses = []
    for training, validation in splits:
        coefficients, intercepts = ridge_path_eigendecomposition(
            x[training], y[training], penalties, device=device
        )
        prediction = x[validation] @ coefficients.T + intercepts[None]
        fold_losses.append(
            np.mean((prediction - y[validation, None]) ** 2, axis=0)
        )
    mean_mse = np.mean(np.stack(fold_losses), axis=0)
    selected = int(np.argmin(mean_mse))
    coefficients, intercepts = ridge_path_eigendecomposition(
        x, y, penalties, device=device
    )
    return TorchRidgeCVResult(
        coef_=coefficients[selected].astype(np.float32),
        intercept_=float(intercepts[selected]),
        alpha_=float(penalties[selected]),
        alphas_=penalties,
        mean_validation_mse_=mean_mse.astype(np.float32),
    )
