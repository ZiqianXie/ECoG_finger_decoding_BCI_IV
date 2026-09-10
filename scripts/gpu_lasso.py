#!/usr/bin/env python3
"""GPU-accelerated Lasso path selection for split-local decoder initialization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TorchLassoCVResult:
    coef_: np.ndarray
    intercept_: float
    alpha_: float
    alphas_: np.ndarray
    mean_validation_mse_: np.ndarray
    iterations_: int


def spectral_lipschitz(gram: torch.Tensor, iterations: int = 24) -> torch.Tensor:
    """Estimate the largest eigenvalue of a positive semidefinite Gram matrix."""
    vector = torch.ones(gram.shape[1], dtype=gram.dtype, device=gram.device)
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(1.0e-12)
    for _ in range(iterations):
        vector = gram @ vector
        vector = vector / torch.linalg.vector_norm(vector).clamp_min(1.0e-12)
    return torch.dot(vector, gram @ vector)


def lasso_path_fista(
    features: np.ndarray,
    target: np.ndarray,
    alphas: np.ndarray,
    *,
    device: torch.device,
    max_iter: int = 300,
    tolerance: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit an entire Lasso path in parallel with batched FISTA updates."""
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    alpha = torch.as_tensor(alphas, dtype=x.dtype, device=device)[:, None]
    x_mean = x.mean(dim=0)
    y_mean = y.mean()
    centered_x = x - x_mean
    centered_y = y - y_mean
    gram = centered_x.T @ centered_x / centered_x.shape[0]
    cross = centered_y @ centered_x / centered_x.shape[0]
    step = spectral_lipschitz(gram).clamp_min(1.0e-8).reciprocal()

    weights = torch.zeros(
        (alpha.shape[0], centered_x.shape[1]), dtype=x.dtype, device=device
    )
    accelerated = weights.clone()
    momentum = torch.ones((), dtype=x.dtype, device=device)
    completed = max_iter
    for iteration in range(max_iter):
        gradient = accelerated @ gram - cross[None]
        gradient_step = accelerated - step * gradient
        candidate = torch.sign(gradient_step) * torch.relu(
            torch.abs(gradient_step) - step * alpha
        )
        next_momentum = (1.0 + torch.sqrt(1.0 + 4.0 * momentum.square())) / 2.0
        next_accelerated = candidate + (
            (momentum - 1.0) / next_momentum
        ) * (candidate - weights)
        if iteration % 10 == 9:
            change = torch.linalg.vector_norm(candidate - weights, dim=1)
            scale = torch.linalg.vector_norm(candidate, dim=1).clamp_min(1.0)
            if bool(torch.all(change <= tolerance * scale)):
                weights = candidate
                completed = iteration + 1
                break
        weights = candidate
        accelerated = next_accelerated
        momentum = next_momentum

    intercepts = y_mean - weights @ x_mean
    return (
        weights.detach().cpu().numpy(),
        intercepts.detach().cpu().numpy(),
        completed,
    )


def fit_torch_lasso_cv(
    features: np.ndarray,
    target: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    device: torch.device,
    alpha_count: int = 32,
    alpha_ratio: float = 1.0e-3,
    max_iter: int = 300,
    tolerance: float = 1.0e-5,
) -> TorchLassoCVResult:
    """Select Lasso alpha inside supplied folds and refit on all rows."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("features must be 2D and target must match its rows")
    if alpha_count < 2 or not 0.0 < alpha_ratio < 1.0:
        raise ValueError("alpha path requires alpha_count >= 2 and 0 < ratio < 1")
    centered_x = x - x.mean(axis=0, keepdims=True)
    centered_y = y - y.mean()
    alpha_max = float(np.max(np.abs(centered_x.T @ centered_y)) / x.shape[0])
    alpha_max = max(alpha_max, np.finfo(np.float32).eps)
    alphas = alpha_max * np.geomspace(1.0, alpha_ratio, alpha_count).astype(
        np.float32
    )

    fold_mse = []
    iterations = 0
    for training, validation in splits:
        coefficients, intercepts, used = lasso_path_fista(
            x[training],
            y[training],
            alphas,
            device=device,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        estimate = x[validation] @ coefficients.T + intercepts[None]
        fold_mse.append(np.mean((estimate - y[validation, None]) ** 2, axis=0))
        iterations = max(iterations, used)
    mean_mse = np.mean(np.stack(fold_mse), axis=0)
    selected_index = int(np.argmin(mean_mse))
    coefficients, intercepts, used = lasso_path_fista(
        x,
        y,
        alphas,
        device=device,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    iterations = max(iterations, used)
    return TorchLassoCVResult(
        coef_=coefficients[selected_index].astype(np.float32),
        intercept_=float(intercepts[selected_index]),
        alpha_=float(alphas[selected_index]),
        alphas_=alphas,
        mean_validation_mse_=mean_mse.astype(np.float32),
        iterations_=iterations,
    )
