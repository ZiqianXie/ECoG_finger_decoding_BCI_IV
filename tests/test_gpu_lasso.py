from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Lasso

from gpu_lasso import fit_torch_lasso_cv, lasso_path_fista


def sparse_problem(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(120, 12)).astype(np.float32)
    coefficients = np.zeros(12, dtype=np.float32)
    coefficients[[1, 6, 9]] = (1.4, -0.9, 0.65)
    target = features @ coefficients + 0.07 * rng.normal(size=features.shape[0])
    return features, target.astype(np.float32)


def test_fista_matches_sklearn_lasso_at_fixed_alpha() -> None:
    features, target = sparse_problem()
    alpha = np.asarray([0.035], dtype=np.float32)
    coefficients, intercepts, _ = lasso_path_fista(
        features,
        target,
        alpha,
        device=torch.device("cpu"),
        max_iter=800,
        tolerance=1.0e-7,
    )
    reference = Lasso(alpha=float(alpha[0]), max_iter=20_000, tol=1.0e-6).fit(
        features, target
    )
    assert np.allclose(coefficients[0], reference.coef_, atol=2.0e-4)
    assert np.isclose(intercepts[0], reference.intercept_, atol=2.0e-4)


def test_gpu_lasso_cv_is_deterministic_and_predictive_on_cpu() -> None:
    features, target = sparse_problem()
    indices = np.arange(features.shape[0])
    splits = [
        (indices[40:], indices[:40]),
        (np.concatenate((indices[:40], indices[80:])), indices[40:80]),
        (indices[:80], indices[80:]),
    ]
    first = fit_torch_lasso_cv(
        features, target, splits, device=torch.device("cpu"), max_iter=300
    )
    second = fit_torch_lasso_cv(
        features, target, splits, device=torch.device("cpu"), max_iter=300
    )
    prediction = features @ first.coef_ + first.intercept_
    assert np.corrcoef(prediction, target)[0, 1] > 0.99
    assert 2 <= np.count_nonzero(np.abs(first.coef_) > 1.0e-7) <= 12
    assert first.alpha_ == second.alpha_
    assert np.array_equal(first.coef_, second.coef_)
