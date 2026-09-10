from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gpu_ridge import fit_torch_ridge_cv, ridge_path_eigendecomposition


def regression_problem(seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(120, 18)).astype(np.float32)
    coefficients = rng.normal(size=18).astype(np.float32)
    target = features @ coefficients + 0.1 * rng.normal(size=features.shape[0])
    return features, target.astype(np.float32)


def test_gpu_ridge_path_matches_standardized_sklearn_fit() -> None:
    features, target = regression_problem()
    alphas = np.asarray((0.1, 3.0, 100.0), dtype=np.float32)
    coefficients, intercepts = ridge_path_eigendecomposition(
        features, target, alphas, device=torch.device("cpu")
    )
    for index, alpha in enumerate(alphas):
        reference = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha))).fit(
            features, target
        )
        prediction = features @ coefficients[index] + intercepts[index]
        assert np.allclose(prediction, reference.predict(features), atol=2.0e-4)


def test_gpu_ridge_cv_selects_a_deterministic_predictive_model() -> None:
    features, target = regression_problem()
    indices = np.arange(features.shape[0])
    splits = [
        (indices[40:], indices[:40]),
        (np.concatenate((indices[:40], indices[80:])), indices[40:80]),
        (indices[:80], indices[80:]),
    ]
    options = np.logspace(-2, 3, 6).astype(np.float32)
    first = fit_torch_ridge_cv(
        features, target, splits, alphas=options, device=torch.device("cpu")
    )
    second = fit_torch_ridge_cv(
        features, target, splits, alphas=options, device=torch.device("cpu")
    )
    assert np.corrcoef(first.predict(features), target)[0, 1] > 0.99
    assert first.alpha_ == second.alpha_
    assert np.array_equal(first.coef_, second.coef_)
