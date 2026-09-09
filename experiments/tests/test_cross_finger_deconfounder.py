import numpy as np

from scripts.fit_oof_cross_finger_deconfounder import (
    apply_model,
    fit_diagonal_prior_ridge,
    smooth_nonnegative,
)


def test_ridge_removes_known_cross_finger_interference() -> None:
    rng = np.random.default_rng(4)
    predictors = rng.normal(size=(500, 5))
    target = predictors[:, 2] - 0.7 * predictors[:, 1]
    model = fit_diagonal_prior_ridge(predictors, target, 0.01, 2)
    estimate = apply_model(model, predictors)
    assert np.corrcoef(estimate, target)[0, 1] > 0.999
    assert model["standardized_coefficients"][1] < -0.4


def test_smooth_nonnegative_is_nonnegative_and_near_identity_when_positive() -> None:
    values = np.asarray([-2.0, 0.0, 2.0])
    transformed = smooth_nonnegative(values)
    assert np.all(transformed >= 0.0)
    assert transformed[0] < 1.0e-3
    assert abs(transformed[-1] - 2.0) < 1.0e-3
