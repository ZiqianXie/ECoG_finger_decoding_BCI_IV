import numpy as np

from scripts.apply_validation_soft_latent_gate import forward_backward, soft_gated_prediction
from scripts.apply_validation_multibase_latent_gate import temper_probability


def test_forward_backward_returns_normalized_finite_posteriors() -> None:
    emission = np.asarray([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]])
    transition = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    posterior = forward_backward(emission, transition, np.asarray([0.5, 0.5]))
    assert np.isfinite(posterior).all()
    assert np.allclose(posterior.sum(axis=1), 1.0)
    assert posterior[-1, 1] > posterior[-1, 0]


def test_soft_superunit_gate_strength_is_clipped_at_zero() -> None:
    prediction = np.asarray([0.2, 0.4])
    posterior = np.zeros((2, 6))
    posterior[:, 0] = 1.0
    activity = np.zeros((6, 1))
    assert np.array_equal(
        soft_gated_prediction(prediction, posterior, activity, 0, 1.5),
        np.zeros_like(prediction),
    )


def test_probability_temperature_preserves_normalization_and_sharpens() -> None:
    probability = np.asarray([[0.1, 0.9]])
    sharpened = temper_probability(probability, 0.5)
    assert np.allclose(sharpened.sum(axis=1), 1.0)
    assert sharpened[0, 1] > probability[0, 1]
