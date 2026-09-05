import numpy as np

from scripts.fit_oof_latent_movement_gate import (
    gated_prediction,
    intended_state,
    transition_model,
    viterbi,
)


def test_intended_state_keeps_largest_finger_as_latent_not_as_cleaned_output() -> None:
    target = np.asarray([[0.0, 0.0], [0.2, 0.1], [0.1, 0.3]])
    assert np.array_equal(intended_state(target, 0.08), [0, 1, 2])


def test_viterbi_uses_persistence_to_bridge_one_ambiguous_bin() -> None:
    state = np.asarray([0, 1, 1, 1, 0])
    transition, prior = transition_model(state, np.ones(state.size, dtype=bool))
    probability = np.full((3, 6), 1.0e-6)
    probability[:, 1] = [0.9, 0.4, 0.9]
    probability[1, 0] = 0.6
    probability /= probability.sum(axis=1, keepdims=True)
    decoded = viterbi(probability, transition, prior)
    assert np.array_equal(decoded, [1, 1, 1])


def test_gate_strength_zero_is_identity() -> None:
    prediction = np.asarray([0.2, 0.4])
    state = np.asarray([0, 1])
    activity = np.zeros((6, 1))
    assert np.array_equal(gated_prediction(prediction, state, activity, 0, 0.0), prediction)


def test_superunit_gate_strength_cannot_flip_prediction_sign() -> None:
    prediction = np.asarray([0.2, 0.4])
    state = np.asarray([0, 1])
    activity = np.zeros((6, 1))
    assert np.array_equal(
        gated_prediction(prediction, state, activity, 0, 1.5),
        np.zeros_like(prediction),
    )
