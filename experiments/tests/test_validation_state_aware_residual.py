import numpy as np

from scripts.apply_validation_multibase_latent_gate import select_diverse_candidates
from scripts.fit_validation_state_aware_residual import atom_features


def test_candidate_ranking_respects_training_mask() -> None:
    target = np.zeros((6, 5))
    target[:, 0] = np.arange(6)
    first = np.zeros_like(target)
    second = np.zeros_like(target)
    first[:3, 0] = target[:3, 0]
    first[3:, 0] = -target[3:, 0]
    second[:3, 0] = -target[:3, 0]
    second[3:, 0] = target[3:, 0]
    candidates = [
        {"path": "first", "validation": first},
        {"path": "second", "validation": second},
    ]
    selected = select_diverse_candidates(
        candidates, target, 1, 1.0, mask=np.asarray([1, 1, 1, 0, 0, 0], dtype=bool)
    )
    assert selected[0]["path"] == "first"


def test_atom_features_include_state_conditioned_candidate_residuals() -> None:
    base = np.zeros((4, 5))
    prediction = np.ones((4, 5))
    probability = np.zeros((4, 6))
    probability[:, 0] = 0.25
    probability[:, 1] = 0.75
    features = atom_features(
        base,
        [{"validation": prediction}],
        probability,
        finger=0,
        split="validation",
    )
    assert features.shape == (4, 14)
    assert np.allclose(features[:, 0], 1.0)
    assert np.allclose(features[:, 1], 0.75)
    assert np.allclose(features[:, 2], 0.25)
