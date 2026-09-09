import numpy as np

from scripts.fit_event_finger_mixture import fit_mixture


def test_soft_mixture_recovers_clear_single_finger_events() -> None:
    generator = np.random.default_rng(4)
    labels = np.repeat(np.arange(5), 12)
    glove = np.eye(5)[labels] + generator.normal(0.0, 0.01, (labels.size, 5))
    neural = np.eye(5)[labels] + generator.normal(0.0, 0.02, (labels.size, 5))
    glove = np.maximum(glove, 0.0)
    neural = np.maximum(neural, 0.0)
    glove /= np.linalg.norm(glove, axis=1, keepdims=True)
    neural /= np.linalg.norm(neural, axis=1, keepdims=True)
    posterior, glove_map, neural_map, _, _, state_names = fit_mixture(
        glove,
        neural,
        iterations=50,
        temperature=0.05,
        neural_weight=1.0,
        identity_strength=4.0,
    )
    predicted = np.argmax(posterior, axis=1)
    expected = labels + 1  # state zero is idle
    assert np.mean(predicted == expected) > 0.95
    assert np.trace(glove_map) > 4.5
    assert np.trace(neural_map) > 4.5
    assert state_names[1:6] == ["thumb", "index", "middle", "ring", "little"]
