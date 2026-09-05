"""Compatibility check for the historical nonlinear LSTM initializer."""

import numpy as np
import torch
from torch import nn

from scripts.train_exact_window_end_to_end import ExactWindowFingerDecoder, make_windows


def test_historical_initializer_remains_a_standard_nonlinear_lstm() -> None:
    model = ExactWindowFingerDecoder(
        input_channels=2,
        component_count=2,
        selected_indices=np.arange(3),
        feature_mean=np.zeros(3, dtype=np.float32),
        feature_scale=np.ones(3, dtype=np.float32),
        hidden_size=4,
        head_initialization="lars_linear_regime",
    )

    assert type(model.lstm) is nn.LSTM


def test_softplus_output_preserves_gradient_below_zero() -> None:
    model = ExactWindowFingerDecoder(
        input_channels=2,
        component_count=2,
        selected_indices=np.arange(3),
        feature_mean=np.zeros(3, dtype=np.float32),
        feature_scale=np.ones(3, dtype=np.float32),
        hidden_size=4,
        output_activation="softplus",
        softplus_beta=5.0,
    )
    with torch.no_grad():
        model.direct.weight.zero_()
        model.direct.bias.fill_(-1.0)
        model.temporal.weight.zero_()
        model.temporal.bias.zero_()

    prediction = model.decode(torch.zeros(1, 3, 3))
    prediction.sum().backward()

    assert torch.all(prediction > 0)
    assert model.direct.bias.grad is not None
    assert model.direct.bias.grad.abs().item() > 0


def test_hurdle_output_is_gate_times_positive_amplitude() -> None:
    model = ExactWindowFingerDecoder(
        input_channels=2,
        component_count=2,
        selected_indices=np.arange(3),
        feature_mean=np.zeros(3, dtype=np.float32),
        feature_scale=np.ones(3, dtype=np.float32),
        hidden_size=4,
        output_activation="hurdle",
    )
    prediction, state_logit, amplitude = model.decode_with_hurdle(
        torch.zeros(1, 5, 3)
    )

    torch.testing.assert_close(prediction, torch.sigmoid(state_logit) * amplitude)
    assert torch.all(amplitude > 0)
    prediction.sum().backward()
    assert model.movement_head.weight.grad is not None


def test_gpu_unfold_matches_numpy_stride_windows() -> None:
    recording = np.arange(40, dtype=np.float32).reshape(20, 2)
    expected = make_windows(recording, window_samples=5, stride_samples=2)
    observed = torch.from_numpy(recording).unfold(0, 5, 2).numpy()

    np.testing.assert_array_equal(observed, expected)
