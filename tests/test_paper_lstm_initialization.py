import numpy as np
import torch
from torch import nn

from scripts.train_exact_window_end_to_end import ExactWindowFingerDecoder


def test_lars_initialization_uses_standard_nonlinear_lstm() -> None:
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


def test_lars_initialization_reproduces_linear_prediction_at_epoch_zero() -> None:
    torch.manual_seed(7)
    coefficients = np.asarray([0.4, -0.2, 0.1], dtype=np.float32)
    intercept = 0.05
    model = ExactWindowFingerDecoder(
        input_channels=2,
        component_count=2,
        selected_indices=np.arange(coefficients.size),
        feature_mean=np.zeros(coefficients.size, dtype=np.float32),
        feature_scale=np.ones(coefficients.size, dtype=np.float32),
        hidden_size=4,
        head_initialization="lars_linear_regime",
    )
    model.initialize_lars_linear_regime(coefficients, intercept)
    hidden = model.lstm.hidden_size
    candidate_row = 2 * hidden
    assert torch.allclose(
        model.lstm.weight_ih_l0[candidate_row],
        torch.from_numpy(coefficients),
    )
    near_zero = torch.cat(
        (
            model.lstm.weight_ih_l0[0],
            model.lstm.weight_ih_l0[hidden],
            model.lstm.weight_ih_l0[3 * hidden],
            model.lstm.weight_hh_l0[:, 0],
            model.temporal.weight[0, 1:],
        )
    )
    assert torch.count_nonzero(near_zero) == near_zero.numel()
    assert float(near_zero.abs().max()) < 5.0e-3
    assert model.lstm.bias_ih_l0[0] == 3.0
    assert model.lstm.bias_ih_l0[hidden] == -3.0
    assert model.lstm.bias_ih_l0[3 * hidden] == 3.0
    features = torch.randn(2, 20, coefficients.size) * 0.5

    observed = model.decode(features).detach().numpy()
    expected = torch.relu(
        features @ torch.from_numpy(coefficients) + intercept
    ).numpy()
    correlation = np.corrcoef(observed.ravel(), expected.ravel())[0, 1]

    assert correlation > 0.98
    assert np.sqrt(np.mean((observed - expected) ** 2)) < 5.0e-2
    assert model.direct.weight.requires_grad is False
