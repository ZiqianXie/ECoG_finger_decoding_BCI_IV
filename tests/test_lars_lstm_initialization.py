"""Compatibility check for the historical nonlinear LSTM initializer."""

import numpy as np
from torch import nn

from scripts.train_exact_window_end_to_end import ExactWindowFingerDecoder


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
