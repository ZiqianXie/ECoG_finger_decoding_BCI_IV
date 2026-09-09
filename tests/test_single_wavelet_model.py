import numpy as np
import torch

from single_wavelet_model import SingleWaveletDecoder


def make_initialization(coefficients: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "selected_indices": np.arange(coefficients.size),
        "feature_mean": np.zeros(coefficients.size, dtype=np.float32),
        "feature_scale": np.ones(coefficients.size, dtype=np.float32),
        "coefficients": coefficients,
        "intercept": np.asarray(0.04, dtype=np.float32),
    }


def test_lars_is_embedded_in_a_standard_nonlinear_lstm() -> None:
    torch.manual_seed(13)
    coefficients = np.asarray([0.35, -0.20, 0.10, 0.05], dtype=np.float32)
    model = SingleWaveletDecoder(
        np.eye(2, dtype=np.float32),
        make_initialization(coefficients),
        hidden_size=6,
        lars_candidate_scale=0.1,
        near_zero_std=1.0e-3,
        open_gate_bias=5.0,
        forget_gate_bias=-5.0,
        output_activation="linear",
        energy_window_samples=40,
        tap_resample_up=5,
        tap_resample_down=2,
    )
    features = torch.randn(3, 40, coefficients.size) * 0.5
    expected = model.direct_features(features).detach()
    actual = model.decode_features(features).detach()

    correlation = np.corrcoef(expected.numpy().ravel(), actual.numpy().ravel())[0, 1]
    assert correlation > 0.999
    assert torch.sqrt(torch.mean((expected - actual).square())) < 1.0e-2
    assert model.direct.weight.requires_grad is False
    assert type(model.lstm) is torch.nn.LSTM
    hidden = model.lstm.hidden_size
    torch.testing.assert_close(
        model.lstm.weight_ih_l0[2 * hidden],
        0.1 * torch.from_numpy(coefficients),
    )


def test_release_topology_and_nonnegative_output() -> None:
    coefficients = np.asarray([0.2, -0.1], dtype=np.float32)
    model = SingleWaveletDecoder(
        np.eye(2, dtype=np.float32),
        make_initialization(coefficients),
        hidden_size=4,
    )
    assert len(model.wavelet.layers) == 3
    assert model.wavelet.tap_resample_up == 5
    assert model.wavelet.tap_resample_down == 2
    assert not any("lmp" in name.lower() for name, _ in model.named_modules())
    assert not hasattr(model, "movement_gate")
    prediction = model.decode_features(torch.randn(2, 12, 2))
    assert torch.all(prediction >= 0)
