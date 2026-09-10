import numpy as np
import torch

from single_wavelet_model import PaperEquationLSTM, SingleWaveletDecoder


def make_initialization(coefficients: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "selected_indices": np.arange(coefficients.size),
        "feature_mean": np.zeros(coefficients.size, dtype=np.float32),
        "feature_scale": np.ones(coefficients.size, dtype=np.float32),
        "coefficients": coefficients,
        "intercept": np.asarray(0.04, dtype=np.float32),
    }


def make_candidate_initialization(
    coefficients: np.ndarray,
) -> dict[str, np.ndarray]:
    initialization = make_initialization(coefficients)
    initialization.update(
        {
            "candidate_indices": np.arange(coefficients.size + 2),
            "candidate_feature_mean": np.zeros(
                coefficients.size + 2, dtype=np.float32
            ),
            "candidate_feature_scale": np.ones(
                coefficients.size + 2, dtype=np.float32
            ),
            "selected_candidate_positions": np.arange(coefficients.size),
        }
    )
    return initialization


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
    assert "candidate_indices" not in model.state_dict()
    prediction = model.decode_features(torch.randn(2, 12, 2))
    assert torch.all(prediction >= 0)


def test_paper_equation_cell_keeps_gate_nonlinearity_without_state_tanh() -> None:
    cell = PaperEquationLSTM(input_size=1, hidden_size=1)
    with torch.no_grad():
        cell.weight_ih_l0.zero_()
        cell.weight_hh_l0.zero_()
        cell.bias_ih_l0.zero_()
        cell.bias_hh_l0.zero_()
        # PyTorch gate order is input, forget, candidate, output.  With all
        # gates at sigmoid(0)=0.5 and candidate=2, c1=1 and h1=0.5.
        cell.bias_ih_l0[2] = 2.0
    output, (hidden, state) = cell(torch.zeros(1, 2, 1))
    torch.testing.assert_close(output[0, :, 0], torch.tensor([0.5, 0.75]))
    torch.testing.assert_close(hidden[0, 0, 0], torch.tensor(0.75))
    torch.testing.assert_close(state[0, 0, 0], torch.tensor(1.5))


def test_decoder_can_use_paper_equation_cell_with_lars_initialization() -> None:
    coefficients = np.asarray([0.35, -0.20], dtype=np.float32)
    model = SingleWaveletDecoder(
        np.eye(2, dtype=np.float32),
        make_initialization(coefficients),
        hidden_size=4,
        recurrent_cell="paper_equations",
        lars_candidate_scale=0.1,
        near_zero_std=1.0e-3,
        open_gate_bias=3.0,
        forget_gate_bias=-3.0,
        output_activation="linear",
    )
    assert isinstance(model.lstm, PaperEquationLSTM)
    expected = model.direct_features(torch.zeros(2, 5, 2)).detach()
    observed = model.decode_features(torch.zeros(2, 5, 2)).detach()
    assert torch.isfinite(observed).all()
    assert torch.sqrt(torch.mean((expected - observed).square())) < 2.0e-2


def test_residual_lstm_starts_exactly_at_softplus_lars() -> None:
    coefficients = np.asarray([0.35, -0.20], dtype=np.float32)
    model = SingleWaveletDecoder(
        np.eye(2, dtype=np.float32),
        make_initialization(coefficients),
        hidden_size=5,
        recurrent_cell="residual_lstm",
        output_activation="softplus",
    )
    features = torch.randn(2, 12, 2)

    torch.testing.assert_close(
        model.decode_features(features), model.direct_features(features)
    )
    assert model.head_initialization == "zero_residual_on_lars"
    assert isinstance(model.lstm, torch.nn.LSTM)


def test_auxiliary_velocity_uses_the_same_recurrent_state() -> None:
    coefficients = np.asarray([0.35, -0.20], dtype=np.float32)
    model = SingleWaveletDecoder(
        np.eye(2, dtype=np.float32),
        make_initialization(coefficients),
        hidden_size=5,
        recurrent_cell="residual_lstm",
        movement_head=True,
        velocity_head=True,
        output_activation="softplus",
    )
    features = torch.randn(2, 12, 2)

    trajectory, movement_logit, velocity = model.decode_features_with_auxiliary(
        features
    )

    torch.testing.assert_close(trajectory, model.decode_features(features))
    assert movement_logit.shape == trajectory.shape
    assert velocity.shape == trajectory.shape
    assert model.velocity_output.in_features == model.lstm.hidden_size


def test_residual_gru_starts_exactly_at_softplus_lars_and_can_learn() -> None:
    coefficients = np.asarray([0.35, -0.20], dtype=np.float32)
    model = SingleWaveletDecoder(
        np.eye(2, dtype=np.float32),
        make_initialization(coefficients),
        hidden_size=5,
        recurrent_cell="residual_gru",
        output_activation="softplus",
    )
    features = torch.randn(2, 12, 2)
    target = torch.rand(2, 12)

    torch.testing.assert_close(
        model.decode_features(features), model.direct_features(features)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (model.decode_features(features) - target).square().mean()
        loss.backward()
        optimizer.step()
    assert torch.count_nonzero(model.output.weight).item() > 0
    assert torch.count_nonzero(model.lstm.weight_ih_l0.grad).item() > 0


def test_candidate_pool_residual_starts_exactly_at_selected_lars() -> None:
    coefficients = np.asarray([0.35, -0.20], dtype=np.float32)
    model = SingleWaveletDecoder(
        np.eye(2, dtype=np.float32),
        make_candidate_initialization(coefficients),
        hidden_size=5,
        recurrent_cell="residual_lstm",
        residual_input="candidate",
        output_activation="softplus",
    )
    features = torch.randn(2, 12, 4)

    torch.testing.assert_close(
        model.decode_features(features), model.direct_features(features)
    )
    assert model.lstm.input_size == 4
    assert model.direct.in_features == 2


def test_candidate_direct_path_is_fixed_width_and_matches_selected_scaling() -> None:
    coefficients = np.asarray([0.35, -0.20], dtype=np.float32)
    initialization = make_candidate_initialization(coefficients)
    initialization["feature_mean"] = np.asarray([2.0, -1.0], dtype=np.float32)
    initialization["feature_scale"] = np.asarray([0.5, 4.0], dtype=np.float32)
    initialization["candidate_feature_mean"] = np.asarray(
        [1.5, -2.0, 3.0, 4.0], dtype=np.float32
    )
    initialization["candidate_feature_scale"] = np.asarray(
        [2.0, 0.25, 5.0, 6.0], dtype=np.float32
    )
    model = SingleWaveletDecoder(
        np.eye(2, dtype=np.float32),
        initialization,
        hidden_size=5,
        recurrent_cell="residual_lstm",
        residual_input="candidate",
        output_activation="linear",
    )
    features = torch.randn(2, 12, 4)
    selected = features[..., :2]
    expected = torch.nn.functional.linear(
        (selected - model.feature_mean) / model.feature_scale,
        model.direct.weight,
        model.direct.bias,
    ).squeeze(-1)

    torch.testing.assert_close(model.direct_features(features), expected)
    assert model.candidate_direct_weight.shape == (1, 4)
    assert torch.count_nonzero(model.candidate_direct_weight[..., 2:]).item() == 0
    assert "candidate_direct_weight" not in model.state_dict()


def test_near_zero_residual_initialization_trains_recurrent_weights_immediately() -> None:
    torch.manual_seed(19)
    coefficients = np.asarray([0.35, -0.20], dtype=np.float32)
    model = SingleWaveletDecoder(
        np.eye(2, dtype=np.float32),
        make_candidate_initialization(coefficients),
        hidden_size=5,
        recurrent_cell="residual_lstm",
        residual_input="candidate",
        residual_output_init_std=1.0e-3,
        output_activation="linear",
    )
    features = torch.randn(2, 12, 4)
    direct = model.direct_features(features).detach()
    prediction = model.decode_features(features)

    delta = torch.sqrt(torch.mean((prediction.detach() - direct).square()))
    assert 0 < delta < 1.0e-2
    assert model.head_initialization == "near_zero_residual_on_lars"

    prediction.square().mean().backward()
    assert torch.count_nonzero(model.lstm.weight_ih_l0.grad).item() > 0


def test_candidate_pool_residual_can_use_a_feature_excluded_by_lars() -> None:
    coefficients = np.asarray([0.35, -0.20], dtype=np.float32)
    model = SingleWaveletDecoder(
        np.eye(2, dtype=np.float32),
        make_candidate_initialization(coefficients),
        hidden_size=5,
        recurrent_cell="residual_lstm",
        residual_input="candidate",
        output_activation="linear",
    )
    with torch.no_grad():
        model.output.weight.fill_(0.1)
    features = torch.randn(2, 12, 4, requires_grad=True)

    model.decode_features(features).sum().backward()

    assert torch.count_nonzero(features.grad[..., 2:]).item() > 0


def test_auxiliary_movement_head_preserves_lars_trajectory_and_trains_lstm() -> None:
    coefficients = np.asarray([0.35, -0.20], dtype=np.float32)
    model = SingleWaveletDecoder(
        np.eye(2, dtype=np.float32),
        make_candidate_initialization(coefficients),
        hidden_size=5,
        recurrent_cell="residual_lstm",
        residual_input="candidate",
        movement_head=True,
        output_activation="softplus",
    )
    features = torch.randn(2, 12, 4)

    trajectory, movement_logit = model.decode_features_with_movement(features)
    torch.testing.assert_close(trajectory, model.direct_features(features))
    assert movement_logit.shape == trajectory.shape

    movement_target = torch.randint(0, 2, movement_logit.shape).float()
    torch.nn.functional.binary_cross_entropy_with_logits(
        movement_logit, movement_target
    ).backward()
    assert torch.count_nonzero(model.movement_output.weight.grad).item() > 0
    assert torch.count_nonzero(model.lstm.weight_ih_l0.grad).item() > 0
