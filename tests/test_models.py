from __future__ import annotations

import numpy as np
import torch

from ecog_decoding.models import (
    CausalLinearAttentionBlock,
    DiagonalSSMBlock,
    DilatedWaveletFilterBank,
    EcogTrajectoryDecoder,
    WaveletPacketEnergy,
    compact_wavelet_taps,
    fit_fastica_spatial_weights,
    fixed_length_wavelet_taps,
)
from ecog_decoding.training import (
    align_causal_sequence,
    joint_trajectory_loss,
    trajectory_metrics,
)


def test_bior22_omits_zero_padding_but_keeps_wavelet_taps() -> None:
    taps = compact_wavelet_taps("bior2.2", "decomposition_highpass")
    expected = np.array([0.3535533905932738, -0.7071067811865476, 0.3535533905932738])
    np.testing.assert_allclose(taps, expected)
    assert np.all(taps != 0.0)


def test_dilated_bank_preserves_time_and_shares_filters_over_electrodes() -> None:
    # Keep explicit coverage for the compact bior2.2 compatibility path while
    # the public default follows the paper's bior6.8 initialization.
    bank = DilatedWaveletFilterBank(wavelet="bior2.2", dilations=(1, 2, 4))
    x = torch.zeros(2, 5, 101)
    x[:, :, 50] = 1.0
    output = bank(x)
    assert output.shape == (2, 5, 3, 101)
    torch.testing.assert_close(output[:, 0], output[:, 4])

    for scale, dilation in enumerate(bank.dilations):
        support = torch.nonzero(output[0, 0, scale], as_tuple=False).flatten()
        torch.testing.assert_close(
            support,
            torch.tensor([50 - dilation, 50, 50 + dilation]),
        )


def test_wavelet_kernels_are_independently_trainable_by_scale() -> None:
    bank = DilatedWaveletFilterBank(dilations=(1, 4))
    x = torch.randn(3, 2, 80)
    loss = bank(x)[:, :, 1].square().mean()
    loss.backward()
    assert bank.kernel_taps.grad is not None
    assert torch.count_nonzero(bank.kernel_taps.grad[0]) == 0
    assert torch.count_nonzero(bank.kernel_taps.grad[1]) > 0


def test_highpass_initialization_rejects_constant_signal_away_from_edges() -> None:
    bank = DilatedWaveletFilterBank(dilations=(1, 8, 32), trainable=False)
    output = bank(torch.ones(1, 2, 256))
    torch.testing.assert_close(output[..., 64:-64], torch.zeros_like(output[..., 64:-64]))


def test_bior68_analysis_filters_are_symmetric_17_tap_kernels() -> None:
    for branch in ("decomposition_lowpass", "decomposition_highpass"):
        taps = fixed_length_wavelet_taps("bior6.8", branch, 17)
        assert taps.shape == (17,)
        np.testing.assert_allclose(taps, taps[::-1], atol=1e-12)


def test_wavelet_defaults_follow_paper_bior68_initialization() -> None:
    expected = compact_wavelet_taps("bior6.8", "decomposition_highpass")
    np.testing.assert_allclose(compact_wavelet_taps(), expected)
    bank = DilatedWaveletFilterBank(dilations=(1,))
    np.testing.assert_allclose(bank.kernel_taps.detach().numpy()[0, 0], expected)


def test_wavelet_packet_calculates_eight_log_energy_bands_at_25_hz() -> None:
    tree = WaveletPacketEnergy(
        levels=3, energy_window_samples=40, energy_stride_samples=40
    )
    x = torch.randn(2, 3, 4000)
    energy = tree(x)
    assert energy.shape == (2, 3, 8, 100)
    assert tree.band_names == (
        "LLL",
        "LLH",
        "LHL",
        "LHH",
        "HLL",
        "HLH",
        "HHL",
        "HHH",
    )
    assert torch.all(energy >= 0)


def test_wavelet_packet_high_frequency_energy_is_zero_for_constant_interior() -> None:
    tree = WaveletPacketEnergy(
        levels=3,
        energy_window_samples=32,
        energy_stride_samples=32,
        trainable=False,
    )
    energy = tree(torch.ones(1, 1, 4096))
    high_frequency_bands = energy[:, :, 1:, 4:-4]
    torch.testing.assert_close(high_frequency_bands, torch.zeros_like(high_frequency_bands))


def test_wavelet_packet_zero_cross_connections_are_trainable() -> None:
    tree = WaveletPacketEnergy(
        levels=3, energy_window_samples=20, energy_stride_samples=20
    )
    assert torch.count_nonzero(tree.layers[1].weight[0, 1]) == 0
    tree(torch.randn(2, 2, 1000)).mean().backward()
    assert all(layer.weight.grad is not None for layer in tree.layers)
    assert torch.count_nonzero(tree.layers[1].weight.grad[0, 1]) > 0


def test_diagonal_ssm_is_causal() -> None:
    torch.manual_seed(4)
    block = DiagonalSSMBlock(width=8, state_size=4).eval()
    x = torch.randn(2, 40, 8)
    changed = x.clone()
    changed[:, 25:] += 10.0
    with torch.no_grad():
        before = block(x)
        after = block(changed)
    torch.testing.assert_close(before[:, :25], after[:, :25], atol=1e-5, rtol=1e-5)


def test_linear_attention_is_causal_and_nonlinear() -> None:
    torch.manual_seed(5)
    block = CausalLinearAttentionBlock(width=8, heads=2).eval()
    x = torch.randn(2, 40, 8)
    changed_future = x.clone()
    changed_future[:, 25:] += 10.0
    with torch.no_grad():
        before = block(x)
        after = block(changed_future)
        doubled = block(2.0 * x)
    torch.testing.assert_close(before[:, :25], after[:, :25], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(doubled, 2.0 * before)


def test_joint_decoder_keeps_cnn_and_outputs_five_25hz_trajectories() -> None:
    frontend = WaveletPacketEnergy(
        levels=3,
        energy_window_samples=40,
        energy_stride_samples=40,
    )
    model = EcogTrajectoryDecoder(
        input_channels=6,
        spatial_components=4,
        feature_width=16,
        temporal_layers=2,
        state_size=4,
        output_fingers=5,
        cnn_history_bins=5,
        wavelet_frontend=frontend,
    )
    output = model(torch.randn(2, 6, 400))
    assert output.shape == (2, 6, 5)
    assert torch.all((0.0 <= output) & (output <= 1.0))
    output.mean().backward()
    assert model.spatial_projection.weight.grad is not None


def test_direct_head_exactly_reproduces_standardized_ridge() -> None:
    torch.manual_seed(21)
    frontend = WaveletPacketEnergy(
        levels=1,
        kernel_size=17,
        energy_window_samples=40,
        energy_stride_samples=40,
    )
    model = EcogTrajectoryDecoder(
        input_channels=2,
        spatial_components=2,
        feature_width=4,
        temporal_layers=1,
        state_size=2,
        output_fingers=2,
        cnn_history_bins=2,
        wavelet_frontend=frontend,
        direct_linear_head=True,
        output_activation="identity",
        zero_initialize_residual=True,
    ).eval()
    fits = [
        (
            np.array([0, 3]),
            np.array([0.2, 0.5]),
            np.array([2.0, 4.0]),
            0.1,
            np.array([0.6, -0.4]),
        ),
        (
            np.array([1, 6]),
            np.array([-0.3, 0.8]),
            np.array([1.5, 0.5]),
            -0.2,
            np.array([0.25, 0.75]),
        ),
    ]
    model.initialize_direct_output_from_ridge(fits)
    x = torch.randn(1, 2, 160)
    with torch.no_grad():
        spatial = model.spatial_projection(x)
        energy = model.wavelet_frontend(spatial)
        per_bin = energy.permute(0, 3, 1, 2).flatten(start_dim=2)
        history = per_bin.unfold(1, 2, 1).permute(0, 1, 3, 2).flatten(start_dim=2)
        prediction = model(x)
    expected = np.empty((1, history.shape[1], 2), dtype=np.float32)
    history_numpy = history.numpy()
    for finger, (indices, mean, scale, target_mean, weight) in enumerate(fits):
        expected[..., finger] = (
            (history_numpy[..., indices] - mean) / scale @ weight + target_mean
        )
    np.testing.assert_allclose(prediction.numpy(), expected, atol=1e-5)


def test_spatial_convolution_can_be_initialized_from_fastica() -> None:
    rng = np.random.default_rng(8)
    sources = rng.laplace(size=(3000, 4))
    mixing = rng.normal(size=(4, 6))
    ecog = sources @ mixing
    weights = fit_fastica_spatial_weights(
        ecog, n_components=4, max_samples=2000, random_state=2
    )
    model = EcogTrajectoryDecoder(
        input_channels=6,
        spatial_components=4,
        feature_width=8,
        temporal_layers=1,
        state_size=2,
    )
    copied = model.initialize_spatial_from_fastica(
        ecog, max_samples=2000, random_state=2
    )
    assert weights.shape == (4, 6)
    np.testing.assert_allclose(copied, weights)
    np.testing.assert_allclose(
        model.spatial_projection.weight.detach().numpy()[:, :, 0], weights
    )


def test_torch_fastica_backend_returns_unit_variance_components() -> None:
    rng = np.random.default_rng(12)
    sources = rng.laplace(size=(4000, 4))
    mixing = rng.normal(size=(4, 6))
    ecog = sources @ mixing
    weights = fit_fastica_spatial_weights(
        ecog,
        n_components=4,
        max_samples=None,
        random_state=3,
        backend="torch",
        device="cpu",
    )
    transformed = (ecog - ecog.mean(axis=0)) @ weights.T
    assert weights.shape == (4, 6)
    np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=2.0e-4)


def test_causal_alignment_keeps_context_without_extra_labels() -> None:
    ecog = np.arange(80 * 2, dtype=np.float32).reshape(80, 2)
    target = np.arange(20 * 5, dtype=np.float32).reshape(20, 5)
    sequence = align_causal_sequence(
        ecog, target, start_bin=7, stop_bin=12, history_bins=4, samples_per_bin=4
    )
    assert sequence.ecog.shape == (2, 32)
    np.testing.assert_array_equal(sequence.ecog[:, 0], ecog[16])
    np.testing.assert_array_equal(sequence.target, target[7:12])


def test_joint_loss_and_metrics_are_finite() -> None:
    target = torch.tensor(
        [[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.5, 0.0]]]
    )
    prediction = (0.9 * target + 0.02).requires_grad_()
    loss, parts = joint_trajectory_loss(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert all(torch.isfinite(value) for value in parts.values())
    metrics = trajectory_metrics(
        prediction.detach().squeeze(0).numpy(), target.squeeze(0).numpy()
    )
    assert metrics["pearson_by_finger"]["thumb"] > 0.9
    assert metrics["pearson_by_finger"]["index"] == 0.0


def test_joint_huber_loss_is_finite_and_backpropagates() -> None:
    target = torch.zeros(1, 8, 5)
    target[:, 2:5, 1] = 1.0
    prediction = torch.full_like(target, 0.2, requires_grad=True)
    loss, parts = joint_trajectory_loss(
        prediction,
        target,
        level_kind="huber",
        huber_delta=0.1,
        correlation_weight=0.3,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert all(torch.isfinite(value) for value in parts.values())
