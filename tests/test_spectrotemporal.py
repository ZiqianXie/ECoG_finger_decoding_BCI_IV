from __future__ import annotations

import torch

from ecog_decoding.spectrotemporal import (
    ContinuousMorletDecoder,
    GroupedTopKAtomGate,
    HilbertQuadrature,
    MorletAtomBank,
    SoftScaleSpectroTemporalConv2d,
    SpectroTemporalWaveletDecoder,
    WaveletPacketSpectrogram,
    analytic_signal,
    label_independent_morlet_centers,
    wavelet_packet_frequency_order,
)


def test_wavelet_frequency_order_uses_gray_code() -> None:
    assert wavelet_packet_frequency_order(3) == (0, 1, 3, 2, 6, 7, 5, 4)


def test_analytic_signal_suppresses_negative_frequency() -> None:
    time = torch.arange(128, dtype=torch.float32)
    signal = torch.cos(2.0 * torch.pi * 9.0 * time / time.numel())
    observed = analytic_signal(signal)
    expected = torch.exp(2.0j * torch.pi * 9.0 * time / time.numel())
    torch.testing.assert_close(observed, expected, atol=2.0e-5, rtol=2.0e-5)


def test_real_hilbert_pair_is_quadrature_in_the_signal_interior() -> None:
    time = torch.arange(512, dtype=torch.float32)
    signal = torch.cos(2.0 * torch.pi * 31.0 * time / time.numel())
    shifted = HilbertQuadrature(kernel_size=129)(signal[None, None]).squeeze()
    expected = torch.sin(2.0 * torch.pi * 31.0 * time / time.numel())
    correlation = torch.corrcoef(torch.stack((shifted[80:-80], expected[80:-80])))[0, 1]
    assert correlation > 0.99


def test_energy_only_spectrogram_matches_frequency_ordered_packet() -> None:
    torch.manual_seed(30)
    module = WaveletPacketSpectrogram(
        levels=2,
        kernel_size=17,
        trainable=False,
        energy_window_samples=20,
        energy_stride_samples=20,
    )
    x = torch.randn(2, 3, 160)
    observed = module(x)
    expected = module.packet(x).index_select(2, module.frequency_order)
    torch.testing.assert_close(observed, expected)


def test_phase_spectrogram_is_finite_and_differentiable() -> None:
    torch.manual_seed(31)
    module = WaveletPacketSpectrogram(
        levels=2,
        energy_window_samples=20,
        energy_stride_samples=20,
        include_phase=True,
    )
    x = torch.randn(2, 3, 160, requires_grad=True)
    observed = module(x)
    assert observed.shape == (2, 9, 4, 8)
    assert torch.isfinite(observed).all()
    observed.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(layer.weight.grad is not None for layer in module.packet.layers)


def test_soft_scale_convolution_mixes_planes_and_selects_kernel_shapes() -> None:
    torch.manual_seed(32)
    module = SoftScaleSpectroTemporalConv2d(
        input_planes=6,
        output_units=4,
        kernel_sizes=((1, 3), (3, 5), (5, 7)),
    )
    x = torch.randn(2, 6, 8, 25, requires_grad=True)
    observed = module(x)
    assert observed.shape == (2, 4, 8, 25)
    probabilities = module.scale_probabilities()
    torch.testing.assert_close(probabilities.sum(dim=0), torch.ones(4))
    with torch.no_grad():
        module.scale_logits[:, 0] = torch.tensor((-4.0, 6.0, -4.0))
    assert module.selected_kernel_sizes()[0] == (3, 5)
    observed.square().mean().backward()
    assert x.grad is not None
    assert module.scale_logits.grad is not None
    assert all(branch.weight.grad is not None for branch in module.branches)


def test_spectrotemporal_map_does_not_normalize_away_amplitude() -> None:
    module = SoftScaleSpectroTemporalConv2d(
        input_planes=2,
        output_units=3,
        kernel_sizes=((1, 1),),
        include_frequency_coordinates=False,
        activation="identity",
    )
    x = torch.randn(2, 2, 4, 5)
    with torch.no_grad():
        module.branches[0].bias.zero_()
        baseline = module(x)
        scaled = module(3.0 * x)
    torch.testing.assert_close(scaled, 3.0 * baseline)


def test_joint_spectrotemporal_decoder_backpropagates_end_to_end() -> None:
    torch.manual_seed(33)
    model = SpectroTemporalWaveletDecoder(
        input_channels=5,
        window_samples=160,
        spatial_components=3,
        wavelet_levels=2,
        energy_window_samples=20,
        energy_stride_samples=20,
        include_phase=True,
        convolution_units=4,
        kernel_sizes=((1, 3), (3, 5)),
        feature_width=8,
        hidden_size=6,
        output_fingers=2,
    )
    prediction = model(torch.randn(2, 4, 5, 160))
    assert prediction.shape == (2, 4, 2)
    assert torch.all(prediction >= 0)
    prediction.square().mean().backward()
    assert model.spatial_projection.weight.grad is not None
    assert model.spectrotemporal.scale_logits.grad is not None
    assert model.recurrent.weight_ih_l0.grad is not None
    assert model.direct_output.weight.grad is not None


def test_decoder_can_extract_in_chunks_then_decode_once() -> None:
    torch.manual_seed(34)
    model = SpectroTemporalWaveletDecoder(
        input_channels=3,
        window_samples=80,
        spatial_components=2,
        wavelet_levels=2,
        energy_window_samples=20,
        energy_stride_samples=20,
        convolution_units=3,
        kernel_sizes=((1, 3),),
        feature_width=5,
        hidden_size=4,
        output_fingers=1,
        output_activation="linear",
    ).eval()
    windows = torch.randn(1, 6, 3, 80)
    with torch.no_grad():
        direct = model(windows)
        features = torch.cat(
            (model.extract(windows[:, :2]), model.extract(windows[:, 2:])), dim=1
        )
        chunked = model.decode(features)
    torch.testing.assert_close(chunked, direct)


def test_direct_head_starts_near_zero_without_dead_softplus() -> None:
    torch.manual_seed(35)
    model = SpectroTemporalWaveletDecoder(
        input_channels=3,
        window_samples=80,
        spatial_components=2,
        wavelet_levels=2,
        energy_window_samples=20,
        energy_stride_samples=20,
        convolution_units=3,
        kernel_sizes=((1, 3),),
        feature_width=8,
        hidden_size=4,
    )
    assert model.direct_output.weight.std() < 2.0e-3
    torch.testing.assert_close(
        model.direct_output.bias,
        torch.zeros_like(model.direct_output.bias),
    )


def test_label_independent_morlet_centers_cover_requested_spectrum() -> None:
    centers = label_independent_morlet_centers(16, 4.0, 450.0)
    assert centers.shape == (16,)
    assert torch.all(centers[1:] > centers[:-1])
    torch.testing.assert_close(centers[[0, -1]], torch.tensor((4.0, 450.0)))


def test_morlet_atoms_are_normalized_and_differentiable() -> None:
    torch.manual_seed(36)
    bank = MorletAtomBank(
        atom_count=6,
        kernel_size=65,
        pool_samples=20,
        maximum_hz=400.0,
    )
    real, imaginary = bank.complex_kernels()
    torch.testing.assert_close(real.mean(dim=-1), torch.zeros(6), atol=2e-7, rtol=0)
    torch.testing.assert_close(
        (real.square() + imaginary.square()).sum(dim=-1),
        torch.ones(6),
        atol=2e-6,
        rtol=2e-6,
    )
    x = torch.randn(2, 3, 200, requires_grad=True)
    representations, broadband = bank(x)
    assert representations.shape == (2, 3, 4, 6, 10)
    assert broadband.shape == (2, 6, 10)
    (representations.square().mean() + broadband.mean()).backward()
    assert bank.center_logits.grad is not None
    assert bank.width_logits.grad is not None
    assert x.grad is not None


def test_long_morlet_fft_matches_direct_causal_convolution() -> None:
    torch.manual_seed(38)
    bank = MorletAtomBank(
        atom_count=4,
        kernel_size=257,
        pool_samples=20,
        maximum_hz=400.0,
        trainable=False,
    )
    x = torch.randn(1, 2, 400)
    observed, _ = bank(x)
    real_kernel, imaginary_kernel = bank.complex_kernels()
    kernels = torch.cat((real_kernel, imaginary_kernel), dim=0)[:, None]
    direct = torch.nn.functional.conv1d(
        torch.nn.functional.pad(x.reshape(2, 1, 400), (256, 0)), kernels
    ).reshape(1, 2, 2, 4, 400)
    real = direct[:, :, 0]
    imaginary = direct[:, :, 1]
    amplitude = torch.sqrt(real.square() + imaginary.square() + bank.epsilon**2)
    phase_gate = torch.log1p(amplitude)

    def pool(values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool1d(
            values.reshape(2, 4, 400), 20, 20
        ).reshape(1, 2, 4, 20)

    expected = torch.stack(
        (
            pool(real),
            torch.log1p(pool(real.square() + imaginary.square())),
            pool(phase_gate * real / amplitude),
            pool(phase_gate * imaginary / amplitude),
        ),
        dim=2,
    )
    torch.testing.assert_close(observed, expected, atol=2e-5, rtol=2e-5)


def test_grouped_topk_gate_anneals_without_losing_gradients() -> None:
    gate = GroupedTopKAtomGate(atom_count=8, top_k=3)
    dense = gate()
    torch.testing.assert_close(dense, torch.ones(8))
    with torch.no_grad():
        gate.logits.copy_(torch.arange(8, dtype=torch.float32))
    gate.set_schedule(temperature=0.25, hard_fraction=1.0)
    hard = gate()
    assert torch.count_nonzero(hard).item() == 3
    assert gate.selected_indices().tolist() == [5, 6, 7]
    hard.sum().backward()
    assert gate.logits.grad is not None


def test_continuous_morlet_decoder_backpropagates_through_broadband_skip() -> None:
    torch.manual_seed(37)
    model = ContinuousMorletDecoder(
        input_components=3,
        input_channels=5,
        atom_count=6,
        top_k=3,
        kernel_size=65,
        convolution_units=4,
        kernel_sizes=((1, 3), (3, 5)),
        feature_width=8,
        hidden_size=5,
    )
    prediction = model(torch.randn(2, 5, 200))
    assert prediction.shape == (2, 5)
    assert torch.all(prediction >= 0)
    prediction.square().mean().backward()
    assert model.morlet.center_logits.grad is not None
    assert model.atom_gate is not None and model.atom_gate.logits.grad is not None
    assert model.broadband_projection.weight.grad is not None
    assert isinstance(model.spatial_projection, torch.nn.Conv1d)
    assert model.spatial_projection.weight.grad is not None
