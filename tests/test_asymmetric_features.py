import numpy as np
import torch
from torch.nn import functional as F

from ecog_decoding.models import AsymmetricWaveletPacketEnergy, WaveletPacketEnergy
from scripts.generate_asymmetric_wavelet_features import (
    choose_split_parents,
    lowpass_fir,
)


def test_lowpass_fir_is_symmetric_with_unit_dc_gain() -> None:
    kernel = lowpass_fir(taps=201)
    np.testing.assert_allclose(kernel, kernel[::-1], atol=1.0e-7)
    np.testing.assert_allclose(kernel.sum(), 1.0, atol=1.0e-6)


def test_split_selection_uses_target_band_power() -> None:
    frequency = np.array([0.0, 50.0, 100.0, 150.0, 250.0])
    magnitude = np.array([[5.0, 2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 2.0, 0.0]])
    selected, fraction = choose_split_parents(
        frequency, magnitude, low_hz=60.0, high_hz=200.0, minimum_power_fraction=0.5
    )
    np.testing.assert_array_equal(selected, [1])
    assert fraction[0] == 0.0
    assert fraction[1] == 1.0


def test_trainable_asymmetric_frontend_shape_and_names() -> None:
    frontend = AsymmetricWaveletPacketEnergy(trainable=True)
    output = frontend(torch.randn(2, 3, 1000))
    assert output.shape == (2, 3, 11, 25)
    assert frontend.band_names[-1] == "LMP_0_5_HZ_SIGNED"
    assert frontend.effective_kernel_size == 241
    assert frontend.lmp.weight.requires_grad


def test_asymmetric_initialization_matches_full_packet_children() -> None:
    torch.manual_seed(3)
    values = torch.randn(1, 2, 1000)
    asymmetric = AsymmetricWaveletPacketEnergy(trainable=False)
    depth3 = WaveletPacketEnergy(levels=3, trainable=False, padding_mode="constant")
    depth4 = WaveletPacketEnergy(levels=4, trainable=False, padding_mode="constant")
    observed = asymmetric(values)
    expected_depth3 = depth3(values).index_select(2, torch.tensor([0, 2, 4, 5, 6, 7]))
    expected_depth4 = depth4(values).index_select(2, torch.tensor([2, 3, 6, 7]))
    torch.testing.assert_close(observed[:, :, :6], expected_depth3)
    torch.testing.assert_close(observed[:, :, 6:10], expected_depth4)
    kernel = torch.from_numpy(lowpass_fir())[None, None]
    flat = values.flatten(0, 1).unsqueeze(1)
    expected_lmp = F.avg_pool1d(
        F.conv1d(F.pad(flat, (100, 100), mode="reflect"), kernel), 40, 40
    ).reshape(1, 2, 1, 25)
    torch.testing.assert_close(observed[:, :, 10:], expected_lmp)
