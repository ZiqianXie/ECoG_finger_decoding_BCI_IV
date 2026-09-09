import numpy as np
import torch

from ecog_decoding.models import WaveletPacketEnergy
from ecog_decoding.spectral_audit import summarize_responses, wavelet_path_responses


def test_interpolated_taps_compress_tree_to_200_hz_at_1000_hz():
    frontend = WaveletPacketEnergy(
        trainable=False,
        tap_resample_up=5,
        tap_resample_down=2,
    ).eval()
    frequency, magnitude = wavelet_path_responses(
        frontend, sampling_rate_hz=1000.0, fft_size=32768
    )
    summaries = summarize_responses(frequency, magnitude, frontend.band_names)
    centroids = np.asarray([item["power_centroid_hz"] for item in summaries])
    assert frontend.kernel_size == 41
    assert np.min(centroids) < 25.0
    assert np.max(centroids) < 210.0
    assert np.ptp(centroids) > 150.0


def test_default_wavelet_initialization_is_unchanged():
    default = WaveletPacketEnergy(trainable=False)
    explicit = WaveletPacketEnergy(
        trainable=False, tap_resample_up=1, tap_resample_down=1
    )
    for default_layer, explicit_layer in zip(default.layers, explicit.layers):
        np.testing.assert_array_equal(
            default_layer.weight.detach().numpy(), explicit_layer.weight.detach().numpy()
        )
