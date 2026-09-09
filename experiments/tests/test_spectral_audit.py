import numpy as np

from ecog_decoding.models import WaveletPacketEnergy
from ecog_decoding.spectral_audit import (
    normalized_response_similarity,
    summarize_responses,
    wavelet_path_responses,
)
from scripts.synthetic_filter_recovery import make_scenario, movement_control


def test_full_path_response_shapes_and_self_similarity() -> None:
    frontend = WaveletPacketEnergy(trainable=False)
    frequency, magnitude = wavelet_path_responses(frontend, fft_size=2048)
    assert magnitude.shape == (8, 1025)
    assert frequency.shape == (1025,)
    np.testing.assert_allclose(
        normalized_response_similarity(magnitude, magnitude), np.ones(8), atol=1e-12
    )
    summaries = summarize_responses(frequency, magnitude, frontend.band_names)
    assert len(summaries) == 8
    assert all(0.0 <= item["peak_hz"] <= 500.0 for item in summaries)


def test_one_extra_wavelet_level_doubles_paths_and_expands_support() -> None:
    depth3 = WaveletPacketEnergy(levels=3, trainable=False)
    depth4 = WaveletPacketEnergy(levels=4, trainable=False)
    assert depth3.effective_kernel_size == 113
    assert depth4.effective_kernel_size == 241
    frequency, magnitude = wavelet_path_responses(depth4, fft_size=2048)
    assert magnitude.shape == (16, frequency.size)


def test_synthetic_scenarios_are_reproducible_and_aligned_to_bins() -> None:
    first = movement_control(200, np.random.default_rng(4))
    second = movement_control(200, np.random.default_rng(4))
    np.testing.assert_array_equal(first, second)
    values, target, bands = make_scenario("spatial_mix", 200, 1000.0, 4)
    assert values.shape == (8000, 6)
    assert target.shape == (200,)
    assert bands == [[78.0, 88.0]]
    assert np.isfinite(values).all()
