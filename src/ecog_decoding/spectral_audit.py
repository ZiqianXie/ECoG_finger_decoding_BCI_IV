"""Frequency-response measurements for the complete dilated wavelet paths."""

from __future__ import annotations

import numpy as np
import torch

from .models import WaveletPacketEnergy


@torch.inference_mode()
def wavelet_path_responses(
    frontend: WaveletPacketEnergy,
    sampling_rate_hz: float = 1000.0,
    fft_size: int = 16384,
    impulse_amplitude: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency and magnitude for every full dilated tree path."""
    if fft_size < 1024 or fft_size % 2:
        raise ValueError("fft_size must be an even integer of at least 1024")
    parameter = next(frontend.parameters())
    values = torch.zeros(2, 1, fft_size, dtype=parameter.dtype, device=parameter.device)
    values[1, ..., fft_size // 2] = impulse_amplitude
    response = values
    for layer in frontend.layers:
        response = frontend._same_filter(response, layer)
        response = 1.7156 * torch.tanh((2.0 / 3.0) * response)
    # Learned biases create a large DC term that is not part of the local
    # input-to-output transfer function.  Subtract the zero-input response so
    # the FFT measures the finite-difference Jacobian around rest.
    response = (response[1] - response[0]).double().cpu().numpy() / impulse_amplitude
    frequency = np.fft.rfftfreq(fft_size, d=1.0 / sampling_rate_hz)
    magnitude = np.abs(np.fft.rfft(response, axis=1))
    return frequency, magnitude


def summarize_responses(
    frequency: np.ndarray, magnitude: np.ndarray, names: tuple[str, ...]
) -> list[dict[str, object]]:
    """Summarize peak and power support for each path."""
    power = np.square(magnitude)
    result: list[dict[str, object]] = []
    for index, name in enumerate(names):
        cumulative = np.cumsum(power[index])
        cumulative /= max(float(cumulative[-1]), 1.0e-12)
        lower = int(np.searchsorted(cumulative, 0.05))
        upper = int(np.searchsorted(cumulative, 0.95))
        result.append(
            {
                "path": name,
                "peak_hz": float(frequency[int(np.argmax(magnitude[index]))]),
                "power_centroid_hz": float(
                    np.sum(frequency * power[index])
                    / max(float(np.sum(power[index])), 1.0e-12)
                ),
                "central_90_percent_power_hz": [
                    float(frequency[lower]),
                    float(frequency[upper]),
                ],
            }
        )
    return result


def normalized_response_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosine similarity of magnitude-squared responses, path by path."""
    left = np.square(np.asarray(left, dtype=np.float64))
    right = np.square(np.asarray(right, dtype=np.float64))
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("responses must be same-shaped path-by-frequency matrices")
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(left.shape[0], dtype=np.float64),
        where=denominator > 0,
    )
