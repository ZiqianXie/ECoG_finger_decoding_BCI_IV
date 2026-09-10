#!/usr/bin/env python3
"""Measure the initialized wavelet packet's implemented frequency responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ecog_decoding.models import WaveletPacketEnergy
from single_wavelet_model import OvercompleteWaveletPacketEnergy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-rate", type=float, default=1000.0)
    parser.add_argument("--tap-resample-up", type=int, default=1)
    parser.add_argument("--tap-resample-down", type=int, default=1)
    parser.add_argument("--fft-size", type=int, default=16384)
    parser.add_argument("--response-points", type=int, default=513)
    parser.add_argument(
        "--frontend",
        choices=("depth3", "overcomplete_depth3_depth4"),
        default="depth3",
    )
    parser.add_argument("--output", default="outputs/audit/wavelet_frequency_response.json")
    args = parser.parse_args()
    if args.fft_size < 1024 or args.fft_size % 2:
        raise ValueError("fft-size must be an even integer of at least 1024")

    frontend_type = (
        OvercompleteWaveletPacketEnergy
        if args.frontend == "overcomplete_depth3_depth4"
        else WaveletPacketEnergy
    )
    frontend = frontend_type(
        trainable=False,
        tap_resample_up=args.tap_resample_up,
        tap_resample_down=args.tap_resample_down,
    ).eval()
    amplitude = 1.0e-4
    impulse = torch.zeros(1, 1, args.fft_size, dtype=torch.float32)
    impulse[..., args.fft_size // 2] = amplitude
    with torch.inference_mode():
        bands = frontend.transform(impulse)
    response = bands[0].double().numpy() / amplitude
    frequency = np.fft.rfftfreq(args.fft_size, d=1.0 / args.sampling_rate)
    magnitude = np.abs(np.fft.rfft(response, axis=1))
    power = magnitude**2
    normalized = magnitude / np.maximum(magnitude.max(axis=1, keepdims=True), 1.0e-12)
    sample_indices = np.linspace(0, frequency.size - 1, args.response_points, dtype=int)

    summaries: list[dict[str, object]] = []
    for index, name in enumerate(frontend.band_names):
        cumulative = np.cumsum(power[index])
        cumulative /= cumulative[-1]
        lower = int(np.searchsorted(cumulative, 0.05))
        upper = int(np.searchsorted(cumulative, 0.95))
        centroid = float(np.sum(frequency * power[index]) / np.sum(power[index]))
        summaries.append(
            {
                "band": name,
                "peak_hz": float(frequency[int(np.argmax(magnitude[index]))]),
                "power_centroid_hz": centroid,
                "central_90_percent_power_hz": [
                    float(frequency[lower]),
                    float(frequency[upper]),
                ],
                "dc_gain_relative_to_peak": float(normalized[index, 0]),
                "nyquist_gain_relative_to_peak": float(normalized[index, -1]),
            }
        )

    result = {
        "sampling_rate_hz": args.sampling_rate,
        "tap_interpolation": {
            "up": args.tap_resample_up,
            "down": args.tap_resample_down,
            "method": "polyphase_kaiser_8.6",
        },
        "fft_size": args.fft_size,
        "impulse_amplitude": amplitude,
        "frontend": args.frontend,
        "method": (
            "actual initialized PyTorch cascade with scaled tanh in its linear "
            "regime; the overcomplete variant retains depth-3 parents and their "
            "depth-4 children from one tree"
        ),
        "band_order": list(frontend.band_names),
        "frequency_hz": np.round(frequency[sample_indices], 5).tolist(),
        "normalized_magnitude": np.round(normalized[:, sample_indices], 6).tolist(),
        "summaries": summaries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
