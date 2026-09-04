#!/usr/bin/env python3
"""Audit public files and quantify power-line contamination."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal

from ecog_decoding.io import load_subject


def line_peak_ratios(x: np.ndarray, fs: float) -> dict[str, float]:
    frequencies, power = signal.welch(x, fs=fs, nperseg=10_000, axis=0)
    median_power = np.median(power, axis=1)
    output: dict[str, float] = {}
    for frequency in (60.0, 120.0, 180.0):
        center = int(np.argmin(np.abs(frequencies - frequency)))
        lower = int(np.argmin(np.abs(frequencies - (frequency - 1.0))))
        upper = int(np.argmin(np.abs(frequencies - (frequency + 1.0))))
        reference = np.sqrt(median_power[lower] * median_power[upper])
        output[str(int(frequency))] = float(median_power[center] / reference)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/bci_competition_iv_ds4"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/audit/dataset.json"))
    args = parser.parse_args()

    report: dict[str, object] = {"sampling_rate_hz": 1000, "subjects": {}}
    for subject in (1, 2, 3):
        data = load_subject(args.data_root, subject)
        report["subjects"][str(subject)] = {
            "train_ecog_shape": list(data.train_ecog.shape),
            "test_ecog_shape": list(data.test_ecog.shape),
            "train_glove_shape": list(data.train_glove.shape),
            "test_glove_shape": list(data.test_glove.shape),
            "line_peak_ratio_vs_59_61_hz_neighbors": line_peak_ratios(
                data.train_ecog, 1000.0
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
