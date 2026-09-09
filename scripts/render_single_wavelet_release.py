#!/usr/bin/env python3
"""Render the measured initialization of the released wavelet tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.input.read_text())
    frequency = np.asarray(report["frequency_hz"], dtype=np.float64)
    magnitude = np.asarray(report["normalized_magnitude"], dtype=np.float64)
    summaries = report["summaries"]
    order = np.argsort([entry["power_centroid_hz"] for entry in summaries])

    figure, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(order)))
    for color, index in zip(colors, order):
        entry = summaries[int(index)]
        centroid = float(entry["power_centroid_hz"])
        axis.plot(
            frequency,
            magnitude[int(index)],
            color=color,
            linewidth=1.5,
            label=f"{entry['band']}  ({centroid:.1f} Hz centroid)",
        )
    axis.axvline(200.0, color="0.25", linestyle="--", linewidth=1.0)
    axis.text(198.0, 1.02, "200 Hz design edge", ha="right", va="bottom")
    axis.set_xlim(0.0, 225.0)
    axis.set_ylim(0.0, 1.08)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Normalized magnitude")
    axis.set_title("Measured 1 kHz initialization of the eight-leaf wavelet tree")
    axis.legend(frameon=False, ncol=2, fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
