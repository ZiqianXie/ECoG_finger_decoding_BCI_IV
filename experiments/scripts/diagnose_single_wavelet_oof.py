#!/usr/bin/env python3
"""Inspect one merged single-wavelet OOF trajectory without released labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    return float(left_centered @ right_centered / denominator) if denominator > 0 else 0.0


def state_f1(prediction: np.ndarray, target: np.ndarray, threshold: float) -> float:
    predicted = prediction >= threshold
    observed = target >= threshold
    true_positive = int(np.sum(predicted & observed))
    false_positive = int(np.sum(predicted & ~observed))
    false_negative = int(np.sum(~predicted & observed))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return float(2 * precision * recall / max(precision + recall, 1.0e-12))


def strongest_windows(target: np.ndarray, width: int, count: int) -> list[int]:
    score = np.convolve(np.maximum(target, 0.0), np.ones(width), mode="valid")
    starts: list[int] = []
    for candidate in np.argsort(score)[::-1]:
        start = int(candidate)
        if all(abs(start - previous) >= width for previous in starts):
            starts.append(start)
        if len(starts) == count:
            break
    return sorted(starts)


def worst_rest_window(
    prediction: np.ndarray,
    target: np.ndarray,
    width: int,
    threshold: float,
) -> int:
    rest = target < threshold
    score = np.convolve(np.abs(prediction) * rest, np.ones(width), mode="valid")
    rest_count = np.convolve(rest.astype(np.int32), np.ones(width), mode="valid")
    fully_resting = np.flatnonzero(rest_count == width)
    if fully_resting.size:
        return int(fully_resting[np.argmax(score[fully_resting])])
    best_coverage = np.flatnonzero(rest_count == np.max(rest_count))
    return int(best_coverage[np.argmax(score[best_coverage])])


def peak_summary(
    prediction: np.ndarray,
    target: np.ndarray,
    radius: int,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    peaks, _ = signal.find_peaks(target, height=0.35, prominence=0.25, distance=8)
    peaks = peaks[(peaks >= radius) & (peaks + radius < target.size)]
    if not peaks.size:
        empty = np.empty(0, dtype=np.float32)
        return {"peak_count": 0}, empty, empty
    target_average = np.stack(
        [target[peak - radius : peak + radius + 1] for peak in peaks]
    ).mean(axis=0)
    prediction_average = np.stack(
        [prediction[peak - radius : peak + radius + 1] for peak in peaks]
    ).mean(axis=0)
    correlations = np.correlate(
        prediction_average - prediction_average.mean(),
        target_average - target_average.mean(),
        mode="full",
    )
    return (
        {
            "peak_count": int(peaks.size),
            "aligned_waveform_pcc": pearson(prediction_average, target_average),
            "prediction_to_target_peak_ratio": float(
                prediction_average.max() / max(target_average.max(), 1.0e-12)
            ),
            "best_lag_bins": int(np.argmax(correlations) - (target_average.size - 1)),
        },
        prediction_average,
        target_average,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--window-bins", type=int, default=200)
    parser.add_argument("--peak-radius", type=int, default=20)
    parser.add_argument("--rate", type=float, default=25.0)
    args = parser.parse_args()

    initialized = np.load(args.root / "initialized_oof.npy")
    selected = np.load(args.root / "selected_oof.npy")
    target = np.load(args.root / "target_oof.npy")
    observed = np.isfinite(initialized) & np.isfinite(selected) & np.isfinite(target)
    initialized = initialized[observed]
    selected = selected[observed]
    target = target[observed]
    if not target.size:
        raise RuntimeError("no finite OOF observations")

    moving = target >= args.threshold
    initialized_peak, _, _ = peak_summary(initialized, target, args.peak_radius)
    selected_peak, selected_average, target_average = peak_summary(
        selected, target, args.peak_radius
    )
    report = {
        "data_scope": "merged event-grouped training-data OOF predictions only",
        "observations": int(target.size),
        "initialized": {
            "cleaned_pcc": pearson(initialized, target),
            "velocity_pcc": pearson(np.diff(initialized), np.diff(target)),
            "rest_rms": float(np.sqrt(np.mean(initialized[~moving] ** 2))),
            "movement_rmse": float(np.sqrt(np.mean((initialized[moving] - target[moving]) ** 2))),
            "state_f1": state_f1(initialized, target, args.threshold),
            "peak_triggered": initialized_peak,
        },
        "selected": {
            "cleaned_pcc": pearson(selected, target),
            "velocity_pcc": pearson(np.diff(selected), np.diff(target)),
            "rest_rms": float(np.sqrt(np.mean(selected[~moving] ** 2))),
            "movement_rmse": float(np.sqrt(np.mean((selected[moving] - target[moving]) ** 2))),
            "state_f1": state_f1(selected, target, args.threshold),
            "peak_triggered": selected_peak,
        },
    }

    width = min(args.window_bins, target.size)
    starts = strongest_windows(target, width, count=2)
    starts.append(worst_rest_window(selected, target, width, args.threshold))
    figure, axes = plt.subplots(4, 1, figsize=(13, 9), constrained_layout=True)
    for index, (axis, start) in enumerate(zip(axes[:3], starts, strict=True)):
        stop = start + width
        time = np.arange(width) / args.rate + start / args.rate
        axis.plot(time, target[start:stop], color="black", linewidth=2.2, label="cleaned target")
        axis.plot(time, initialized[start:stop], color="#2563eb", linewidth=1.3, label="LARS initialization")
        axis.plot(time, selected[start:stop], color="#dc2626", linewidth=1.4, label="residual LSTM")
        axis.axhline(args.threshold, color="#777", linestyle="--", linewidth=0.8)
        axis.set_title("strong movement window" if index < 2 else "worst all-rest window")
        axis.set_ylabel("flexion")
        axis.grid(alpha=0.2)
    axes[2].set_xlabel("OOF time after history offset (s)")

    if target_average.size:
        peak_time = np.arange(-args.peak_radius, args.peak_radius + 1) / args.rate
        axes[3].plot(peak_time, target_average, color="black", linewidth=2.2, label="cleaned target")
        axes[3].plot(peak_time, selected_average, color="#dc2626", linewidth=1.5, label="residual LSTM")
        axes[3].axvline(0.0, color="#777", linestyle="--", linewidth=0.8)
        axes[3].set_title(f"target-aligned mean of {selected_peak['peak_count']} movement peaks")
        axes[3].set_xlabel("time relative to target peak (s)")
        axes[3].set_ylabel("flexion")
        axes[3].grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)

    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output / "oof_trajectory_diagnostic.png", dpi=160)
    plt.close(figure)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
