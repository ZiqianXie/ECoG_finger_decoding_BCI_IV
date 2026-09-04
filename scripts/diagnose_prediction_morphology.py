#!/usr/bin/env python3
"""Quantify and plot trajectory morphology beyond aggregate PCC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
    return float(xc @ yc / denominator) if denominator > 0 else 0.0


def parse_method(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("method must be NAME=OUTPUT_DIRECTORY")
    return name, Path(path)


def morphology(prediction: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, object]:
    report: dict[str, object] = {}
    per_finger: dict[str, object] = {}
    for finger, name in enumerate(FINGER_NAMES):
        truth = target[:, finger]
        estimate = prediction[:, finger]
        moving = truth >= threshold
        predicted_moving = estimate >= threshold
        tp = np.sum(moving & predicted_moving)
        fp = np.sum(~moving & predicted_moving)
        fn = np.sum(moving & ~predicted_moving)
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        peak_truth = float(np.quantile(truth[moving], 0.95)) if moving.any() else 0.0
        peak_prediction = float(np.quantile(estimate[moving], 0.95)) if moving.any() else 0.0
        per_finger[name] = {
            "pcc_cleaned": pearson(estimate, truth),
            "derivative_pcc": pearson(np.diff(estimate), np.diff(truth)),
            "rest_rms": float(np.sqrt(np.mean(estimate[~moving] ** 2))),
            "movement_mae": float(np.mean(np.abs(estimate[moving] - truth[moving]))),
            "false_positive_rate": float(fp / max(np.sum(~moving), 1)),
            "state_precision": precision,
            "state_recall": recall,
            "state_f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "movement_peak_ratio": peak_prediction / max(peak_truth, 1e-12),
        }
    report["per_finger"] = per_finger
    report["macro_rest_rms"] = float(np.mean([v["rest_rms"] for v in per_finger.values()]))
    report["macro_state_f1"] = float(np.mean([v["state_f1"] for v in per_finger.values()]))
    report["macro_derivative_pcc"] = float(np.mean([v["derivative_pcc"] for v in per_finger.values()]))
    return report


def peak_triggered_summary(
    prediction: np.ndarray, target: np.ndarray, radius: int = 20
) -> dict[str, object]:
    result: dict[str, object] = {}
    for finger, name in enumerate(FINGER_NAMES):
        peaks, _ = signal.find_peaks(
            target[:, finger], height=0.35, prominence=0.25, distance=8
        )
        peaks = peaks[(peaks >= radius) & (peaks + radius < target.shape[0])]
        if peaks.size == 0:
            result[name] = {"peak_count": 0}
            continue
        truth = np.stack([target[p - radius : p + radius + 1, finger] for p in peaks]).mean(0)
        estimate = np.stack(
            [prediction[p - radius : p + radius + 1, finger] for p in peaks]
        ).mean(0)
        correlations = np.correlate(
            estimate - estimate.mean(), truth - truth.mean(), mode="full"
        )
        lag = int(np.argmax(correlations) - (truth.size - 1))
        result[name] = {
            "peak_count": int(peaks.size),
            "aligned_waveform_pcc": pearson(estimate, truth),
            "peak_ratio": float(estimate.max() / max(truth.max(), 1e-12)),
            "best_lag_bins": lag,
        }
    return result


def best_window(signal: np.ndarray, width: int) -> int:
    score = np.convolve(signal, np.ones(width), mode="valid")
    return int(np.argmax(score))


def best_rest_window(signal: np.ndarray, rest: np.ndarray, width: int) -> int:
    """Choose a high-error window that is genuinely all-finger stationary."""
    kernel = np.ones(width)
    score = np.convolve(signal, kernel, mode="valid")
    rest_count = np.convolve(rest.astype(np.float32), kernel, mode="valid")
    valid = rest_count == width
    if valid.any():
        candidates = np.flatnonzero(valid)
        return int(candidates[np.argmax(score[candidates])])
    # Short recordings may not contain a full window of uninterrupted rest.
    # Prefer maximum stationary coverage, then the largest prediction error.
    candidates = np.flatnonzero(rest_count == rest_count.max())
    return int(candidates[np.argmax(score[candidates])])


def plot_movement_windows(
    target: np.ndarray,
    methods: dict[str, np.ndarray],
    output: Path,
    width: int,
    rate: float,
) -> None:
    colors = ["#1473e6", "#db5f57", "#2a9d70", "#8e62c6", "#d68c15"]
    figure, axes = plt.subplots(5, 1, figsize=(14, 11), constrained_layout=True)
    for finger, (axis, finger_name) in enumerate(zip(axes, FINGER_NAMES, strict=True)):
        dominance = np.maximum(target[:, finger] - np.max(np.delete(target, finger, axis=1), axis=1), 0)
        start = best_window(dominance, width)
        stop = start + width
        time = np.arange(width) / rate + start / rate
        axis.plot(time, target[start:stop, finger], color="black", linewidth=2.4, label="cleaned glove target")
        for color, (name, prediction) in zip(colors, methods.items()):
            axis.plot(time, prediction[start:stop, finger], color=color, linewidth=1.35, alpha=0.9, label=name)
        axis.set_title(f"{finger_name}: strongest relatively isolated movement window")
        axis.set_ylabel("flexion")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("test time after history offset (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_false_positive_windows(
    target: np.ndarray,
    methods: dict[str, np.ndarray],
    output: Path,
    width: int,
    rate: float,
    threshold: float,
) -> None:
    colors = ["#1473e6", "#db5f57", "#2a9d70", "#8e62c6", "#d68c15"]
    global_rest = np.max(target, axis=1) < threshold
    figure, axes = plt.subplots(5, 1, figsize=(14, 11), constrained_layout=True)
    for finger, (axis, finger_name) in enumerate(zip(axes, FINGER_NAMES, strict=True)):
        score = np.zeros(target.shape[0])
        for prediction in methods.values():
            score += np.abs(prediction[:, finger])
        score *= global_rest
        start = best_rest_window(score, global_rest, width)
        stop = start + width
        rest_fraction = float(global_rest[start:stop].mean())
        time = np.arange(width) / rate + start / rate
        axis.plot(time, target[start:stop, finger], color="black", linewidth=2.4, label="cleaned glove target")
        for color, (name, prediction) in zip(colors, methods.items()):
            axis.plot(time, prediction[start:stop, finger], color=color, linewidth=1.35, alpha=0.9, label=name)
        axis.axhline(threshold, color="#777", linestyle="--", linewidth=0.8)
        axis.set_title(
            f"{finger_name}: worst mostly-rest window ({rest_fraction:.0%} all-finger rest)"
        )
        axis.set_ylabel("flexion")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("test time after history offset (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_peak_triggered(
    target: np.ndarray,
    methods: dict[str, np.ndarray],
    output: Path,
    radius: int,
    rate: float,
) -> None:
    colors = ["#1473e6", "#db5f57", "#2a9d70", "#8e62c6", "#d68c15"]
    figure, axes = plt.subplots(5, 1, figsize=(12, 11), constrained_layout=True)
    time = np.arange(-radius, radius + 1) / rate
    for finger, (axis, finger_name) in enumerate(zip(axes, FINGER_NAMES, strict=True)):
        peaks, _ = signal.find_peaks(
            target[:, finger], height=0.35, prominence=0.25, distance=8
        )
        peaks = peaks[(peaks >= radius) & (peaks + radius < target.shape[0])]
        truth = np.stack(
            [target[p - radius : p + radius + 1, finger] for p in peaks]
        ).mean(0)
        axis.plot(time, truth, color="black", linewidth=2.4, label="cleaned glove target")
        for color, (name, prediction) in zip(colors, methods.items()):
            estimate = np.stack(
                [prediction[p - radius : p + radius + 1, finger] for p in peaks]
            ).mean(0)
            axis.plot(time, estimate, color=color, linewidth=1.4, label=name)
        axis.axvline(0, color="#777", linestyle="--", linewidth=0.8)
        axis.set_title(f"{finger_name}: mean of {peaks.size} target-aligned flexion peaks")
        axis.set_ylabel("flexion")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("time relative to target peak (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--target", default="local_w4_q20")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction_morphology_v2/sub3"))
    args = parser.parse_args()

    root = args.prepared_root / f"sub{args.subject}"
    offset = args.history - 1
    cleaned_target = np.load(root / f"test_glove_{args.target}.npy")[offset:]
    raw_target = np.load(root / "test_glove_25hz_raw.npy")[offset:]
    methods = {
        name: np.load(path if path.suffix == ".npy" else path / "test_prediction.npy")
        for name, path in args.method
    }
    for name, prediction in methods.items():
        if prediction.shape != cleaned_target.shape:
            raise ValueError(f"{name} has shape {prediction.shape}, expected {cleaned_target.shape}")

    report = {
        "subject": args.subject,
        "visual_target": args.target,
        "movement_threshold": args.threshold,
        "methods": {
            name: {
                "raw_metrics": trajectory_metrics(prediction, raw_target),
                "cleaned_metrics": trajectory_metrics(prediction, cleaned_target),
                "morphology": morphology(prediction, cleaned_target, args.threshold),
                "peak_triggered": peak_triggered_summary(prediction, cleaned_target),
            }
            for name, prediction in methods.items()
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_movement_windows(cleaned_target, methods, args.output / "movement_windows.png", 250, 25.0)
    plot_false_positive_windows(
        cleaned_target,
        methods,
        args.output / "rest_false_positives.png",
        200,
        25.0,
        args.threshold,
    )
    plot_peak_triggered(cleaned_target, methods, args.output / "peak_triggered.png", 20, 25.0)
    print(args.output)


if __name__ == "__main__":
    main()
