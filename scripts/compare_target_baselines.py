#!/usr/bin/env python3
"""Generate and audit fine-grained glove-baseline alternatives.

The comparison is intentionally target-only.  It exports each candidate under
an explicit name and never overwrites the paper target, so downstream decoder
experiments can select a target without losing provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from ecog_decoding.preprocessing import local_baseline_correct


FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


def normalized_target(
    raw: np.ndarray,
    baseline: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return np.clip((raw - baseline) / scale, 0.0, 2.0).astype(np.float32)


def target_metrics(target: np.ndarray, sampling_rate_hz: float) -> dict[str, object]:
    target = np.asarray(target, dtype=np.float64)
    frequencies, power = signal.periodogram(target, fs=sampling_rate_hz, axis=0)
    total_power = np.maximum(power.sum(axis=0), 1e-12)
    very_slow = power[frequencies < 0.15].sum(axis=0) / total_power
    peaks = [
        int(
            signal.find_peaks(
                target[:, finger],
                height=0.15,
                distance=max(1, int(round(0.20 * sampling_rate_hz))),
            )[0].size
        )
        for finger in range(target.shape[1])
    ]
    return {
        "active_fraction_gt_0p10": np.mean(target > 0.10, axis=0).tolist(),
        "near_zero_fraction_lt_0p02": np.mean(target < 0.02, axis=0).tolist(),
        "very_slow_power_fraction_lt_0p15hz": very_slow.tolist(),
        "peak_count": peaks,
        "q50": np.quantile(target, 0.50, axis=0).tolist(),
        "q95": np.quantile(target, 0.95, axis=0).tolist(),
    }


def plot_comparison(
    raw: np.ndarray,
    paper_baseline: np.ndarray,
    paper_target: np.ndarray,
    candidates: dict[str, tuple[np.ndarray, np.ndarray]],
    sampling_rate_hz: float,
    output_path: Path,
    reference_name: str,
) -> None:
    if reference_name not in candidates:
        raise ValueError(f"plot candidate {reference_name!r} was not generated")
    local_baseline, local_target = candidates[reference_name]
    disagreement = np.mean(np.abs(local_target - paper_target), axis=1)
    center = int(np.argmax(np.convolve(disagreement, np.ones(100), mode="same")))
    half_width = int(round(5.0 * sampling_rate_hz))
    start = max(0, center - half_width)
    stop = min(raw.shape[0], center + half_width)
    time = np.arange(start, stop) / sampling_rate_hz

    fig, axes = plt.subplots(5, 2, figsize=(15, 13), sharex="col")
    for finger, name in enumerate(FINGER_NAMES):
        left = axes[finger, 0]
        left.plot(time, raw[start:stop, finger], color="#334155", lw=1.1, label="raw glove")
        left.plot(time, paper_baseline[start:stop, finger], color="#d97706", lw=1.1, label="paper baseline")
        left.plot(time, local_baseline[start:stop, finger], color="#2563eb", lw=1.1, label=f"local {reference_name}")
        left.set_ylabel(name)
        left.grid(alpha=0.18)

        right = axes[finger, 1]
        right.plot(time, paper_target[start:stop, finger], color="#d97706", lw=1.2, label="paper residual")
        right.plot(time, local_target[start:stop, finger], color="#2563eb", lw=1.2, label="local residual")
        right.grid(alpha=0.18)
    axes[0, 0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0, 1].legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle("Largest paper-vs-local baseline disagreement window", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/target_baseline_audit"))
    parser.add_argument("--sampling-rate", type=float, default=25.0)
    parser.add_argument("--windows", type=float, nargs="+", default=[0.6, 1.0, 1.5, 2.5, 4.0])
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--smoothing-seconds", type=float, default=0.16)
    parser.add_argument("--plot-candidate", default="local_w1p5_q10")
    args = parser.parse_args()

    complete_report: dict[str, object] = {
        "selection_rule": "Choose target preprocessing only by downstream validation correlation; test labels are visualization-only.",
        "subjects": {},
    }
    for subject in args.subjects:
        root = args.prepared_root / f"sub{subject}"
        metadata = json.loads((root / "metadata.json").read_text())
        split = int(metadata["target_fit_samples_25hz"])
        train_raw = np.load(root / "train_glove_25hz_raw.npy")
        test_raw = np.load(root / "test_glove_25hz_raw.npy")
        paper_train = np.load(root / "train_glove_paper_baseline_only.npy")
        paper_test = np.load(root / "test_glove_paper_baseline_only.npy")
        paper_baseline = np.load(root / "test_glove_paper_baseline.npy")

        subject_report: dict[str, object] = {
            "paper": {
                "train_fit": target_metrics(paper_train[:split], args.sampling_rate),
                "validation": target_metrics(paper_train[split:], args.sampling_rate),
                "test": target_metrics(paper_test, args.sampling_rate),
            },
            "candidates": {},
        }
        plot_candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for window in args.windows:
            for quantile in args.quantiles:
                key = f"local_w{window:g}_q{int(round(100 * quantile)):02d}".replace(".", "p")
                train_result = local_baseline_correct(
                    train_raw,
                    sampling_rate_hz=args.sampling_rate,
                    window_seconds=window,
                    quantile=quantile,
                    smoothing_seconds=args.smoothing_seconds,
                )
                test_result = local_baseline_correct(
                    test_raw,
                    sampling_rate_hz=args.sampling_rate,
                    window_seconds=window,
                    quantile=quantile,
                    smoothing_seconds=args.smoothing_seconds,
                )
                scale = np.quantile(train_result.corrected[:split], 0.995, axis=0)
                scale_floor = max(float(np.max(scale)) * 0.01, np.finfo(np.float64).eps)
                scale = np.maximum(scale, scale_floor)
                train_target = normalized_target(train_raw, train_result.baseline, scale)
                test_target = normalized_target(test_raw, test_result.baseline, scale)
                np.save(root / f"train_glove_{key}.npy", train_target, allow_pickle=False)
                np.save(root / f"test_glove_{key}.npy", test_target, allow_pickle=False)
                np.save(root / f"train_glove_{key}_baseline.npy", train_result.baseline.astype(np.float32), allow_pickle=False)
                np.save(root / f"test_glove_{key}_baseline.npy", test_result.baseline.astype(np.float32), allow_pickle=False)
                subject_report["candidates"][key] = {
                    "window_seconds": window,
                    "quantile": quantile,
                    "smoothing_seconds": args.smoothing_seconds,
                    "scale": scale.tolist(),
                    "train_fit": target_metrics(train_target[:split], args.sampling_rate),
                    "validation": target_metrics(train_target[split:], args.sampling_rate),
                    "test": target_metrics(test_target, args.sampling_rate),
                }
                plot_candidates[key] = (test_result.baseline, test_target)

        subject_output = args.output_root / f"sub{subject}"
        subject_output.mkdir(parents=True, exist_ok=True)
        (subject_output / "metrics.json").write_text(json.dumps(subject_report, indent=2) + "\n")
        plot_comparison(
            test_raw,
            paper_baseline,
            paper_test,
            plot_candidates,
            args.sampling_rate,
            subject_output / "largest_disagreement.png",
            args.plot_candidate,
        )
        complete_report["subjects"][str(subject)] = subject_report
        print(f"subject={subject} wrote {len(plot_candidates)} candidates", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "metrics.json").write_text(json.dumps(complete_report, indent=2) + "\n")


if __name__ == "__main__":
    main()
