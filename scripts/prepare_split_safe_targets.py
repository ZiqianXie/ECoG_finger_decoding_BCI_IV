#!/usr/bin/env python3
"""Generate local glove targets without crossing the final validation boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.preprocessing import local_baseline_correct


def normalize(corrected: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip(corrected / scale, 0.0, 2.0).astype(np.float32)


def split_safe_local_target(
    raw: np.ndarray,
    split: int,
    *,
    sampling_rate_hz: float,
    window_seconds: float,
    quantile: float,
    smoothing_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process fit and final-validation segments independently."""
    if not 2 <= split <= raw.shape[0] - 2:
        raise ValueError("split must leave at least two samples on both sides")
    fit = local_baseline_correct(
        raw[:split],
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=window_seconds,
        quantile=quantile,
        smoothing_seconds=smoothing_seconds,
    )
    validation = local_baseline_correct(
        raw[split:],
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=window_seconds,
        quantile=quantile,
        smoothing_seconds=smoothing_seconds,
    )
    scale = np.quantile(fit.corrected, 0.995, axis=0)
    floor = max(float(np.max(scale)) * 0.01, np.finfo(np.float64).eps)
    scale = np.maximum(scale, floor)
    target = np.concatenate(
        (normalize(fit.corrected, scale), normalize(validation.corrected, scale)),
        axis=0,
    )
    baseline = np.concatenate((fit.baseline, validation.baseline), axis=0)
    return target, baseline.astype(np.float32), scale


def method_name(window: float, quantile: float) -> str:
    window_label = f"{window:g}".replace(".", "p")
    return f"local_w{window_label}_q{int(round(100 * quantile)):02d}_split_safe"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument(
        "--windows", type=float, nargs="+", default=(1, 2, 3, 4, 5, 6)
    )
    parser.add_argument(
        "--quantiles", type=float, nargs="+", default=(0.10, 0.20, 0.30)
    )
    parser.add_argument("--sampling-rate", type=float, default=25.0)
    parser.add_argument("--smoothing-seconds", type=float, default=0.16)
    args = parser.parse_args()

    for subject in args.subjects:
        root = args.prepared_root / f"sub{subject}"
        metadata = json.loads((root / "metadata.json").read_text())
        split = int(metadata["target_fit_samples_25hz"])
        raw = np.load(root / "train_glove_25hz_raw.npy")
        records: dict[str, object] = {}
        for window in args.windows:
            for quantile in args.quantiles:
                name = method_name(window, quantile)
                target, baseline, scale = split_safe_local_target(
                    raw,
                    split,
                    sampling_rate_hz=args.sampling_rate,
                    window_seconds=window,
                    quantile=quantile,
                    smoothing_seconds=args.smoothing_seconds,
                )
                np.save(root / f"train_glove_{name}.npy", target, allow_pickle=False)
                np.save(
                    root / f"train_glove_{name}_baseline.npy",
                    baseline,
                    allow_pickle=False,
                )
                records[name] = {
                    "window_seconds": window,
                    "quantile": quantile,
                    "smoothing_seconds": args.smoothing_seconds,
                    "scale_fit_partition_only": scale.tolist(),
                }
        report = {
            "subject": subject,
            "fit_validation_boundary": split,
            "policy": (
                "fit and final-validation glove segments processed independently; "
                "scale estimated from fit segment only"
            ),
            "targets": records,
        }
        (root / "split_safe_target_metadata.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps({"subject": subject, "targets": len(records)}), flush=True)


if __name__ == "__main__":
    main()
