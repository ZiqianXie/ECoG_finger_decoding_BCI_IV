#!/usr/bin/env python3
"""Outer-split signed low-frequency-potential ridge decoder.

Unlike the established log-energy routes, this preserves signal polarity and
therefore low-frequency phase. Feature screening and ridge selection are
repeated inside every outer-training split; released test labels are untouched.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import signal
import torch
from sklearn.model_selection import GroupKFold

from cross_validate_electrode_bandpower_ridge import (
    ALPHAS,
    contextual_features,
    grouped_rows,
    pearson,
    rows_from_intervals,
    screen_context_features,
)
from ecog_decoding.training import FINGER_NAMES
from gpu_ridge import fit_torch_ridge_cv
from single_wavelet_support import OFFSET


LMP_BANDS_HZ = ((0.5, 4.0), (4.0, 8.0), (8.0, 15.0))


def binned_signed_mean(values: np.ndarray, samples_per_bin: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("values must have shape (samples, channels)")
    usable = values.shape[0] // samples_per_bin * samples_per_bin
    return values[:usable].reshape(
        -1, samples_per_bin, values.shape[1]
    ).mean(axis=1, dtype=np.float64).astype(np.float32)


def load_signed_lmp(
    prepared_root: Path,
    subject: int,
    samples_per_bin: int,
    cache: Path | None,
) -> np.ndarray:
    if cache is not None and cache.is_file():
        return np.load(cache, mmap_mode="r")
    ecog = np.load(
        prepared_root / f"sub{subject}" / "train_ecog.npy", mmap_mode="r"
    )
    parts = []
    for low, high in LMP_BANDS_HZ:
        sos = signal.butter(
            4, (low, high), btype="bandpass", fs=1000.0, output="sos"
        )
        filtered = signal.sosfiltfilt(sos, ecog, axis=0).astype(np.float32)
        parts.append(binned_signed_mean(filtered, samples_per_bin))
    lmp = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(f".{cache.name}.{subject}.tmp.npy")
        np.save(temporary, lmp, allow_pickle=False)
        temporary.replace(cache)
    return lmp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--finger", choices=FINGER_NAMES, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--lmp-cache", type=Path, default=None)
    parser.add_argument("--samples-per-bin", type=int, default=40)
    parser.add_argument("--past-bins", type=int, default=24)
    parser.add_argument("--future-bins", type=int, default=16)
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument(
        "--ridge-selection-metric", choices=("mse", "pcc"), default="pcc"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixed-reference-pcc", type=float, required=True)
    parser.add_argument("--gain-threshold", type=float, default=0.095)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.samples_per_bin, args.top_features) <= 0:
        parser.error("samples-per-bin and top-features must be positive")
    if args.past_bins < 0 or args.future_bins < 0:
        parser.error("context widths must be nonnegative")

    finger_index = list(FINGER_NAMES).index(args.finger)
    lmp = load_signed_lmp(
        args.prepared_root, args.subject, args.samples_per_bin, args.lmp_cache
    )
    rows = int(
        np.load(
            args.cache_root / "cache" / "outer0" / "outer" / "target.npy",
            mmap_mode="r",
        ).shape[0]
    )
    raw = np.asarray(
        np.load(
            args.prepared_root
            / f"sub{args.subject}"
            / "train_glove_25hz_raw.npy",
            mmap_mode="r",
        )[OFFSET : OFFSET + rows, finger_index],
        dtype=np.float32,
    )
    prediction_oof = np.full(rows, np.nan, dtype=np.float32)
    device = torch.device(args.device)
    outer_records = []
    for outer_fold in range(3):
        started = time.perf_counter()
        split = json.loads(
            (
                args.cache_root
                / "cache"
                / f"outer{outer_fold}"
                / "outer"
                / "split.json"
            ).read_text()
        )
        training, groups = grouped_rows(split["training_intervals"])
        validation = rows_from_intervals(split["validation_intervals"])
        selected, screen_pcc = screen_context_features(
            lmp,
            raw,
            training,
            args.past_bins,
            args.future_bins,
            args.top_features,
        )
        training_features = contextual_features(
            lmp,
            training,
            selected,
            args.past_bins,
            args.future_bins,
        )
        splitter = GroupKFold(n_splits=min(5, len(split["training_intervals"])))
        inner_splits = list(splitter.split(training, raw[training], groups))
        model = fit_torch_ridge_cv(
            training_features,
            raw[training],
            inner_splits,
            alphas=ALPHAS,
            device=device,
            selection_metric=args.ridge_selection_metric,
        )
        validation_features = contextual_features(
            lmp,
            validation,
            selected,
            args.past_bins,
            args.future_bins,
        )
        estimate = np.maximum(model.predict(validation_features), 0.0)
        prediction_oof[validation] = estimate
        record = {
            "outer_fold": outer_fold,
            "selected_alpha": model.alpha_,
            "outer_raw_pcc": pearson(estimate, raw[validation]),
            "mean_inner_pcc": {
                str(float(alpha)): float(value)
                for alpha, value in zip(model.alphas_, model.mean_validation_pcc_)
            },
            "screen_abs_pcc_min": float(np.min(np.abs(screen_pcc))),
            "screen_abs_pcc_max": float(np.max(np.abs(screen_pcc))),
            "runtime_seconds": time.perf_counter() - started,
        }
        outer_records.append(record)
        print(json.dumps(record), flush=True)

    observed = np.isfinite(prediction_oof) & np.isfinite(raw)
    pcc = pearson(prediction_oof[observed], raw[observed])
    gain = pcc - args.fixed_reference_pcc
    report = {
        "protocol": (
            "outer-split signed 0.5-15 Hz low-frequency-potential ridge; lag-feature "
            "screen and grouped alpha selection refit inside each outer training "
            "scope; released test untouched"
        ),
        "released_test_touched": False,
        "outer_validation_used_for_selection": False,
        "subject": args.subject,
        "finger": args.finger,
        "bands_hz": [list(band) for band in LMP_BANDS_HZ],
        "past_bins": args.past_bins,
        "future_bins": args.future_bins,
        "top_features": args.top_features,
        "ridge_selection_metric": args.ridge_selection_metric,
        "outer_records": outer_records,
        "selected_oof_raw_pcc": pcc,
        "fixed_pre_ablation_reference_pcc": args.fixed_reference_pcc,
        "gain_over_fixed_reference": gain,
        "required_gain_over_fixed_reference": args.gain_threshold,
        "passes_working_threshold": gain >= args.gain_threshold,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "prediction_oof.npy", prediction_oof, allow_pickle=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
