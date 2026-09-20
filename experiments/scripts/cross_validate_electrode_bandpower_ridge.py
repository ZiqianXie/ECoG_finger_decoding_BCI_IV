#!/usr/bin/env python3
"""Outer-split all-electrode band-power ridge decoder.

Every outer fold performs its own correlation screen and grouped inner-CV
ridge selection.  The released competition test set is never loaded.
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

from benchmark_csp_ridge import BANDS_HZ
from ecog_decoding.training import FINGER_NAMES
from gpu_ridge import fit_torch_ridge_cv
from single_wavelet_support import OFFSET


ALPHAS = np.logspace(-2, 5, 8, dtype=np.float32)


def rows_from_intervals(intervals: list[list[int]]) -> np.ndarray:
    return np.concatenate(
        [np.arange(start, stop, dtype=np.int64) for start, stop in intervals]
    )


def grouped_rows(intervals: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    rows = rows_from_intervals(intervals)
    groups = np.concatenate(
        [
            np.full(stop - start, group, dtype=np.int64)
            for group, (start, stop) in enumerate(intervals)
        ]
    )
    return rows, groups


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.corrcoef(np.asarray(first), np.asarray(second))[0, 1])


def binned_log_energy(filtered_bands: np.ndarray, samples_per_bin: int) -> np.ndarray:
    if filtered_bands.ndim != 3:
        raise ValueError("filtered bands must have shape (bands, samples, channels)")
    usable = filtered_bands.shape[1] // samples_per_bin * samples_per_bin
    result = []
    for band in range(filtered_bands.shape[0]):
        values = np.asarray(filtered_bands[band, :usable], dtype=np.float32)
        bins = values.reshape(-1, samples_per_bin, values.shape[1])
        result.append(np.log1p(np.sqrt(np.sum(bins * bins, axis=1))))
    return np.concatenate(result, axis=1).astype(np.float32)


def load_band_energy(
    prepared_root: Path,
    subject: int,
    samples_per_bin: int,
    cache: Path | None,
) -> np.ndarray:
    if cache is not None and cache.is_file():
        return np.load(cache, mmap_mode="r")
    subject_root = prepared_root / f"sub{subject}"
    filtered_path = subject_root / "train_filtered_bands.npy"
    if filtered_path.is_file():
        energy = binned_log_energy(
            np.load(filtered_path, mmap_mode="r"), samples_per_bin
        )
    else:
        ecog = np.load(subject_root / "train_ecog.npy", mmap_mode="r")
        usable = ecog.shape[0] // samples_per_bin * samples_per_bin
        parts = []
        for low, high in BANDS_HZ:
            sos = signal.butter(
                4, (low, high), btype="bandpass", fs=1000.0, output="sos"
            )
            filtered = signal.sosfiltfilt(sos, ecog, axis=0).astype(np.float32)
            bins = filtered[:usable].reshape(
                -1, samples_per_bin, filtered.shape[1]
            )
            parts.append(np.log1p(np.sqrt(np.sum(bins * bins, axis=1))))
        energy = np.concatenate(parts, axis=1).astype(np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(f".{cache.name}.{subject}.tmp.npy")
        np.save(temporary, energy, allow_pickle=False)
        temporary.replace(cache)
    return energy


def contextual_features(
    energy: np.ndarray,
    rows: np.ndarray,
    selected: np.ndarray,
    past_bins: int,
    future_bins: int,
) -> np.ndarray:
    base_features = energy.shape[1]
    lag_positions = selected // base_features
    feature_positions = selected % base_features
    lags = lag_positions - past_bins
    indices = np.clip(
        rows[:, None] + OFFSET + lags[None], 0, energy.shape[0] - 1
    )
    return np.asarray(energy[indices, feature_positions[None]], dtype=np.float32)


def screen_context_features(
    energy: np.ndarray,
    target: np.ndarray,
    training: np.ndarray,
    past_bins: int,
    future_bins: int,
    top_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    centered_target = target[training] - target[training].mean()
    target_norm = np.sqrt(np.sum(centered_target * centered_target))
    correlations = []
    for lag in range(-past_bins, future_bins + 1):
        indices = np.clip(training + OFFSET + lag, 0, energy.shape[0] - 1)
        values = np.asarray(energy[indices], dtype=np.float32)
        centered = values - values.mean(axis=0, keepdims=True)
        denominator = np.sqrt(np.sum(centered * centered, axis=0)) * target_norm
        correlations.append(
            np.divide(
                centered.T @ centered_target,
                denominator,
                out=np.zeros(values.shape[1], dtype=np.float32),
                where=denominator > 0,
            )
        )
    scores = np.concatenate(correlations)
    count = min(top_features, scores.size)
    selected = np.argpartition(np.abs(scores), -count)[-count:]
    selected = selected[np.argsort(np.abs(scores[selected]))[::-1]]
    return selected.astype(np.int64), scores[selected].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--finger", choices=tuple(FINGER_NAMES), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--band-energy-cache", type=Path, default=None)
    parser.add_argument("--samples-per-bin", type=int, default=40)
    parser.add_argument("--past-bins", type=int, default=24)
    parser.add_argument("--future-bins", type=int, default=0)
    parser.add_argument("--top-features", type=int, default=1024)
    parser.add_argument(
        "--ridge-selection-metric", choices=("mse", "pcc"), default="pcc"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixed-reference-pcc", type=float, required=True)
    parser.add_argument("--gain-threshold", type=float, default=0.095)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.past_bins < 0 or args.future_bins < 0 or args.top_features <= 0:
        parser.error("context widths must be nonnegative and top-features positive")
    finger_index = list(FINGER_NAMES).index(args.finger)
    energy = load_band_energy(
        args.prepared_root,
        args.subject,
        args.samples_per_bin,
        args.band_energy_cache,
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
    outer_records = []
    device = torch.device(args.device)
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
            energy,
            raw,
            training,
            args.past_bins,
            args.future_bins,
            args.top_features,
        )
        train_x = contextual_features(
            energy,
            training,
            selected,
            args.past_bins,
            args.future_bins,
        )
        splitter = GroupKFold(n_splits=min(5, len(split["training_intervals"])))
        inner_splits = list(splitter.split(training, raw[training], groups))
        model = fit_torch_ridge_cv(
            train_x,
            raw[training],
            inner_splits,
            alphas=ALPHAS,
            device=device,
            selection_metric=args.ridge_selection_metric,
        )
        validation_x = contextual_features(
            energy,
            validation,
            selected,
            args.past_bins,
            args.future_bins,
        )
        estimate = np.maximum(model.predict(validation_x), 0.0)
        prediction_oof[validation] = estimate
        record = {
            "outer_fold": outer_fold,
            "selected_alpha": model.alpha_,
            "outer_raw_pcc": pearson(estimate, raw[validation]),
            "mean_inner_mse": {
                str(float(alpha)): float(value)
                for alpha, value in zip(model.alphas_, model.mean_validation_mse_)
            },
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
            "outer-split all-electrode seven-band log-energy ridge; feature screen "
            "and grouped alpha selection refit inside each outer training scope; "
            "released test untouched"
        ),
        "released_test_touched": False,
        "outer_validation_used_for_selection": False,
        "subject": args.subject,
        "finger": args.finger,
        "past_bins": args.past_bins,
        "future_bins": args.future_bins,
        "top_features": args.top_features,
        "ridge_selection_metric": args.ridge_selection_metric,
        "outer_records": outer_records,
        "selected_oof_raw_pcc": pcc,
        "fixed_pre_ablation_reference_pcc": args.fixed_reference_pcc,
        "gain_over_fixed_reference": gain,
        "required_gain_over_fixed_reference": args.gain_threshold,
        "passes_fixed_reference_gain_threshold": gain >= args.gain_threshold,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "selected_oof.npy", prediction_oof, allow_pickle=False)
    np.save(args.output / "target_oof.npy", raw, allow_pickle=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
