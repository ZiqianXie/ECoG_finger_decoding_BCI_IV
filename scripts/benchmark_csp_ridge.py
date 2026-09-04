#!/usr/bin/env python3
"""Evaluate band-specific CSP-style spatial filters with ridge decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy import signal
import torch

from benchmark_ridge_target_variants import fit_candidate, lagged


BANDS_HZ = (
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 30.0),
    (30.0, 55.0),
    (65.0, 95.0),
    (105.0, 145.0),
    (155.0, 195.0),
)

BAND_PROFILES = {
    "standard": BANDS_HZ,
    # The paper notes that the local motor potential below 5 Hz is highly
    # informative. Keep separate narrow delta/LMP views rather than merging
    # them into the first 4-8 Hz rhythm band.
    "low_extended": (
        (0.5, 4.0),
        (0.5, 5.0),
        (2.0, 6.0),
        *BANDS_HZ,
    ),
}


def regularized_covariance(values: np.ndarray, shrinkage: float = 0.05) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    covariance = values.T @ values / max(1, values.shape[0] - 1)
    isotropic = np.trace(covariance) / covariance.shape[0]
    return (1.0 - shrinkage) * covariance + shrinkage * isotropic * np.eye(covariance.shape[0])


def csp_filters(
    filtered_fit: np.ndarray,
    target_fit: np.ndarray,
    samples_per_bin: int,
    components_per_tail: int,
    include_global_movement: bool = False,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    rest_bins = np.max(target_fit, axis=1) < 0.05
    rows: list[np.ndarray] = []
    audit: list[dict[str, object]] = []
    for finger in range(target_fit.shape[1]):
        active_bins = target_fit[:, finger] > 0.20
        active = np.repeat(active_bins, samples_per_bin)
        rest = np.repeat(rest_bins, samples_per_bin)
        active_cov = regularized_covariance(filtered_fit[active])
        rest_cov = regularized_covariance(filtered_fit[rest])
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            active_cov,
            active_cov + rest_cov,
            check_finite=False,
        )
        order = np.r_[
            np.arange(components_per_tail),
            np.arange(eigenvalues.size - components_per_tail, eigenvalues.size),
        ]
        selected = eigenvectors[:, order].T
        selected /= np.linalg.norm(selected, axis=1, keepdims=True).clip(min=1e-12)
        rows.append(selected)
        audit.append(
            {
                "finger": finger,
                "active_bins": int(active_bins.sum()),
                "rest_bins": int(rest_bins.sum()),
                "selected_generalized_eigenvalues": eigenvalues[order].tolist(),
            }
        )
    if include_global_movement:
        # Shared onset/offset features: discriminate any intended movement from
        # true rest, leaving finger identity to the five finger-specific CSPs.
        active_bins = np.max(target_fit, axis=1) > 0.20
        active = np.repeat(active_bins, samples_per_bin)
        rest = np.repeat(rest_bins, samples_per_bin)
        active_cov = regularized_covariance(filtered_fit[active])
        rest_cov = regularized_covariance(filtered_fit[rest])
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            active_cov,
            active_cov + rest_cov,
            check_finite=False,
        )
        order = np.r_[
            np.arange(components_per_tail),
            np.arange(eigenvalues.size - components_per_tail, eigenvalues.size),
        ]
        selected = eigenvectors[:, order].T
        selected /= np.linalg.norm(selected, axis=1, keepdims=True).clip(min=1e-12)
        rows.append(selected)
        audit.append(
            {
                "finger": "any_movement",
                "active_bins": int(active_bins.sum()),
                "rest_bins": int(rest_bins.sum()),
                "selected_generalized_eigenvalues": eigenvalues[order].tolist(),
            }
        )
    return np.concatenate(rows, axis=0).astype(np.float32), audit


def binned_log_energy(
    filtered: np.ndarray,
    spatial_filters: np.ndarray,
    samples_per_bin: int,
) -> np.ndarray:
    projected = filtered @ spatial_filters.T
    usable = projected[: projected.shape[0] // samples_per_bin * samples_per_bin]
    reshaped = usable.reshape(-1, samples_per_bin, projected.shape[1])
    return np.log1p(np.sqrt(np.sum(reshaped * reshaped, axis=1))).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/csp_ridge_v2"))
    parser.add_argument("--target", default="local_w4_q20")
    parser.add_argument("--sampling-rate", type=float, default=1000.0)
    parser.add_argument("--band-profile", choices=tuple(BAND_PROFILES), default="standard")
    parser.add_argument("--samples-per-bin", type=int, default=40)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--components-per-tail", type=int, default=2)
    parser.add_argument("--include-global-movement-csp", action="store_true")
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = args.prepared_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    train_ecog = np.load(root / "train_ecog.npy", mmap_mode="r")
    test_ecog = np.load(root / "test_ecog.npy", mmap_mode="r")
    train_target = np.load(root / f"train_glove_{args.target}.npy")
    train_raw = np.load(root / "train_glove_25hz_raw.npy")
    test_raw = np.load(root / "test_glove_25hz_raw.npy")

    train_features: list[np.ndarray] = []
    test_features: list[np.ndarray] = []
    filter_audit: list[dict[str, object]] = []
    fit_samples = split * args.samples_per_bin
    bands_hz = BAND_PROFILES[args.band_profile]
    for low, high in bands_hz:
        sos = signal.butter(
            4,
            (low, high),
            btype="bandpass",
            fs=args.sampling_rate,
            output="sos",
        )
        joined = np.concatenate((np.asarray(train_ecog), np.asarray(test_ecog)), axis=0)
        filtered = signal.sosfiltfilt(sos, joined, axis=0).astype(np.float32)
        filtered_train = filtered[: train_ecog.shape[0]]
        filtered_test = filtered[train_ecog.shape[0] :]
        spatial, audit = csp_filters(
            filtered_train[:fit_samples],
            train_target[:split],
            args.samples_per_bin,
            args.components_per_tail,
            args.include_global_movement_csp,
        )
        train_features.append(binned_log_energy(filtered_train, spatial, args.samples_per_bin))
        test_features.append(binned_log_energy(filtered_test, spatial, args.samples_per_bin))
        filter_audit.append({"band_hz": [low, high], "filters": audit})
        np.save(output / f"csp_{low:g}_{high:g}hz.npy", spatial, allow_pickle=False)
        print(f"band={low:g}-{high:g}Hz features={spatial.shape[0]}", flush=True)

    train_energy = np.concatenate(train_features, axis=1)
    test_energy = np.concatenate(test_features, axis=1)
    np.save(output / "train_csp_energy.npy", train_energy, allow_pickle=False)
    np.save(output / "test_csp_energy.npy", test_energy, allow_pickle=False)
    train_x_all = lagged(train_energy, args.history)
    test_x = lagged(test_energy, args.history)
    offset = args.history - 1
    train_count = split - offset
    result, validation_prediction, test_prediction = fit_candidate(
        train_x_all[:train_count],
        train_target[offset:split],
        train_x_all[train_count:],
        train_target[split:],
        train_raw[split:],
        test_x,
        test_raw[offset:],
        args.top_features,
        torch.device(args.device),
    )
    report = {
        "subject": args.subject,
        "method": "finger-specific plus optional shared movement-vs-rest CSP, log energy, screened GPU ridge",
        "target": args.target,
        "band_profile": args.band_profile,
        "bands_hz": bands_hz,
        "components_per_tail": args.components_per_tail,
        "include_global_movement_csp": args.include_global_movement_csp,
        "feature_count_per_bin": int(train_energy.shape[1]),
        "filter_audit": filter_audit,
        "result": result,
    }
    np.save(output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
    np.save(output / "test_prediction.npy", test_prediction, allow_pickle=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result["validation_raw_metrics"]), flush=True)
    print(json.dumps(result["test_raw_metrics"]), flush=True)


if __name__ == "__main__":
    main()
