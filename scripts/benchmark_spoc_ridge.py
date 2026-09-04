#!/usr/bin/env python3
"""Decode with continuous-target SPoC spatial filters and fixed band energy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy import signal
import torch

from benchmark_csp_ridge import BAND_PROFILES, binned_log_energy, regularized_covariance
from benchmark_ridge_target_variants import fit_candidate, lagged
from ecog_decoding.training import FINGER_NAMES


def spoc_filters(
    filtered_fit: np.ndarray,
    target_fit: np.ndarray,
    samples_per_bin: int,
    components_per_finger: int,
    shrinkage: float = 0.05,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    bin_count = target_fit.shape[0]
    values = np.asarray(filtered_fit[: bin_count * samples_per_bin], dtype=np.float64)
    values = values.reshape(bin_count, samples_per_bin, values.shape[1])
    # Normalize each short-window covariance by total power, as in SPoC, so
    # isolated high-amplitude windows do not dominate the generalized problem.
    trace = np.sum(values * values, axis=(1, 2)).clip(min=1.0e-10)
    normalized = values / np.sqrt(trace[:, None, None])
    flat = normalized.reshape(-1, normalized.shape[-1])
    mean_cov = regularized_covariance(flat, shrinkage=shrinkage)
    rows: list[np.ndarray] = []
    audit: list[dict[str, object]] = []
    for finger, name in enumerate(FINGER_NAMES):
        weight = target_fit[:, finger].astype(np.float64)
        weight = (weight - weight.mean()) / max(weight.std(), 1.0e-8)
        weighted = normalized * weight[:, None, None]
        target_cov = np.einsum("btc,btd->cd", normalized, weighted, optimize=True)
        target_cov /= bin_count * samples_per_bin
        target_cov = 0.5 * (target_cov + target_cov.T)
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            target_cov,
            mean_cov,
            check_finite=False,
        )
        order = np.argsort(np.abs(eigenvalues))[::-1][:components_per_finger]
        selected = eigenvectors[:, order].T
        selected /= np.linalg.norm(selected, axis=1, keepdims=True).clip(min=1.0e-12)
        rows.append(selected)
        audit.append(
            {
                "finger": name,
                "selected_generalized_eigenvalues": eigenvalues[order].tolist(),
            }
        )
    return np.concatenate(rows, axis=0).astype(np.float32), audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/spoc_ridge_v1"))
    parser.add_argument("--target", required=True)
    parser.add_argument("--sampling-rate", type=float, default=1000.0)
    parser.add_argument("--samples-per-bin", type=int, default=40)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--band-profile", choices=tuple(BAND_PROFILES), default="standard")
    parser.add_argument("--components-per-finger", type=int, default=4)
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    prepared = args.prepared_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((prepared / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    train_ecog = np.load(prepared / "train_ecog.npy", mmap_mode="r")
    test_ecog = np.load(prepared / "test_ecog.npy", mmap_mode="r")
    target = np.load(prepared / f"train_glove_{args.target}.npy")
    raw_train = np.load(prepared / "train_glove_25hz_raw.npy")
    raw_test = np.load(prepared / "test_glove_25hz_raw.npy")
    joined = np.concatenate((np.asarray(train_ecog), np.asarray(test_ecog)), axis=0)
    train_features = []
    test_features = []
    audit = []
    fit_samples = split * args.samples_per_bin
    for low, high in BAND_PROFILES[args.band_profile]:
        sos = signal.butter(4, (low, high), btype="bandpass", fs=args.sampling_rate, output="sos")
        filtered = signal.sosfiltfilt(sos, joined, axis=0).astype(np.float32)
        filtered_train = filtered[: train_ecog.shape[0]]
        filtered_test = filtered[train_ecog.shape[0] :]
        spatial, detail = spoc_filters(
            filtered_train[:fit_samples],
            target[:split],
            args.samples_per_bin,
            args.components_per_finger,
        )
        train_features.append(binned_log_energy(filtered_train, spatial, args.samples_per_bin))
        test_features.append(binned_log_energy(filtered_test, spatial, args.samples_per_bin))
        audit.append({"band_hz": [low, high], "filters": detail})
        np.save(output / f"spoc_{low:g}_{high:g}hz.npy", spatial, allow_pickle=False)
        print(f"band={low:g}-{high:g}Hz filters={spatial.shape[0]}", flush=True)
    train_energy = np.concatenate(train_features, axis=1)
    test_energy = np.concatenate(test_features, axis=1)
    np.save(output / "train_spoc_energy.npy", train_energy, allow_pickle=False)
    np.save(output / "test_spoc_energy.npy", test_energy, allow_pickle=False)
    offset = args.history - 1
    train_x_all = lagged(train_energy, args.history)
    test_x = lagged(test_energy, args.history)
    train_count = split - offset
    result, validation_prediction, test_prediction = fit_candidate(
        train_x_all[:train_count],
        target[offset:split],
        train_x_all[train_count:],
        target[split:],
        raw_train[split:],
        test_x,
        raw_test[offset:],
        args.top_features,
        torch.device(args.device),
    )
    np.save(output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
    np.save(output / "test_prediction.npy", test_prediction, allow_pickle=False)
    report = {
        "subject": args.subject,
        "method": "continuous-target SPoC spatial filters, fixed band energy, screened ridge",
        "target": args.target,
        "band_profile": args.band_profile,
        "components_per_finger": args.components_per_finger,
        "filter_audit": audit,
        "result": result,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"validation": result["validation_raw_metrics"], "test": result["test_raw_metrics"]}), flush=True)


if __name__ == "__main__":
    main()
