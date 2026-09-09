#!/usr/bin/env python3
"""Add within-bout flexion-phase CSP filters to movement/rest CSP features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy import signal
import torch

from benchmark_csp_ridge import BANDS_HZ, binned_log_energy, csp_filters, regularized_covariance
from benchmark_ridge_target_variants import fit_candidate, lagged


def phase_csp_filters(
    filtered_fit: np.ndarray,
    target_fit: np.ndarray,
    samples_per_bin: int,
    components_per_tail: int,
    context_bins: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    rows: list[np.ndarray] = []
    audit: list[dict[str, object]] = []
    kernel = np.ones(context_bins, dtype=np.float32)
    for finger in range(target_fit.shape[1]):
        trajectory = target_fit[:, finger]
        active = trajectory >= 0.1
        context = np.convolve(active.astype(np.float32), kernel, mode="same") > 0
        context_values = trajectory[context]
        low_cut, high_cut = np.quantile(context_values, (0.30, 0.70))
        low_bins = context & (trajectory <= low_cut)
        high_bins = context & (trajectory >= high_cut)
        low = np.repeat(low_bins, samples_per_bin)
        high = np.repeat(high_bins, samples_per_bin)
        low_cov = regularized_covariance(filtered_fit[low])
        high_cov = regularized_covariance(filtered_fit[high])
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            high_cov,
            high_cov + low_cov,
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
                "low_bins": int(low_bins.sum()),
                "high_bins": int(high_bins.sum()),
                "low_cut": float(low_cut),
                "high_cut": float(high_cut),
                "selected_generalized_eigenvalues": eigenvalues[order].tolist(),
            }
        )
    return np.concatenate(rows, axis=0).astype(np.float32), audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase_csp_ridge_v2"))
    parser.add_argument("--target", default="local_w4_q20")
    parser.add_argument("--sampling-rate", type=float, default=1000.0)
    parser.add_argument("--samples-per-bin", type=int, default=40)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--components-per-tail", type=int, default=2)
    parser.add_argument("--context-bins", type=int, default=25)
    parser.add_argument("--top-features", type=int, default=768)
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
    joined = np.concatenate((np.asarray(train_ecog), np.asarray(test_ecog)), axis=0)
    fit_samples = split * args.samples_per_bin
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    audits: list[dict[str, object]] = []
    for low, high in BANDS_HZ:
        sos = signal.butter(4, (low, high), btype="bandpass", fs=args.sampling_rate, output="sos")
        filtered = signal.sosfiltfilt(sos, joined, axis=0).astype(np.float32)
        filtered_train = filtered[: train_ecog.shape[0]]
        filtered_test = filtered[train_ecog.shape[0] :]
        state_filters, state_audit = csp_filters(
            filtered_train[:fit_samples], train_target[:split], args.samples_per_bin, args.components_per_tail
        )
        phase_filters, phase_audit = phase_csp_filters(
            filtered_train[:fit_samples],
            train_target[:split],
            args.samples_per_bin,
            args.components_per_tail,
            args.context_bins,
        )
        spatial = np.concatenate((state_filters, phase_filters), axis=0)
        train_parts.append(binned_log_energy(filtered_train, spatial, args.samples_per_bin))
        test_parts.append(binned_log_energy(filtered_test, spatial, args.samples_per_bin))
        np.save(output / f"csp_{low:g}_{high:g}hz.npy", spatial, allow_pickle=False)
        audits.append({"band_hz": [low, high], "state": state_audit, "phase": phase_audit})
        print(f"band={low:g}-{high:g}Hz filters={spatial.shape[0]}", flush=True)

    train_energy = np.concatenate(train_parts, axis=1)
    test_energy = np.concatenate(test_parts, axis=1)
    np.save(output / "train_csp_energy.npy", train_energy, allow_pickle=False)
    np.save(output / "test_csp_energy.npy", test_energy, allow_pickle=False)
    offset = args.history - 1
    train_x_all = lagged(train_energy, args.history)
    test_x = lagged(test_energy, args.history)
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
        "method": "movement-rest plus within-bout flexion-phase CSP, seven-band log energy, GPU ridge",
        "target": args.target,
        "feature_count_per_bin": int(train_energy.shape[1]),
        "history": args.history,
        "filter_audit": audits,
        "result": result,
    }
    np.save(output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
    np.save(output / "test_prediction.npy", test_prediction, allow_pickle=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result["validation_raw_metrics"]), flush=True)
    print(json.dumps(result["test_raw_metrics"]), flush=True)


if __name__ == "__main__":
    main()
