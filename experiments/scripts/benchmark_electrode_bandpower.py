#!/usr/bin/env python3
"""Decode trajectories from per-electrode band power without spatial compression.

This diagnostic keeps every electrode in every frequency band.  It tests
whether movement-cycle information is lost by the movement-vs-rest CSP
projection before a more complicated temporal model is considered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal
import torch

from benchmark_csp_ridge import BANDS_HZ
from benchmark_ridge_target_variants import fit_candidate, lagged


def binned_log_energy(values: np.ndarray, samples_per_bin: int) -> np.ndarray:
    usable = values[: values.shape[0] // samples_per_bin * samples_per_bin]
    bins = usable.reshape(-1, samples_per_bin, values.shape[1])
    return np.log1p(np.sqrt(np.sum(bins * bins, axis=1))).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/electrode_bandpower_v2"))
    parser.add_argument("--target", default="local_w4_q20")
    parser.add_argument("--sampling-rate", type=float, default=1000.0)
    parser.add_argument("--samples-per-bin", type=int, default=40)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--top-features", type=int, default=1024)
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

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    joined = np.concatenate((np.asarray(train_ecog), np.asarray(test_ecog)), axis=0)
    for low, high in BANDS_HZ:
        sos = signal.butter(4, (low, high), btype="bandpass", fs=args.sampling_rate, output="sos")
        filtered = signal.sosfiltfilt(sos, joined, axis=0).astype(np.float32)
        train_parts.append(binned_log_energy(filtered[: train_ecog.shape[0]], args.samples_per_bin))
        test_parts.append(binned_log_energy(filtered[train_ecog.shape[0] :], args.samples_per_bin))
        print(f"band={low:g}-{high:g}Hz", flush=True)

    train_energy = np.concatenate(train_parts, axis=1)
    test_energy = np.concatenate(test_parts, axis=1)
    np.save(output / "train_band_energy.npy", train_energy, allow_pickle=False)
    np.save(output / "test_band_energy.npy", test_energy, allow_pickle=False)

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
        "method": "all-electrode seven-band log energy plus screened GPU ridge",
        "target": args.target,
        "bands_hz": BANDS_HZ,
        "feature_count_per_bin": int(train_energy.shape[1]),
        "history": args.history,
        "top_features_per_finger": args.top_features,
        "result": result,
    }
    np.save(output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
    np.save(output / "test_prediction.npy", test_prediction, allow_pickle=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result["validation_raw_metrics"]), flush=True)
    print(json.dumps(result["test_raw_metrics"]), flush=True)


if __name__ == "__main__":
    main()
