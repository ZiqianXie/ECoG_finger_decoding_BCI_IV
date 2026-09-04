#!/usr/bin/env python3
"""Rank train/validation/test channel distribution shifts for one subject."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, dtype=np.float64)
    std = np.std(values, axis=0, dtype=np.float64)
    absolute_max = np.max(np.abs(values), axis=0)
    return mean, std, absolute_max


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/channel_shift_audit_v1"))
    args = parser.parse_args()

    root = args.prepared_root / f"sub{args.subject}"
    metadata = json.loads((root / "metadata.json").read_text())
    split = int(metadata["normalization_fit_samples_1khz"])
    train_all = np.load(root / "train_ecog.npy", mmap_mode="r")
    test = np.load(root / "test_ecog.npy", mmap_mode="r")
    fit = train_all[:split]
    validation = train_all[split:]
    fit_mean, fit_std, fit_max = moments(fit)
    validation_mean, validation_std, validation_max = moments(validation)
    test_mean, test_std, test_max = moments(test)
    safe = np.maximum(fit_std, 1.0e-8)
    retained = metadata["retained_channels_one_based"]
    rows = []
    for index, physical in enumerate(retained):
        rows.append(
            {
                "retained_index": index,
                "physical_channel_one_based": physical,
                "validation_mean_z": float((validation_mean[index] - fit_mean[index]) / safe[index]),
                "test_mean_z": float((test_mean[index] - fit_mean[index]) / safe[index]),
                "validation_std_ratio": float(validation_std[index] / safe[index]),
                "test_std_ratio": float(test_std[index] / safe[index]),
                "validation_max_ratio": float(validation_max[index] / max(fit_max[index], 1.0e-8)),
                "test_max_ratio": float(test_max[index] / max(fit_max[index], 1.0e-8)),
            }
        )
    rows.sort(
        key=lambda row: max(
            abs(np.log(max(row["test_std_ratio"], 1.0e-8))),
            abs(row["test_mean_z"]),
            np.log(max(row["test_max_ratio"], 1.0e-8)),
        ),
        reverse=True,
    )
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "subject": args.subject,
        "normalization_fit_samples": split,
        "top_test_shift_channels": rows[:20],
        "all_channels": rows,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows[:12], indent=2), flush=True)


if __name__ == "__main__":
    main()
