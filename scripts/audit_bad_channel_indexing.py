#!/usr/bin/env python3
"""Check whether paper-reported bad channels were zero- or one-based indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.io import load_subject


PAPER_INDICES = {1: (55,), 2: (21, 38), 3: (49,)}


def channel_stats(train: np.ndarray, test: np.ndarray) -> list[dict[str, float | int]]:
    train_std = train.std(axis=0)
    test_std = test.std(axis=0)
    train_max = np.max(np.abs(train - np.median(train, axis=0)), axis=0)
    test_max = np.max(np.abs(test - np.median(test, axis=0)), axis=0)
    ratio = test_std / np.maximum(train_std, 1e-12)
    rows = []
    for index in range(train.shape[1]):
        rows.append({
            "zero_based_index": index,
            "one_based_channel": index + 1,
            "train_std": float(train_std[index]),
            "test_std": float(test_std[index]),
            "test_train_std_ratio": float(ratio[index]),
            "train_abs_max_from_median": float(train_max[index]),
            "test_abs_max_from_median": float(test_max[index]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/bci_competition_iv_ds4"))
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/bad_channel_indexing.json"))
    args = parser.parse_args()
    output = {}
    for subject in (1, 2, 3):
        data = load_subject(args.data_root, subject)
        rows = channel_stats(data.train_ecog, data.test_ecog)
        paper = PAPER_INDICES[subject]
        candidates = sorted(set(paper) | {value - 1 for value in paper} | {value + 1 for value in paper})
        output[str(subject)] = {
            "paper_reported_numbers": paper,
            "candidate_rows": [rows[index] for index in candidates if 0 <= index < len(rows)],
            "largest_test_train_std_ratios": sorted(rows, key=lambda row: row["test_train_std_ratio"], reverse=True)[:8],
            "largest_train_outliers": sorted(rows, key=lambda row: row["train_abs_max_from_median"], reverse=True)[:8],
            "largest_test_outliers": sorted(rows, key=lambda row: row["test_abs_max_from_median"], reverse=True)[:8],
        }
        print(json.dumps({
            "subject": subject,
            "top_test_train_ratio": output[str(subject)]["largest_test_train_std_ratios"][:3],
            "paper_neighborhood": output[str(subject)]["candidate_rows"],
        }), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
