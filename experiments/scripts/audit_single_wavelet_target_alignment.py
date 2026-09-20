#!/usr/bin/env python3
"""Audit how split-local cleaned OOF targets align with raw glove trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES
from single_wavelet_support import OFFSET, pearson


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    pairs: dict[str, dict[str, float]] = {}
    for subject in (1, 2, 3):
        raw_matrix = np.load(
            args.prepared_root / f"sub{subject}" / "train_glove_25hz_raw.npy",
            mmap_mode="r",
        )
        for finger_index, finger in enumerate(FINGER_NAMES):
            target_path = args.root / f"sub{subject}" / finger / "target_oof.npy"
            if not target_path.is_file():
                continue
            target = np.load(target_path)
            raw = np.asarray(
                raw_matrix[OFFSET : OFFSET + target.size, finger_index]
            )
            observed = np.isfinite(target) & np.isfinite(raw)
            pairs[f"S{subject}_{finger}"] = {
                "observed_bins": int(observed.sum()),
                "cleaned_vs_raw_pcc": pearson(target[observed], raw[observed]),
                "cleaned_sd": float(np.std(target[observed])),
                "raw_sd": float(np.std(raw[observed])),
            }

    report = {
        "description": "split-local OOF cleaned target versus raw glove trajectory",
        "pairs": pairs,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
