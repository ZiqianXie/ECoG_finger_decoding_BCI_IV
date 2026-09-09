#!/usr/bin/env python3
"""Combine per-fold fixed-LARS spatial-bank gates without reading test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES
from single_wavelet_support import OFFSET, pearson


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    args = parser.parse_args()

    raw = np.load(
        args.prepared_root / f"sub{args.subject}" / "train_glove_25hz_raw.npy",
        mmap_mode="r",
    )[OFFSET:]
    report: dict[str, object] = {
        "protocol": "three purged development folds; fixed split-local LARS; released test untouched",
        "subject": args.subject,
        "released_test_touched": False,
        "fingers": {},
    }
    for finger_index, finger in enumerate(FINGER_NAMES):
        summaries = [
            json.loads((args.root / finger / f"fold{fold}" / "summary.json").read_text())
            for fold in range(3)
        ]
        variants = tuple(summaries[0]["variants"])
        finger_report: dict[str, object] = {}
        for variant in variants:
            parts = [
                np.load(args.root / finger / f"fold{fold}" / f"{variant}_oof.npy")
                for fold in range(3)
            ]
            stitched = np.full(parts[0].shape, np.nan, dtype=np.float32)
            for part in parts:
                observed = np.isfinite(part)
                if np.any(np.isfinite(stitched[observed])):
                    raise RuntimeError(f"overlapping folds for {finger} {variant}")
                stitched[observed] = part[observed]
            observed = np.isfinite(stitched)
            fold_scores = [
                float(summary["aggregate"][variant]["stitched_softplus_raw_pcc"])
                for summary in summaries
            ]
            finger_report[variant] = {
                "stitched_oof_raw_pcc": pearson(stitched[observed], raw[: stitched.size, finger_index][observed]),
                "mean_fold_raw_pcc": float(np.mean(fold_scores)),
                "fold_raw_pcc": fold_scores,
            }
        selected = max(finger_report, key=lambda name: finger_report[name]["stitched_oof_raw_pcc"])
        report["fingers"][finger] = {
            "selected_by_stitched_oof_raw_pcc": selected,
            "variants": finger_report,
        }

    (args.root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
