#!/usr/bin/env python3
"""Merge independently run experimental outer folds into one selection report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES
from single_wavelet_support import OFFSET, pearson


ARRAYS = ("initialized_oof.npy", "selected_oof.npy", "target_oof.npy")


def merge_array(paths: list[Path]) -> np.ndarray:
    parts = [np.load(path) for path in paths]
    merged = np.full(parts[0].shape, np.nan, dtype=np.float32)
    for part in parts:
        observed = np.isfinite(part)
        if np.any(np.isfinite(merged[observed])):
            raise RuntimeError(f"overlapping finite rows while merging {paths[0].name}")
        merged[observed] = part[observed]
    if not np.isfinite(merged).any():
        raise RuntimeError(f"no finite rows while merging {paths[0].name}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--fingers", nargs="+", choices=tuple(FINGER_NAMES), required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    args = parser.parse_args()

    raw = np.load(
        args.prepared_root / f"sub{args.subject}" / "train_glove_25hz_raw.npy",
        mmap_mode="r",
    )[OFFSET:]
    for finger in args.fingers:
        finger_index = list(FINGER_NAMES).index(finger)
        part_directories = [args.parts_root / finger / f"outer{fold}" for fold in range(3)]
        reports = [json.loads((directory / "summary.json").read_text()) for directory in part_directories]
        outer_records = [item for report in reports for item in report["outer_folds"]]
        if sorted(int(item["outer_fold"]) for item in outer_records) != [0, 1, 2]:
            raise RuntimeError(f"{finger}: expected exactly outer folds 0, 1, and 2")
        outer_records.sort(key=lambda item: int(item["outer_fold"]))
        output = args.output_root / f"sub{args.subject}" / finger
        output.mkdir(parents=True, exist_ok=True)
        merged_arrays = {
            name: merge_array([directory / name for directory in part_directories])
            for name in ARRAYS
        }
        for name, values in merged_arrays.items():
            np.save(output / name, values, allow_pickle=False)

        report = reports[0]
        report["outer_folds"] = outer_records
        report["mean_initialized_outer_raw_pcc"] = float(
            np.mean([item["initialized_outer_metrics"]["raw_pcc"] for item in outer_records])
        )
        report["mean_selected_outer_raw_pcc"] = float(
            np.mean([item["selected_outer_metrics"]["raw_pcc"] for item in outer_records])
        )
        report["mean_initialized_outer_event_macro_nmse"] = float(
            np.mean([item["initialized_outer_metrics"]["event_macro_nmse"] for item in outer_records])
        )
        report["mean_selected_outer_event_macro_nmse"] = float(
            np.mean([item["selected_outer_metrics"]["event_macro_nmse"] for item in outer_records])
        )
        observed = np.isfinite(merged_arrays["selected_oof.npy"])
        raw_target = np.asarray(raw[: observed.size, finger_index])
        report["stitched_initialized_raw_pcc"] = pearson(
            merged_arrays["initialized_oof.npy"][observed], raw_target[observed]
        )
        report["stitched_selected_raw_pcc"] = pearson(
            merged_arrays["selected_oof.npy"][observed], raw_target[observed]
        )
        report["runtime_seconds"] = float(sum(item["runtime_seconds"] for item in reports))
        report["settings"]["outer_folds"] = [0, 1, 2]
        report["settings"]["output"] = str(output)
        report["merged_from"] = [str(directory) for directory in part_directories]
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "finger": finger,
                    "initialized_oof_raw_pcc": report["stitched_initialized_raw_pcc"],
                    "selected_oof_raw_pcc": report["stitched_selected_raw_pcc"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
