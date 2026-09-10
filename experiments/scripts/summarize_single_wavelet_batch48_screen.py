#!/usr/bin/env python3
"""Summarize merged per-finger outputs from the fixed batch-48 screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES


DIAGNOSTIC_KEYS = (
    "event_macro_nmse",
    "rest_false_positive_rms",
    "movement_rmse",
    "amplitude_slope",
    "velocity_pcc",
    "movement_balanced_accuracy",
)


def summarize_report(report: dict[str, object]) -> dict[str, object]:
    outer = list(report["outer_folds"])
    initialized = float(report["stitched_initialized_raw_pcc"])
    selected = float(report["stitched_selected_raw_pcc"])
    return {
        "initialization_pcc": initialized,
        "tuned_pcc": selected,
        "gain": selected - initialized,
        "outer_raw_pcc": [
            float(item["selected_outer_metrics"]["raw_pcc"]) for item in outer
        ],
        "selected_schedules": [str(item["selected_schedule"]) for item in outer],
        "mean_initialized_outer_diagnostics": {
            key: float(
                np.mean([item["initialized_outer_metrics"][key] for item in outer])
            )
            for key in DIAGNOSTIC_KEYS
        },
        "mean_selected_outer_diagnostics": {
            key: float(
                np.mean([item["selected_outer_metrics"][key] for item in outer])
            )
            for key in DIAGNOSTIC_KEYS
        },
        "runtime_seconds_sum": float(report["runtime_seconds"]),
    }


def pair_key(path: Path) -> tuple[int, int, str]:
    subject = int(path.parent.parent.name.removeprefix("sub"))
    finger = path.parent.name
    return subject, list(FINGER_NAMES).index(finger), f"S{subject}_{finger}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gain-threshold", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summaries = sorted(args.root.glob("sub*/*/summary.json"), key=pair_key)
    pairs = {
        pair_key(path)[2]: summarize_report(json.loads(path.read_text()))
        for path in summaries
    }
    report = {
        "protocol": "fixed single-branch current-bin batch-48 residual-LSTM screen",
        "released_test_touched": False,
        "gain_threshold": args.gain_threshold,
        "completed_pairs": len(pairs),
        "passing_pairs": [
            pair for pair, result in pairs.items() if result["gain"] >= args.gain_threshold
        ],
        "pairs": pairs,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
