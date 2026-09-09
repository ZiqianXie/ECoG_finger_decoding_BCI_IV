#!/usr/bin/env python3
"""Summarize per-finger seed ensembles from event-grouped outer folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES
from summarize_event_lars_lstm_cv import pearson


def ensemble_oof(root: Path, subject: int, finger: str, seeds: list[int]) -> dict[str, object] | None:
    predictions = []
    raw_targets = []
    cleaned_targets = []
    fold_scores = []
    for fold in range(3):
        members = []
        fold_root = root / f"sub{subject}" / finger / f"fold{fold}"
        for seed in seeds:
            path = fold_root / f"seed{seed}" / "validation_prediction.npy"
            if not path.exists():
                return None
            members.append(np.load(path))
        prediction = np.mean(np.stack(members), axis=0)
        raw = np.load(fold_root / f"seed{seeds[0]}" / "validation_raw_target.npy")
        cleaned = np.load(
            fold_root / f"seed{seeds[0]}" / "validation_cleaned_target.npy"
        )
        predictions.append(prediction)
        raw_targets.append(raw)
        cleaned_targets.append(cleaned)
        fold_scores.append(pearson(prediction, raw))
    prediction = np.concatenate(predictions)
    raw = np.concatenate(raw_targets)
    cleaned = np.concatenate(cleaned_targets)
    return {
        "raw_pcc": pearson(prediction, raw),
        "cleaned_pcc": pearson(prediction, cleaned),
        "fold_raw_pcc": fold_scores,
        "prediction_sd": float(np.std(prediction)),
        "rows": int(prediction.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument(
        "--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=tuple(FINGER_NAMES)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report: dict[str, object] = {"subjects": {}, "complete_models": 0}
    for subject in args.subjects:
        subject_report = {}
        for finger in args.fingers:
            current = ensemble_oof(args.input_root, subject, finger, list(args.seeds))
            baseline = (
                ensemble_oof(args.baseline_root, subject, finger, list(args.seeds))
                if args.baseline_root is not None
                else None
            )
            record: dict[str, object] = {"current": current, "baseline": baseline}
            if current is not None:
                report["complete_models"] = int(report["complete_models"]) + 1
            if current is not None and baseline is not None:
                record["raw_pcc_delta"] = current["raw_pcc"] - baseline["raw_pcc"]
            subject_report[finger] = record
        report["subjects"][str(subject)] = subject_report

    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
