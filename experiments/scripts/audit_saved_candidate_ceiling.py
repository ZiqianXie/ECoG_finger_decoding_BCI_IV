#!/usr/bin/env python3
"""Audit the retrospective ceiling of saved five-finger test predictions.

This is deliberately not a model-selection script: released-test labels define
the reported oracle ceiling, so its output may only diagnose whether an already
saved representation contains the missing signal.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES
from summarize_event_lars_lstm_cv import pearson


SUBJECT = re.compile(r"(?:^|/)sub([123])(?:/|$)")


def aligned_target(raw: np.ndarray, rows: int, *, tail: bool = False) -> np.ndarray:
    if rows == raw.shape[0]:
        return raw
    aligned = raw[24:]
    if rows > aligned.shape[0]:
        raise ValueError("prediction is longer than the available target")
    return aligned[-rows:] if tail else aligned[:rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/saved_candidate_ceiling_v1/summary.json"))
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    raw_test = {
        subject: np.load(args.prepared_root / f"sub{subject}" / "test_glove_25hz_raw.npy")
        for subject in (1, 2, 3)
    }
    raw_train = {
        subject: np.load(args.prepared_root / f"sub{subject}" / "train_glove_25hz_raw.npy")
        for subject in (1, 2, 3)
    }
    records = {subject: {finger: [] for finger in FINGER_NAMES} for subject in (1, 2, 3)}
    scanned = 0
    for path in args.output_root.rglob("test_prediction*.npy"):
        relative = path.as_posix()
        match = SUBJECT.search(relative)
        if match is None:
            continue
        subject = int(match.group(1))
        try:
            prediction = np.load(path, mmap_mode="r")
        except (OSError, ValueError):
            continue
        if (
            prediction.ndim != 2
            or prediction.shape[1] != 5
            or prediction.shape[0] < 2
            or not np.isfinite(prediction).all()
        ):
            continue
        try:
            target = aligned_target(raw_test[subject], prediction.shape[0])
        except ValueError:
            continue
        validation_path = path.with_name(path.name.replace("test_", "validation_", 1))
        validation_scores = None
        if validation_path.exists():
            validation = np.load(validation_path, mmap_mode="r")
            if (
                validation.ndim == 2
                and validation.shape[1] == 5
                and validation.shape[0] >= 2
                and np.isfinite(validation).all()
            ):
                validation_target = aligned_target(
                    raw_train[subject], validation.shape[0], tail=True
                )
                validation_scores = [
                    pearson(validation[:, finger], validation_target[:, finger])
                    for finger in range(5)
                ]
        scanned += 1
        for finger_index, finger in enumerate(FINGER_NAMES):
            records[subject][finger].append({
                "path": relative,
                "test_raw_pcc": pearson(prediction[:, finger_index], target[:, finger_index]),
                "validation_raw_pcc": (
                    validation_scores[finger_index]
                    if validation_scores is not None
                    else None
                ),
            })
    report: dict[str, object] = {
        "protocol": "released-test oracle audit of previously saved candidate arrays; invalid for selection",
        "scanned_five_finger_arrays": scanned,
        "subjects": {},
    }
    for subject in (1, 2, 3):
        subject_report = {}
        for finger in FINGER_NAMES:
            ordered = sorted(
                records[subject][finger],
                key=lambda item: item["test_raw_pcc"],
                reverse=True,
            )
            validation_available = [
                item for item in records[subject][finger]
                if item["validation_raw_pcc"] is not None
            ]
            validation_selected = max(
                validation_available,
                key=lambda item: item["validation_raw_pcc"],
                default=None,
            )
            subject_report[finger] = {
                "oracle_top": ordered[: args.top],
                "validation_selected_from_saved_candidates": validation_selected,
            }
        report["subjects"][str(subject)] = subject_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"scanned": scanned, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
