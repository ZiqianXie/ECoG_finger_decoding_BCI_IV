#!/usr/bin/env python3
"""Evaluate equal-weight ensembles of independently trained seed predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 0 else 0.0


def checked_stack(values: list[np.ndarray], label: str) -> np.ndarray:
    """Stack matching prediction arrays with an actionable shape error."""
    reference = values[0].shape
    mismatched = [value.shape for value in values if value.shape != reference]
    if mismatched:
        raise ValueError(
            f"{label} prediction shapes must match; expected {reference}, "
            f"found {mismatched}"
        )
    return np.stack(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument(
        "--finger",
        action="append",
        choices=FINGER_NAMES,
        required=True,
        help="finger to average; repeat to replace several or all fingers",
    )
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help="subject output directory supplying unchanged finger columns",
    )
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate) < 2:
        parser.error("at least two --candidate directories are required")

    prepared = args.prepared_root / f"sub{args.subject}"
    metadata = json.loads((prepared / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    selected_fingers = list(dict.fromkeys(args.finger))
    selected_indices = [list(FINGER_NAMES).index(name) for name in selected_fingers]
    validation_target_full = np.load(prepared / "train_glove_25hz_raw.npy")[split:]
    test_target_full = np.load(prepared / "test_glove_25hz_raw.npy")[offset:]
    validation_full = [np.load(path / "validation_prediction.npy") for path in args.candidate]
    test_full = [np.load(path / "test_prediction.npy") for path in args.candidate]
    validation_all = checked_stack(validation_full, "validation")
    test_all = checked_stack(test_full, "test")
    if validation_all.shape[2] != len(FINGER_NAMES) or test_all.shape[2] != len(FINGER_NAMES):
        raise ValueError("prediction arrays must have one column for each finger")
    validation_output = np.load(args.base / "validation_prediction.npy").copy()
    test_output = np.load(args.base / "test_prediction.npy").copy()
    if validation_output.shape != validation_target_full.shape:
        raise ValueError("base validation prediction shape does not match the target")
    if test_output.shape != test_target_full.shape:
        raise ValueError("base test prediction shape does not match the target")
    per_finger: dict[str, object] = {}
    for name, finger in zip(selected_fingers, selected_indices, strict=True):
        validation = validation_all[:, :, finger]
        test = test_all[:, :, finger]
        validation_mean = validation.mean(axis=0)
        test_mean = test.mean(axis=0)
        validation_output[:, finger] = validation_mean
        test_output[:, finger] = test_mean
        per_finger[name] = {
            "validation_raw_r": pearson(
                validation_mean, validation_target_full[:, finger]
            ),
            "test_raw_r_descriptive_only": pearson(
                test_mean, test_target_full[:, finger]
            ),
            "validation_prediction_correlation": np.corrcoef(validation).tolist(),
        }
    validation_per_finger = [
        pearson(validation_output[:, index], validation_target_full[:, index])
        for index in range(len(FINGER_NAMES))
    ]
    test_per_finger = [
        pearson(test_output[:, index], test_target_full[:, index])
        for index in range(len(FINGER_NAMES))
    ]
    report = {
        "subject": args.subject,
        "fingers": selected_fingers,
        "method": "equal-weight seed ensemble",
        "base": str(args.base),
        "candidates": [str(path) for path in args.candidate],
        "per_finger": per_finger,
        "subject_validation_raw_r_by_finger": dict(zip(FINGER_NAMES, validation_per_finger)),
        "subject_validation_raw_macro_five": float(np.mean(validation_per_finger)),
        "subject_test_raw_r_by_finger_descriptive_only": dict(zip(FINGER_NAMES, test_per_finger)),
        "subject_test_raw_macro_five_descriptive_only": float(np.mean(test_per_finger)),
        "released_test_used_for_selection": False,
    }
    if len(selected_fingers) == 1:
        name = selected_fingers[0]
        report["finger"] = name
        report.update(per_finger[name])
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "validation_prediction.npy", validation_output, allow_pickle=False)
    np.save(args.output / "test_prediction.npy", test_output, allow_pickle=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
