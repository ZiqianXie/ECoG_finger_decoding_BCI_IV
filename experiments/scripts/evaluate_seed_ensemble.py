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


def candidate_status(
    directory: Path, finger_name: str, validation_prediction: np.ndarray
) -> dict[str, object]:
    """Classify a seed using training metadata and validation predictions only."""
    if not np.isfinite(validation_prediction).all():
        return {"eligible": False, "reason": "nonfinite_validation_prediction"}
    if float(np.std(validation_prediction)) <= 1.0e-8:
        return {"eligible": False, "reason": "constant_validation_prediction"}
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return {"eligible": False, "reason": "missing_training_summary"}
    summary = json.loads(summary_path.read_text())
    metadata = summary.get("per_finger", {}).get(finger_name)
    if not isinstance(metadata, dict):
        return {"eligible": False, "reason": "missing_finger_training_metadata"}
    best_epoch = int(metadata.get("best_epoch", 0))
    if best_epoch <= 0:
        return {
            "eligible": False,
            "reason": "protected_epoch0_baseline",
            "best_epoch": best_epoch,
        }
    return {"eligible": True, "reason": "trained_checkpoint", "best_epoch": best_epoch}


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
    parser.add_argument(
        "--exclude-collapsed",
        action="store_true",
        help=(
            "exclude a seed independently for each finger when its best checkpoint "
            "is the protected epoch-0 initializer or its validation prediction is invalid"
        ),
    )
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
        statuses = [
            candidate_status(path, name, validation_all[index, :, finger])
            for index, path in enumerate(args.candidate)
        ]
        eligible = [
            index
            for index, status in enumerate(statuses)
            if not args.exclude_collapsed or bool(status["eligible"])
        ]
        candidate_audit = [
            {"directory": str(path), **status}
            for path, status in zip(args.candidate, statuses, strict=True)
        ]
        if not eligible:
            per_finger[name] = {
                "status": "fallback_to_base_no_noncollapsed_seed",
                "included_candidates": [],
                "candidate_audit": candidate_audit,
                "validation_raw_r": pearson(
                    validation_output[:, finger], validation_target_full[:, finger]
                ),
                "test_raw_r_descriptive_only": pearson(
                    test_output[:, finger], test_target_full[:, finger]
                ),
                "validation_prediction_correlation": [],
            }
            continue
        validation = validation_all[eligible, :, finger]
        test = test_all[eligible, :, finger]
        validation_mean = validation.mean(axis=0)
        test_mean = test.mean(axis=0)
        validation_output[:, finger] = validation_mean
        test_output[:, finger] = test_mean
        per_finger[name] = {
            "status": "filtered_seed_mean",
            "included_candidates": [str(args.candidate[index]) for index in eligible],
            "candidate_audit": candidate_audit,
            "validation_raw_r": pearson(
                validation_mean, validation_target_full[:, finger]
            ),
            "test_raw_r_descriptive_only": pearson(
                test_mean, test_target_full[:, finger]
            ),
            "validation_prediction_correlation": (
                np.corrcoef(validation).tolist() if len(eligible) > 1 else [[1.0]]
            ),
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
        "exclude_collapsed": args.exclude_collapsed,
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
