#!/usr/bin/env python3
"""Validation-fitted affine normalization for prediction amplitude and offset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_prediction_constrained import finger_metrics
from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def fit_positive_affine(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return least-squares scale and offset with a nonnegative scale."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    centered = prediction - prediction.mean()
    denominator = float(centered @ centered)
    if denominator <= np.finfo(np.float64).eps:
        return 0.0, float(target.mean())
    scale = max(float(centered @ (target - target.mean()) / denominator), 0.0)
    offset = float(target.mean() - scale * prediction.mean())
    return scale, offset


def apply_affine(prediction: np.ndarray, scale: float, offset: float) -> np.ndarray:
    return scale * prediction + offset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--validation-name", default="validation_prediction.npy")
    parser.add_argument("--test-name", default="test_prediction.npy")
    parser.add_argument("--target", required=True)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument(
        "--finger",
        action="append",
        choices=FINGER_NAMES,
        help="finger to normalize; repeat as needed. Defaults to all fingers.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prepared = args.prepared_root / f"sub{args.subject}"
    metadata = json.loads((prepared / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    validation_target = np.load(prepared / f"train_glove_{args.target}.npy")[split:]
    test_target = np.load(prepared / f"test_glove_{args.target}.npy")[offset:]
    validation_raw = np.load(prepared / "train_glove_25hz_raw.npy")[split:]
    test_raw = np.load(prepared / "test_glove_25hz_raw.npy")[offset:]
    validation_input = np.load(args.prediction_root / args.validation_name)
    test_input = np.load(args.prediction_root / args.test_name)
    if validation_input.shape != validation_target.shape:
        raise ValueError("validation prediction and target shapes differ")
    if test_input.shape != test_target.shape:
        raise ValueError("test prediction and target shapes differ")

    requested = set(args.finger or FINGER_NAMES)
    validation_output = validation_input.copy()
    test_output = test_input.copy()
    selection: dict[str, object] = {}
    for finger, name in enumerate(FINGER_NAMES):
        if name not in requested:
            selection[name] = {"status": "unchanged"}
            continue
        scale, intercept = fit_positive_affine(
            validation_input[:, finger], validation_target[:, finger]
        )
        validation_output[:, finger] = apply_affine(
            validation_input[:, finger], scale, intercept
        )
        test_output[:, finger] = apply_affine(test_input[:, finger], scale, intercept)
        selection[name] = {
            "status": "normalized",
            "scale": scale,
            "offset": intercept,
            "validation_before": finger_metrics(
                validation_input[:, finger], validation_target[:, finger], args.threshold
            ),
            "validation_after": finger_metrics(
                validation_output[:, finger], validation_target[:, finger], args.threshold
            ),
            "test_before": finger_metrics(
                test_input[:, finger], test_target[:, finger], args.threshold
            ),
            "test_after": finger_metrics(
                test_output[:, finger], test_target[:, finger], args.threshold
            ),
        }

    report = {
        "subject": args.subject,
        "method": (
            "positive affine least-squares normalization fit on the cleaned "
            "chronological validation target only; no clipping"
        ),
        "target": args.target,
        "selection": selection,
        "validation_raw_metrics": trajectory_metrics(validation_output, validation_raw),
        "test_raw_metrics": trajectory_metrics(test_output, test_raw),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "validation_prediction.npy", validation_output, allow_pickle=False)
    np.save(args.output / "test_prediction.npy", test_output, allow_pickle=False)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
