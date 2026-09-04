#!/usr/bin/env python3
"""Validation-only smooth shrinkage/gain calibration for arbitrary predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from blend_prediction_candidates import score
from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--validation-name", default="validation_prediction_selected.npy")
    parser.add_argument("--test-name", default="test_prediction_selected.npy")
    parser.add_argument("--target", default="local_w4_q20")
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.prepared_root / f"sub{args.subject}"
    split = int(json.loads((root / "metadata.json").read_text())["target_fit_samples_25hz"])
    offset = args.history - 1
    validation_target = np.load(root / f"train_glove_{args.target}.npy")[split:]
    validation_raw = np.load(root / "train_glove_25hz_raw.npy")[split:]
    test_target = np.load(root / f"test_glove_{args.target}.npy")[offset:]
    test_raw = np.load(root / "test_glove_25hz_raw.npy")[offset:]
    validation = np.load(args.prediction_root / args.validation_name)
    test = np.load(args.prediction_root / args.test_name)
    validation_out = np.zeros_like(validation)
    test_out = np.zeros_like(test)
    selection: dict[str, object] = {}
    for finger, name in enumerate(FINGER_NAMES):
        best = (-np.inf, 1.0, 1.0, 0.0)
        for floor in (0.1, 0.25, 0.5, 0.75, 1.0):
            for gain in (0.5, 1.0, 1.5, 2.0, 3.0):
                for deadzone in (0.0, 0.025, 0.05, 0.10):
                    estimate = floor * validation[:, finger] + gain * np.maximum(validation[:, finger] - deadzone, 0)
                    candidate = (score(estimate, validation_target[:, finger], args.threshold), floor, gain, deadzone)
                    if candidate[0] > best[0]:
                        best = candidate
        _, floor, gain, deadzone = best
        validation_out[:, finger] = floor * validation[:, finger] + gain * np.maximum(validation[:, finger] - deadzone, 0)
        test_out[:, finger] = floor * test[:, finger] + gain * np.maximum(test[:, finger] - deadzone, 0)
        selection[name] = {"floor": floor, "gain": gain, "deadzone": deadzone, "validation_morphology": best[0]}

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "validation_prediction.npy", validation_out)
    np.save(args.output / "test_prediction.npy", test_out)
    report = {
        "selection": selection,
        "validation_cleaned_metrics": trajectory_metrics(validation_out, validation_target),
        "validation_raw_metrics": trajectory_metrics(validation_out, validation_raw),
        "test_cleaned_metrics": trajectory_metrics(test_out, test_target),
        "test_raw_metrics": trajectory_metrics(test_out, test_raw),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
