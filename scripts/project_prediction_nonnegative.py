#!/usr/bin/env python3
"""Make nonnegative predictions the default while retaining raw outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.postprocessing import project_nonnegative
from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prepared = args.prepared_root / f"sub{args.subject}"
    metadata = json.loads((prepared / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    validation_raw = np.load(prepared / "train_glove_25hz_raw.npy")[split:]
    validation_cleaned = np.load(prepared / f"train_glove_{args.target}.npy")[split:]
    test_raw = np.load(prepared / "test_glove_25hz_raw.npy")[offset:]
    test_cleaned = np.load(prepared / f"test_glove_{args.target}.npy")[offset:]
    validation = np.load(args.prediction_root / "validation_prediction.npy")
    test = np.load(args.prediction_root / "test_prediction.npy")
    if validation.shape != validation_raw.shape or test.shape != test_raw.shape:
        raise ValueError("prediction and target shapes do not agree")

    projected_validation = project_nonnegative(validation)
    projected_test = project_nonnegative(test)
    report = {
        "subject": args.subject,
        "source": str(args.prediction_root),
        "constraint": "pointwise maximum(prediction, 0); no fitted parameters",
        "negative_excursions": {
            name: {
                "validation_fraction": float(np.mean(validation[:, finger] < 0)),
                "validation_minimum": float(np.min(validation[:, finger])),
                "test_fraction": float(np.mean(test[:, finger] < 0)),
                "test_minimum": float(np.min(test[:, finger])),
            }
            for finger, name in enumerate(FINGER_NAMES)
        },
        "before": {
            "validation_raw": trajectory_metrics(validation, validation_raw),
            "validation_cleaned": trajectory_metrics(validation, validation_cleaned),
            "test_raw": trajectory_metrics(test, test_raw),
            "test_cleaned": trajectory_metrics(test, test_cleaned),
        },
        "after": {
            "validation_raw": trajectory_metrics(projected_validation, validation_raw),
            "validation_cleaned": trajectory_metrics(projected_validation, validation_cleaned),
            "test_raw": trajectory_metrics(projected_test, test_raw),
            "test_cleaned": trajectory_metrics(projected_test, test_cleaned),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    # Standard filenames are the constrained, public-facing predictions.
    np.save(args.output / "validation_prediction.npy", projected_validation, allow_pickle=False)
    np.save(args.output / "test_prediction.npy", projected_test, allow_pickle=False)
    # The exact model outputs remain available for diagnostics and ablations.
    np.save(args.output / "validation_prediction_unconstrained.npy", validation, allow_pickle=False)
    np.save(args.output / "test_prediction_unconstrained.npy", test, allow_pickle=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
