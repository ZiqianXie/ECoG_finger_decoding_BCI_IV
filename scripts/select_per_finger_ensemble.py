#!/usr/bin/env python3
"""Select a different validated decoder for each finger and combine outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
    return float(xc @ yc / denominator) if denominator > 0 else 0.0


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("method must be NAME=OUTPUT_DIRECTORY")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("method must be NAME=OUTPUT_DIRECTORY")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/per_finger_ensemble_v2/sub3"))
    args = parser.parse_args()

    root = args.prepared_root / f"sub{args.subject}"
    metadata = json.loads((root / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    validation_truth = np.load(root / "train_glove_25hz_raw.npy")[split:]
    test_truth = np.load(root / "test_glove_25hz_raw.npy")[offset:]
    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, directory in args.method:
        validation = np.load(directory / "validation_prediction.npy")
        test = np.load(directory / "test_prediction.npy")
        if validation.shape != validation_truth.shape or test.shape != test_truth.shape:
            raise ValueError(
                f"{name} prediction shapes {validation.shape}, {test.shape} do not match "
                f"truth {validation_truth.shape}, {test_truth.shape}"
            )
        candidates[name] = (validation, test)

    validation_output = np.zeros_like(validation_truth, dtype=np.float32)
    test_output = np.zeros_like(test_truth, dtype=np.float32)
    selection: dict[str, object] = {}
    for finger, finger_name in enumerate(FINGER_NAMES):
        validation_scores = {
            name: pearson(prediction[0][:, finger], validation_truth[:, finger])
            for name, prediction in candidates.items()
        }
        winner = max(validation_scores, key=validation_scores.get)
        validation_output[:, finger] = candidates[winner][0][:, finger]
        test_output[:, finger] = candidates[winner][1][:, finger]
        selection[finger_name] = {
            "selected_method": winner,
            "validation_r": validation_scores[winner],
            "test_r": pearson(test_output[:, finger], test_truth[:, finger]),
            "all_validation_r": validation_scores,
        }

    report = {
        "subject": args.subject,
        "selection_rule": "highest validation PCC independently for each finger",
        "candidate_methods": list(candidates),
        "selection": selection,
        "validation_raw_metrics": trajectory_metrics(validation_output, validation_truth),
        "test_raw_metrics": trajectory_metrics(test_output, test_truth),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "validation_prediction.npy", validation_output, allow_pickle=False)
    np.save(args.output / "test_prediction.npy", test_output, allow_pickle=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
