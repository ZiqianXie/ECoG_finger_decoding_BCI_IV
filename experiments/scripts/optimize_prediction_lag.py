#!/usr/bin/env python3
"""Select a separate prediction delay per finger on validation data only."""

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


def overlap(values: np.ndarray, target: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    # Positive lag delays the prediction; negative lag advances it.
    if lag > 0:
        return values[:-lag], target[lag:]
    if lag < 0:
        return values[-lag:], target[:lag]
    return values, target


def shifted(values: np.ndarray, lag: int) -> np.ndarray:
    result = np.empty_like(values)
    if lag > 0:
        result[:lag] = values[0]
        result[lag:] = values[:-lag]
    elif lag < 0:
        result[lag:] = values[-1]
        result[:lag] = values[-lag:]
    else:
        result[:] = values
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--max-lag-bins", type=int, default=10)
    parser.add_argument("--sampling-rate", type=float, default=25.0)
    parser.add_argument("--history", type=int, default=25)
    args = parser.parse_args()

    root = args.prepared_root / f"sub{args.subject}"
    prediction_root = args.prediction_root / f"sub{args.subject}"
    metadata = json.loads((root / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    validation_prediction = np.load(prediction_root / "validation_prediction.npy")
    test_prediction = np.load(prediction_root / "test_prediction.npy")
    validation_true = np.load(root / "train_glove_25hz_raw.npy")[split:]
    test_true = np.load(root / "test_glove_25hz_raw.npy")[offset:]
    adjusted_validation = np.empty_like(validation_prediction)
    adjusted_test = np.empty_like(test_prediction)
    finger_report: dict[str, object] = {}
    for finger, name in enumerate(FINGER_NAMES):
        scores = {}
        best_lag = 0
        best_score = -float("inf")
        for lag in range(-args.max_lag_bins, args.max_lag_bins + 1):
            prediction_overlap, target_overlap = overlap(
                validation_prediction[:, finger],
                validation_true[:, finger],
                lag,
            )
            score = pearson(prediction_overlap, target_overlap)
            scores[str(lag)] = score
            if score > best_score:
                best_score = score
                best_lag = lag
        adjusted_validation[:, finger] = shifted(validation_prediction[:, finger], best_lag)
        adjusted_test[:, finger] = shifted(test_prediction[:, finger], best_lag)
        test_overlap, test_target_overlap = overlap(
            test_prediction[:, finger],
            test_true[:, finger],
            best_lag,
        )
        finger_report[name] = {
            "selected_lag_bins": best_lag,
            "selected_lag_seconds": best_lag / args.sampling_rate,
            "validation_r": best_score,
            "test_r_overlap": pearson(test_overlap, test_target_overlap),
            "validation_lag_scores": scores,
        }
    report = {
        "subject": args.subject,
        "selection": "independent per-finger lag selected on chronological validation only",
        "original_validation_metrics": trajectory_metrics(validation_prediction, validation_true),
        "adjusted_validation_metrics": trajectory_metrics(adjusted_validation, validation_true),
        "original_test_metrics": trajectory_metrics(test_prediction, test_true),
        "adjusted_test_metrics": trajectory_metrics(adjusted_test, test_true),
        "fingers": finger_report,
    }
    np.save(prediction_root / "validation_prediction_lag_adjusted.npy", adjusted_validation, allow_pickle=False)
    np.save(prediction_root / "test_prediction_lag_adjusted.npy", adjusted_test, allow_pickle=False)
    (prediction_root / "lag_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
