#!/usr/bin/env python3
"""Validation-only amplitude calibration with explicit morphology constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    return float(x_centered @ y_centered / denominator) if denominator > 0 else 0.0


def calibrate(
    prediction: np.ndarray,
    floor: float,
    gain: float,
    deadzone: float,
) -> np.ndarray:
    """Retain a linear floor and amplify only activity above a dead zone."""
    return floor * prediction + gain * np.maximum(prediction - deadzone, 0.0)


def finger_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    moving = target >= threshold
    predicted_moving = prediction >= threshold
    true_positive = int(np.sum(moving & predicted_moving))
    false_positive = int(np.sum(~moving & predicted_moving))
    false_negative = int(np.sum(moving & ~predicted_moving))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    truth_peak = float(np.quantile(target[moving], 0.95)) if moving.any() else 0.0
    prediction_peak = (
        float(np.quantile(prediction[moving], 0.95)) if moving.any() else 0.0
    )
    rest_rms = (
        float(np.sqrt(np.mean(prediction[~moving] ** 2)))
        if (~moving).any()
        else 0.0
    )
    return {
        "pcc": pearson(prediction, target),
        "derivative_pcc": pearson(np.diff(prediction), np.diff(target)),
        "state_precision": float(precision),
        "state_recall": float(recall),
        "state_f1": float(f1),
        "movement_peak_ratio": prediction_peak / max(truth_peak, 1.0e-12),
        "rest_rms": rest_rms,
    }


def morphology_score(metrics: dict[str, float]) -> float:
    peak_score = np.exp(
        -abs(np.log(max(metrics["movement_peak_ratio"], 1.0e-6)))
    )
    return float(
        0.45 * metrics["pcc"]
        + 0.20 * metrics["derivative_pcc"]
        + 0.15 * metrics["state_f1"]
        + 0.15 * peak_score
        - 0.05 * metrics["rest_rms"]
    )


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
        help="Finger to calibrate; repeat as needed. Defaults to all fingers.",
    )
    parser.add_argument("--max-pcc-drop", type=float, default=0.02)
    parser.add_argument("--max-derivative-pcc-drop", type=float, default=0.02)
    parser.add_argument("--min-state-recall", type=float, default=0.50)
    parser.add_argument("--min-peak-ratio", type=float, default=0.50)
    parser.add_argument("--max-peak-ratio", type=float, default=1.25)
    parser.add_argument("--max-rest-rms", type=float, default=0.12)
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
        raise ValueError(
            f"validation prediction {validation_input.shape} does not match "
            f"target {validation_target.shape}"
        )
    if test_input.shape != test_target.shape:
        raise ValueError(
            f"test prediction {test_input.shape} does not match target {test_target.shape}"
        )

    requested = set(args.finger or FINGER_NAMES)
    validation_output = validation_input.copy()
    test_output = test_input.copy()
    report: dict[str, object] = {
        "subject": args.subject,
        "selection_policy": (
            "parameters and feasibility selected on cleaned chronological "
            "validation target only"
        ),
        "target": args.target,
        "constraints": {
            "max_pcc_drop": args.max_pcc_drop,
            "max_derivative_pcc_drop": args.max_derivative_pcc_drop,
            "min_state_recall": args.min_state_recall,
            "min_peak_ratio": args.min_peak_ratio,
            "max_peak_ratio": args.max_peak_ratio,
            "max_rest_rms": args.max_rest_rms,
        },
        "selection": {},
    }
    floors = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    gains = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
    deadzones = (0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2)

    for finger, name in enumerate(FINGER_NAMES):
        baseline = finger_metrics(
            validation_input[:, finger], validation_target[:, finger], args.threshold
        )
        detail: dict[str, object] = {"baseline_validation": baseline}
        if name not in requested:
            detail["status"] = "unchanged"
            report["selection"][name] = detail
            continue

        feasible: list[
            tuple[float, float, float, float, dict[str, float], np.ndarray]
        ] = []
        for floor in floors:
            for gain in gains:
                for deadzone in deadzones:
                    estimate = calibrate(
                        validation_input[:, finger], floor, gain, deadzone
                    )
                    metrics = finger_metrics(
                        estimate, validation_target[:, finger], args.threshold
                    )
                    if (
                        metrics["pcc"] >= baseline["pcc"] - args.max_pcc_drop
                        and metrics["derivative_pcc"]
                        >= baseline["derivative_pcc"]
                        - args.max_derivative_pcc_drop
                        and metrics["state_recall"] >= args.min_state_recall
                        and args.min_peak_ratio
                        <= metrics["movement_peak_ratio"]
                        <= args.max_peak_ratio
                        and metrics["rest_rms"] <= args.max_rest_rms
                    ):
                        feasible.append(
                            (
                                morphology_score(metrics),
                                floor,
                                gain,
                                deadzone,
                                metrics,
                                estimate,
                            )
                        )

        if not feasible:
            detail["status"] = "unchanged_no_feasible_calibration"
            detail["feasible_candidate_count"] = 0
            report["selection"][name] = detail
            continue

        best = max(feasible, key=lambda candidate: candidate[0])
        score, floor, gain, deadzone, validation_metrics, validation_estimate = best
        test_estimate = calibrate(test_input[:, finger], floor, gain, deadzone)
        validation_output[:, finger] = validation_estimate
        test_output[:, finger] = test_estimate
        detail.update(
            {
                "status": "calibrated",
                "feasible_candidate_count": len(feasible),
                "floor": floor,
                "gain": gain,
                "deadzone": deadzone,
                "validation_morphology_score": score,
                "validation": validation_metrics,
                "test_cleaned": finger_metrics(
                    test_estimate, test_target[:, finger], args.threshold
                ),
            }
        )
        report["selection"][name] = detail

    report["validation_raw_metrics"] = trajectory_metrics(
        validation_output, validation_raw
    )
    report["test_raw_metrics"] = trajectory_metrics(test_output, test_raw)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(
        args.output / "validation_prediction.npy",
        validation_output,
        allow_pickle=False,
    )
    np.save(args.output / "test_prediction.npy", test_output, allow_pickle=False)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
