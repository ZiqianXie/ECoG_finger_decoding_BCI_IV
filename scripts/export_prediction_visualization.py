#!/usr/bin/env python3
"""Export compact held-out windows from explicitly selected prediction arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.postprocessing import project_nonnegative
from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def parse_mapping(value: str) -> tuple[int, str]:
    subject, separator, item = value.partition("=")
    if not separator or not item:
        raise argparse.ArgumentTypeError("expected SUBJECT=VALUE")
    return int(subject), item


def select_windows(target: np.ndarray, width: int, count: int) -> list[int]:
    width = min(width, target.shape[0])
    activity = np.max(target, axis=1)
    scores = np.convolve(activity, np.ones(width), mode="valid")
    starts: list[int] = []
    for _ in range(min(count, scores.size)):
        start = int(np.argmax(scores))
        if not np.isfinite(scores[start]):
            break
        starts.append(start)
        left = max(0, start - width)
        right = min(scores.size, start + width)
        scores[left:right] = -np.inf
    return sorted(starts)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 0 else 0.0


def morphology(prediction: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    rest_rms: list[float] = []
    state_f1: list[float] = []
    derivative_pcc: list[float] = []
    for finger in range(target.shape[1]):
        truth = target[:, finger]
        estimate = prediction[:, finger]
        moving = truth >= threshold
        predicted_moving = estimate >= threshold
        tp = np.sum(moving & predicted_moving)
        fp = np.sum(~moving & predicted_moving)
        fn = np.sum(moving & ~predicted_moving)
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        rest_rms.append(float(np.sqrt(np.mean(estimate[~moving] ** 2))))
        state_f1.append(2 * precision * recall / max(precision + recall, 1e-12))
        derivative_pcc.append(pearson(np.diff(estimate), np.diff(truth)))
    return {
        "macro_rest_rms": float(np.mean(rest_rms)),
        "macro_state_f1": float(np.mean(state_f1)),
        "macro_derivative_pcc": float(np.mean(derivative_pcc)),
    }


def rounded(values: np.ndarray) -> list[object]:
    return np.round(values, 4).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=parse_mapping, action="append", required=True)
    parser.add_argument("--method", type=parse_mapping, action="append", default=[])
    parser.add_argument("--prepared-root", default="outputs/preprocessed_v2")
    parser.add_argument("--target-name", default="test_glove_local_w4_q20.npy")
    parser.add_argument("--raw-target-name", default="test_glove_25hz_raw.npy")
    parser.add_argument("--excluded-initial-bins", type=int, default=24)
    parser.add_argument("--sampling-rate-hz", type=float, default=25.0)
    parser.add_argument("--movement-threshold", type=float, default=0.1)
    parser.add_argument("--output", default="outputs/prediction_viz_current/data.json")
    parser.add_argument("--window-seconds", type=float, default=12.0)
    parser.add_argument("--windows-per-subject", type=int, default=3)
    parser.add_argument(
        "--allow-negative",
        action="store_true",
        help="display unconstrained model outputs instead of the default nonnegative projection",
    )
    args = parser.parse_args()

    predictions = dict(args.prediction)
    methods = dict(args.method)
    result: dict[str, object] = {
        "finger_names": [name.title() for name in FINGER_NAMES],
        "sampling_rate_hz": args.sampling_rate_hz,
        "subjects": {},
    }
    width = int(round(args.window_seconds * args.sampling_rate_hz))
    for subject, prediction_path in sorted(predictions.items()):
        prepared = Path(args.prepared_root) / f"sub{subject}"
        unconstrained_prediction = np.load(prediction_path)
        prediction = (
            unconstrained_prediction
            if args.allow_negative
            else project_nonnegative(unconstrained_prediction)
        )
        cleaned = np.load(prepared / args.target_name)[args.excluded_initial_bins :]
        raw = np.load(prepared / args.raw_target_name)[args.excluded_initial_bins :]
        if prediction.shape != cleaned.shape or prediction.shape != raw.shape:
            raise RuntimeError(
                f"subject {subject}: prediction {prediction.shape}, cleaned {cleaned.shape}, raw {raw.shape}"
            )
        windows: list[dict[str, object]] = []
        for start in select_windows(cleaned, width, args.windows_per_subject):
            stop = start + min(width, cleaned.shape[0])
            pred_window = prediction[start:stop]
            true_window = cleaned[start:stop]
            windows.append(
                {
                    "start_seconds": round((start + args.excluded_initial_bins) / args.sampling_rate_hz, 2),
                    "prediction": rounded(pred_window),
                    "unconstrained_prediction": rounded(
                        unconstrained_prediction[start:stop]
                    ),
                    "target": rounded(true_window),
                    "metrics": trajectory_metrics(pred_window, true_window),
                }
            )
        result["subjects"][str(subject)] = {
            "method": methods.get(subject, Path(prediction_path).parent.name),
            "output_constraint": (
                "none" if args.allow_negative else "pointwise maximum(prediction, 0)"
            ),
            "test_raw_metrics": trajectory_metrics(prediction, raw),
            "test_cleaned_metrics": trajectory_metrics(prediction, cleaned),
            "unconstrained_test_raw_metrics": trajectory_metrics(
                unconstrained_prediction, raw
            ),
            "morphology": morphology(prediction, cleaned, args.movement_threshold),
            "windows": windows,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(",", ":")))
    print(output)


if __name__ == "__main__":
    main()
