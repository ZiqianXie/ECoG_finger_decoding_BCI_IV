#!/usr/bin/env python3
"""Choose validation-only convex blends of prediction candidates per finger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 0 else 0.0


def score(estimate: np.ndarray, truth: np.ndarray, threshold: float) -> float:
    moving = truth >= threshold
    predicted = estimate >= threshold
    tp = np.sum(moving & predicted)
    fp = np.sum(~moving & predicted)
    fn = np.sum(moving & ~predicted)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    truth_peak = np.quantile(truth[moving], .95) if moving.any() else 1.0
    estimate_peak = np.quantile(estimate[moving], .95) if moving.any() else 0.0
    peak = np.exp(-abs(np.log(max(estimate_peak / max(truth_peak, 1e-6), 1e-6))))
    rest = np.sqrt(np.mean(estimate[~moving] ** 2)) if (~moving).any() else 0.0
    return float(
        .50 * pearson(estimate, truth)
        + .20 * pearson(np.diff(estimate), np.diff(truth))
        + .15 * f1
        + .10 * peak
        - .05 * rest / max(float(truth_peak), 1e-6)
    )


def parse_candidate(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("candidate must be NAME=OUTPUT_DIRECTORY")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--target", default="local_w4_q20")
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=.1)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.prepared_root / f"sub{args.subject}"
    split = int(json.loads((root / "metadata.json").read_text())["target_fit_samples_25hz"])
    offset = args.history - 1
    validation_target = np.load(root / f"train_glove_{args.target}.npy")[split:]
    validation_raw = np.load(root / "train_glove_25hz_raw.npy")[split:]
    test_target = np.load(root / f"test_glove_{args.target}.npy")[offset:]
    test_raw = np.load(root / "test_glove_25hz_raw.npy")[offset:]
    values = {
        name: (
            np.load(path / "validation_prediction.npy"),
            np.load(path / "test_prediction.npy"),
        )
        for name, path in args.candidate
    }
    weights = np.linspace(0, 1, 5)
    validation_out = np.zeros_like(validation_target)
    test_out = np.zeros_like(test_target)
    selection: dict[str, object] = {}
    names = list(values)
    for finger, finger_name in enumerate(FINGER_NAMES):
        best = (-np.inf, names[0], names[0], 1.0)
        for left_index, left in enumerate(names):
            for right in names[left_index:]:
                for weight in weights:
                    estimate = weight * values[left][0][:, finger] + (1 - weight) * values[right][0][:, finger]
                    candidate = (score(estimate, validation_target[:, finger], args.threshold), left, right, float(weight))
                    if candidate[0] > best[0]:
                        best = candidate
        _, left, right, weight = best
        validation_out[:, finger] = weight * values[left][0][:, finger] + (1 - weight) * values[right][0][:, finger]
        test_out[:, finger] = weight * values[left][1][:, finger] + (1 - weight) * values[right][1][:, finger]
        selection[finger_name] = {"left": left, "right": right, "left_weight": weight, "validation_morphology": best[0]}

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
