#!/usr/bin/env python3
"""Visualize whether decoder outputs consistently match the wrong glove finger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from ecog_decoding.training import FINGER_NAMES


def correlation_matrix(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = prediction - prediction.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    numerator = prediction.T @ target
    denominator = np.sqrt(
        np.sum(prediction * prediction, axis=0)[:, None]
        * np.sum(target * target, axis=0)[None, :]
    )
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def matrix_dict(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        FINGER_NAMES[row]: {
            FINGER_NAMES[column]: float(matrix[row, column]) for column in range(5)
        }
        for row in range(5)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--target", default="local_w4_q20")
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.prepared_root / f"sub{args.subject}"
    metadata = json.loads((root / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    validation_prediction = np.load(args.prediction_root / "validation_prediction.npy")
    test_prediction = np.load(args.prediction_root / "test_prediction.npy")
    validation_target = np.load(root / f"train_glove_{args.target}.npy")[split:]
    test_target = np.load(root / f"test_glove_{args.target}.npy")[offset:]
    validation = correlation_matrix(validation_prediction, validation_target)
    test = correlation_matrix(test_prediction, test_target)
    predicted_rows, target_columns = linear_sum_assignment(-validation)
    assignment = {
        FINGER_NAMES[row]: FINGER_NAMES[column]
        for row, column in zip(predicted_rows, target_columns, strict=True)
    }
    diagonal_validation = float(np.mean(np.diag(validation)))
    assigned_validation = float(np.mean(validation[predicted_rows, target_columns]))
    diagonal_test = float(np.mean(np.diag(test)))
    assigned_test = float(np.mean(test[predicted_rows, target_columns]))

    report = {
        "subject": args.subject,
        "validation_matrix": matrix_dict(validation),
        "test_matrix": matrix_dict(test),
        "validation_selected_assignment": assignment,
        "validation_diagonal_mean": diagonal_validation,
        "validation_assigned_mean": assigned_validation,
        "test_diagonal_mean": diagonal_test,
        "test_validation_assigned_mean": assigned_test,
        "assignment_gain_validation": assigned_validation - diagonal_validation,
        "assignment_gain_test": assigned_test - diagonal_test,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    for axis, title, matrix in zip(
        axes, ("chronological validation", "held-out test"), (validation, test), strict=True
    ):
        image = axis.imshow(matrix, vmin=-0.2, vmax=0.8, cmap="RdBu_r")
        axis.set_xticks(range(5), FINGER_NAMES, rotation=35, ha="right")
        axis.set_yticks(range(5), FINGER_NAMES)
        axis.set_xlabel("observed glove finger")
        axis.set_ylabel("decoder output")
        axis.set_title(title)
        for row in range(5):
            for column in range(5):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axes, shrink=0.9, label="Pearson correlation")
    figure.suptitle(f"Subject {args.subject}: finger identity diagnostic")
    figure.savefig(args.output / "finger_confusion.png", dpi=160)
    plt.close(figure)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
