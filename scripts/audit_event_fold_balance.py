#!/usr/bin/env python3
"""Visualize per-finger event-fold balance before model comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecog_decoding.training import FINGER_NAMES


METRICS = ("seconds", "moving_fraction", "movement_mass", "mean_active")


def coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    return float(np.std(values, ddof=0) / mean) if mean > 0 else 0.0


def fold_values(definition: dict[str, object], finger: str) -> dict[str, list[float]]:
    result = {metric: [] for metric in METRICS}
    for fold in definition["folds"]:
        summary = fold["validation_summary"]
        finger_summary = summary["per_finger"][finger]
        result["seconds"].append(float(summary["seconds"]))
        for metric in METRICS[1:]:
            result[metric].append(float(finger_summary[metric]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-root", type=Path, default=Path("outputs/event_stratified_folds_v1")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/event_fold_balance_v1")
    )
    args = parser.parse_args()

    cv = np.empty((3, 5, len(METRICS)), dtype=np.float64)
    report: dict[str, object] = {"subjects": {}}
    for subject in (1, 2, 3):
        subject_report: dict[str, object] = {}
        for finger_index, finger in enumerate(FINGER_NAMES):
            definition = json.loads(
                (args.fold_root / f"sub{subject}" / finger / "folds.json").read_text()
            )
            values = fold_values(definition, finger)
            metric_cv = {
                metric: coefficient_of_variation(np.asarray(metric_values))
                for metric, metric_values in values.items()
            }
            for metric_index, metric in enumerate(METRICS):
                cv[subject - 1, finger_index, metric_index] = metric_cv[metric]
            subject_report[finger] = {
                "fold_values": values,
                "coefficient_of_variation": metric_cv,
                "group_count": int(definition["group_count"]),
                "assignment_objective": float(definition["assignment_objective"]),
            }
        report["subjects"][str(subject)] = subject_report

    args.output_root.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(METRICS), figsize=(17, 4.2), constrained_layout=True)
    maximum = max(0.25, float(np.max(cv)))
    for metric_index, (axis, metric) in enumerate(zip(axes, METRICS)):
        values = cv[:, :, metric_index]
        image = axis.imshow(values, cmap="magma", vmin=0.0, vmax=maximum)
        for row in range(3):
            for column in range(5):
                value = values[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.5 * maximum else "black",
                    fontsize=8,
                )
        axis.set_title(f"fold CV: {metric.replace('_', ' ')}")
        axis.set_xticks(range(5), FINGER_NAMES, rotation=35, ha="right")
        axis.set_yticks(range(3), ("S1", "S2", "S3"))
        figure.colorbar(image, ax=axis, shrink=0.78)
    figure.savefig(args.output_root / "fold_balance_cv.png", dpi=180)
    plt.close(figure)

    report["summary"] = {
        "mean_cv_by_metric": {
            metric: float(np.mean(cv[:, :, index]))
            for index, metric in enumerate(METRICS)
        },
        "maximum_cv_by_metric": {
            metric: float(np.max(cv[:, :, index]))
            for index, metric in enumerate(METRICS)
        },
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
