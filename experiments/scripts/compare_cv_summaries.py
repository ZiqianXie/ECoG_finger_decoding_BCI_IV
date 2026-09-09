#!/usr/bin/env python3
"""Render a compact matched comparison between two event-OOF summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecog_decoding.training import FINGER_NAMES


def score_matrix(
    report: dict[str, object], subjects: tuple[int, ...]
) -> np.ndarray:
    return np.asarray(
        [
            [
                report["subjects"][str(subject)]["per_finger"][finger][
                    "ensemble_oof_pcc"
                ]
                for finger in FINGER_NAMES
            ]
            for subject in subjects
        ],
        dtype=np.float64,
    )


def annotate(axis: plt.Axes, values: np.ndarray, digits: int = 3) -> None:
    threshold = float(np.nanmax(np.abs(values))) * 0.55
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            color = "white" if abs(value) >= threshold else "black"
            axis.text(
                column,
                row,
                f"{value:.{digits}f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    baseline_report = json.loads(args.baseline_summary.read_text())
    candidate_report = json.loads(args.candidate_summary.read_text())
    subjects = tuple(
        sorted(
            int(subject)
            for subject in (
                set(baseline_report["subjects"])
                & set(candidate_report["subjects"])
            )
        )
    )
    if not subjects:
        raise RuntimeError("summaries have no subjects in common")
    baseline = score_matrix(baseline_report, subjects)
    candidate = score_matrix(candidate_report, subjects)
    delta = candidate - baseline
    args.output_root.mkdir(parents=True, exist_ok=True)

    limit = max(0.02, float(np.max(np.abs(delta))))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for axis, values, title in (
        (axes[0], baseline, args.baseline_label),
        (axes[1], candidate, args.candidate_label),
    ):
        image = axis.imshow(values, cmap="viridis", vmin=0.25, vmax=0.75)
        annotate(axis, values)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.78, label="event-OOF PCC")
    image = axes[2].imshow(delta, cmap="RdBu_r", vmin=-limit, vmax=limit)
    annotate(axes[2], delta, digits=3)
    axes[2].set_title(f"{args.candidate_label} minus {args.baseline_label}")
    figure.colorbar(image, ax=axes[2], shrink=0.78, label="PCC difference")
    for axis in axes:
        axis.set_xticks(range(5), FINGER_NAMES, rotation=35, ha="right")
        axis.set_yticks(range(len(subjects)), [f"S{subject}" for subject in subjects])
    figure.savefig(args.output_root / "matched_pcc_comparison.png", dpi=180)
    plt.close(figure)

    report = {
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "subjects": list(subjects),
        "baseline": baseline.tolist(),
        "candidate": candidate.tolist(),
        "delta": delta.tolist(),
        "baseline_subject_macro": baseline.mean(axis=1).tolist(),
        "candidate_subject_macro": candidate.mean(axis=1).tolist(),
        "model_count": int(delta.size),
        "mean_delta_models": float(delta.mean()),
        "wins": int(np.sum(delta > 0)),
        "ties": int(np.sum(delta == 0)),
        "losses": int(np.sum(delta < 0)),
        "largest_gain": float(np.max(delta)),
        "largest_loss": float(np.min(delta)),
    }
    (args.output_root / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
