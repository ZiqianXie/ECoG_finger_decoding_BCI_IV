#!/usr/bin/env python3
"""Measure how far trained spatial filters moved from their ICA initialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def spatial_change_metrics(
    initial: np.ndarray,
    trained: np.ndarray,
    selected_components: np.ndarray,
) -> dict[str, float | int | list[int]]:
    initial = np.asarray(initial, dtype=np.float64)
    trained = np.asarray(trained, dtype=np.float64)
    selected_components = np.asarray(selected_components, dtype=np.int64)
    if initial.shape != trained.shape or initial.ndim != 2:
        raise ValueError("initial and trained weights must be matching matrices")
    if selected_components.ndim != 1 or selected_components.size == 0:
        raise ValueError("selected_components must be a nonempty vector")

    delta = trained - initial
    initial_row_norm = np.linalg.norm(initial, axis=1)
    trained_row_norm = np.linalg.norm(trained, axis=1)
    denominator = np.maximum(initial_row_norm * trained_row_norm, 1.0e-12)
    cosine = np.sum(initial * trained, axis=1) / denominator
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    selected_relative = (
        np.linalg.norm(delta[selected_components], axis=1)
        / np.maximum(initial_row_norm[selected_components], 1.0e-12)
    )
    return {
        "relative_frobenius_change": float(
            np.linalg.norm(delta) / np.maximum(np.linalg.norm(initial), 1.0e-12)
        ),
        "max_absolute_change": float(np.max(np.abs(delta))),
        "median_row_angle_degrees": float(np.median(angles)),
        "max_row_angle_degrees": float(np.max(angles)),
        "rows_over_1_degree": int(np.sum(angles > 1.0)),
        "rows_over_5_degrees": int(np.sum(angles > 5.0)),
        "median_row_norm_ratio": float(
            np.median(trained_row_norm / np.maximum(initial_row_norm, 1.0e-12))
        ),
        "selected_spatial_components": int(selected_components.size),
        "selected_component_indices": selected_components.tolist(),
        "selected_row_relative_change_median": float(np.median(selected_relative)),
        "selected_row_relative_change_max": float(np.max(selected_relative)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--ica-path", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=range(6))
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="optional nested-CV fold directories below checkpoint-root",
    )
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--features-per-component", type=int, default=11 * 25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    initial = np.load(args.ica_path)
    runs: list[dict[str, object]] = []
    run_locations = (
        [
            (fold, seed, args.checkpoint_root / f"fold{fold}" / f"seed{seed}")
            for fold in args.folds
            for seed in args.seeds
        ]
        if args.folds is not None
        else [(None, seed, args.checkpoint_root / f"seed{seed}") for seed in args.seeds]
    )
    for fold, seed, run in run_locations:
        payload = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
        state = payload["model_state_dict"]
        trained = state["spatial.weight"][:, :, 0].numpy()
        selected = np.asarray(payload["feature_indices"], dtype=np.int64)
        selected_components = np.unique(selected // args.features_per_component)
        summary_path = run / "training_summary.json"
        if not summary_path.exists():
            summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text())
        selected_epoch = int(payload["selected_epoch"])
        run_result: dict[str, object] = {
            "seed": seed,
            "selected_epoch": selected_epoch,
            "warmup_epochs": args.warmup_epochs,
            "unfrozen_epochs": max(0, selected_epoch - args.warmup_epochs),
            "spatial_learning_rate": summary["configuration"]["spatial_learning_rate"],
            **spatial_change_metrics(initial, trained, selected_components),
        }
        if fold is not None:
            run_result["fold"] = fold
        runs.append(run_result)

    frobenius = np.asarray([run["relative_frobenius_change"] for run in runs])
    selected_median = np.asarray(
        [run["selected_row_relative_change_median"] for run in runs]
    )
    max_angles = np.asarray([run["max_row_angle_degrees"] for run in runs])
    unfrozen_epochs = np.asarray([run["unfrozen_epochs"] for run in runs])
    effectively_initialized = bool(
        max_angles.max() < 0.5 and selected_median.max() < 0.005
    )
    result = {
        "checkpoint_root": str(args.checkpoint_root),
        "ica_initialization": str(args.ica_path),
        "definition": "trained spatial convolution minus its FastICA initialization",
        "runs": runs,
        "aggregate": {
            "run_count": len(runs),
            "seed_count": len(set(args.seeds)),
            "runs_with_spatial_updates": int(np.sum(unfrozen_epochs > 0)),
            "unfrozen_epoch_range": [
                int(unfrozen_epochs.min()),
                int(unfrozen_epochs.max()),
            ],
            "relative_frobenius_change_range": [
                float(frobenius.min()),
                float(frobenius.max()),
            ],
            "selected_row_median_relative_change_range": [
                float(selected_median.min()),
                float(selected_median.max()),
            ],
            "maximum_row_angle_degrees": float(max_angles.max()),
            "interpretation": (
                "The spatial layer was technically unfrozen but remained effectively at its "
                "FastICA initialization."
                if effectively_initialized
                else "At least one selected checkpoint moved measurably from its FastICA "
                "initialization."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
