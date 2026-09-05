#!/usr/bin/env python3
"""Screen paper-style cleaning for the little target on development folds only.

The paper baseline and five-finger winner mask are used only to construct the
little-finger training target.  Thumb through ring remain untouched.  Every
decoder is fitted on purged training intervals and scored on held-out complete
development events against the original glove trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from audit_little_finger_training_only import (
    FINGERS,
    LITTLE,
    interval_indices,
    load_targets,
    pearson,
    ridge_gcv_multioutput,
)


def paper_little_target(values: np.ndarray, threshold: float) -> np.ndarray:
    thresholded = np.asarray(values, dtype=np.float64).copy()
    thresholded[thresholded < threshold] = 0.0
    winner = np.argmax(thresholded, axis=1)
    little = thresholded[:, LITTLE].copy()
    little[winner != LITTLE] = 0.0
    return little


def subject_probe(subject: int, args: argparse.Namespace, target_map: dict) -> dict:
    definition = json.loads(
        (args.fold_root / f"sub{subject}" / "little" / "folds.json").read_text()
    )
    rows = int(definition["training_rows"])
    offset = int(definition["history_offset"])
    current = load_targets(
        args.prepared_root, target_map, subject, rows, offset
    )[:, LITTLE]
    paper_path = (
        args.prepared_root / f"sub{subject}" / "train_glove_paper_baseline_only.npy"
    )
    if "test" in paper_path.name.lower():
        raise RuntimeError(f"refusing non-development target: {paper_path}")
    paper = np.asarray(
        np.load(paper_path, mmap_mode="r")[offset : offset + rows],
        dtype=np.float64,
    )
    candidate_targets = {
        "current_local": current,
        "paper_baseline_no_wta": paper[:, LITTLE],
    }
    for threshold in args.thresholds:
        candidate_targets[f"paper_little_wta_{threshold:g}"] = paper_little_target(
            paper, threshold
        )
    names = list(candidate_targets)
    predictions = {
        name: np.full(rows, np.nan, dtype=np.float64) for name in names
    }
    raw_target = np.full(rows, np.nan, dtype=np.float64)
    fold_metrics = {name: [] for name in names}
    fold_details = []
    features = np.load(
        args.feature_root / f"sub{subject}" / "train_initialized_window_features.npy",
        mmap_mode="r",
    )
    for fold in definition["folds"]:
        fold_number = int(fold["fold"])
        training = interval_indices(fold["training_intervals_after_purge"])
        validation = interval_indices(fold["validation_intervals"])
        checkpoint_root = (
            args.input_root
            / f"sub{subject}"
            / "little"
            / f"fold{fold_number}"
            / "seed0"
        )
        checkpoint = torch.load(
            checkpoint_root / "model.pt",
            map_location="cpu",
            weights_only=False,
        )
        selected = np.asarray(checkpoint["feature_indices"], dtype=np.int64)
        selected_features = np.asarray(features[:, selected], dtype=np.float64)
        train_x = selected_features[training]
        validation_x = selected_features[validation]
        mean = np.mean(train_x, axis=0)
        scale = np.std(train_x, axis=0)
        scale[scale < 1.0e-8] = 1.0
        train_x = (train_x - mean) / scale
        validation_x = (validation_x - mean) / scale
        all_targets = np.column_stack([candidate_targets[name] for name in names])
        estimate, alphas = ridge_gcv_multioutput(
            train_x,
            all_targets[training],
            validation_x,
            np.logspace(-3, 5, 17),
        )
        fold_raw = np.load(checkpoint_root / "validation_raw_target.npy")
        raw_target[validation] = fold_raw
        for index, name in enumerate(names):
            predictions[name][validation] = estimate[:, index]
            fold_metrics[name].append(
                {
                    "fold": fold_number,
                    "alpha": float(alphas[index]),
                    "raw_glove_pcc": pearson(estimate[:, index], fold_raw),
                    "candidate_target_pcc": pearson(
                        estimate[:, index], all_targets[validation, index]
                    ),
                }
            )
        fold_details.append(
            {"fold": fold_number, "selected_feature_count": int(selected.size)}
        )
    if not np.isfinite(raw_target).all():
        raise RuntimeError(f"incomplete raw OOF target for S{subject}")
    results = {}
    for name in names:
        if not np.isfinite(predictions[name]).all():
            raise RuntimeError(f"incomplete OOF prediction for S{subject} {name}")
        results[name] = {
            "fold_metrics": fold_metrics[name],
            "mean_fold_raw_glove_pcc": float(
                np.mean([item["raw_glove_pcc"] for item in fold_metrics[name]])
            ),
            "stitched_raw_glove_pcc": pearson(predictions[name], raw_target),
            "active_fraction": float(
                np.mean(candidate_targets[name] >= args.movement_threshold)
            ),
            "target_mass_fraction_of_current": float(
                np.sum(candidate_targets[name]) / max(float(np.sum(current)), 1.0e-12)
            ),
        }
    ranking = sorted(
        results, key=lambda name: results[name]["mean_fold_raw_glove_pcc"], reverse=True
    )
    return {
        "subject": subject,
        "target_scope": "little only; thumb through ring unchanged",
        "folds": fold_details,
        "ranking": ranking,
        "results": results,
    }


def make_figure(subjects: dict[str, dict], output: Path) -> None:
    subject_ids = sorted(subjects, key=int)
    figure, axes = plt.subplots(
        1, len(subject_ids), figsize=(4.7 * len(subject_ids), 4.2), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for axis, subject_key in zip(axes, subject_ids):
        subject = int(subject_key)
        results = subjects[subject_key]["results"]
        names = [name for name in results if name.startswith("paper_little_wta_")]
        thresholds = [float(name.rsplit("_", 1)[1]) for name in names]
        scores = [results[name]["mean_fold_raw_glove_pcc"] for name in names]
        axis.plot(thresholds, scores, marker="o", label="paper baseline + little WTA")
        axis.axhline(
            results["current_local"]["mean_fold_raw_glove_pcc"],
            color="#2563eb",
            linestyle="--",
            label="current local target",
        )
        axis.axhline(
            results["paper_baseline_no_wta"]["mean_fold_raw_glove_pcc"],
            color="#16a34a",
            linestyle=":",
            label="paper baseline, no WTA",
        )
        axis.set_title(f"S{subject} little")
        axis.set_xlabel("small-movement threshold")
        axis.set_ylabel("held-out raw-glove PCC")
        if subject == 1:
            axis.legend(fontsize=8)
    figure.suptitle(
        "Development-only paper-style little-finger target screen", fontsize=14
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=(0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25),
    )
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    parser.add_argument(
        "--input-root", type=Path, default=Path("outputs/event_lars_e2e_fulldev_seq50_v1")
    )
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path("outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1"),
    )
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("outputs/windowed_ica_wavelet_asymmetric_v1"),
    )
    parser.add_argument(
        "--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/results/little-paper-target-oof.json")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("docs/figures/little-paper-target-oof.png")
    )
    args = parser.parse_args()
    target_map = yaml.safe_load(args.target_map.read_text())
    subjects = {
        str(subject): subject_probe(subject, args, target_map)
        for subject in args.subjects
    }
    report = {
        "protocol": "purged event OOF paper-style little-target screen",
        "released_test_loaded": False,
        "selection_warning": (
            "The threshold sweep is a development screen. A threshold must be selected "
            "inside nested folds before its OOF score is treated as unbiased."
        ),
        "subjects": subjects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    make_figure(subjects, args.figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
