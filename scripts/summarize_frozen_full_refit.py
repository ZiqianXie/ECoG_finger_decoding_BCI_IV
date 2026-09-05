#!/usr/bin/env python3
"""Aggregate the frozen full-data seed ensembles and diagnose released-test traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from ecog_decoding.training import FINGER_NAMES
from evaluate_cv_ensemble_final_validation import movement_groups
from refit_frozen_event_model import plot_events, plot_full_trajectory, resolve_options
from summarize_event_lars_lstm_cv import morphology_metrics, pearson


PAPER_PCC = {
    1: (0.750, 0.790, 0.170, 0.600, 0.470),
    2: (0.620, 0.380, 0.270, 0.470, 0.300),
    3: (0.740, 0.550, 0.460, 0.410, 0.750),
}


def plot_subject_overview(
    path: Path,
    cleaned: np.ndarray,
    prediction: np.ndarray,
    subject: int,
) -> None:
    time_axis = np.arange(cleaned.shape[0]) / 25.0
    figure, axes = plt.subplots(5, 1, figsize=(16, 11), sharex=True)
    for finger, name, axis in zip(range(5), FINGER_NAMES, axes):
        axis.plot(time_axis, cleaned[:, finger], color="black", linewidth=0.55, label="cleaned target")
        axis.plot(time_axis, prediction[:, finger], color="#2563eb", linewidth=0.55, label="prediction")
        axis.set_ylabel(name)
        axis.axhline(0.0, color="#cbd5e1", linewidth=0.4)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("released-test time (s)")
    figure.suptitle(f"Subject {subject}: full-development six-seed ensemble")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def cross_finger_matrix(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [pearson(prediction[:, decoded], target[:, observed]) for observed in range(5)]
            for decoded in range(5)
        ],
        dtype=np.float64,
    )


def standardized_attribution(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Unique standardized target weights explaining each decoded output."""
    target_z = (target - target.mean(axis=0)) / np.maximum(target.std(axis=0), 1.0e-8)
    prediction_z = (prediction - prediction.mean(axis=0)) / np.maximum(
        prediction.std(axis=0), 1.0e-8
    )
    gram = target_z.T @ target_z
    coefficients = np.linalg.solve(
        gram + 1.0e-3 * np.eye(gram.shape[0]), target_z.T @ prediction_z
    )
    return coefficients.T


def best_lag(prediction: np.ndarray, target: np.ndarray, maximum: int = 25) -> dict[str, float]:
    records = []
    for lag in range(-maximum, maximum + 1):
        if lag < 0:
            score = pearson(prediction[-lag:], target[:lag])
        elif lag > 0:
            score = pearson(prediction[:-lag], target[lag:])
        else:
            score = pearson(prediction, target)
        records.append((score, lag))
    score, lag = max(records)
    return {"lag_bins": int(lag), "lag_seconds": lag / 25.0, "pcc": float(score)}


def plot_cross_finger_matrices(
    path: Path,
    raw: np.ndarray,
    cleaned: np.ndarray,
    target_cross: np.ndarray,
    subject: int,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for axis, matrix, title in zip(
        axes,
        (raw, cleaned, target_cross),
        ("prediction vs raw glove", "prediction vs cleaned target", "cleaned target vs target"),
    ):
        image = axis.imshow(matrix, vmin=-0.2, vmax=0.8, cmap="coolwarm")
        axis.set_xticks(range(5), FINGER_NAMES, rotation=35, ha="right")
        axis.set_yticks(range(5), FINGER_NAMES)
        axis.set_xlabel("observed finger")
        axis.set_ylabel("decoded model")
        axis.set_title(title)
        for row in range(5):
            for column in range(5):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle(f"Subject {subject}: cross-finger released-test correlation")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs/frozen_event_full_refit_v1"))
    parser.add_argument("--ensemble-map", type=Path, default=Path("configs/final_event_ensemble.yaml"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/frozen_event_full_refit_v1/ensemble"))
    args = parser.parse_args()

    ensemble_map = yaml.safe_load(args.ensemble_map.read_text())
    target_map = yaml.safe_load(args.target_map.read_text())
    report: dict[str, object] = {
        "protocol": "OOF-selected per-finger six-seed ensemble after full-development refit",
        "official_final_validation_incorporated_into_training": True,
        "released_test_touched": True,
        "released_test_role": "retrospective paper-comparison evaluation",
        "released_test_used_for_current_pipeline_selection": False,
        "released_test_previously_inspected_during_reconstruction": True,
        "subjects": {},
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    for subject in (1, 2, 3):
        prepared = args.prepared_root / f"sub{subject}"
        raw_all = np.load(prepared / "test_glove_25hz_raw.npy")[24:]
        subject_targets = target_map.get(subject, target_map.get(str(subject)))
        predictions = np.empty_like(raw_all, dtype=np.float32)
        cleaned_all = np.empty_like(raw_all, dtype=np.float32)
        per_finger: dict[str, object] = {}
        subject_output = args.output_root / f"sub{subject}"
        subject_output.mkdir(parents=True, exist_ok=True)
        for finger_index, finger in enumerate(FINGER_NAMES):
            options = resolve_options(ensemble_map, subject, finger)
            seeds = options["seeds"]
            members = []
            member_reports = []
            for seed in seeds:
                root = args.input_root / f"sub{subject}" / finger / f"seed{seed}"
                member = np.load(root / "released_test_prediction.npy")
                members.append(member)
                summary = json.loads((root / "summary.json").read_text())
                member_reports.append({
                    "seed": seed,
                    "selected_epoch": summary["selected_epoch"],
                    "raw_pcc": summary["released_test_metrics"]["raw_pcc"],
                    "cleaned_pcc": summary["released_test_metrics"]["cleaned_pcc"],
                    "prediction_sd": float(np.std(member)),
                })
            stacked = np.stack(members)
            prediction = np.mean(stacked, axis=0)
            target_policy = str(subject_targets[finger]).removesuffix("_split_safe")
            cleaned = np.load(prepared / f"test_glove_{target_policy}.npy")[24:, finger_index]
            raw = raw_all[:, finger_index]
            groups = movement_groups(cleaned, 0.08)
            metrics = morphology_metrics(prediction, cleaned, groups)
            metrics["raw_pcc"] = pearson(prediction, raw)
            diversity = [
                pearson(stacked[first], stacked[second])
                for first in range(stacked.shape[0])
                for second in range(first + 1, stacked.shape[0])
            ]
            member_raw = [float(item["raw_pcc"]) for item in member_reports]
            paper_value = PAPER_PCC[subject][finger_index]
            per_finger[finger] = {
                "target_policy": target_policy,
                "seeds": list(seeds),
                "metrics": metrics,
                "members": member_reports,
                "mean_pairwise_seed_prediction_pcc": float(np.mean(diversity)) if diversity else None,
                "ensemble_gain_over_best_seed_raw_pcc": metrics["raw_pcc"] - max(member_raw),
                "paper_raw_pcc": paper_value,
                "delta_vs_paper": metrics["raw_pcc"] - paper_value,
                "beats_paper": bool(metrics["raw_pcc"] > paper_value),
            }
            predictions[:, finger_index] = prediction
            cleaned_all[:, finger_index] = cleaned
            np.save(subject_output / f"{finger}_released_test_prediction.npy", prediction)
            plot_full_trajectory(
                subject_output / f"{finger}_full_trajectory.png",
                raw,
                cleaned,
                prediction,
                f"S{subject} {finger}: full-development six-seed ensemble",
            )
            plot_events(
                subject_output / f"{finger}_strongest_events.png",
                raw,
                cleaned,
                prediction,
                f"S{subject} {finger}: strongest released-test events",
            )
        raw_scores = [per_finger[name]["metrics"]["raw_pcc"] for name in FINGER_NAMES]
        paper_scores = list(PAPER_PCC[subject])
        subject_report = {
            "per_finger": per_finger,
            "raw_pcc": raw_scores,
            "macro5_raw_pcc": float(np.mean(raw_scores)),
            "paper_raw_pcc": paper_scores,
            "paper_macro5_raw_pcc": float(np.mean(paper_scores)),
            "macro5_delta_vs_paper": float(np.mean(raw_scores) - np.mean(paper_scores)),
            "fingers_beating_paper": int(sum(score > paper for score, paper in zip(raw_scores, paper_scores))),
        }
        raw_cross = cross_finger_matrix(predictions, raw_all)
        cleaned_cross = cross_finger_matrix(predictions, cleaned_all)
        target_cross = cross_finger_matrix(cleaned_all, cleaned_all)
        attribution = standardized_attribution(predictions, cleaned_all)
        subject_report["cross_finger_raw_pcc"] = raw_cross.tolist()
        subject_report["cross_finger_cleaned_pcc"] = cleaned_cross.tolist()
        subject_report["cleaned_target_cross_finger_pcc"] = target_cross.tolist()
        subject_report["standardized_unique_target_attribution"] = attribution.tolist()
        subject_report["strongest_unique_target_attribution"] = {
            FINGER_NAMES[row]: FINGER_NAMES[int(np.argmax(attribution[row]))]
            for row in range(5)
        }
        subject_report["best_matching_raw_finger"] = {
            FINGER_NAMES[row]: FINGER_NAMES[int(np.argmax(raw_cross[row]))]
            for row in range(5)
        }
        subject_report["diagonal_margin_over_best_off_diagonal"] = {
            FINGER_NAMES[row]: float(
                raw_cross[row, row] - np.max(np.delete(raw_cross[row], row))
            )
            for row in range(5)
        }
        subject_report["best_diagonal_lag"] = {
            FINGER_NAMES[finger]: best_lag(
                predictions[:, finger], raw_all[:, finger]
            )
            for finger in range(5)
        }
        report["subjects"][str(subject)] = subject_report
        plot_subject_overview(
            subject_output / "all_fingers_full_trajectory.png",
            cleaned_all,
            predictions,
            subject,
        )
        plot_cross_finger_matrices(
            subject_output / "cross_finger_correlation.png",
            raw_cross,
            cleaned_cross,
            target_cross,
            subject,
        )
    all_deltas = [
        report["subjects"][str(subject)]["per_finger"][finger]["delta_vs_paper"]
        for subject in (1, 2, 3)
        for finger in FINGER_NAMES
    ]
    report["overall"] = {
        "finger_models_beating_paper": int(sum(delta > 0 for delta in all_deltas)),
        "finger_models_total": len(all_deltas),
        "mean_delta_vs_paper": float(np.mean(all_deltas)),
    }
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
