#!/usr/bin/env python3
"""Fit an OOF-only joint decoder correction for cross-finger interference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ecog_decoding.training import FINGER_NAMES
from evaluate_cv_ensemble_final_validation import movement_groups
from refit_frozen_event_model import resolve_options
from summarize_event_lars_lstm_cv import morphology_metrics, pearson
from summarize_frozen_full_refit import PAPER_PCC, plot_subject_overview
from train_event_grouped_lars_lstm import indices_from_intervals


def fit_diagonal_prior_ridge(
    predictors: np.ndarray,
    target: np.ndarray,
    alpha: float,
    finger: int,
) -> dict[str, np.ndarray | float]:
    mean = predictors.mean(axis=0)
    scale = np.maximum(predictors.std(axis=0), 1.0e-8)
    target_mean = float(target.mean())
    target_scale = max(float(target.std()), 1.0e-8)
    x = (predictors - mean) / scale
    y = (target - target_mean) / target_scale
    prior = np.zeros(predictors.shape[1], dtype=np.float64)
    prior[finger] = 1.0
    coefficients = np.linalg.solve(
        x.T @ x + alpha * np.eye(x.shape[1]),
        x.T @ y + alpha * prior,
    )
    return {
        "predictor_mean": mean,
        "predictor_scale": scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "standardized_coefficients": coefficients,
    }


def apply_model(model: dict[str, np.ndarray | float], predictors: np.ndarray) -> np.ndarray:
    standardized = (
        predictors - np.asarray(model["predictor_mean"])
    ) / np.asarray(model["predictor_scale"])
    return (
        standardized @ np.asarray(model["standardized_coefficients"])
        * float(model["target_scale"])
        + float(model["target_mean"])
    )


def smooth_nonnegative(values: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """Smooth ReLU with exactly nonnegative output and a small transition width."""
    return 0.5 * (values + np.sqrt(np.square(values) + epsilon**2))


def load_oof_predictors(
    ensemble_map: dict[str, object], subject: int, row_count: int
) -> np.ndarray:
    predictors = np.full((row_count, 5), np.nan, dtype=np.float64)
    for finger_index, finger in enumerate(FINGER_NAMES):
        options = resolve_options(ensemble_map, subject, finger)
        for fold in range(3):
            member_predictions = []
            reference_indices = None
            for seed in options["seeds"]:
                root = options["input_root"] / f"sub{subject}" / finger / f"fold{fold}" / f"seed{seed}"
                indices = np.load(root / "validation_indices.npy")
                if reference_indices is None:
                    reference_indices = indices
                elif not np.array_equal(indices, reference_indices):
                    raise RuntimeError(f"seed OOF indices differ for S{subject} {finger} fold {fold}")
                member_predictions.append(np.load(root / "validation_prediction.npy"))
            predictors[reference_indices, finger_index] = np.mean(member_predictions, axis=0)
    if not np.isfinite(predictors).all():
        raise RuntimeError(f"incomplete OOF predictor matrix for S{subject}")
    return predictors


def choose_alpha(
    predictors: np.ndarray,
    target: np.ndarray,
    definition: dict[str, object],
    finger: int,
    alphas: tuple[float, ...],
    minimum_gain: float,
) -> tuple[float | None, list[dict[str, float]]]:
    candidates: list[dict[str, float]] = []
    self_scores = []
    ridge_scores = {alpha: [] for alpha in alphas}
    for fold in definition["folds"]:
        validation = indices_from_intervals(fold["validation_intervals"])
        training_mask = np.ones(predictors.shape[0], dtype=bool)
        training_mask[validation] = False
        self_scores.append(pearson(predictors[validation, finger], target[validation]))
        for alpha in alphas:
            model = fit_diagonal_prior_ridge(
                predictors[training_mask], target[training_mask], alpha, finger
            )
            estimate = apply_model(model, predictors[validation])
            ridge_scores[alpha].append(pearson(estimate, target[validation]))
    self_mean = float(np.mean(self_scores))
    candidates.append({"alpha": -1.0, "mean_fold_pcc": self_mean})
    for alpha in alphas:
        candidates.append({"alpha": alpha, "mean_fold_pcc": float(np.mean(ridge_scores[alpha]))})
    best = max(candidates[1:], key=lambda item: item["mean_fold_pcc"])
    if best["mean_fold_pcc"] < self_mean + minimum_gain:
        return None, candidates
    return float(best["alpha"]), candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-map", type=Path, default=Path("configs/final_event_ensemble.yaml"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--fold-root", type=Path, default=Path("outputs/event_stratified_folds_targetsafe_conservative_v1"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--full-refit-root", type=Path, default=Path("outputs/frozen_event_full_refit_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/oof_cross_finger_deconfounder_v1"))
    parser.add_argument("--minimum-oof-gain", type=float, default=0.005)
    args = parser.parse_args()
    ensemble_map = yaml.safe_load(args.ensemble_map.read_text())
    target_map = yaml.safe_load(args.target_map.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "protocol": "post-test diagnostic method; correction and regularization fitted using training OOF rows only",
        "released_test_used_for_selection": False,
        "minimum_oof_gain_for_cross_finger_model": args.minimum_oof_gain,
        "subjects": {},
    }
    for subject in (1, 2, 3):
        first_definition = json.loads(
            (args.fold_root / f"sub{subject}" / "thumb" / "folds.json").read_text()
        )
        row_count = int(first_definition["training_rows"])
        oof = load_oof_predictors(ensemble_map, subject, row_count)
        test = np.column_stack([
            np.load(args.full_refit_root / "ensemble" / f"sub{subject}" / f"{finger}_released_test_prediction.npy")
            for finger in FINGER_NAMES
        ])
        prepared = args.prepared_root / f"sub{subject}"
        raw_train = np.load(prepared / "train_glove_25hz_raw.npy")[24 : 24 + row_count]
        raw_test = np.load(prepared / "test_glove_25hz_raw.npy")[24 : 24 + test.shape[0]]
        subject_targets = target_map.get(subject, target_map.get(str(subject)))
        cleaned_train = np.column_stack([
            np.load(prepared / f"train_glove_{str(subject_targets[finger])}.npy")[24 : 24 + row_count, index]
            for index, finger in enumerate(FINGER_NAMES)
        ])
        cleaned_test = np.column_stack([
            np.load(prepared / f"test_glove_{str(subject_targets[finger]).removesuffix('_split_safe')}.npy")[24 : 24 + test.shape[0], index]
            for index, finger in enumerate(FINGER_NAMES)
        ])
        corrected_raw = np.empty_like(test)
        corrected_cleaned = np.empty_like(test)
        per_finger: dict[str, object] = {}
        alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
        for finger_index, finger in enumerate(FINGER_NAMES):
            definition = json.loads(
                (args.fold_root / f"sub{subject}" / finger / "folds.json").read_text()
            )
            alpha, candidates = choose_alpha(
                oof,
                raw_train[:, finger_index],
                definition,
                finger_index,
                alphas,
                args.minimum_oof_gain,
            )
            if alpha is None:
                raw_prediction = test[:, finger_index]
                cleaned_prediction = smooth_nonnegative(test[:, finger_index])
                coefficients = np.eye(5)[finger_index]
                method = "self_decoder_only"
            else:
                raw_model = fit_diagonal_prior_ridge(
                    oof, raw_train[:, finger_index], alpha, finger_index
                )
                cleaned_model = fit_diagonal_prior_ridge(
                    oof, cleaned_train[:, finger_index], alpha, finger_index
                )
                raw_prediction = apply_model(raw_model, test)
                cleaned_prediction = smooth_nonnegative(apply_model(cleaned_model, test))
                coefficients = np.asarray(raw_model["standardized_coefficients"])
                method = "diagonal_prior_cross_finger_ridge"
            corrected_raw[:, finger_index] = raw_prediction
            corrected_cleaned[:, finger_index] = cleaned_prediction
            groups = movement_groups(cleaned_test[:, finger_index], 0.08)
            metrics = morphology_metrics(
                cleaned_prediction, cleaned_test[:, finger_index], groups
            )
            metrics["raw_pcc"] = pearson(raw_prediction, raw_test[:, finger_index])
            baseline_pcc = pearson(test[:, finger_index], raw_test[:, finger_index])
            per_finger[finger] = {
                "method": method,
                "selected_alpha": alpha,
                "oof_candidates": candidates,
                "standardized_raw_coefficients": coefficients.tolist(),
                "baseline_test_raw_pcc": baseline_pcc,
                "metrics": metrics,
                "test_raw_pcc_delta": metrics["raw_pcc"] - baseline_pcc,
                "paper_raw_pcc": PAPER_PCC[subject][finger_index],
            }
        subject_output = args.output_root / f"sub{subject}"
        subject_output.mkdir(parents=True, exist_ok=True)
        np.save(subject_output / "released_test_raw_coordinate.npy", corrected_raw)
        np.save(subject_output / "released_test_cleaned.npy", corrected_cleaned)
        plot_subject_overview(
            subject_output / "all_fingers_full_trajectory.png",
            cleaned_test,
            corrected_cleaned,
            subject,
        )
        scores = [per_finger[finger]["metrics"]["raw_pcc"] for finger in FINGER_NAMES]
        baseline_scores = [per_finger[finger]["baseline_test_raw_pcc"] for finger in FINGER_NAMES]
        report["subjects"][str(subject)] = {
            "per_finger": per_finger,
            "raw_pcc": scores,
            "baseline_raw_pcc": baseline_scores,
            "macro5_raw_pcc": float(np.mean(scores)),
            "baseline_macro5_raw_pcc": float(np.mean(baseline_scores)),
            "paper_macro5_raw_pcc": float(np.mean(PAPER_PCC[subject])),
        }
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
