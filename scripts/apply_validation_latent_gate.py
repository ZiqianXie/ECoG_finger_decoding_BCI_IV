#!/usr/bin/env python3
"""Apply the latent intended-finger gate to stronger retrospective bases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ecog_decoding.training import FINGER_NAMES
from evaluate_cv_ensemble_final_validation import movement_groups
from fit_oof_latent_movement_gate import (
    activity_emission,
    fit_classifier,
    gated_prediction,
    intended_state,
    state_probabilities,
    temporal_features,
    transition_model,
    viterbi,
)
from summarize_event_lars_lstm_cv import morphology_metrics, pearson
from summarize_frozen_full_refit import PAPER_PCC, plot_subject_overview
from train_event_grouped_lars_lstm_nested import intervals_from_mask


def event_balanced_folds(state: np.ndarray, folds: int = 5) -> list[np.ndarray]:
    """Assign complete latent-state runs across folds, balanced within state."""
    change = np.r_[0, np.flatnonzero(np.diff(state) != 0) + 1, state.size]
    assignments = [np.zeros(state.size, dtype=bool) for _ in range(folds)]
    loads = np.zeros((folds, 6), dtype=np.int64)
    groups: list[tuple[int, int, int]] = []
    for start, stop in zip(change[:-1], change[1:]):
        latent = int(state[start])
        maximum = 50 if latent == 0 else stop - start
        for begin in range(int(start), int(stop), maximum):
            groups.append((begin, min(int(stop), begin + maximum), latent))
    for start, stop, latent in sorted(groups, key=lambda value: value[1] - value[0], reverse=True):
        fold = int(np.argmin(loads[:, latent]))
        assignments[fold][start:stop] = True
        loads[fold, latent] += stop - start
    if not np.all(np.sum(np.stack(assignments), axis=0) == 1):
        raise RuntimeError("event-balanced folds do not partition every row exactly once")
    return assignments


def choose_strengths(
    validation_prediction: np.ndarray,
    raw_target: np.ndarray,
    cleaned_target: np.ndarray,
    threshold: float,
    minimum_gain: float,
) -> tuple[list[float], dict[str, object]]:
    state = intended_state(cleaned_target, threshold)
    features = temporal_features(validation_prediction)
    folds = event_balanced_folds(state)
    strengths = (0.0, 0.25, 0.5, 0.75, 1.0)
    scores = [{strength: [] for strength in strengths} for _ in range(5)]
    accuracies = []
    for validation_mask in folds:
        training_mask = ~validation_mask
        scaler, classifier = fit_classifier(features, state, training_mask)
        probability = state_probabilities(scaler, classifier, features)
        transition, prior = transition_model(state, training_mask)
        activity = activity_emission(state, cleaned_target, training_mask, threshold)
        intervals = intervals_from_mask(validation_mask)
        order_parts = []
        state_parts = []
        for start, stop in intervals:
            order_parts.append(np.arange(start, stop, dtype=np.int64))
            state_parts.append(viterbi(probability[start:stop], transition, prior))
        order = np.concatenate(order_parts)
        decoded = np.concatenate(state_parts)
        accuracies.append(float(np.mean(decoded == state[order])))
        for finger in range(5):
            for strength in strengths:
                estimate = gated_prediction(
                    validation_prediction[order, finger],
                    decoded,
                    activity,
                    finger,
                    strength,
                )
                scores[finger][strength].append(
                    pearson(estimate, raw_target[order, finger])
                )
    selected = []
    audit: dict[str, object] = {
        "mean_latent_state_accuracy": float(np.mean(accuracies)),
        "per_finger": {},
    }
    for finger, name in enumerate(FINGER_NAMES):
        means = {
            strength: float(np.mean(values))
            for strength, values in scores[finger].items()
        }
        best = max(strengths, key=lambda value: means[value])
        if means[best] < means[0.0] + minimum_gain:
            best = 0.0
        selected.append(best)
        audit["per_finger"][name] = {
            "selected_strength": best,
            "mean_fold_pcc_by_strength": {str(key): value for key, value in means.items()},
        }
    return selected, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-map", type=Path, default=Path("configs/retrospective_base_predictions.yaml"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/validation_latent_gate_v1"))
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    parser.add_argument("--minimum-validation-cv-gain", type=float, default=0.005)
    args = parser.parse_args()
    base_map = yaml.safe_load(args.base_map.read_text())
    target_map = yaml.safe_load(args.target_map.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "protocol": "retrospective validation-trained latent gate; released test used only for terminal reporting within this script",
        "released_test_used_for_selection": False,
        "subjects": {},
    }
    for subject in (1, 2, 3):
        base = Path(str(base_map.get(subject, base_map.get(str(subject)))))
        validation = np.load(base / "validation_prediction.npy")
        test = np.load(base / "test_prediction.npy")
        if validation.ndim != 2 or validation.shape[1] != 5 or test.ndim != 2 or test.shape[1] != 5:
            raise ValueError(f"S{subject} base predictions must be [time, 5]")
        prepared = args.prepared_root / f"sub{subject}"
        raw_train = np.load(prepared / "train_glove_25hz_raw.npy")
        raw_validation = raw_train[24:][-validation.shape[0] :]
        raw_test_all = np.load(prepared / "test_glove_25hz_raw.npy")
        raw_test = raw_test_all[24:][: test.shape[0]]
        subject_targets = target_map.get(subject, target_map.get(str(subject)))
        cleaned_validation = np.column_stack([
            np.load(prepared / f"train_glove_{subject_targets[finger]}.npy")[24:][-validation.shape[0] :, index]
            for index, finger in enumerate(FINGER_NAMES)
        ])
        cleaned_test = np.column_stack([
            np.load(prepared / f"test_glove_{str(subject_targets[finger]).removesuffix('_split_safe')}.npy")[24:][: test.shape[0], index]
            for index, finger in enumerate(FINGER_NAMES)
        ])
        strengths, cv_audit = choose_strengths(
            validation,
            raw_validation,
            cleaned_validation,
            args.movement_threshold,
            args.minimum_validation_cv_gain,
        )
        state = intended_state(cleaned_validation, args.movement_threshold)
        mask = np.ones(validation.shape[0], dtype=bool)
        scaler, classifier = fit_classifier(temporal_features(validation), state, mask)
        transition, prior = transition_model(state, mask)
        activity = activity_emission(
            state, cleaned_validation, mask, args.movement_threshold
        )
        probability = state_probabilities(
            scaler, classifier, temporal_features(test)
        )
        decoded = viterbi(probability, transition, prior)
        prediction = np.column_stack([
            gated_prediction(test[:, finger], decoded, activity, finger, strengths[finger])
            for finger in range(5)
        ])
        per_finger = {}
        for finger, name in enumerate(FINGER_NAMES):
            groups = movement_groups(cleaned_test[:, finger], args.movement_threshold)
            metrics = morphology_metrics(prediction[:, finger], cleaned_test[:, finger], groups)
            metrics["raw_pcc"] = pearson(prediction[:, finger], raw_test[:, finger])
            baseline = pearson(test[:, finger], raw_test[:, finger])
            per_finger[name] = {
                "selected_gate_strength": strengths[finger],
                "baseline_test_raw_pcc": baseline,
                "metrics": metrics,
                "test_raw_pcc_delta": metrics["raw_pcc"] - baseline,
                "paper_raw_pcc": PAPER_PCC[subject][finger],
            }
        subject_output = args.output_root / f"sub{subject}"
        subject_output.mkdir(parents=True, exist_ok=True)
        np.save(subject_output / "test_prediction.npy", prediction)
        np.save(subject_output / "test_latent_state.npy", decoded)
        plot_subject_overview(
            subject_output / "all_fingers_full_trajectory.png",
            cleaned_test,
            prediction,
            subject,
        )
        scores = [per_finger[name]["metrics"]["raw_pcc"] for name in FINGER_NAMES]
        baseline_scores = [per_finger[name]["baseline_test_raw_pcc"] for name in FINGER_NAMES]
        report["subjects"][str(subject)] = {
            "base": str(base),
            "cross_validation_audit": cv_audit,
            "per_finger": per_finger,
            "raw_pcc": scores,
            "baseline_raw_pcc": baseline_scores,
            "macro5_raw_pcc": float(np.mean(scores)),
            "baseline_macro5_raw_pcc": float(np.mean(baseline_scores)),
        }
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
