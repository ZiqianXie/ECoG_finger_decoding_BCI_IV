#!/usr/bin/env python3
"""Fit an OOF-only latent intended-finger gate with co-movement emissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ecog_decoding.training import FINGER_NAMES
from evaluate_cv_ensemble_final_validation import movement_groups
from fit_oof_cross_finger_deconfounder import load_oof_predictors
from summarize_event_lars_lstm_cv import morphology_metrics, pearson
from summarize_frozen_full_refit import PAPER_PCC, plot_subject_overview
from train_event_grouped_lars_lstm import indices_from_intervals


def intended_state(target: np.ndarray, threshold: float) -> np.ndarray:
    state = np.zeros(target.shape[0], dtype=np.int64)
    active = np.max(target, axis=1) >= threshold
    state[active] = np.argmax(target[active], axis=1) + 1
    return state


def temporal_features(prediction: np.ndarray) -> np.ndarray:
    velocity = np.diff(prediction, axis=0, prepend=prediction[[0]])
    return np.concatenate((prediction, velocity), axis=1)


def transition_model(state: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.ones((6, 6), dtype=np.float64)
    adjacent = mask[:-1] & mask[1:]
    np.add.at(counts, (state[:-1][adjacent], state[1:][adjacent]), 1.0)
    transition = counts / counts.sum(axis=1, keepdims=True)
    prior = np.bincount(state[mask], minlength=6).astype(np.float64) + 1.0
    prior /= prior.sum()
    return transition, prior


def activity_emission(
    state: np.ndarray, target: np.ndarray, mask: np.ndarray, threshold: float
) -> np.ndarray:
    emission = np.empty((6, 5), dtype=np.float64)
    for latent in range(6):
        selected = mask & (state == latent)
        if not np.any(selected):
            emission[latent] = 0.0
        else:
            active = target[selected] >= threshold
            emission[latent] = (active.sum(axis=0) + 1.0) / (active.shape[0] + 2.0)
    return emission


def fit_classifier(
    features: np.ndarray, state: np.ndarray, mask: np.ndarray
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(features[mask])
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=0,
    ).fit(scaler.transform(features[mask]), state[mask])
    return scaler, classifier


def state_probabilities(
    scaler: StandardScaler, classifier: LogisticRegression, features: np.ndarray
) -> np.ndarray:
    partial = classifier.predict_proba(scaler.transform(features))
    probabilities = np.full((features.shape[0], 6), 1.0e-8, dtype=np.float64)
    probabilities[:, classifier.classes_.astype(int)] = partial
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def viterbi(probability: np.ndarray, transition: np.ndarray, prior: np.ndarray) -> np.ndarray:
    log_emission = np.log(np.maximum(probability, 1.0e-12))
    log_transition = np.log(np.maximum(transition, 1.0e-12))
    score = np.empty_like(log_emission)
    back = np.empty_like(log_emission, dtype=np.int64)
    score[0] = np.log(np.maximum(prior, 1.0e-12)) + log_emission[0]
    back[0] = 0
    for step in range(1, probability.shape[0]):
        candidate = score[step - 1][:, None] + log_transition
        back[step] = np.argmax(candidate, axis=0)
        score[step] = candidate[back[step], np.arange(6)] + log_emission[step]
    state = np.empty(probability.shape[0], dtype=np.int64)
    state[-1] = int(np.argmax(score[-1]))
    for step in range(probability.shape[0] - 1, 0, -1):
        state[step - 1] = back[step, state[step]]
    return state


def decode_intervals(
    probability: np.ndarray,
    intervals: list[list[int]],
    transition: np.ndarray,
    prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = []
    states = []
    for start, stop in intervals:
        indices.append(np.arange(start, stop, dtype=np.int64))
        states.append(viterbi(probability[start:stop], transition, prior))
    return np.concatenate(indices), np.concatenate(states)


def gated_prediction(
    prediction: np.ndarray,
    latent_state: np.ndarray,
    activity: np.ndarray,
    finger: int,
    strength: float,
) -> np.ndarray:
    gate = activity[latent_state, finger]
    factor = np.maximum((1.0 - strength) + strength * gate, 0.0)
    return prediction * factor


def choose_strength(
    predictors: np.ndarray,
    target_raw: np.ndarray,
    target_cleaned: np.ndarray,
    state: np.ndarray,
    definition: dict[str, object],
    finger: int,
    threshold: float,
    minimum_gain: float,
) -> tuple[float, dict[str, object]]:
    features = temporal_features(predictors)
    strengths = (0.0, 0.25, 0.5, 0.75, 1.0)
    scores = {strength: [] for strength in strengths}
    state_accuracy = []
    for fold in definition["folds"]:
        intervals = fold["validation_intervals"]
        validation = indices_from_intervals(intervals)
        mask = np.ones(predictors.shape[0], dtype=bool)
        mask[validation] = False
        scaler, classifier = fit_classifier(features, state, mask)
        probability = state_probabilities(scaler, classifier, features)
        transition, prior = transition_model(state, mask)
        activity = activity_emission(state, target_cleaned, mask, threshold)
        order, decoded_state = decode_intervals(probability, intervals, transition, prior)
        state_accuracy.append(float(np.mean(decoded_state == state[order])))
        for strength in strengths:
            estimate = gated_prediction(
                predictors[order, finger], decoded_state, activity, finger, strength
            )
            scores[strength].append(pearson(estimate, target_raw[order, finger]))
    means = {strength: float(np.mean(values)) for strength, values in scores.items()}
    best = max(strengths, key=lambda value: means[value])
    if means[best] < means[0.0] + minimum_gain:
        best = 0.0
    return best, {
        "mean_fold_pcc_by_strength": {str(key): value for key, value in means.items()},
        "mean_latent_state_accuracy": float(np.mean(state_accuracy)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-map", type=Path, default=Path("configs/final_event_ensemble.yaml"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--fold-root", type=Path, default=Path("outputs/event_stratified_folds_targetsafe_conservative_v1"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--full-refit-root", type=Path, default=Path("outputs/frozen_event_full_refit_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/oof_latent_movement_gate_v1"))
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    parser.add_argument("--minimum-oof-gain", type=float, default=0.005)
    args = parser.parse_args()
    ensemble_map = yaml.safe_load(args.ensemble_map.read_text())
    target_map = yaml.safe_load(args.target_map.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "protocol": "post-test diagnostic method; latent gate and strength fitted with training OOF predictions only",
        "released_test_used_for_selection": False,
        "subjects": {},
    }
    for subject in (1, 2, 3):
        definition = json.loads(
            (args.fold_root / f"sub{subject}" / "thumb" / "folds.json").read_text()
        )
        rows = int(definition["training_rows"])
        predictors = load_oof_predictors(ensemble_map, subject, rows)
        test = np.column_stack([
            np.load(args.full_refit_root / "ensemble" / f"sub{subject}" / f"{finger}_released_test_prediction.npy")
            for finger in FINGER_NAMES
        ])
        prepared = args.prepared_root / f"sub{subject}"
        raw_train = np.load(prepared / "train_glove_25hz_raw.npy")[24 : 24 + rows]
        raw_test = np.load(prepared / "test_glove_25hz_raw.npy")[24 : 24 + test.shape[0]]
        subject_targets = target_map.get(subject, target_map.get(str(subject)))
        cleaned_train = np.column_stack([
            np.load(prepared / f"train_glove_{subject_targets[finger]}.npy")[24 : 24 + rows, index]
            for index, finger in enumerate(FINGER_NAMES)
        ])
        cleaned_test = np.column_stack([
            np.load(prepared / f"test_glove_{str(subject_targets[finger]).removesuffix('_split_safe')}.npy")[24 : 24 + test.shape[0], index]
            for index, finger in enumerate(FINGER_NAMES)
        ])
        state = intended_state(cleaned_train, args.movement_threshold)
        selected_strengths = []
        audits = []
        for finger_index, finger in enumerate(FINGER_NAMES):
            finger_definition = json.loads(
                (args.fold_root / f"sub{subject}" / finger / "folds.json").read_text()
            )
            strength, audit = choose_strength(
                predictors,
                raw_train,
                cleaned_train,
                state,
                finger_definition,
                finger_index,
                args.movement_threshold,
                args.minimum_oof_gain,
            )
            selected_strengths.append(strength)
            audits.append(audit)
        all_mask = np.ones(rows, dtype=bool)
        features = temporal_features(predictors)
        scaler, classifier = fit_classifier(features, state, all_mask)
        transition, prior = transition_model(state, all_mask)
        activity = activity_emission(
            state, cleaned_train, all_mask, args.movement_threshold
        )
        test_probability = state_probabilities(
            scaler, classifier, temporal_features(test)
        )
        test_state = viterbi(test_probability, transition, prior)
        prediction = np.column_stack([
            gated_prediction(
                test[:, finger],
                test_state,
                activity,
                finger,
                selected_strengths[finger],
            )
            for finger in range(5)
        ])
        true_test_state = intended_state(cleaned_test, args.movement_threshold)
        per_finger = {}
        for finger_index, finger in enumerate(FINGER_NAMES):
            groups = movement_groups(cleaned_test[:, finger_index], args.movement_threshold)
            metrics = morphology_metrics(
                prediction[:, finger_index], cleaned_test[:, finger_index], groups
            )
            metrics["raw_pcc"] = pearson(
                prediction[:, finger_index], raw_test[:, finger_index]
            )
            baseline = pearson(test[:, finger_index], raw_test[:, finger_index])
            per_finger[finger] = {
                "selected_gate_strength": selected_strengths[finger_index],
                "oof_audit": audits[finger_index],
                "baseline_test_raw_pcc": baseline,
                "metrics": metrics,
                "test_raw_pcc_delta": metrics["raw_pcc"] - baseline,
                "paper_raw_pcc": PAPER_PCC[subject][finger_index],
            }
        subject_output = args.output_root / f"sub{subject}"
        subject_output.mkdir(parents=True, exist_ok=True)
        np.save(subject_output / "released_test_prediction.npy", prediction)
        np.save(subject_output / "released_test_latent_state.npy", test_state)
        plot_subject_overview(
            subject_output / "all_fingers_full_trajectory.png",
            cleaned_test,
            prediction,
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
            "test_latent_state_accuracy_for_diagnosis_only": float(
                np.mean(test_state == true_test_state)
            ),
            "training_activity_emission": activity.tolist(),
        }
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
