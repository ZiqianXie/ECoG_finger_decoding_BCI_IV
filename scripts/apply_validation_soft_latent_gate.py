#!/usr/bin/env python3
"""Compare hard and forward-backward soft latent gates by validation CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from apply_validation_latent_gate import event_balanced_folds
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


def forward_backward(
    emission: np.ndarray, transition: np.ndarray, prior: np.ndarray
) -> np.ndarray:
    forward = np.empty_like(emission, dtype=np.float64)
    backward = np.empty_like(emission, dtype=np.float64)
    forward[0] = prior * emission[0]
    forward[0] /= max(float(forward[0].sum()), 1.0e-12)
    for step in range(1, emission.shape[0]):
        forward[step] = emission[step] * (forward[step - 1] @ transition)
        forward[step] /= max(float(forward[step].sum()), 1.0e-12)
    backward[-1] = 1.0
    for step in range(emission.shape[0] - 2, -1, -1):
        backward[step] = transition @ (emission[step + 1] * backward[step + 1])
        backward[step] /= max(float(backward[step].sum()), 1.0e-12)
    posterior = forward * backward
    posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1.0e-12)
    return posterior


def soft_gated_prediction(
    prediction: np.ndarray,
    posterior: np.ndarray,
    activity: np.ndarray,
    finger: int,
    strength: float,
) -> np.ndarray:
    gate = posterior @ activity[:, finger]
    factor = np.maximum((1.0 - strength) + strength * gate, 0.0)
    return prediction * factor


def choose_specs(
    validation: np.ndarray,
    raw_target: np.ndarray,
    cleaned_target: np.ndarray,
    threshold: float,
    minimum_gain: float,
) -> tuple[list[tuple[str, float]], dict[str, object]]:
    state = intended_state(cleaned_target, threshold)
    features = temporal_features(validation)
    folds = event_balanced_folds(state)
    strengths = (0.0, 0.25, 0.5, 0.75, 1.0)
    modes = ("hard", "posterior", "classifier_probability")
    scores = [{(mode, strength): [] for mode in modes for strength in strengths} for _ in range(5)]
    for validation_mask in folds:
        training_mask = ~validation_mask
        scaler, classifier = fit_classifier(features, state, training_mask)
        probability = state_probabilities(scaler, classifier, features)
        transition, prior = transition_model(state, training_mask)
        activity = activity_emission(state, cleaned_target, training_mask, threshold)
        order_parts = []
        hard_parts = []
        posterior_parts = []
        for start, stop in intervals_from_mask(validation_mask):
            order_parts.append(np.arange(start, stop, dtype=np.int64))
            hard_parts.append(viterbi(probability[start:stop], transition, prior))
            posterior_parts.append(
                forward_backward(probability[start:stop], transition, prior)
            )
        order = np.concatenate(order_parts)
        hard = np.concatenate(hard_parts)
        posterior = np.concatenate(posterior_parts)
        for finger in range(5):
            for strength in strengths:
                scores[finger][("hard", strength)].append((
                    pearson(
                        gated_prediction(validation[order, finger], hard, activity, finger, strength),
                        raw_target[order, finger],
                    ),
                    order.size,
                ))
                scores[finger][("posterior", strength)].append((
                    pearson(
                        soft_gated_prediction(validation[order, finger], posterior, activity, finger, strength),
                        raw_target[order, finger],
                    ),
                    order.size,
                ))
                scores[finger][("classifier_probability", strength)].append((
                    pearson(
                        soft_gated_prediction(validation[order, finger], probability[order], activity, finger, strength),
                        raw_target[order, finger],
                    ),
                    order.size,
                ))
    specs = []
    audit = {}
    for finger, name in enumerate(FINGER_NAMES):
        means = {
            key: float(np.average([value for value, _ in records], weights=[weight for _, weight in records]))
            for key, records in scores[finger].items()
        }
        baseline = means[("hard", 0.0)]
        best = max(means, key=means.get)
        if means[best] < baseline + minimum_gain:
            best = ("hard", 0.0)
        specs.append(best)
        audit[name] = {
            "selected_mode": best[0],
            "selected_strength": best[1],
            "baseline_validation_cv_pcc": baseline,
            "selected_validation_cv_pcc": means[best],
            "all_scores": {f"{mode}:{strength}": score for (mode, strength), score in means.items()},
        }
    return specs, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-map", type=Path, default=Path("configs/retrospective_base_predictions.yaml"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/validation_soft_latent_gate_v1"))
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    parser.add_argument("--minimum-validation-cv-gain", type=float, default=0.005)
    args = parser.parse_args()
    base_map = yaml.safe_load(args.base_map.read_text())
    target_map = yaml.safe_load(args.target_map.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "protocol": "retrospective hard-versus-soft latent gate selected by event-balanced validation CV",
        "released_test_used_for_selection": False,
        "subjects": {},
    }
    for subject in (1, 2, 3):
        base = Path(str(base_map.get(subject, base_map.get(str(subject)))))
        validation = np.load(base / "validation_prediction.npy")
        test = np.load(base / "test_prediction.npy")
        prepared = args.prepared_root / f"sub{subject}"
        raw_train = np.load(prepared / "train_glove_25hz_raw.npy")
        raw_validation = raw_train[24:][-validation.shape[0] :]
        raw_test = np.load(prepared / "test_glove_25hz_raw.npy")[24 : 24 + test.shape[0]]
        subject_targets = target_map.get(subject, target_map.get(str(subject)))
        cleaned_validation = np.column_stack([
            np.load(prepared / f"train_glove_{subject_targets[finger]}.npy")[24:][-validation.shape[0] :, index]
            for index, finger in enumerate(FINGER_NAMES)
        ])
        cleaned_test = np.column_stack([
            np.load(prepared / f"test_glove_{str(subject_targets[finger]).removesuffix('_split_safe')}.npy")[24 : 24 + test.shape[0], index]
            for index, finger in enumerate(FINGER_NAMES)
        ])
        specs, audit = choose_specs(
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
        activity = activity_emission(state, cleaned_validation, mask, args.movement_threshold)
        probability = state_probabilities(scaler, classifier, temporal_features(test))
        hard = viterbi(probability, transition, prior)
        posterior = forward_backward(probability, transition, prior)
        prediction = np.empty_like(test, dtype=np.float64)
        per_finger = {}
        for finger, name in enumerate(FINGER_NAMES):
            mode, strength = specs[finger]
            if mode == "hard":
                estimate = gated_prediction(test[:, finger], hard, activity, finger, strength)
            elif mode == "posterior":
                estimate = soft_gated_prediction(test[:, finger], posterior, activity, finger, strength)
            else:
                estimate = soft_gated_prediction(test[:, finger], probability, activity, finger, strength)
            prediction[:, finger] = estimate
            groups = movement_groups(cleaned_test[:, finger], args.movement_threshold)
            metrics = morphology_metrics(estimate, cleaned_test[:, finger], groups)
            metrics["raw_pcc"] = pearson(estimate, raw_test[:, finger])
            baseline = pearson(test[:, finger], raw_test[:, finger])
            per_finger[name] = {
                **audit[name],
                "baseline_test_raw_pcc": baseline,
                "metrics": metrics,
                "test_raw_pcc_delta": metrics["raw_pcc"] - baseline,
                "paper_raw_pcc": PAPER_PCC[subject][finger],
            }
        subject_output = args.output_root / f"sub{subject}"
        subject_output.mkdir(parents=True, exist_ok=True)
        np.save(subject_output / "test_prediction.npy", prediction)
        plot_subject_overview(
            subject_output / "all_fingers_full_trajectory.png",
            cleaned_test,
            prediction,
            subject,
        )
        scores = [per_finger[name]["metrics"]["raw_pcc"] for name in FINGER_NAMES]
        report["subjects"][str(subject)] = {
            "base": str(base),
            "per_finger": per_finger,
            "raw_pcc": scores,
            "macro5_raw_pcc": float(np.mean(scores)),
        }
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
