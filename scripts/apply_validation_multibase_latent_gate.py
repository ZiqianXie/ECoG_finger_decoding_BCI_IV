#!/usr/bin/env python3
"""Infer latent intended-finger state from a diverse bank of saved decoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from apply_validation_latent_gate import event_balanced_folds
from apply_validation_soft_latent_gate import forward_backward, soft_gated_prediction
from ecog_decoding.training import FINGER_NAMES
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
from select_validation_latent_gate_candidate import discover
from summarize_event_lars_lstm_cv import pearson
from summarize_frozen_full_refit import PAPER_PCC
from train_event_grouped_lars_lstm_nested import intervals_from_mask


def temper_probability(probability: np.ndarray, temperature: float) -> np.ndarray:
    values = np.clip(probability, 1.0e-12, 1.0)
    values = values ** (1.0 / temperature)
    return values / values.sum(axis=1, keepdims=True)


def select_diverse_candidates(
    candidates: list[dict[str, object]],
    target: np.ndarray,
    count: int,
    correlation_ceiling: float,
    mask: np.ndarray | None = None,
) -> list[dict[str, object]]:
    rows = np.ones(target.shape[0], dtype=bool) if mask is None else mask
    ranked = sorted(
        candidates,
        key=lambda item: float(np.mean([
            pearson(item["validation"][rows, finger], target[rows, finger])
            for finger in range(5)
        ])),
        reverse=True,
    )
    selected = []
    for candidate in ranked:
        flat = np.asarray(candidate["validation"])[rows].ravel()
        if selected and max(
            abs(pearson(flat, np.asarray(item["validation"])[rows].ravel()))
            for item in selected
        ) > correlation_ceiling:
            continue
        selected.append(candidate)
        if len(selected) == count:
            break
    return selected


def choose_specs(
    base: np.ndarray,
    classifier_features: np.ndarray,
    raw_target: np.ndarray,
    cleaned_target: np.ndarray,
    threshold: float,
    minimum_gain: float,
) -> tuple[list[tuple[str, float, float]], dict[str, object]]:
    state = intended_state(cleaned_target, threshold)
    folds = event_balanced_folds(state)
    strengths = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    mode_temperatures = (
        [("hard", 1.0)]
        + [("posterior", value) for value in (0.5, 0.75, 1.0, 1.5, 2.0)]
        + [("classifier_probability", value) for value in (0.5, 0.75, 1.0, 1.5, 2.0)]
    )
    scores = [
        {(mode, temperature, strength): [] for mode, temperature in mode_temperatures for strength in strengths}
        for _ in range(5)
    ]
    accuracies = []
    for validation_mask in folds:
        training_mask = ~validation_mask
        scaler, classifier = fit_classifier(classifier_features, state, training_mask)
        probability = state_probabilities(scaler, classifier, classifier_features)
        transition, prior = transition_model(state, training_mask)
        activity = activity_emission(state, cleaned_target, training_mask, threshold)
        order_parts, hard_parts, posterior_parts = [], [], []
        for start, stop in intervals_from_mask(validation_mask):
            order_parts.append(np.arange(start, stop, dtype=np.int64))
            hard_parts.append(viterbi(probability[start:stop], transition, prior))
            posterior_parts.append(forward_backward(probability[start:stop], transition, prior))
        order = np.concatenate(order_parts)
        hard = np.concatenate(hard_parts)
        posterior = np.concatenate(posterior_parts)
        accuracies.append(float(np.mean(hard == state[order])))
        for finger in range(5):
            for strength in strengths:
                scores[finger][("hard", 1.0, strength)].append(
                    pearson(gated_prediction(base[order, finger], hard, activity, finger, strength), raw_target[order, finger])
                )
                for temperature in (0.5, 0.75, 1.0, 1.5, 2.0):
                    scores[finger][("posterior", temperature, strength)].append(
                        pearson(soft_gated_prediction(base[order, finger], temper_probability(posterior, temperature), activity, finger, strength), raw_target[order, finger])
                    )
                    scores[finger][("classifier_probability", temperature, strength)].append(
                        pearson(soft_gated_prediction(base[order, finger], temper_probability(probability[order], temperature), activity, finger, strength), raw_target[order, finger])
                    )
    specs, audit = [], {"mean_latent_state_accuracy": float(np.mean(accuracies)), "per_finger": {}}
    for finger, name in enumerate(FINGER_NAMES):
        means = {key: float(np.mean(value)) for key, value in scores[finger].items()}
        baseline = means[("hard", 1.0, 0.0)]
        best = max(means, key=means.get)
        if means[best] < baseline + minimum_gain:
            best = ("hard", 1.0, 0.0)
        specs.append(best)
        audit["per_finger"][name] = {
            "selected_mode": best[0],
            "selected_temperature": best[1],
            "selected_strength": best[2],
            "baseline_validation_cv_pcc": baseline,
            "selected_validation_cv_pcc": means[best],
        }
    return specs, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--base-map", type=Path, default=Path("configs/retrospective_base_predictions.yaml"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/validation_multibase_latent_gate_v1"))
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--candidate-correlation-ceiling", type=float, default=0.995)
    parser.add_argument(
        "--evidence-summary",
        type=Path,
        default=None,
        help="reuse an already frozen evidence-candidate list from a prior summary",
    )
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    parser.add_argument("--minimum-validation-cv-gain", type=float, default=0.005)
    args = parser.parse_args()
    base_map = yaml.safe_load(args.base_map.read_text())
    target_map = yaml.safe_load(args.target_map.read_text())
    evidence_report = (
        json.loads(args.evidence_summary.read_text())
        if args.evidence_summary is not None else None
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "protocol": "retrospective multibase latent-state evidence selected on validation only",
        "released_test_used_for_selection": False,
        "subjects": {},
    }
    for subject in args.subjects:
        prepared = args.prepared_root / f"sub{subject}"
        metadata = json.loads((prepared / "metadata.json").read_text())
        split = int(metadata["target_fit_samples_25hz"])
        raw_train = np.load(prepared / "train_glove_25hz_raw.npy")
        raw_validation = raw_train[split:]
        raw_test = np.load(prepared / "test_glove_25hz_raw.npy")[24:]
        subject_targets = target_map.get(subject, target_map.get(str(subject)))
        cleaned_validation = np.column_stack([
            np.load(prepared / f"train_glove_{subject_targets[finger]}.npy")[-raw_validation.shape[0] :, index]
            for index, finger in enumerate(FINGER_NAMES)
        ])
        base_root = Path(str(base_map.get(subject, base_map.get(str(subject)))))
        base_validation = np.load(base_root / "validation_prediction.npy")
        base_test = np.load(base_root / "test_prediction.npy")
        candidates = discover(args.root, subject, raw_validation.shape, raw_test.shape)
        if evidence_report is None:
            selected_candidates = select_diverse_candidates(
                candidates, raw_validation, args.candidate_count, args.candidate_correlation_ceiling
            )
        else:
            requested = evidence_report["subjects"][str(subject)]["evidence_candidates"]
            by_path = {str(candidate["path"]): candidate for candidate in candidates}
            missing = [path for path in requested if path not in by_path]
            if missing:
                raise FileNotFoundError("missing frozen evidence candidates: " + ", ".join(missing))
            selected_candidates = [by_path[path] for path in requested]
        validation_evidence = temporal_features(np.concatenate([
            np.asarray(candidate["validation"]) for candidate in selected_candidates
        ], axis=1))
        test_evidence = temporal_features(np.concatenate([
            np.asarray(candidate["test"]) for candidate in selected_candidates
        ], axis=1))
        specs, audit = choose_specs(
            base_validation,
            validation_evidence,
            raw_validation,
            cleaned_validation,
            args.movement_threshold,
            args.minimum_validation_cv_gain,
        )
        state = intended_state(cleaned_validation, args.movement_threshold)
        mask = np.ones(state.size, dtype=bool)
        scaler, classifier = fit_classifier(validation_evidence, state, mask)
        transition, prior = transition_model(state, mask)
        activity = activity_emission(state, cleaned_validation, mask, args.movement_threshold)
        probability = state_probabilities(scaler, classifier, test_evidence)
        hard = viterbi(probability, transition, prior)
        posterior = forward_backward(probability, transition, prior)
        prediction = np.empty_like(base_test, dtype=np.float64)
        per_finger = {}
        for finger, name in enumerate(FINGER_NAMES):
            mode, temperature, strength = specs[finger]
            if mode == "hard":
                estimate = gated_prediction(base_test[:, finger], hard, activity, finger, strength)
            elif mode == "posterior":
                estimate = soft_gated_prediction(base_test[:, finger], temper_probability(posterior, temperature), activity, finger, strength)
            else:
                estimate = soft_gated_prediction(base_test[:, finger], temper_probability(probability, temperature), activity, finger, strength)
            prediction[:, finger] = estimate
            score = pearson(estimate, raw_test[:, finger])
            per_finger[name] = {
                **audit["per_finger"][name],
                "test_raw_pcc": score,
                "paper_raw_pcc": PAPER_PCC[subject][finger],
                "delta_vs_paper": score - PAPER_PCC[subject][finger],
            }
        subject_output = args.output_root / f"sub{subject}"
        subject_output.mkdir(parents=True, exist_ok=True)
        np.save(subject_output / "test_prediction.npy", prediction)
        report["subjects"][str(subject)] = {
            "base": str(base_root),
            "evidence_candidates": [candidate["path"] for candidate in selected_candidates],
            "mean_latent_state_cv_accuracy": audit["mean_latent_state_accuracy"],
            "per_finger": per_finger,
            "test_raw_pcc": [per_finger[name]["test_raw_pcc"] for name in FINGER_NAMES],
        }
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
