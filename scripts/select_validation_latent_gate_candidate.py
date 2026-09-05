#!/usr/bin/env python3
"""Select a saved joint base plus latent gate using validation CV only."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml

from apply_validation_latent_gate import choose_strengths
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
from summarize_event_lars_lstm_cv import pearson
from summarize_frozen_full_refit import PAPER_PCC


SUBJECT = re.compile(r"(?:^|/)sub([123])(?:/|$)")


def discover(
    root: Path,
    subject: int,
    validation_shape: tuple[int, int],
    test_shape: tuple[int, int],
) -> list[dict[str, object]]:
    candidates = []
    for test_path in root.rglob("test_prediction*.npy"):
        match = SUBJECT.search(test_path.as_posix())
        if match is None or int(match.group(1)) != subject:
            continue
        validation_path = test_path.with_name(
            test_path.name.replace("test_", "validation_", 1)
        )
        if not validation_path.exists():
            continue
        try:
            validation = np.load(validation_path)
            test = np.load(test_path)
        except (OSError, ValueError):
            continue
        if validation.shape != validation_shape or test.shape != test_shape:
            continue
        if not np.isfinite(validation).all() or not np.isfinite(test).all():
            continue
        if np.any(np.std(validation, axis=0) < 1.0e-6):
            continue
        candidates.append({
            "path": test_path.as_posix(),
            "validation": validation,
            "test": test,
        })
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 3))
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/validation_latent_gate_candidate_v1"))
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    parser.add_argument("--minimum-validation-cv-gain", type=float, default=0.005)
    args = parser.parse_args()
    target_map = yaml.safe_load(args.target_map.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "protocol": "retrospective candidate and latent-gate selection by event-balanced validation CV only",
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
        candidates = discover(
            args.root,
            subject,
            raw_validation.shape,
            raw_test.shape,
        )
        audits = []
        for number, candidate in enumerate(candidates):
            strengths, audit = choose_strengths(
                candidate["validation"],
                raw_validation,
                cleaned_validation,
                args.movement_threshold,
                args.minimum_validation_cv_gain,
            )
            audits.append({
                "candidate_number": number,
                "path": candidate["path"],
                "strengths": strengths,
                "audit": audit,
            })
        selected = {}
        for finger, name in enumerate(FINGER_NAMES):
            def score(item: dict[str, object]) -> float:
                finger_audit = item["audit"]["per_finger"][name]
                strength = str(item["strengths"][finger])
                return float(finger_audit["mean_fold_pcc_by_strength"][strength])

            selected[name] = max(audits, key=score)
        generated = {}
        for candidate_number in sorted({item["candidate_number"] for item in selected.values()}):
            candidate = candidates[candidate_number]
            state = intended_state(cleaned_validation, args.movement_threshold)
            mask = np.ones(raw_validation.shape[0], dtype=bool)
            features = temporal_features(candidate["validation"])
            scaler, classifier = fit_classifier(features, state, mask)
            transition, prior = transition_model(state, mask)
            activity = activity_emission(
                state, cleaned_validation, mask, args.movement_threshold
            )
            probability = state_probabilities(
                scaler, classifier, temporal_features(candidate["test"])
            )
            decoded = viterbi(probability, transition, prior)
            generated[candidate_number] = (candidate, activity, decoded)
        test_prediction = np.empty_like(raw_test, dtype=np.float64)
        per_finger = {}
        for finger, name in enumerate(FINGER_NAMES):
            item = selected[name]
            candidate, activity, decoded = generated[item["candidate_number"]]
            strength = float(item["strengths"][finger])
            estimate = gated_prediction(
                candidate["test"][:, finger], decoded, activity, finger, strength
            )
            test_prediction[:, finger] = estimate
            validation_score = item["audit"]["per_finger"][name][
                "mean_fold_pcc_by_strength"
            ][str(strength)]
            per_finger[name] = {
                "candidate": candidate["path"],
                "selected_strength": strength,
                "validation_cv_pcc": validation_score,
                "test_raw_pcc": pearson(estimate, raw_test[:, finger]),
                "paper_raw_pcc": PAPER_PCC[subject][finger],
            }
        subject_output = args.output_root / f"sub{subject}"
        subject_output.mkdir(parents=True, exist_ok=True)
        np.save(subject_output / "test_prediction.npy", test_prediction)
        report["subjects"][str(subject)] = {
            "candidate_count": len(candidates),
            "per_finger": per_finger,
            "test_raw_pcc": [per_finger[name]["test_raw_pcc"] for name in FINGER_NAMES],
        }
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
