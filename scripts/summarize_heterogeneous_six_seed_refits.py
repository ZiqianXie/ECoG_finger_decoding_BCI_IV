#!/usr/bin/env python3
"""Summarize promoted heterogeneous ensembles and the OOF-routed hybrid result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ecog_decoding.training import FINGER_NAMES
from evaluate_cv_ensemble_final_validation import movement_groups
from refit_frozen_event_model import plot_events, plot_full_trajectory
from summarize_event_lars_lstm_cv import morphology_metrics, pearson
from summarize_frozen_full_refit import plot_subject_overview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/heterogeneous_six_seed_refit.yaml"))
    parser.add_argument("--input-root", type=Path, default=Path("outputs/heterogeneous_six_seed_refit_v1"))
    parser.add_argument("--baseline-root", type=Path, default=Path("outputs/full_development_event_refit_v1/ensemble"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/heterogeneous_six_seed_refit_v1/ensemble"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    target_map = yaml.safe_load(args.target_map.read_text())
    seeds = [int(seed) for seed in config["seeds"]]
    promoted = {
        int(subject): list(fingers) for subject, fingers in config["targets"].items()
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "protocol": "OOF-routed heterogeneous fixed-feature six-seed ensembles",
        "routing_rule": (
            "promote pairs where the joint dictionary exceeded the better individual "
            "feature family in full-development event-grouped OOF PCC"
        ),
        "released_test_used_for_routing": False,
        "released_test_role": "retrospective paper-comparison evaluation",
        "seeds": seeds,
        "subjects": {},
    }
    for subject in (1, 2, 3):
        prepared = args.prepared_root / f"sub{subject}"
        raw = np.load(prepared / "test_glove_25hz_raw.npy")[24:]
        hybrid = np.column_stack([
            np.load(args.baseline_root / f"sub{subject}" / f"{finger}_released_test_prediction.npy")
            for finger in FINGER_NAMES
        ])
        baseline_scores = [pearson(hybrid[:, index], raw[:, index]) for index in range(5)]
        subject_pairs: dict[str, object] = {}
        subject_output = args.output_root / f"sub{subject}"
        subject_output.mkdir(parents=True, exist_ok=True)
        for finger in promoted.get(subject, []):
            finger_index = list(FINGER_NAMES).index(finger)
            members = []
            audits = []
            for seed in seeds:
                root = args.input_root / f"sub{subject}" / finger / f"seed{seed}"
                prediction = np.load(root / "released_test_prediction.npy")
                summary = json.loads((root / "summary.json").read_text())
                eligible = bool(
                    np.isfinite(prediction).all()
                    and float(np.std(prediction)) > 1.0e-8
                    and np.isfinite(summary["fitted_full_train_raw_pcc"])
                )
                audits.append({
                    "seed": seed,
                    "eligible": eligible,
                    "selected_epoch": int(summary["selected_epoch"]),
                    "raw_pcc": float(summary["released_test_metrics"]["raw_pcc"]),
                    "prediction_sd": float(np.std(prediction)),
                    "training_runtime_seconds": float(summary["runtime_seconds"]),
                })
                if eligible:
                    members.append(prediction)
            if not members:
                raise RuntimeError(f"all heterogeneous members collapsed for S{subject} {finger}")
            stacked = np.stack(members)
            prediction = stacked.mean(axis=0)
            hybrid[:, finger_index] = prediction
            subject_targets = target_map.get(subject, target_map.get(str(subject)))
            target_policy = str(subject_targets[finger]).removesuffix("_split_safe")
            cleaned = np.load(prepared / f"test_glove_{target_policy}.npy")[24:, finger_index]
            metrics = morphology_metrics(prediction, cleaned, movement_groups(cleaned, 0.08))
            metrics["raw_pcc"] = pearson(prediction, raw[:, finger_index])
            diversity = [
                pearson(stacked[first], stacked[second])
                for first in range(stacked.shape[0])
                for second in range(first + 1, stacked.shape[0])
            ]
            subject_pairs[finger] = {
                "target_policy": target_policy,
                "members": audits,
                "included_seeds": [audit["seed"] for audit in audits if audit["eligible"]],
                "collapsed_seeds": [audit["seed"] for audit in audits if not audit["eligible"]],
                "metrics": metrics,
                "baseline_raw_pcc": baseline_scores[finger_index],
                "delta_vs_baseline": metrics["raw_pcc"] - baseline_scores[finger_index],
                "mean_pairwise_seed_prediction_pcc": float(np.mean(diversity)),
            }
            np.save(subject_output / f"{finger}_released_test_prediction.npy", prediction)
            plot_full_trajectory(
                subject_output / f"{finger}_full_trajectory.png",
                raw[:, finger_index], cleaned, prediction,
                f"S{subject} {finger}: joint dictionary six-seed ensemble",
            )
            plot_events(
                subject_output / f"{finger}_strongest_events.png",
                raw[:, finger_index], cleaned, prediction,
                f"S{subject} {finger}: joint dictionary strongest events",
            )
        hybrid_scores = [pearson(hybrid[:, index], raw[:, index]) for index in range(5)]
        np.save(subject_output / "hybrid_released_test_prediction.npy", hybrid)
        plot_subject_overview(
            subject_output / "hybrid_all_fingers_full_trajectory.png", raw, hybrid, subject
        )
        report["subjects"][str(subject)] = {
            "promoted_pairs": subject_pairs,
            "baseline_raw_pcc": dict(zip(FINGER_NAMES, baseline_scores)),
            "baseline_macro5_raw_pcc": float(np.mean(baseline_scores)),
            "hybrid_raw_pcc": dict(zip(FINGER_NAMES, hybrid_scores)),
            "hybrid_macro5_raw_pcc": float(np.mean(hybrid_scores)),
            "macro5_gain": float(np.mean(hybrid_scores) - np.mean(baseline_scores)),
        }
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        subject: values["hybrid_macro5_raw_pcc"]
        for subject, values in report["subjects"].items()
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
