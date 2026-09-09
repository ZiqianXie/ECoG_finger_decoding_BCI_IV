#!/usr/bin/env python3
"""Summarize independent single-path refits and visualize their mean prediction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from single_wavelet_support import pearson
from cross_validate_single_wavelet import (
    OFFSET,
    TARGET_POLICIES,
    correct_local_intervals,
    scale_from_training,
)
from ecog_decoding.training import FINGER_NAMES
from evaluate_cv_ensemble_final_validation import movement_groups
from summarize_event_lars_lstm_cv import morphology_metrics


def resolve_route(
    route_map: dict[str, object] | None,
    subject: int,
    finger: str,
    default_root: Path,
    default_seeds: tuple[int, ...] | list[int],
) -> tuple[Path, tuple[int, ...], dict[str, object]]:
    """Resolve one fixed-model six-refit route for a subject/finger pair."""
    options: dict[str, object] = {}
    if route_map is not None:
        defaults = route_map.get("default", {})
        subjects = route_map.get("subjects", {})
        if not isinstance(defaults, dict) or not isinstance(subjects, dict):
            raise TypeError("route map default and subjects entries must be mappings")
        options.update(defaults)
        subject_map = subjects.get(subject, subjects.get(str(subject), {}))
        if not isinstance(subject_map, dict):
            raise TypeError(f"route map subject {subject} must be a mapping")
        finger_map = subject_map.get(finger, {})
        if not isinstance(finger_map, dict):
            raise TypeError(f"route map S{subject} {finger} must be a mapping")
        options.update(finger_map)
    root = Path(str(options.get("ensemble_root", default_root)))
    seeds = tuple(int(seed) for seed in options.get("seeds", default_seeds))
    if not seeds:
        raise ValueError(f"route map S{subject} {finger} has no seeds")
    return root, seeds, options


def best_subject_window(targets: list[np.ndarray], width: int) -> tuple[int, int]:
    score = np.sum(np.abs(np.stack(targets, axis=1)), axis=1)
    width = min(width, score.size)
    rolling = np.convolve(score, np.ones(width), mode="valid")
    start = int(np.argmax(rolling))
    return start, start + width


def plot_full_subject(
    path: Path,
    subject: int,
    fingers: tuple[str, ...] | list[str],
    traces: list[tuple[np.ndarray, np.ndarray]],
    score_summaries: dict[str, tuple[float, float]],
) -> None:
    figure, axes = plt.subplots(
        len(fingers), 1, figsize=(16, 2.2 * len(fingers)), sharex=True,
        constrained_layout=True, squeeze=False,
    )
    for axis, finger, (target, prediction) in zip(axes[:, 0], fingers, traces):
        time = np.arange(target.size) / 25.0
        axis.plot(
            time,
            target,
            color="black",
            linewidth=0.55,
            label="post-hoc cleaned glove",
        )
        axis.plot(
            time,
            prediction,
            color="#2878d0",
            linewidth=0.55,
            label="mean prediction (six seeds)",
        )
        axis.set_ylabel(finger.title())
        mean_pcc, sd_pcc = score_summaries[finger]
        axis.set_title(
            f"seed PCC {mean_pcc:.3f} ± {sd_pcc:.4f}",
            loc="right",
            fontsize=9,
        )
        axis.axhline(0.0, color="#cbd5e1", linewidth=0.45)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("released-test time (s)")
    figure.suptitle(f"S{subject}: complete released-test trajectories")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/single_wavelet_1000hz_tap5over2_six_seed_refit_v1"),
    )
    parser.add_argument("--route-map", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="write the combined report and figures here; defaults to --root",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4, 5))
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument(
        "--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=tuple(FINGER_NAMES)
    )
    parser.add_argument("--collapse-sd", type=float, default=1.0e-8)
    parser.add_argument("--window-bins", type=int, default=1250)
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    args = parser.parse_args()
    route_map = yaml.safe_load(args.route_map.read_text()) if args.route_map else None
    artifact_root = args.output_root or args.root
    artifact_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    pair_reports: dict[str, object] = {}
    series: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    mean_prediction_scores: dict[int, dict[str, float]] = {}
    seed_score_summaries: dict[int, dict[str, tuple[float, float]]] = {}
    seed_scores: dict[int, dict[str, dict[int, float]]] = {}
    configurations: list[dict[str, object]] = []
    for subject in args.subjects:
        mean_prediction_scores[subject] = {}
        seed_score_summaries[subject] = {}
        seed_scores[subject] = {}
        series[subject] = []
        prepared = args.prepared_root / f"sub{subject}"
        train_full = np.load(prepared / "train_glove_25hz_raw.npy", mmap_mode="r")
        test_full = np.load(prepared / "test_glove_25hz_raw.npy", mmap_mode="r")
        train_raw = np.asarray(train_full[OFFSET:], dtype=np.float64)
        test_raw_matrix = np.asarray(test_full[OFFSET:], dtype=np.float64)
        window_seconds, quantile = TARGET_POLICIES[subject]
        train_corrected = correct_local_intervals(
            train_raw, [[0, train_raw.shape[0]]], window_seconds, quantile
        )
        train_scale = scale_from_training(
            train_corrected, np.arange(train_raw.shape[0], dtype=np.int64)
        )
        test_corrected = correct_local_intervals(
            test_raw_matrix,
            [[0, test_raw_matrix.shape[0]]],
            window_seconds,
            quantile,
        )
        test_cleaned = np.clip(test_corrected / train_scale, 0.0, 2.0).astype(
            np.float32
        )
        for finger in args.fingers:
            finger_index = list(FINGER_NAMES).index(finger)
            ensemble_root, pair_seeds, route = resolve_route(
                route_map, subject, finger, args.root, args.seeds
            )
            directory = ensemble_root / f"sub{subject}" / finger
            artifact_directory = artifact_root / f"sub{subject}" / finger
            target: np.ndarray | None = None
            eligible_predictions = []
            audits = []
            pair_configuration: dict[str, object] | None = None
            for seed in pair_seeds:
                seed_directory = directory / f"seed{seed}"
                summary = json.loads((seed_directory / "summary.json").read_text())
                if pair_configuration is None:
                    pair_configuration = {
                        "decoder": summary.get("decoder"),
                        "frontend": summary.get("frontend"),
                        "training_objective": summary.get("training_objective")
                        or {
                            "trajectory": "normalized mean squared error",
                            "model_outputs": ["trajectory"],
                        },
                        "target_policy": summary.get("target_policy")
                        or {
                            "little_event_decontamination": False,
                            "little_event_ratio_low": 0.8,
                            "little_event_ratio_high": 1.2,
                        },
                        "output_activation": summary.get("output_activation"),
                        "softplus_beta": summary.get("softplus_beta"),
                    }
                eligible = bool(
                    summary["full_development_prediction_finite"]
                    and np.isfinite(summary["full_development_raw_pcc"])
                    and float(summary["full_development_prediction_sd"])
                    > args.collapse_sd
                )
                prediction = np.load(seed_directory / "test_prediction.npy")
                if target is None:
                    target = np.load(seed_directory / "test_raw_target.npy")
                audits.append(
                    {
                        "seed": seed,
                        "eligible_from_development_only": eligible,
                        "development_prediction_sd": float(
                            summary["full_development_prediction_sd"]
                        ),
                        "development_raw_pcc": float(
                            summary["full_development_raw_pcc"]
                        ),
                        "released_test_pcc_descriptive": float(
                            summary["released_test_pcc"]
                        ),
                    }
                )
                if eligible:
                    eligible_predictions.append(prediction)
            if not eligible_predictions or target is None:
                raise RuntimeError(f"all seeds collapsed for S{subject} {finger}")
            stacked = np.stack(eligible_predictions)
            ensemble = stacked.mean(axis=0)
            score = pearson(ensemble, target)
            member_scores = np.asarray(
                [
                    audit["released_test_pcc_descriptive"]
                    for audit in audits
                    if audit["eligible_from_development_only"]
                ],
                dtype=np.float64,
            )
            diversity = [
                pearson(stacked[left], stacked[right])
                for left in range(stacked.shape[0])
                for right in range(left + 1, stacked.shape[0])
            ]
            mean_prediction_scores[subject][finger] = score
            eligible_audits = [
                audit for audit in audits if audit["eligible_from_development_only"]
            ]
            seed_scores[subject][finger] = {
                int(audit["seed"]): float(audit["released_test_pcc_descriptive"])
                for audit in eligible_audits
            }
            seed_score_summaries[subject][finger] = (
                float(member_scores.mean()),
                float(member_scores.std(ddof=0)),
            )
            cleaned_target = test_cleaned[: target.size, finger_index]
            morphology = morphology_metrics(
                ensemble,
                cleaned_target,
                movement_groups(cleaned_target, 0.08),
            )
            morphology["raw_pcc"] = score
            row = {
                "subject": subject,
                "finger": finger,
                "mean_prediction_pcc": score,
                "seed_pcc_mean": float(member_scores.mean()),
                "seed_pcc_sd": float(member_scores.std(ddof=0)),
                "seed_pcc_sd_ddof": 0,
                "mean_pairwise_prediction_pcc": float(np.mean(diversity)),
                "included_seed_count": int(stacked.shape[0]),
                "collapsed_seed_count": int(len(pair_seeds) - stacked.shape[0]),
                "morphology": morphology,
            }
            rows.append(row)
            assert pair_configuration is not None
            configurations.append(pair_configuration)
            pair_reports[f"S{subject}_{finger}"] = {
                **row,
                "ensemble_root": str(ensemble_root),
                "route": route,
                "configuration": pair_configuration,
                "members": audits,
            }
            artifact_directory.mkdir(parents=True, exist_ok=True)
            np.save(
                artifact_directory / "ensemble_prediction.npy",
                ensemble,
                allow_pickle=False,
            )
            np.save(artifact_directory / "raw_target.npy", target, allow_pickle=False)
            np.save(
                artifact_directory / "cleaned_target_visual_only.npy",
                cleaned_target,
                allow_pickle=False,
            )
            series[subject].append(
                (cleaned_target, ensemble)
            )

    subjects = {}
    complete_finger_set = (
        set(args.fingers) == set(FINGER_NAMES) and len(args.fingers) == 5
    )
    for subject in args.subjects:
        mean_prediction_score = float(
            np.mean(
                [mean_prediction_scores[subject][finger] for finger in args.fingers]
            )
        )
        common_seeds = sorted(
            set.intersection(
                *(set(seed_scores[subject][finger]) for finger in args.fingers)
            )
        )
        if not common_seeds:
            raise RuntimeError(
                f"S{subject} has no seed eligible for every requested finger"
            )
        seed_macro_scores = {
            seed: float(
                np.mean(
                    [seed_scores[subject][finger][seed] for finger in args.fingers]
                )
            )
            for seed in common_seeds
        }
        macro_values = np.asarray(list(seed_macro_scores.values()), dtype=np.float64)
        subjects[f"S{subject}"] = {
            "fingers": list(args.fingers),
            "seed_macro_pcc_mean": float(macro_values.mean()),
            "seed_macro_pcc_sd": float(macro_values.std(ddof=0)),
            "seed_macro_pcc_sd_ddof": 0,
            "seed_macro_pcc_by_seed": {
                str(seed): value for seed, value in seed_macro_scores.items()
            },
            "mean_prediction_macro_pcc": mean_prediction_score,
        }
        if complete_finger_set:
            subjects[f"S{subject}"]["metric_name"] = "Macro-5 PCC"
    aggregate = {
        "protocol": (
            "six fixed-configuration full-development refits per pair; collapse "
            "eligibility uses only development prediction validity and variance; "
            "released test is used only for final scoring"
        ),
        "reporting": {
            "primary": (
                "mean and population standard deviation of released-test PCC "
                "across six independently refitted seeds"
            ),
            "subject_macro": (
                "compute the five-finger Macro-5 PCC within each seed, then "
                "report mean and population standard deviation across seeds"
            ),
            "visualization": "arithmetic mean of the six saved predictions",
        },
        "model": (
            configurations[0]
            if all(item == configurations[0] for item in configurations)
            else {"varies_by_pair": True, "see_pair_configuration": True}
        ),
        "selection": {
            "development_samples_per_subject": 400000,
            "event_grouped_outer_folds": 3,
            "purge_bins": 95,
            "bin_rate_hz": 25,
            "metric": "raw PCC",
            "rule": "best inner-fold checkpoint",
        },
        "seeds": list(args.seeds),
        "route_map": str(args.route_map) if args.route_map is not None else None,
        "collapse_sd": args.collapse_sd,
        "subjects": subjects,
        "pairs": pair_reports,
    }
    (artifact_root / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    with (artifact_root / "aggregate_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(
        len(args.subjects),
        len(args.fingers),
        figsize=(3.6 * len(args.fingers), 2.8 * len(args.subjects)),
        sharex="row",
        squeeze=False,
        constrained_layout=True,
    )
    for subject_row, subject in enumerate(args.subjects):
        start, stop = best_subject_window(
            [target for target, _ in series[subject]], args.window_bins
        )
        time_axis = np.arange(start, stop) / 25.0
        for finger_column, finger in enumerate(args.fingers):
            target, prediction = series[subject][finger_column]
            axis = axes[subject_row, finger_column]
            axis.plot(
                time_axis,
                target[start:stop],
                color="black",
                linewidth=0.8,
                label="post-hoc baseline-corrected glove",
            )
            axis.plot(
                time_axis,
                prediction[start:stop],
                color="#2878d0",
                linewidth=0.8,
                label="mean prediction (six seeds)",
            )
            mean_pcc, sd_pcc = seed_score_summaries[subject][finger]
            axis.set_title(
                f"S{subject} {finger.title()}  "
                f"seed PCC {mean_pcc:.3f} ± {sd_pcc:.4f}"
            )
            if finger_column == 0:
                axis.set_ylabel("flexion")
            if subject_row == len(args.subjects) - 1:
                axis.set_xlabel("released-test time (s)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Six independent single-path refits: released-test trajectories")
    figure.savefig(artifact_root / "aggregate_test_trajectories.png", dpi=180)
    plt.close(figure)
    for subject in args.subjects:
        plot_full_subject(
            artifact_root / f"sub{subject}_full_test_trajectories.png",
            subject,
            list(args.fingers),
            series[subject],
            seed_score_summaries[subject],
        )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
