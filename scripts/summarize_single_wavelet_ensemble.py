#!/usr/bin/env python3
"""Average eligible single-path seeds and visualize requested predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from single_wavelet_support import pearson
from cross_validate_single_wavelet import (
    OFFSET,
    TARGET_POLICIES,
    correct_local_intervals,
    scale_from_training,
)
from ecog_decoding.training import FINGER_NAMES


def best_subject_window(targets: list[np.ndarray], width: int) -> tuple[int, int]:
    score = np.sum(np.abs(np.stack(targets, axis=1)), axis=1)
    width = min(width, score.size)
    rolling = np.convolve(score, np.ones(width), mode="valid")
    start = int(np.argmax(rolling))
    return start, start + width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/single_wavelet_1000hz_tap5over2_six_seed_refit_v1"),
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

    rows: list[dict[str, object]] = []
    pair_reports: dict[str, object] = {}
    series: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    scores: dict[int, dict[str, float]] = {}
    for subject in args.subjects:
        scores[subject] = {}
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
            directory = args.root / f"sub{subject}" / finger
            target: np.ndarray | None = None
            eligible_predictions = []
            audits = []
            for seed in args.seeds:
                seed_directory = directory / f"seed{seed}"
                summary = json.loads((seed_directory / "summary.json").read_text())
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
            scores[subject][finger] = score
            row = {
                "subject": subject,
                "finger": finger,
                "ensemble_pcc": score,
                "member_pcc_mean": float(member_scores.mean()),
                "member_pcc_sd": float(member_scores.std(ddof=0)),
                "mean_pairwise_prediction_pcc": float(np.mean(diversity)),
                "included_seed_count": int(stacked.shape[0]),
                "collapsed_seed_count": int(len(args.seeds) - stacked.shape[0]),
            }
            rows.append(row)
            pair_reports[f"S{subject}_{finger}"] = {**row, "members": audits}
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / "ensemble_prediction.npy", ensemble, allow_pickle=False)
            np.save(directory / "raw_target.npy", target, allow_pickle=False)
            np.save(
                directory / "cleaned_target_visual_only.npy",
                test_cleaned[: target.size, finger_index],
                allow_pickle=False,
            )
            series[subject].append(
                (test_cleaned[: target.size, finger_index], ensemble)
            )

    subjects = {
        f"S{subject}": {
            "macro_5_pcc": float(
                np.mean([scores[subject][finger] for finger in args.fingers])
            ),
        }
        for subject in args.subjects
    }
    aggregate = {
        "protocol": (
            "six fixed-configuration full-development refits per pair; collapse "
            "eligibility uses only development prediction validity and variance; "
            "released test is used only for final scoring"
        ),
        "model": {
            "input_rate_hz": 1000,
            "spatial_initialization": (
                "full-rank split-local FastICA plus one split-local, finger-specific "
                "movement-versus-rest CSP row fitted from HHL/HHH samples"
            ),
            "frontend": (
                "one trainable depth-3 bior6.8 wavelet-packet tree; first filter pair "
                "interpolated 5/2; deeper dilations 5 and 10; eight energy leaves; "
                "no auxiliary branch"
            ),
            "temporal_decoder": (
                "one nonlinear LSTM initialized in a near-linear regime from LARS; "
                "not a residual or frozen linear skip"
            ),
            "output": "Softplus(beta=10)",
        },
        "selection": {
            "development_samples_per_subject": 400000,
            "event_grouped_outer_folds": 3,
            "purge_bins": 95,
            "bin_rate_hz": 25,
            "metric": "raw PCC",
            "rule": "best inner-fold checkpoint",
        },
        "seeds": list(args.seeds),
        "collapse_sd": args.collapse_sd,
        "subjects": subjects,
        "pairs": pair_reports,
    }
    (args.root / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    with (args.root / "aggregate_summary.csv").open("w", newline="") as handle:
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
                label="exact Softplus ensemble",
            )
            axis.set_title(
                f"S{subject} {finger.title()}  PCC {scores[subject][finger]:.3f}"
            )
            if finger_column == 0:
                axis.set_ylabel("flexion")
            if subject_row == len(args.subjects) - 1:
                axis.set_xlabel("released-test time (s)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Single-path six-seed ensembles: released-test trajectories")
    figure.savefig(args.root / "aggregate_test_trajectories.png", dpi=180)
    plt.close(figure)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
