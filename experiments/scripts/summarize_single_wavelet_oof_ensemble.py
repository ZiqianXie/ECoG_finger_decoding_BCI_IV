#!/usr/bin/env python3
"""Summarize a fixed-configuration OOF ensemble without target-based pruning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ecog_decoding.training import FINGER_NAMES
from cross_validate_single_wavelet import validation_metrics
from single_wavelet_support import OFFSET, pearson


def finite_intervals(*arrays: np.ndarray) -> list[list[int]]:
    """Return contiguous intervals observed in every supplied OOF array."""
    observed = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    padded = np.pad(observed.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [[int(start), int(stop)] for start, stop in edges.reshape(-1, 2)]


def summarize_predictions(
    initialized: np.ndarray,
    target: np.ndarray,
    raw: np.ndarray,
    predictions: dict[int, np.ndarray],
    collapse_sd: float,
) -> tuple[dict[str, object], np.ndarray]:
    observed = np.isfinite(initialized) & np.isfinite(target) & np.isfinite(raw)
    eligible: dict[int, np.ndarray] = {}
    members = []
    for seed, prediction in sorted(predictions.items()):
        finite = bool(np.isfinite(prediction[observed]).all())
        prediction_sd = float(np.std(prediction[observed])) if finite else float("nan")
        included = finite and prediction_sd > collapse_sd
        member = {
            "seed": seed,
            "eligible_without_target": included,
            "prediction_finite": finite,
            "prediction_sd": prediction_sd,
        }
        if included:
            member["oof_raw_pcc_after_inclusion"] = pearson(
                prediction[observed], raw[observed]
            )
            eligible[seed] = prediction
        members.append(member)
    if not eligible:
        raise ValueError("every seed is nonfinite or collapsed")
    ensemble = np.mean(np.stack(list(eligible.values())), axis=0)
    initialized_pcc = pearson(initialized[observed], raw[observed])
    ensemble_pcc = pearson(ensemble[observed], raw[observed])
    member_pcc = np.asarray(
        [item["oof_raw_pcc_after_inclusion"] for item in members if item["eligible_without_target"]]
    )
    report = {
        "collapse_rule": (
            "exclude only nonfinite or effectively constant OOF predictions; "
            "held-out target agreement is not used for eligibility"
        ),
        "collapse_sd": collapse_sd,
        "members": members,
        "included_seed_count": len(eligible),
        "collapsed_seed_count": len(predictions) - len(eligible),
        "initialized_oof_raw_pcc": initialized_pcc,
        "member_oof_raw_pcc_mean": float(member_pcc.mean()),
        "member_oof_raw_pcc_population_sd": float(member_pcc.std(ddof=0)),
        "ensemble_oof_raw_pcc": ensemble_pcc,
        "ensemble_gain_over_initialization": ensemble_pcc - initialized_pcc,
    }
    return report, ensemble.astype(np.float32)


def plot_oof(
    path: Path,
    raw: np.ndarray,
    target: np.ndarray,
    initialized: np.ndarray,
    ensemble: np.ndarray,
) -> None:
    observed = np.isfinite(ensemble)
    rows = np.flatnonzero(observed)
    figure, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    axes[0].plot(rows, raw[observed], color="black", linewidth=0.7, label="raw glove")
    axes[0].plot(rows, target[observed], color="#999999", linewidth=0.7, label="cleaned target")
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("target")
    axes[1].plot(rows, initialized[observed], linewidth=0.7, label="LARS initialization")
    axes[1].plot(rows, ensemble[observed], linewidth=0.7, label="six-seed mean")
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("prediction")
    axes[1].set_xlabel("25 Hz bin")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--finger", choices=tuple(FINGER_NAMES), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4, 5))
    parser.add_argument("--collapse-sd", type=float, default=1.0e-8)
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seed_roots = {
        seed: args.root / f"seed{seed}" / f"sub{args.subject}" / args.finger
        for seed in args.seeds
    }
    initialized = np.load(seed_roots[args.seeds[0]] / "initialized_oof.npy")
    target = np.load(seed_roots[args.seeds[0]] / "target_oof.npy")
    predictions = {}
    selected_schedules = {}
    for seed, root in seed_roots.items():
        candidate_initialized = np.load(root / "initialized_oof.npy")
        candidate_target = np.load(root / "target_oof.npy")
        if not np.allclose(initialized, candidate_initialized, equal_nan=True, atol=1e-6):
            raise ValueError(f"seed {seed} has a different initialization")
        if not np.allclose(target, candidate_target, equal_nan=True, atol=1e-6):
            raise ValueError(f"seed {seed} has a different target")
        predictions[seed] = np.load(root / "selected_oof.npy")
        summary = json.loads((root / "summary.json").read_text())
        selected_schedules[seed] = [
            item["selected_schedule"] for item in summary["outer_folds"]
        ]

    finger_index = list(FINGER_NAMES).index(args.finger)
    raw = np.asarray(
        np.load(
            args.prepared_root / f"sub{args.subject}" / "train_glove_25hz_raw.npy",
            mmap_mode="r",
        )[OFFSET : OFFSET + target.size, finger_index]
    )
    report, ensemble = summarize_predictions(
        initialized, target, raw, predictions, args.collapse_sd
    )
    intervals = finite_intervals(initialized, target, raw, ensemble)
    report["initialized_oof_morphology"] = validation_metrics(
        initialized, target, raw, intervals, movement_threshold=0.1, rest_threshold=0.05
    )
    report["ensemble_oof_morphology"] = validation_metrics(
        ensemble, target, raw, intervals, movement_threshold=0.1, rest_threshold=0.05
    )
    report.update(
        {
            "protocol": "fixed-configuration event-grouped nested OOF six-seed ensemble",
            "released_test_touched": False,
            "outer_validation_used_for_seed_exclusion": False,
            "subject": args.subject,
            "finger": args.finger,
            "selected_schedules_by_seed": selected_schedules,
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "initialized_oof.npy", initialized, allow_pickle=False)
    np.save(args.output / "target_oof.npy", target, allow_pickle=False)
    np.save(args.output / "ensemble_oof.npy", ensemble, allow_pickle=False)
    plot_oof(args.output / "oof_trace.png", raw, target, initialized, ensemble)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
