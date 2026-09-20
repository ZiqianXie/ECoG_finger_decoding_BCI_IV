#!/usr/bin/env python3
"""Select a single-wavelet learning-rate run without inspecting outer folds.

Each candidate has already selected its optimizer-update schedule from the
inner folds of each outer-training scope.  This script performs the remaining
learning-rate selection using only that selected schedule's inner-fold PCC,
then stitches the corresponding untouched outer predictions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ecog_decoding.training import FINGER_NAMES
from single_wavelet_support import OFFSET, pearson


@dataclass(frozen=True)
class Candidate:
    name: str
    root: Path
    report: dict[str, object]


def candidate_argument(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("candidate must be NAME=SUMMARY_ROOT") from error
    if not name or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=SUMMARY_ROOT")
    return name, Path(path)


def outer_record(report: dict[str, object], outer_fold: int) -> dict[str, object]:
    matches = [
        item
        for item in report["outer_folds"]
        if int(item["outer_fold"]) == outer_fold
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one record for outer fold {outer_fold}")
    return matches[0]


def choose_candidate(
    candidates: list[Candidate], outer_fold: int
) -> tuple[Candidate, dict[str, object], float, float]:
    """Choose LR from inner PCC after each LR's inner schedule selection."""
    proposals = []
    for candidate in candidates:
        record = outer_record(candidate.report, outer_fold)
        schedule = str(record["selected_schedule"])
        inner = record["selection_summary"][schedule]
        proposals.append(
            (
                float(inner["mean_raw_pcc"]),
                -float(inner["sem_raw_pcc"]),
                candidate.name,
                candidate,
                record,
            )
        )
    _, _, _, selected, record = max(proposals, key=lambda item: item[:3])
    inner = record["selection_summary"][str(record["selected_schedule"])]
    return (
        selected,
        record,
        float(inner["mean_raw_pcc"]),
        float(inner["sem_raw_pcc"]),
    )


def interval_indices(intervals: list[list[int]]) -> np.ndarray:
    return np.concatenate(
        [np.arange(int(start), int(stop), dtype=np.int64) for start, stop in intervals]
    )


def plot_oof(
    output: Path,
    raw: np.ndarray,
    target: np.ndarray,
    initialized: np.ndarray,
    selected: np.ndarray,
) -> None:
    observed = np.isfinite(selected)
    rows = np.flatnonzero(observed)
    figure, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    axes[0].plot(rows, raw[observed], color="black", linewidth=0.8, label="raw glove")
    axes[0].plot(rows, target[observed], color="#999999", linewidth=0.8, label="cleaned target")
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("target")
    axes[1].plot(rows, initialized[observed], linewidth=0.8, label="LARS initialization")
    axes[1].plot(rows, selected[observed], linewidth=0.8, label="selected residual LSTM")
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("prediction")
    axes[1].set_xlabel("25 Hz bin")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--finger", choices=tuple(FINGER_NAMES), required=True)
    parser.add_argument("--candidate", action="append", type=candidate_argument, required=True)
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path("outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1"),
    )
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    for name, root in args.candidate:
        report = json.loads((root / "summary.json").read_text())
        if int(report["subject"]) != args.subject or report["finger"] != args.finger:
            raise ValueError(f"{name}: summary subject/finger does not match request")
        candidates.append(Candidate(name=name, root=root, report=report))
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")

    reference_initialized = np.load(candidates[0].root / "initialized_oof.npy")
    reference_target = np.load(candidates[0].root / "target_oof.npy")
    for candidate in candidates[1:]:
        initialized = np.load(candidate.root / "initialized_oof.npy")
        target = np.load(candidate.root / "target_oof.npy")
        if not np.allclose(reference_initialized, initialized, equal_nan=True, atol=1e-6):
            raise ValueError(f"{candidate.name}: initialized OOF values differ")
        if not np.allclose(reference_target, target, equal_nan=True, atol=1e-6):
            raise ValueError(f"{candidate.name}: target OOF values differ")

    fold_definition = json.loads(
        (
            args.fold_root
            / f"sub{args.subject}"
            / args.finger
            / "folds.json"
        ).read_text()
    )
    rows = int(fold_definition["training_rows"])
    if rows != reference_target.size:
        raise ValueError("fold definition and candidate arrays have different lengths")
    selected_oof = np.full(rows, np.nan, dtype=np.float32)
    selections = []
    for outer_fold in range(3):
        candidate, record, mean_pcc, sem_pcc = choose_candidate(candidates, outer_fold)
        validation = interval_indices(
            fold_definition["folds"][outer_fold]["validation_intervals"]
        )
        prediction = np.load(candidate.root / "selected_oof.npy")
        if not np.isfinite(prediction[validation]).all():
            raise ValueError(f"{candidate.name}: missing outer {outer_fold} predictions")
        if np.isfinite(selected_oof[validation]).any():
            raise ValueError(f"outer fold {outer_fold} overlaps an earlier fold")
        selected_oof[validation] = prediction[validation]
        selections.append(
            {
                "outer_fold": outer_fold,
                "candidate": candidate.name,
                "candidate_root": str(candidate.root),
                "selected_schedule": record["selected_schedule"],
                "inner_mean_raw_pcc": mean_pcc,
                "inner_sem_raw_pcc": sem_pcc,
                "outer_metrics_after_selection": record["selected_outer_metrics"],
            }
        )

    finger_index = list(FINGER_NAMES).index(args.finger)
    raw = np.asarray(
        np.load(
            args.prepared_root / f"sub{args.subject}" / "train_glove_25hz_raw.npy",
            mmap_mode="r",
        )[OFFSET : OFFSET + rows, finger_index]
    )
    observed = np.isfinite(selected_oof)
    report = {
        "protocol": (
            "each learning rate first selects its optimizer-update schedule inside each "
            "outer-training scope; learning rate is then selected by the corresponding "
            "inner mean raw PCC; untouched outer predictions are stitched only afterward"
        ),
        "released_test_touched": False,
        "outer_validation_used_for_selection": False,
        "subject": args.subject,
        "finger": args.finger,
        "candidate_roots": {item.name: str(item.root) for item in candidates},
        "outer_selections": selections,
        "stitched_initialized_raw_pcc": pearson(
            reference_initialized[observed], raw[observed]
        ),
        "stitched_selected_raw_pcc": pearson(selected_oof[observed], raw[observed]),
        "stitched_selected_cleaned_pcc": pearson(
            selected_oof[observed], reference_target[observed]
        ),
    }
    report["gain_over_initialization"] = (
        report["stitched_selected_raw_pcc"] - report["stitched_initialized_raw_pcc"]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "initialized_oof.npy", reference_initialized, allow_pickle=False)
    np.save(args.output / "selected_oof.npy", selected_oof, allow_pickle=False)
    np.save(args.output / "target_oof.npy", reference_target, allow_pickle=False)
    plot_oof(
        args.output / "oof_trace.png",
        raw,
        reference_target,
        reference_initialized,
        selected_oof,
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
