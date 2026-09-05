#!/usr/bin/env python3
"""Build purged, multi-finger event-balanced folds in a labeled-data scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

from ecog_decoding.training import FINGER_NAMES


TARGETS = {1: "local_w2_q10", 2: "local_w1_q10", 3: "local_w4_q20"}


def selection_stop(
    *, raw_target_rows: int, model_fit_stop: int, selection_scope: str
) -> int:
    """Return the exclusive raw-target row used by model-selection folds."""
    if not 0 < model_fit_stop <= raw_target_rows:
        raise ValueError("model-fit boundary must lie inside the target recording")
    if selection_scope == "model-fit":
        return model_fit_stop
    if selection_scope == "full-development":
        return raw_target_rows
    raise ValueError(f"unknown selection scope: {selection_scope}")


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    return [
        (int(start), int(stop))
        for start, stop in zip(
            np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        )
    ]


def intervals(mask: np.ndarray) -> list[list[int]]:
    return [[start, stop] for start, stop in runs(mask)]


def event_groups(
    target: np.ndarray,
    trigger_finger: int,
    threshold: float,
    merge_gap_bins: int,
    minimum_event_bins: int,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    smooth = ndimage.gaussian_filter1d(target, sigma=1.0, axis=0, mode="nearest")
    active = smooth >= threshold
    union = ndimage.binary_closing(
        active[:, trigger_finger], structure=np.ones(merge_gap_bins + 1)
    )
    events = [
        (start, stop)
        for start, stop in runs(union)
        if stop - start >= minimum_event_bins
    ]
    if len(events) < 3:
        raise RuntimeError("too few movement events to construct grouped folds")
    boundaries = [0]
    for left, right in zip(events[:-1], events[1:]):
        boundaries.append((left[1] + right[0]) // 2)
    boundaries.append(target.shape[0])
    groups = [(boundaries[index], boundaries[index + 1]) for index in range(len(events))]
    return groups, smooth, active


def group_features(
    groups: list[tuple[int, int]],
    smooth: np.ndarray,
    active: np.ndarray,
    threshold: float,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for start, stop in groups:
        segment = smooth[start:stop]
        segment_active = active[start:stop]
        event_counts = np.asarray(
            [
                sum(end - begin >= 3 for begin, end in runs(segment_active[:, finger]))
                for finger in range(segment.shape[1])
            ],
            dtype=np.float64,
        )
        active_bins = segment_active.sum(axis=0, dtype=np.float64)
        movement_mass = np.maximum(segment - threshold, 0.0).sum(axis=0)
        values.append(
            np.r_[float(stop - start), event_counts, active_bins, movement_mass]
        )
    return np.stack(values)


def balanced_assignment(
    features: np.ndarray,
    folds: int,
    seed: int,
    trials: int,
) -> tuple[np.ndarray, float]:
    target = features.sum(axis=0) / folds
    scale = np.maximum(target, 1.0)
    importance = np.linalg.norm(features / scale, axis=1)
    generator = np.random.default_rng(seed)
    best_assignment: np.ndarray | None = None
    best_score = np.inf
    for _ in range(trials):
        order = np.argsort(-(importance + generator.gumbel(scale=0.15, size=importance.size)))
        totals = np.zeros((folds, features.shape[1]), dtype=np.float64)
        counts = np.zeros(folds, dtype=np.int64)
        assignment = np.full(features.shape[0], -1, dtype=np.int64)
        for position, group in enumerate(order):
            candidates = np.arange(folds) if position >= folds else np.asarray([position])
            scores = []
            for fold in candidates:
                proposed = totals.copy()
                proposed[fold] += features[group]
                normalized = proposed / scale
                feature_imbalance = np.mean(np.var(normalized, axis=0))
                count_imbalance = np.var((counts + (np.arange(folds) == fold)) / max(1, features.shape[0] / folds))
                scores.append(feature_imbalance + 0.05 * count_imbalance)
            chosen = int(candidates[int(np.argmin(scores))])
            assignment[group] = chosen
            totals[chosen] += features[group]
            counts[chosen] += 1
        normalized = totals / scale
        score = float(np.mean(np.var(normalized, axis=0)) + 0.05 * np.var(counts))
        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()
    if best_assignment is None:
        raise RuntimeError("fold assignment failed")
    return best_assignment, best_score


def fold_summary(
    mask: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    selected = target[mask]
    summary: dict[str, object] = {
        "bins": int(mask.sum()),
        "seconds": float(mask.sum() / 25.0),
        "per_finger": {},
    }
    for finger, name in enumerate(FINGER_NAMES):
        values = selected[:, finger]
        moving = values >= threshold
        summary["per_finger"][name] = {
            "moving_fraction": float(moving.mean()),
            "movement_mass": float(np.maximum(values - threshold, 0.0).sum()),
            "mean_active": float(values[moving].mean()) if np.any(moving) else 0.0,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=tuple(FINGER_NAMES))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/event_stratified_folds_v1"))
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--merge-gap-bins", type=int, default=12)
    parser.add_argument("--minimum-event-bins", type=int, default=3)
    parser.add_argument("--purge-bins", type=int, default=25)
    parser.add_argument("--assignment-trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--selection-scope",
        choices=("model-fit", "full-development"),
        default="model-fit",
        help=(
            "build folds inside the first two-thirds model-fitting partition "
            "or across the complete labeled competition training recording"
        ),
    )
    parser.add_argument(
        "--split-safe-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--target-map",
        type=Path,
        default=None,
        help="YAML mapping from subject and finger to target file stem",
    )
    args = parser.parse_args()
    if args.split_safe_targets and args.target_map is not None:
        parser.error("--split-safe-targets and --target-map are mutually exclusive")
    target_map = yaml.safe_load(args.target_map.read_text()) if args.target_map else {}

    for subject in args.subjects:
        prepared = args.prepared_root / f"sub{subject}"
        metadata = json.loads((prepared / "metadata.json").read_text())
        model_fit_stop = int(metadata["target_fit_samples_25hz"])
        offset = args.history - 1
        for finger_name in args.fingers:
            if args.target_map is not None:
                subject_targets = target_map.get(
                    subject, target_map.get(str(subject), {})
                )
                target_name = subject_targets.get(finger_name)
                if target_name is None:
                    raise KeyError(
                        f"target map has no entry for S{subject} {finger_name}"
                    )
            else:
                target_name = TARGETS[subject] + (
                    "_split_safe" if args.split_safe_targets else ""
                )
            complete_target = np.load(prepared / f"train_glove_{target_name}.npy")
            stop = selection_stop(
                raw_target_rows=int(complete_target.shape[0]),
                model_fit_stop=model_fit_stop,
                selection_scope=args.selection_scope,
            )
            target = np.asarray(complete_target[offset:stop], dtype=np.float32)
            trigger_finger = list(FINGER_NAMES).index(finger_name)
            groups, smooth, active = event_groups(
                target,
                trigger_finger,
                args.threshold,
                args.merge_gap_bins,
                args.minimum_event_bins,
            )
            features = group_features(groups, smooth, active, args.threshold)
            # The target finger defines this model's score, so prioritize its
            # event count, active duration, and movement mass while retaining
            # the other fingers as lower-weight confounder-balancing terms.
            priority_columns = np.asarray(
                (
                    1 + trigger_finger,
                    1 + len(FINGER_NAMES) + trigger_finger,
                    1 + 2 * len(FINGER_NAMES) + trigger_finger,
                )
            )
            priority = features[:, priority_columns]
            features = np.concatenate((features, priority, priority, priority), axis=1)
            assignment, objective = balanced_assignment(
                features,
                args.folds,
                args.seed + 100 * subject + trigger_finger,
                args.assignment_trials,
            )
            fold_reports: list[dict[str, object]] = []
            for fold in range(args.folds):
                validation = np.zeros(target.shape[0], dtype=bool)
                chosen = np.flatnonzero(assignment == fold)
                for group in chosen:
                    start, stop = groups[int(group)]
                    validation[start:stop] = True
                excluded = validation.copy()
                for start, stop in runs(validation):
                    excluded[
                        max(0, start - args.purge_bins) : min(
                            target.shape[0], stop + args.purge_bins
                        )
                    ] = True
                training = ~excluded
                fold_reports.append(
                    {
                        "fold": fold,
                        "validation_group_ids": chosen.tolist(),
                        "validation_intervals": intervals(validation),
                        "training_intervals_after_purge": intervals(training),
                        "purged_bins": int((~training & ~validation).sum()),
                        "validation_summary": fold_summary(validation, target, args.threshold),
                        "training_bins_after_purge": int(training.sum()),
                    }
                )
            report = {
                "subject": subject,
                "finger": finger_name,
                "target": target_name,
                "protocol": "per-finger event-grouped stratification with multi-finger balance and fold-specific temporal purge",
                "selection_scope": args.selection_scope,
                "model_fit_stop_raw_target_row": model_fit_stop,
                "selection_stop_raw_target_row": stop,
                "chronological_validation_included_in_selection": (
                    args.selection_scope == "full-development"
                ),
                "official_final_validation_touched": (
                    args.selection_scope == "full-development"
                ),
                "released_test_touched": False,
                "training_rows": int(target.shape[0]),
                "history_offset": offset,
                "threshold": args.threshold,
                "merge_gap_bins": args.merge_gap_bins,
                "purge_bins": args.purge_bins,
                "assignment_trials": args.assignment_trials,
                "assignment_seed": args.seed + 100 * subject + trigger_finger,
                "group_count": len(groups),
                "assignment_objective": objective,
                "groups": [
                    {"group": index, "start": start, "stop": stop, "fold": int(assignment[index])}
                    for index, (start, stop) in enumerate(groups)
                ],
                "folds": fold_reports,
            }
            output = args.output_root / f"sub{subject}" / finger_name
            output.mkdir(parents=True, exist_ok=True)
            (output / "folds.json").write_text(json.dumps(report, indent=2) + "\n")
            compact = {
                "subject": subject,
                "finger": finger_name,
                "groups": len(groups),
                "objective": objective,
                "fold_seconds": [round(item["validation_summary"]["seconds"], 1) for item in fold_reports],
                "target_moving_fraction": [
                    round(item["validation_summary"]["per_finger"][finger_name]["moving_fraction"], 3)
                    for item in fold_reports
                ],
            }
            print(json.dumps(compact), flush=True)


if __name__ == "__main__":
    main()
