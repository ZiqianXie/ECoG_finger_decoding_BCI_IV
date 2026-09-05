#!/usr/bin/env python3
"""Audit little-finger target cross-talk using development data only.

The audit deliberately never opens a released-test target or prediction.  It
combines the full-development event folds, the corresponding OOF little-finger
predictions, and the five baseline-corrected development glove channels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import nnls


FINGERS = ("thumb", "index", "middle", "ring", "little")
LITTLE = FINGERS.index("little")


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.std(left) < 1.0e-12 or np.std(right) < 1.0e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def interval_indices(intervals: list[list[int]]) -> np.ndarray:
    return np.concatenate(
        [np.arange(start, stop, dtype=np.int64) for start, stop in intervals]
    )


def load_targets(
    prepared_root: Path,
    target_map: dict,
    subject: int,
    rows: int,
    history_offset: int,
) -> np.ndarray:
    subject_targets = target_map.get(subject, target_map.get(str(subject)))
    columns = []
    for finger_index, finger in enumerate(FINGERS):
        target_name = str(subject_targets[finger])
        path = prepared_root / f"sub{subject}" / f"train_glove_{target_name}.npy"
        if "test" in path.name.lower():
            raise RuntimeError(f"refusing non-development target: {path}")
        values = np.load(path, mmap_mode="r")
        columns.append(
            np.asarray(
                values[history_offset : history_offset + rows, finger_index],
                dtype=np.float64,
            )
        )
    return np.column_stack(columns)


def load_little_oof(
    input_root: Path, subject: int, rows: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    prediction = np.full(rows, np.nan, dtype=np.float64)
    raw_target = np.full(rows, np.nan, dtype=np.float64)
    fold_id = np.full(rows, -1, dtype=np.int64)
    seed_counts: dict[str, int] = {}
    for fold in range(3):
        fold_root = input_root / f"sub{subject}" / "little" / f"fold{fold}"
        seed_roots = sorted(path for path in fold_root.glob("seed*") if path.is_dir())
        if not seed_roots:
            raise FileNotFoundError(f"no OOF seeds in {fold_root}")
        indices = np.load(seed_roots[0] / "validation_indices.npy")
        raw_target[indices] = np.load(seed_roots[0] / "validation_raw_target.npy")
        members = []
        for seed_root in seed_roots:
            seed_prediction = np.load(seed_root / "validation_prediction.npy")
            if np.std(seed_prediction) < 1.0e-4:
                continue
            members.append(np.asarray(seed_prediction, dtype=np.float64))
        if not members:
            raise RuntimeError(f"all seeds collapsed in {fold_root}")
        if np.any(fold_id[indices] >= 0):
            raise RuntimeError(f"overlapping OOF rows in {fold_root}")
        prediction[indices] = np.mean(members, axis=0)
        fold_id[indices] = fold
        seed_counts[str(fold)] = len(members)
    if (
        not np.isfinite(prediction).all()
        or not np.isfinite(raw_target).all()
        or np.any(fold_id < 0)
    ):
        raise RuntimeError(f"incomplete OOF coverage for S{subject} little")
    return prediction, raw_target, {"included_seed_count_by_fold": seed_counts}


def dominant_state(target: np.ndarray) -> np.ndarray:
    return np.argmax(target, axis=1)


def little_regimes(target: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    little = target[:, LITTLE]
    others = np.max(target[:, :LITTLE], axis=1)
    return {
        "rest": (little < threshold) & (others < threshold),
        "little_isolated": (little >= threshold) & (others < threshold),
        "little_dominant_coupled": (little >= threshold)
        & (others >= threshold)
        & (little >= others),
        "other_dominant_little_active": (little >= threshold)
        & (others >= threshold)
        & (little < others),
        "other_active_little_weak": (little < threshold) & (others >= threshold),
    }


def regime_metrics(
    target: np.ndarray, prediction: np.ndarray, threshold: float
) -> dict[str, dict[str, float | int]]:
    little = target[:, LITTLE]
    result = {}
    for name, mask in little_regimes(target, threshold).items():
        count = int(np.sum(mask))
        result[name] = {
            "bins": count,
            "fraction": float(np.mean(mask)),
            "target_mean": float(np.mean(little[mask])) if count else 0.0,
            "prediction_mean": float(np.mean(prediction[mask])) if count else 0.0,
            "pcc": pearson(prediction[mask], little[mask]) if count > 1 else 0.0,
            "rmse": (
                float(np.sqrt(np.mean((prediction[mask] - little[mask]) ** 2)))
                if count
                else 0.0
            ),
        }
    return result


def crossfit_soft_residual(
    target: np.ndarray,
    prediction: np.ndarray,
    definition: dict,
    threshold: float,
    strengths: tuple[float, ...],
) -> tuple[dict[str, np.ndarray], dict]:
    """Cross-fit a nonnegative passive-coupling estimate, then subtract softly.

    Coefficients are fitted only on each outer fold's purged training rows where
    another finger dominates.  The held-out fold is transformed without using
    its labels to fit those coefficients.  No activity is ever reassigned to a
    different finger; subtraction can only attenuate the little channel.
    """
    others = target[:, :LITTLE]
    little = target[:, LITTLE]
    output = {str(value): np.full(little.shape, np.nan) for value in strengths}
    fold_audit = []
    for fold in definition["folds"]:
        training = interval_indices(fold["training_intervals_after_purge"])
        validation = interval_indices(fold["validation_intervals"])
        other_peak = np.max(others[training], axis=1)
        passive = (other_peak >= threshold) & (little[training] < other_peak)
        fit_rows = training[passive]
        if fit_rows.size < 20:
            coefficient = np.zeros(LITTLE, dtype=np.float64)
        else:
            coefficient, _ = nnls(others[fit_rows], little[fit_rows])
        estimated = np.maximum(others[validation] @ coefficient, 0.0)
        for strength in strengths:
            output[str(strength)][validation] = np.maximum(
                little[validation] - strength * estimated, 0.0
            )
        fold_audit.append(
            {
                "fold": int(fold["fold"]),
                "passive_fit_bins": int(fit_rows.size),
                "coefficients": {
                    FINGERS[index]: float(value)
                    for index, value in enumerate(coefficient)
                },
            }
        )
    if any(not np.isfinite(values).all() for values in output.values()):
        raise RuntimeError("incomplete cross-fitted cleaning coverage")
    candidate_metrics = {}
    original_mass = max(float(np.sum(little)), 1.0e-12)
    for strength, values in output.items():
        candidate_metrics[strength] = {
            "oof_prediction_pcc": pearson(prediction, values),
            "target_mass_removed_fraction": float(1.0 - np.sum(values) / original_mass),
            "active_fraction": float(np.mean(values >= threshold)),
        }
    return output, {"folds": fold_audit, "candidates": candidate_metrics}


def fit_passive_coefficients(
    target: np.ndarray, training: np.ndarray, threshold: float
) -> tuple[np.ndarray, int]:
    others = target[:, :LITTLE]
    little = target[:, LITTLE]
    other_peak = np.max(others[training], axis=1)
    passive = (other_peak >= threshold) & (little[training] < other_peak)
    fit_rows = training[passive]
    if fit_rows.size < 20:
        return np.zeros(LITTLE, dtype=np.float64), int(fit_rows.size)
    coefficient, _ = nnls(others[fit_rows], little[fit_rows])
    return coefficient, int(fit_rows.size)


def selected_feature_ridge_probe(
    subject: int,
    args: argparse.Namespace,
    target: np.ndarray,
    raw_target: np.ndarray,
    definition: dict,
) -> dict:
    """Retrain a fast linear decoder on original-target-selected fold features.

    Feature indices were selected without each outer fold, so the probe remains
    held out with respect to that fold.  Because the dictionary was originally
    selected for the unmodified target, this is a conservative screening test,
    not the final nested model-selection experiment.
    """
    import torch
    features = np.load(
        args.feature_root / f"sub{subject}" / "train_initialized_window_features.npy",
        mmap_mode="r",
    )
    strengths = tuple(args.probe_strengths)
    variant_names = [f"residual_{value}" for value in strengths] + [
        f"dominance_{value}" for value in args.dominance_strengths
    ]
    predictions = {
        name: np.full(target.shape[0], np.nan, dtype=np.float64)
        for name in variant_names
    }
    transformed = {
        name: np.full(target.shape[0], np.nan, dtype=np.float64)
        for name in variant_names
    }
    fold_metrics = {name: [] for name in variant_names}
    fold_details = []
    for fold in definition["folds"]:
        fold_number = int(fold["fold"])
        training = interval_indices(fold["training_intervals_after_purge"])
        validation = interval_indices(fold["validation_intervals"])
        checkpoint = torch.load(
            args.input_root
            / f"sub{subject}"
            / "little"
            / f"fold{fold_number}"
            / "seed0"
            / "model.pt",
            map_location="cpu",
            weights_only=False,
        )
        selected = np.asarray(checkpoint["feature_indices"], dtype=np.int64)
        # Select columns before rows.  Fancy row indexing first would copy the
        # entire 16k-column matrix and then discard almost all of it.
        selected_features = np.asarray(features[:, selected], dtype=np.float64)
        train_x = selected_features[training]
        validation_x = selected_features[validation]
        mean = np.mean(train_x, axis=0)
        scale = np.std(train_x, axis=0)
        scale[scale < 1.0e-8] = 1.0
        train_x = (train_x - mean) / scale
        validation_x = (validation_x - mean) / scale
        coefficient, passive_bins = fit_passive_coefficients(
            target, training, args.threshold
        )
        estimated = np.maximum(target[:, :LITTLE] @ coefficient, 0.0)
        candidate_targets = {
            f"residual_{strength}": np.maximum(
                target[:, LITTLE] - strength * estimated, 0.0
            )
            for strength in strengths
        }
        other_peak = np.max(target[:, :LITTLE], axis=1)
        excess = np.maximum(other_peak - target[:, LITTLE], 0.0)
        dominance_factor = np.exp(-excess / args.dominance_temperature)
        for strength in args.dominance_strengths:
            candidate_targets[f"dominance_{strength}"] = target[:, LITTLE] * (
                (1.0 - strength) + strength * dominance_factor
            )
        cleaned_targets = np.column_stack(
            [candidate_targets[name] for name in variant_names]
        )
        # RidgeCV shares one matrix decomposition across the target variants.
        # alpha_per_target retains independent regularization choices without
        # repeating the expensive solve for each strength.
        estimates, selected_alphas = ridge_gcv_multioutput(
            train_x,
            cleaned_targets[training],
            validation_x,
            np.logspace(-3, 5, 17),
        )
        for variant_index, key in enumerate(variant_names):
            cleaned = cleaned_targets[:, variant_index]
            estimate = estimates[:, variant_index]
            predictions[key][validation] = estimate
            transformed[key][validation] = cleaned[validation]
            fold_metrics[key].append(
                {
                    "fold": fold_number,
                    "alpha": float(selected_alphas[variant_index]),
                    "pcc_original_cleaned_target": pearson(
                        estimate, target[validation, LITTLE]
                    ),
                    "pcc_transformed_target": pearson(
                        estimate, cleaned[validation]
                    ),
                    "pcc_raw_glove_target": pearson(
                        estimate, raw_target[validation]
                    ),
                }
            )
        fold_details.append(
            {
                "fold": fold_number,
                "selected_feature_count": int(selected.size),
                "passive_fit_bins": passive_bins,
                "passive_coefficients": {
                    FINGERS[index]: float(value)
                    for index, value in enumerate(coefficient)
                },
            }
        )
    result = {
        "folds": fold_details,
        "dominance_temperature": args.dominance_temperature,
        "variants": {},
    }
    for key in variant_names:
        if not np.isfinite(predictions[key]).all():
            raise RuntimeError(f"incomplete ridge OOF probe for S{subject}, variant {key}")
        result["variants"][key] = {
            "fold_metrics": fold_metrics[key],
            "mean_fold_pcc_original_cleaned_target": float(
                np.mean([item["pcc_original_cleaned_target"] for item in fold_metrics[key]])
            ),
            "mean_fold_pcc_transformed_target": float(
                np.mean([item["pcc_transformed_target"] for item in fold_metrics[key]])
            ),
            "mean_fold_pcc_raw_glove_target": float(
                np.mean([item["pcc_raw_glove_target"] for item in fold_metrics[key]])
            ),
            "stitched_pcc_original_cleaned_target": pearson(
                predictions[key], target[:, LITTLE]
            ),
            "stitched_pcc_transformed_target": pearson(
                predictions[key], transformed[key]
            ),
            "stitched_pcc_raw_glove_target": pearson(predictions[key], raw_target),
        }
    return result


def ridge_gcv_multioutput(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Primal ridge GCV with one eigendecomposition for all target variants."""
    target_mean = np.mean(train_y, axis=0)
    centered_y = train_y - target_mean
    gram = train_x.T @ train_x
    eigenvalue, eigenvector = np.linalg.eigh(gram)
    eigenvalue = np.maximum(eigenvalue, 0.0)
    projected_train = train_x @ eigenvector
    right_hand = projected_train.T @ centered_y
    scores = np.empty((alphas.size, train_y.shape[1]), dtype=np.float64)
    coefficients = []
    for alpha_index, alpha in enumerate(alphas):
        coefficient = right_hand / (eigenvalue[:, None] + alpha)
        coefficients.append(coefficient)
        residual = centered_y - projected_train @ coefficient
        degrees_of_freedom = float(np.sum(eigenvalue / (eigenvalue + alpha)))
        denominator = max(float(train_x.shape[0]) - degrees_of_freedom, 1.0)
        scores[alpha_index] = np.sum(residual**2, axis=0) / denominator**2
    selected_indices = np.argmin(scores, axis=0)
    projected_validation = validation_x @ eigenvector
    prediction = np.column_stack(
        [
            projected_validation @ coefficients[index][:, target_index]
            + target_mean[target_index]
            for target_index, index in enumerate(selected_indices)
        ]
    )
    return prediction, np.asarray(alphas)[selected_indices]


def subject_audit(
    subject: int,
    args: argparse.Namespace,
    target_map: dict,
) -> tuple[dict, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    fold_path = args.fold_root / f"sub{subject}" / "little" / "folds.json"
    definition = json.loads(fold_path.read_text())
    rows = int(definition["training_rows"])
    offset = int(definition["history_offset"])
    target = load_targets(args.prepared_root, target_map, subject, rows, offset)
    prediction, raw_target, seed_audit = load_little_oof(
        args.input_root, subject, rows
    )
    regimes = little_regimes(target, args.threshold)
    little = target[:, LITTLE]
    active = little >= args.threshold
    other_active = np.max(target[:, :LITTLE], axis=1) >= args.threshold
    other_dominant = dominant_state(target) != LITTLE
    total_little_energy = max(float(np.sum(little**2)), 1.0e-12)
    crossfit, residual_audit = crossfit_soft_residual(
        target,
        prediction,
        definition,
        args.threshold,
        tuple(args.residual_strengths),
    )
    target_correlations = {
        finger: pearson(little, target[:, index])
        for index, finger in enumerate(FINGERS[:LITTLE])
    }
    prediction_correlations = {
        finger: pearson(prediction, target[:, index])
        for index, finger in enumerate(FINGERS)
    }
    result = {
        "subject": subject,
        "rows": rows,
        "target": str(
            target_map.get(subject, target_map.get(str(subject)))["little"]
        ),
        "oof_seed_audit": seed_audit,
        "little_oof_pcc_original_target": pearson(prediction, little),
        "little_target_active_fraction": float(np.mean(active)),
        "little_active_overlap_any_other_fraction": float(
            np.mean(other_active[active]) if np.any(active) else 0.0
        ),
        "little_active_other_dominant_fraction": float(
            np.mean(other_dominant[active]) if np.any(active) else 0.0
        ),
        "little_energy_in_other_dominant_bins_fraction": float(
            np.sum(little[other_dominant] ** 2) / total_little_energy
        ),
        "little_target_correlations": target_correlations,
        "little_decoder_correlations_with_all_targets": prediction_correlations,
        "regimes": regime_metrics(target, prediction, args.threshold),
        "crossfit_soft_residual": residual_audit,
    }
    if args.run_ridge_probe:
        result["selected_feature_ridge_probe"] = selected_feature_ridge_probe(
            subject, args, target, raw_target, definition
        )
    return result, target, prediction, crossfit


def make_figure(
    audits: list[dict],
    arrays: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for row, (audit, (target, prediction, candidates)) in enumerate(zip(audits, arrays)):
        subject = audit["subject"]
        correlations = audit["little_decoder_correlations_with_all_targets"]
        axes[row, 0].bar(FINGERS, [correlations[name] for name in FINGERS], color="#4f46e5")
        axes[row, 0].axhline(0, color="black", linewidth=0.7)
        axes[row, 0].set_ylim(-0.1, 0.85)
        axes[row, 0].set_ylabel(f"S{subject}")
        axes[row, 0].set_title("OOF little decoder vs each target" if row == 0 else "")

        regimes = audit["regimes"]
        regime_names = (
            "little_isolated",
            "little_dominant_coupled",
            "other_dominant_little_active",
        )
        axes[row, 1].bar(
            ("isolated", "little-dominant\ncoupled", "other-dominant\nlittle-active"),
            [regimes[name]["fraction"] for name in regime_names],
            color=("#16a34a", "#eab308", "#dc2626"),
        )
        axes[row, 1].set_ylim(0, 0.35)
        axes[row, 1].set_title("Fraction of development bins" if row == 0 else "")

        metrics = audit["crossfit_soft_residual"]["candidates"]
        strengths = [float(value) for value in metrics]
        pcc = [metrics[str(value)]["oof_prediction_pcc"] for value in strengths]
        removed = [metrics[str(value)]["target_mass_removed_fraction"] for value in strengths]
        axes[row, 2].plot(strengths, pcc, marker="o", color="#2563eb", label="OOF PCC")
        twin = axes[row, 2].twinx()
        twin.plot(strengths, removed, marker="s", color="#dc2626", label="mass removed")
        axes[row, 2].set_ylim(0.15, 0.55)
        twin.set_ylim(0, 0.5)
        axes[row, 2].set_xlabel("passive-coupling subtraction strength")
        if row == 0:
            axes[row, 2].set_title("Cross-fitted little-only cleaning")
        if row == 1:
            axes[row, 2].set_ylabel("OOF PCC", color="#2563eb")
            twin.set_ylabel("target mass removed", color="#dc2626")
    for axis in axes[:, 0]:
        axis.tick_params(axis="x", rotation=25)
    figure.suptitle(
        "Little-finger training-only audit: coupling, decoder specificity, and soft residual cleaning",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def example_center(mask: np.ndarray, score: np.ndarray) -> int:
    candidates = np.flatnonzero(mask)
    if not candidates.size:
        return int(np.argmax(score))
    return int(candidates[np.argmax(score[candidates])])


def make_example_figure(
    audits: list[dict],
    arrays: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    output: Path,
    strength: float = 0.5,
    half_window: int = 100,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(14, 9), constrained_layout=True)
    for row, (audit, (target, prediction, candidates)) in enumerate(zip(audits, arrays)):
        little = target[:, LITTLE]
        other_index = np.argmax(target[:, :LITTLE], axis=1)
        other_peak = np.max(target[:, :LITTLE], axis=1)
        little_active = little >= 0.08
        centers = (
            example_center(little_active & (little >= other_peak), little),
            example_center(little_active & (little < other_peak), little),
        )
        for column, center in enumerate(centers):
            start = max(0, center - half_window)
            stop = min(target.shape[0], center + half_window)
            time = (np.arange(start, stop) - center) / 25.0
            axis = axes[row, column]
            axis.plot(time, little[start:stop], color="black", linewidth=2.0, label="little target")
            axis.plot(
                time,
                candidates[str(strength)][start:stop],
                color="#16a34a",
                linewidth=1.7,
                label=f"little residual ({strength:g})",
            )
            axis.plot(
                time,
                prediction[start:stop],
                color="#2563eb",
                linewidth=1.4,
                label="OOF little prediction",
            )
            dominant = int(other_index[center])
            axis.plot(
                time,
                target[start:stop, dominant],
                color="#dc2626",
                linewidth=1.0,
                alpha=0.8,
                label=f"{FINGERS[dominant]} target",
            )
            axis.axvline(0, color="#6b7280", linewidth=0.7, linestyle="--")
            axis.set_ylabel(f"S{audit['subject']}")
            if row == 0:
                axis.set_title(
                    "Little-dominant development event"
                    if column == 0
                    else "Other-finger-dominant development event"
                )
            if row == 2:
                axis.set_xlabel("time from selected peak (s)")
            if row == 0 and column == 0:
                axis.legend(loc="upper right", fontsize=8)
    figure.suptitle(
        "Representative development events (no released-test labels)", fontsize=15
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path, default=Path("outputs/event_lars_e2e_fulldev_seq50_v1")
    )
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path("outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1"),
    )
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument(
        "--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml")
    )
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--subjects", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument(
        "--residual-strengths", nargs="+", type=float, default=(0.0, 0.25, 0.5, 0.75, 1.0)
    )
    parser.add_argument(
        "--probe-strengths", nargs="+", type=float, default=(0.0, 0.5, 1.0)
    )
    parser.add_argument(
        "--dominance-strengths", nargs="+", type=float, default=(0.5, 1.0)
    )
    parser.add_argument("--dominance-temperature", type=float, default=0.10)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("outputs/windowed_ica_wavelet_asymmetric_v1"),
    )
    parser.add_argument("--run-ridge-probe", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("docs/results/little-finger-training-only-audit.json")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("docs/figures/little-finger-training-only-audit.png")
    )
    parser.add_argument(
        "--example-figure",
        type=Path,
        default=Path("docs/figures/little-finger-training-only-examples.png"),
    )
    args = parser.parse_args()
    target_map = yaml.safe_load(args.target_map.read_text())
    audits = []
    arrays = []
    for subject in args.subjects:
        audit, target, prediction, candidates = subject_audit(subject, args, target_map)
        audits.append(audit)
        arrays.append((target, prediction, candidates))
    report = {
        "protocol": "development-only little-finger audit with purged event OOF predictions",
        "scope": "little-finger target only; the other four targets are untouched",
        "released_test_loaded": False,
        "interpretation_guardrail": (
            "The residual sweep is diagnostic because the same OOF predictions are compared "
            "with several target transforms. A cleaned-target decoder must be retrained and "
            "selected in nested development folds before promotion."
        ),
        "threshold": args.threshold,
        "subjects": {str(item["subject"]): item for item in audits},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    make_figure(audits, arrays, args.figure)
    make_example_figure(audits, arrays, args.example_figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
