#!/usr/bin/env python3
"""Event-safe joint LARS benchmark for CSP/band and ICA/wavelet atoms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import linalg, signal
import yaml

from benchmark_csp_ridge import BANDS_HZ, regularized_covariance
from benchmark_ridge_target_variants import lagged
from ecog_decoding.training import FINGER_NAMES
from train_event_grouped_lars_e2e_nested import fit_or_load_inner_lars
from train_event_grouped_lars_lstm import indices_from_intervals, pearson


def target_name(target_map: dict[object, object], subject: int, finger: str) -> str:
    subject_map = target_map.get(subject, target_map.get(str(subject), {}))
    if not isinstance(subject_map, dict) or finger not in subject_map:
        raise KeyError(f"target map has no entry for S{subject} {finger}")
    return str(subject_map[finger])


def selected_csp_filters(
    filtered_bins: np.ndarray,
    training_rows: np.ndarray,
    active: np.ndarray,
    rest: np.ndarray,
    components_per_tail: int,
) -> tuple[np.ndarray, list[float]]:
    """Fit one movement/rest CSP using only the requested lagged rows."""
    active_rows = training_rows[active[training_rows]]
    rest_rows = training_rows[rest[training_rows]]
    if active_rows.size < 2 or rest_rows.size < 2:
        raise RuntimeError("CSP fold has too few active or rest bins")
    active_values = filtered_bins[active_rows].reshape(-1, filtered_bins.shape[-1])
    rest_values = filtered_bins[rest_rows].reshape(-1, filtered_bins.shape[-1])
    active_covariance = regularized_covariance(active_values)
    rest_covariance = regularized_covariance(rest_values)
    eigenvalues, eigenvectors = linalg.eigh(
        active_covariance,
        active_covariance + rest_covariance,
        check_finite=False,
    )
    order = np.r_[
        np.arange(components_per_tail),
        np.arange(eigenvalues.size - components_per_tail, eigenvalues.size),
    ]
    weights = eigenvectors[:, order].T
    weights /= np.linalg.norm(weights, axis=1, keepdims=True).clip(min=1.0e-12)
    return weights.astype(np.float32), eigenvalues[order].tolist()


def binned_energy(filtered: np.ndarray, weights: np.ndarray, samples_per_bin: int) -> np.ndarray:
    projected = filtered @ weights.T
    bins = projected.shape[0] // samples_per_bin
    values = projected[: bins * samples_per_bin].reshape(
        bins, samples_per_bin, weights.shape[0]
    )
    return np.log1p(np.sqrt(np.sum(values * values, axis=1))).astype(np.float32)


def predict(selection: dict[str, object], features: np.ndarray, rows: np.ndarray) -> np.ndarray:
    chosen = np.asarray(selection["selected_source"], dtype=np.int64)
    mean = np.asarray(selection["feature_mean"], dtype=np.float64)
    scale = np.asarray(selection["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(selection["coefficients"], dtype=np.float64)
    standardized = (np.asarray(features[rows][:, chosen], dtype=np.float64) - mean) / scale
    return (standardized @ coefficients + float(selection["intercept"])).astype(np.float32)


def csp_selection_audit(
    selected: np.ndarray,
    wavelet_features: int,
    csp_features_per_bin: int,
    filters_per_state: int,
) -> dict[str, object]:
    wavelet = selected[selected < wavelet_features]
    csp = selected[selected >= wavelet_features] - wavelet_features
    bands = np.zeros(len(BANDS_HZ), dtype=np.int64)
    state = {"decoded_finger": 0, "any_movement": 0}
    for index in csp:
        atom = int(index % csp_features_per_bin)
        band = atom // (2 * filters_per_state)
        bands[band] += 1
        key = "decoded_finger" if atom % (2 * filters_per_state) < filters_per_state else "any_movement"
        state[key] += 1
    return {
        "ica_wavelet_features": int(wavelet.size),
        "csp_designed_band_features": int(csp.size),
        "csp_selected_by_band_hz": {
            f"{low:g}-{high:g}": int(count)
            for (low, high), count in zip(BANDS_HZ, bands, strict=True)
        },
        "csp_selected_by_spatial_contrast": state,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--wavelet-root", type=Path, default=Path("outputs/windowed_ica_wavelet_asymmetric_v1"))
    parser.add_argument("--fold-root", type=Path, default=Path("outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/event_heterogeneous_dictionary_cache_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/event_heterogeneous_dictionary_v1"))
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--samples-per-bin", type=int, default=40)
    parser.add_argument("--components-per-tail", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=1024)
    args = parser.parse_args()

    prepared = args.prepared_root / f"sub{args.subject}"
    output_subject = args.output_root / f"sub{args.subject}"
    output_subject.mkdir(parents=True, exist_ok=True)
    target_map = yaml.safe_load(args.target_map.read_text())
    definitions = {
        finger: json.loads(
            (args.fold_root / f"sub{args.subject}" / finger / "folds.json").read_text()
        )
        for finger in FINGER_NAMES
    }
    row_counts = {int(value["training_rows"]) for value in definitions.values()}
    if len(row_counts) != 1:
        raise RuntimeError("finger fold definitions disagree on row count")
    rows = row_counts.pop()
    offset = args.history - 1
    ecog = np.load(prepared / "train_ecog.npy", mmap_mode="r")
    available_bins = ecog.shape[0] // args.samples_per_bin
    if rows + offset > available_bins:
        raise RuntimeError("ECoG recording is shorter than the lagged fold definition")
    raw = np.load(prepared / "train_glove_25hz_raw.npy")[offset : offset + rows]
    targets = np.column_stack(
        [
            np.load(prepared / f"train_glove_{target_name(target_map, args.subject, finger)}.npy")[
                offset : offset + rows, finger_index
            ]
            for finger_index, finger in enumerate(FINGER_NAMES)
        ]
    ).astype(np.float32)
    wavelet = np.load(
        args.wavelet_root / f"sub{args.subject}" / "train_initialized_window_features.npy",
        mmap_mode="r",
    )[:rows]

    keys = [(finger, fold) for finger in FINGER_NAMES for fold in range(3)]
    energy_parts: dict[tuple[str, int], list[np.ndarray]] = {key: [] for key in keys}
    filter_audit: dict[tuple[str, int], list[dict[str, object]]] = {key: [] for key in keys}
    target_moving = targets > 0.20
    rest = np.max(targets, axis=1) < 0.05
    raw_bins = np.asarray(ecog[: available_bins * args.samples_per_bin]).reshape(
        available_bins, args.samples_per_bin, ecog.shape[1]
    )
    filters_per_state = 2 * args.components_per_tail
    for low, high in BANDS_HZ:
        sos = signal.butter(
            4, (low, high), btype="bandpass", fs=1000.0, output="sos"
        )
        filtered = signal.sosfiltfilt(
            sos, raw_bins.reshape(-1, ecog.shape[1]), axis=0
        ).astype(np.float32)
        filtered_bins = filtered.reshape(
            available_bins, args.samples_per_bin, ecog.shape[1]
        )[offset : offset + rows]
        for finger_index, finger in enumerate(FINGER_NAMES):
            for fold_index, fold in enumerate(definitions[finger]["folds"]):
                key = (finger, fold_index)
                training = indices_from_intervals(fold["training_intervals_after_purge"])
                own, own_values = selected_csp_filters(
                    filtered_bins,
                    training,
                    target_moving[:, finger_index],
                    rest,
                    args.components_per_tail,
                )
                shared, shared_values = selected_csp_filters(
                    filtered_bins,
                    training,
                    np.max(target_moving, axis=1),
                    rest,
                    args.components_per_tail,
                )
                weights = np.concatenate((own, shared), axis=0)
                energy_parts[key].append(
                    binned_energy(filtered, weights, args.samples_per_bin)
                )
                filter_audit[key].append(
                    {
                        "band_hz": [low, high],
                        "decoded_finger_eigenvalues": own_values,
                        "any_movement_eigenvalues": shared_values,
                    }
                )
        print(f"S{args.subject} filtered and projected {low:g}-{high:g} Hz", flush=True)

    subject_report: dict[str, object] = {
        "subject": args.subject,
        "protocol": "full-development purged event OOF; fold-trained CSP plus fixed ICA/wavelet dictionary",
        "released_test_touched": False,
        "bands_hz": BANDS_HZ,
        "per_finger": {},
    }
    for finger_index, finger in enumerate(FINGER_NAMES):
        stitched = {
            name: np.full(rows, np.nan, dtype=np.float32)
            for name in ("ica_wavelet_only", "csp_designed_bands_only", "joint_dictionary")
        }
        folds: list[dict[str, object]] = []
        for fold_index, fold in enumerate(definitions[finger]["folds"]):
            key = (finger, fold_index)
            csp_energy = np.concatenate(energy_parts[key], axis=1)
            csp = lagged(csp_energy, args.history)[:rows]
            feature_sets = {
                "ica_wavelet_only": wavelet,
                "csp_designed_bands_only": csp,
                "joint_dictionary": np.concatenate((wavelet, csp), axis=1),
            }
            validation = indices_from_intervals(fold["validation_intervals"])
            fold_report: dict[str, object] = {
                "fold": fold_index,
                "filter_audit": filter_audit[key],
                "candidates": {},
            }
            for name, features in feature_sets.items():
                selection = fit_or_load_inner_lars(
                    features_all=features,
                    target_all=targets[:, finger_index],
                    training_intervals=fold["training_intervals_after_purge"],
                    cache=(
                        args.cache_root / f"sub{args.subject}" / finger
                        / f"fold{fold_index}" / f"{name}.npz"
                    ),
                    max_features=args.max_features,
                )
                estimate = predict(selection, features, validation)
                stitched[name][validation] = estimate
                selected = np.asarray(selection["selected_source"], dtype=np.int64)
                candidate_report: dict[str, object] = {
                    "validation_raw_pcc": pearson(estimate, raw[validation, finger_index]),
                    "selected_feature_count": int(selected.size),
                    "selection_method": selection["selection_method"],
                    "alpha": float(selection["alpha"]),
                }
                if name == "joint_dictionary":
                    candidate_report["selected_family_audit"] = csp_selection_audit(
                        selected,
                        wavelet.shape[1],
                        csp_energy.shape[1],
                        filters_per_state,
                    )
                fold_report["candidates"][name] = candidate_report
            folds.append(fold_report)
            destination = output_subject / finger / f"fold{fold_index}"
            destination.mkdir(parents=True, exist_ok=True)
            np.save(destination / "validation_indices.npy", validation)
            for name in stitched:
                np.save(destination / f"{name}_prediction.npy", stitched[name][validation])
            (destination / "summary.json").write_text(
                json.dumps(fold_report, indent=2) + "\n"
            )
        scores = {
            name: pearson(values, raw[:, finger_index]) for name, values in stitched.items()
        }
        subject_report["per_finger"][finger] = {
            "oof_raw_pcc": scores,
            "joint_minus_ica_wavelet": scores["joint_dictionary"] - scores["ica_wavelet_only"],
            "folds": folds,
        }
        print(f"S{args.subject} {finger}: {json.dumps(scores)}", flush=True)
    matrix = np.asarray(
        [
            [subject_report["per_finger"][finger]["oof_raw_pcc"][name] for finger in FINGER_NAMES]
            for name in ("ica_wavelet_only", "csp_designed_bands_only", "joint_dictionary")
        ]
    )
    subject_report["macro_five"] = {
        name: float(matrix[index].mean())
        for index, name in enumerate(
            ("ica_wavelet_only", "csp_designed_bands_only", "joint_dictionary")
        )
    }
    subject_report["joint_wins"] = int(np.sum(matrix[2] > matrix[0]))
    (output_subject / "summary.json").write_text(
        json.dumps(subject_report, indent=2) + "\n"
    )
    print(json.dumps(subject_report["macro_five"], indent=2), flush=True)


if __name__ == "__main__":
    main()
