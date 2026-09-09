#!/usr/bin/env python3
"""Prepare full-development joint ICA-wavelet/CSP features without duplication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from benchmark_event_heterogeneous_dictionary import (
    binned_energy,
    selected_csp_filters,
    target_name,
)
from benchmark_csp_ridge import BANDS_HZ
from benchmark_ridge_target_variants import lagged
from ecog_decoding.training import FINGER_NAMES
from refit_frozen_event_model import lars_chunks
from train_event_grouped_lars_e2e_nested import fit_or_load_inner_lars


def correlations_by_column(
    features: np.ndarray, target: np.ndarray, block_columns: int = 512
) -> np.ndarray:
    """Compute feature correlations in column blocks to keep RAM bounded."""
    centered_target = np.asarray(target, dtype=np.float64) - float(np.mean(target))
    target_norm = float(np.linalg.norm(centered_target))
    result = np.zeros(features.shape[1], dtype=np.float64)
    for start in range(0, features.shape[1], block_columns):
        stop = min(start + block_columns, features.shape[1])
        values = np.asarray(features[:, start:stop], dtype=np.float64)
        values -= values.mean(axis=0, keepdims=True)
        denominator = np.linalg.norm(values, axis=0) * target_norm
        result[start:stop] = np.divide(
            values.T @ centered_target,
            denominator,
            out=np.zeros(stop - start, dtype=np.float64),
            where=denominator > 0,
        )
    return result


def selected_matrix(
    wavelet: np.ndarray, csp: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Gather global joint-dictionary indices without materializing the dictionary."""
    indices = np.asarray(indices, dtype=np.int64)
    result = np.empty((wavelet.shape[0], indices.size), dtype=np.float32)
    wavelet_mask = indices < wavelet.shape[1]
    if wavelet_mask.any():
        result[:, wavelet_mask] = wavelet[:, indices[wavelet_mask]]
    if (~wavelet_mask).any():
        result[:, ~wavelet_mask] = csp[:, indices[~wavelet_mask] - wavelet.shape[1]]
    return result


def prepare_pair(args: argparse.Namespace, subject: int, finger: str) -> None:
    destination = args.output_root / f"sub{subject}" / finger
    summary_path = destination / "summary.json"
    if summary_path.exists():
        print(f"skipping prepared S{subject} {finger}: {summary_path}", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    prepared = args.prepared_root / f"sub{subject}"
    band_root = args.band_cache_root / f"sub{subject}"
    wavelet_root = args.wavelet_root / f"sub{subject}"
    target_map = yaml.safe_load(args.target_map.read_text())
    offset = args.history - 1
    finger_index = list(FINGER_NAMES).index(finger)

    target_columns = []
    for index, name in enumerate(FINGER_NAMES):
        policy = target_name(target_map, subject, name).removesuffix("_split_safe")
        target_columns.append(
            np.load(prepared / f"train_glove_{policy}.npy")[offset:, index]
        )
    targets = np.column_stack(target_columns).astype(np.float32)
    rows = targets.shape[0]
    target = targets[:, finger_index]
    moving = targets > 0.20
    rest = np.max(targets, axis=1) < 0.05
    training_rows = np.arange(rows, dtype=np.int64)

    train_bands = np.load(band_root / "train_filtered_bands.npy", mmap_mode="r")
    test_bands = np.load(band_root / "test_filtered_bands.npy", mmap_mode="r")
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    filters: list[dict[str, object]] = []
    for band, (low, high) in enumerate(BANDS_HZ):
        train_continuous = np.asarray(train_bands[band])
        test_continuous = np.asarray(test_bands[band])
        train_bins_all = train_continuous.reshape(
            -1, args.samples_per_bin, train_bands.shape[-1]
        )
        test_bins_all = test_continuous.reshape(
            -1, args.samples_per_bin, test_bands.shape[-1]
        )
        train_bins = train_bins_all[offset : offset + rows]
        own, own_values = selected_csp_filters(
            train_bins, training_rows, moving[:, finger_index], rest,
            args.components_per_tail,
        )
        shared, shared_values = selected_csp_filters(
            train_bins, training_rows, np.max(moving, axis=1), rest,
            args.components_per_tail,
        )
        weights = np.concatenate((own, shared), axis=0)
        train_parts.append(binned_energy(train_continuous, weights, args.samples_per_bin))
        test_parts.append(binned_energy(test_continuous, weights, args.samples_per_bin))
        filters.append({
            "band_hz": [low, high],
            "weights": weights.tolist(),
            "decoded_finger_eigenvalues": own_values,
            "any_movement_eigenvalues": shared_values,
        })
        print(f"S{subject} {finger}: prepared {low:g}-{high:g} Hz", flush=True)

    csp_train = lagged(np.concatenate(train_parts, axis=1), args.history)[:rows]
    csp_test = lagged(np.concatenate(test_parts, axis=1), args.history)
    wavelet_train = np.load(
        wavelet_root / "train_initialized_window_features.npy", mmap_mode="r"
    )[:rows]
    wavelet_test = np.load(
        wavelet_root / "test_initialized_window_features.npy", mmap_mode="r"
    )[: csp_test.shape[0]]

    correlations = np.concatenate(
        (correlations_by_column(wavelet_train, target), correlations_by_column(csp_train, target))
    )
    prescreen = np.argsort(np.abs(correlations))[::-1][: args.max_features]
    train_candidates = selected_matrix(wavelet_train, csp_train, prescreen)
    test_candidates = selected_matrix(wavelet_test, csp_test, prescreen)
    selection = fit_or_load_inner_lars(
        features_all=train_candidates,
        target_all=target,
        training_intervals=lars_chunks(rows),
        cache=destination / "candidate_lars.npz",
        max_features=prescreen.size,
    )
    chosen_candidate = np.asarray(selection["selected_source"], dtype=np.int64)
    chosen_global = prescreen[chosen_candidate]
    np.save(destination / "train_selected_features.npy", train_candidates[:, chosen_candidate])
    np.save(destination / "test_selected_features.npy", test_candidates[:, chosen_candidate])
    np.savez(
        destination / "selection.npz",
        selected_global_indices=chosen_global,
        feature_mean=np.asarray(selection["feature_mean"]),
        feature_scale=np.asarray(selection["feature_scale"]),
        coefficients=np.asarray(selection["coefficients"]),
        intercept=np.asarray(selection["intercept"]),
        alpha=np.asarray(selection["alpha"]),
        selection_method=np.asarray(selection["selection_method"]),
    )
    report = {
        "subject": subject,
        "finger": finger,
        "protocol": "full-development CSP fit plus joint ICA-wavelet/CSP LARS selection",
        "released_test_labels_used": False,
        "target_policy": target_name(target_map, subject, finger).removesuffix("_split_safe"),
        "train_rows": rows,
        "test_rows": int(csp_test.shape[0]),
        "wavelet_feature_count": int(wavelet_train.shape[1]),
        "csp_feature_count": int(csp_train.shape[1]),
        "prescreened_feature_count": int(prescreen.size),
        "selected_feature_count": int(chosen_global.size),
        "selected_wavelet_feature_count": int(np.sum(chosen_global < wavelet_train.shape[1])),
        "selected_csp_feature_count": int(np.sum(chosen_global >= wavelet_train.shape[1])),
        "lars_alpha": float(selection["alpha"]),
        "selection_method": str(selection["selection_method"]),
        "csp_filters": filters,
    }
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "subject", "finger", "selected_feature_count",
        "selected_wavelet_feature_count", "selected_csp_feature_count",
    )}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--fingers", nargs="+", required=True, choices=tuple(FINGER_NAMES))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--wavelet-root", type=Path, default=Path("outputs/windowed_ica_wavelet_asymmetric_v1"))
    parser.add_argument("--band-cache-root", type=Path, default=Path("/dev/shm/ecog_csp_band_cache"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/heterogeneous_full_features_v1"))
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--samples-per-bin", type=int, default=40)
    parser.add_argument("--components-per-tail", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=1024)
    args = parser.parse_args()
    for finger in args.fingers:
        prepare_pair(args, args.subject, finger)


if __name__ == "__main__":
    main()
