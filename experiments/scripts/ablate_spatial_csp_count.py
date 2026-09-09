#!/usr/bin/env python3
"""Experimentally measure whether finger-specific CSP rows improve fixed LARS.

The experiment reuses each outer fold's split-local FastICA basis and target,
then adds progressively larger CSP banks before the shared wavelet tree.  It
uses development folds only and never loads released-test labels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import linalg
from sklearn.linear_model import LassoLarsCV, RidgeCV
from sklearn.preprocessing import StandardScaler

from single_wavelet_support import (
    HISTORY,
    OFFSET,
    csp_candidate_union,
    extract_energy,
    linear_gamma_leaf_signals,
    pearson,
    regularized_covariance,
    resample_ecog,
    training_ecog_samples,
)
from balanced_nested_tune_s3_little import (
    grouped_lars_subfolds,
    scoped_event_groups,
)
from ecog_decoding.models import WaveletPacketEnergy
from ecog_decoding.regression import lagged
from ecog_decoding.training import FINGER_NAMES
from train_event_grouped_lars_lstm import indices_from_intervals


VARIANTS: dict[str, tuple[int, ...]] = {
    "ica_only": (),
    "movement_1": (-1,),
    "movement_2": (-1, -2),
    "movement_4": (-1, -2, -3, -4),
    "tails_2x2": (0, 1, -2, -1),
    "tails_4x4": (0, 1, 2, 3, -4, -3, -2, -1),
}


def finger_csp_eigensystem(
    filtered_bins: np.ndarray,
    target: np.ndarray,
    training: np.ndarray,
    finger_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rest = np.max(np.nan_to_num(target, nan=np.inf), axis=1) < 0.05
    active = target[:, finger_index] > 0.20
    active_rows = training[active[training]]
    rest_rows = training[rest[training]]
    if active_rows.size < 2 or rest_rows.size < 2:
        raise RuntimeError("too few movement or rest bins for CSP")
    active_values = filtered_bins[active_rows + OFFSET].reshape(
        -1, filtered_bins.shape[-1]
    )
    rest_values = filtered_bins[rest_rows + OFFSET].reshape(
        -1, filtered_bins.shape[-1]
    )
    active_covariance = regularized_covariance(active_values)
    rest_covariance = regularized_covariance(rest_values)
    eigenvalues, eigenvectors = linalg.eigh(
        active_covariance,
        active_covariance + rest_covariance,
        check_finite=False,
    )
    weights = eigenvectors.T.astype(np.float32)
    return eigenvalues, weights, {
        "active_bins": int(active_rows.size),
        "rest_bins": int(rest_rows.size),
    }


def softplus(values: np.ndarray, beta: float = 10.0) -> np.ndarray:
    return np.logaddexp(0.0, beta * values) / beta


def fit_predict(
    features: np.ndarray,
    target: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
    training_groups: list[list[int]],
    candidates: np.ndarray,
    finger_index: int,
) -> tuple[np.ndarray, dict[str, object]]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(features[training][:, candidates]).astype(
        np.float64, copy=False
    )
    splits = grouped_lars_subfolds(training, training_groups, target.shape[0])
    lars = LassoLarsCV(cv=splits, max_iter=500, n_jobs=1)
    lars.fit(train_x, target[training, finger_index])
    nonzero = np.flatnonzero(lars.coef_)
    if nonzero.size:
        prediction = lars.predict(scaler.transform(features[:, candidates]))
        method = "lasso_lars_cv"
        alpha = float(lars.alpha_)
    else:
        centered = target[training, finger_index] - np.mean(
            target[training, finger_index]
        )
        ranked = np.argsort(np.abs(train_x.T @ centered))[::-1]
        nonzero = ranked[: min(64, ranked.size)]
        ridge = RidgeCV(
            alphas=np.logspace(-2, 4, 13),
            cv=splits,
            scoring="neg_mean_squared_error",
        )
        ridge.fit(train_x[:, nonzero], target[training, finger_index])
        transformed = scaler.transform(features[:, candidates])[:, nonzero]
        prediction = ridge.predict(transformed)
        method = "ridge_fallback_after_null_lars"
        alpha = float(ridge.alpha_)
    return prediction.astype(np.float32), {
        "method": method,
        "alpha": alpha,
        "selected_features": int(nonzero.size),
        "validation_rows": int(validation.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--finger", choices=tuple(FINGER_NAMES), default="middle")
    parser.add_argument("--outer-folds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--model-rate", type=int, choices=(400, 1000), default=400)
    parser.add_argument("--tap-resample-up", type=int, default=1)
    parser.add_argument("--tap-resample-down", type=int, default=1)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument(
        "--initialization-root",
        type=Path,
        default=Path("outputs/single_path_csp_wavelet_lstm_nested_subject_targets_v1"),
    )
    parser.add_argument("--resampled-root", type=Path, default=Path("/dev/shm/ecog_wavelet_400hz"))
    parser.add_argument("--leaf-cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/spatial_csp_count_ablation_v1"))
    parser.add_argument("--ica-prescreen", type=int, default=512)
    parser.add_argument("--component-chunk", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    started = time.perf_counter()
    finger_index = list(FINGER_NAMES).index(args.finger)
    device = torch.device(args.device)
    if args.model_rate % 25:
        raise ValueError("model rate must be divisible by 25 Hz")
    if args.model_rate == 400 and (args.tap_resample_up, args.tap_resample_down) != (1, 1):
        raise ValueError("400 Hz input must use the original wavelet taps")
    samples_per_bin = args.model_rate // 25
    subject_cache = args.resampled_root / f"sub{args.subject}"
    prepared = args.prepared_root / f"sub{args.subject}"
    source_ecog = prepared / "train_ecog.npy"
    ecog = (
        np.asarray(np.load(source_ecog, mmap_mode="r"))
        if args.model_rate == 1000
        else np.asarray(resample_ecog(source_ecog, subject_cache / "train_ecog.npy"))
    )
    raw_full = np.load(prepared / "train_glove_25hz_raw.npy", mmap_mode="r")
    rows = int(raw_full.shape[0] - OFFSET)
    raw = np.asarray(raw_full[OFFSET : OFFSET + rows, finger_index], dtype=np.float32)
    frontend = WaveletPacketEnergy(
        wavelet="bior6.8",
        levels=3,
        kernel_size=17,
        trainable=False,
        padding_mode="constant",
        energy_window_samples=samples_per_bin,
        energy_stride_samples=samples_per_bin,
        tap_resample_up=args.tap_resample_up,
        tap_resample_down=args.tap_resample_down,
    ).to(device).eval()
    leaf_cache = args.leaf_cache or Path(
        f"/dev/shm/ecog_wavelet_{args.model_rate}hz_"
        f"taps{args.tap_resample_up}over{args.tap_resample_down}/sub{args.subject}"
    )
    leaf_cache.mkdir(parents=True, exist_ok=True)
    hhl_path = leaf_cache / "linear_hhl_100_125.npy"
    hhh_path = leaf_cache / "linear_hhh_125_150.npy"
    if hhl_path.exists() and hhh_path.exists():
        hhl = np.load(hhl_path, mmap_mode="r")
        hhh = np.load(hhh_path, mmap_mode="r")
    else:
        hhl, hhh = linear_gamma_leaf_signals(ecog, frontend, device)
        np.save(hhl_path, hhl, allow_pickle=False)
        np.save(hhh_path, hhh, allow_pickle=False)
    bins = ecog.shape[0] // samples_per_bin
    joint_bins = np.concatenate(
        (
            hhl[: bins * samples_per_bin].reshape(bins, samples_per_bin, -1),
            hhh[: bins * samples_per_bin].reshape(bins, samples_per_bin, -1),
        ),
        axis=1,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    oof = {name: np.full(rows, np.nan, dtype=np.float32) for name in VARIANTS}
    records: list[dict[str, object]] = []
    for fold in args.outer_folds:
        fold_started = time.perf_counter()
        source = (
            args.initialization_root
            / f"sub{args.subject}"
            / args.finger
            / "cache"
            / f"outer{fold}"
            / "outer"
        )
        split = json.loads((source / "split.json").read_text())
        target = np.load(source / "target.npy")
        training_intervals = split["training_intervals"]
        validation_intervals = split["validation_intervals"]
        training = indices_from_intervals(training_intervals)
        validation = indices_from_intervals(validation_intervals)
        groups, _, _ = scoped_event_groups(
            target, training_intervals, 0.08, 12, 3, 250, finger_index
        )
        training_groups = [[int(start), int(stop)] for start, stop in groups]

        saved_spatial = np.load(source / "spatial.npy")
        channel_count = ecog.shape[1]
        ica = np.asarray(saved_spatial[:channel_count], dtype=np.float32)
        if ica.shape != (channel_count, channel_count):
            raise RuntimeError("cached FastICA bank is not full rank in row count")
        eigenvalues, eigenspatial, csp_audit = finger_csp_eigensystem(
            joint_bins, target, training, finger_index
        )
        requested = sorted({index % eigenspatial.shape[0] for indices in VARIANTS.values() for index in indices})
        normalized: dict[int, np.ndarray] = {}
        training_samples = training_ecog_samples(ecog, training, samples_per_bin)
        for index in requested:
            row = eigenspatial[index]
            projected_std = float(np.std(training_samples @ row, dtype=np.float64))
            if projected_std < 1.0e-7:
                raise RuntimeError("CSP spatial projection has near-zero training variance")
            normalized[index] = (row / projected_std).astype(np.float32)

        ica_energy = extract_energy(ecog, ica, frontend, device, args.component_chunk)
        requested_position = {row: position for position, row in enumerate(requested)}
        all_csp_energy = extract_energy(
            ecog,
            np.stack([normalized[row] for row in requested]),
            frontend,
            device,
            args.component_chunk,
        )
        for name, indices in VARIANTS.items():
            selected_rows = [index % eigenspatial.shape[0] for index in indices]
            if selected_rows:
                csp_energy = all_csp_energy[
                    :, [requested_position[index] for index in selected_rows], :
                ]
                stream = np.concatenate(
                    (ica_energy.reshape(ica_energy.shape[0], -1), csp_energy.reshape(csp_energy.shape[0], -1)),
                    axis=1,
                )
            else:
                stream = ica_energy.reshape(ica_energy.shape[0], -1)
            features = lagged(np.asarray(stream, dtype=np.float32), HISTORY)
            candidates, candidate_audit = csp_candidate_union(
                features,
                target,
                training,
                stream.shape[1],
                ica_energy.shape[1] * 8,
                args.ica_prescreen,
                finger_index,
            )
            prediction, audit = fit_predict(
                features,
                target,
                training,
                validation,
                training_groups,
                candidates,
                finger_index,
            )
            projected = softplus(prediction)
            oof[name][validation] = projected[validation]
            positions = candidates % stream.shape[1]
            csp_candidate_count = int(np.sum(positions >= ica_energy.shape[1] * 8))
            record = {
                "outer_fold": fold,
                "variant": name,
                "ica_rows": int(ica.shape[0]),
                "added_csp_rows": len(selected_rows),
                "spatial_rows": int(ica.shape[0] + len(selected_rows)),
                "csp_eigenvalues": [float(eigenvalues[index]) for index in selected_rows],
                "candidate_features": int(candidates.size),
                "csp_candidate_features": csp_candidate_count,
                "linear_raw_pcc": pearson(prediction[validation], raw[validation]),
                "softplus_raw_pcc": pearson(projected[validation], raw[validation]),
                **csp_audit,
                **candidate_audit,
                **audit,
                "runtime_seconds": time.perf_counter() - fold_started,
            }
            records.append(record)
            print(json.dumps(record), flush=True)

    aggregate = {}
    for name in VARIANTS:
        observed = np.isfinite(oof[name])
        per_fold = [record for record in records if record["variant"] == name]
        aggregate[name] = {
            "mean_fold_softplus_raw_pcc": float(
                np.mean([record["softplus_raw_pcc"] for record in per_fold])
            ),
            "stitched_softplus_raw_pcc": pearson(oof[name][observed], raw[observed]),
            "mean_selected_features": float(
                np.mean([record["selected_features"] for record in per_fold])
            ),
            "mean_added_csp_rows": float(
                np.mean([record["added_csp_rows"] for record in per_fold])
            ),
        }
        np.save(args.output / f"{name}_oof.npy", oof[name], allow_pickle=False)
    report = {
        "protocol": "outer-development-fold spatial-count ablation with split-local target, FastICA, CSP and LARS; released test untouched",
        "subject": args.subject,
        "finger": args.finger,
        "input_rate_hz": args.model_rate,
        "tap_resample_up": args.tap_resample_up,
        "tap_resample_down": args.tap_resample_down,
        "full_rank_ica_rows": int(ecog.shape[1]),
        "variants": {name: list(indices) for name, indices in VARIANTS.items()},
        "aggregate": aggregate,
        "fold_records": records,
        "released_test_touched": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
