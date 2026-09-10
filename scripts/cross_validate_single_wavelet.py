#!/usr/bin/env python3
"""Nested event-grouped validation for one single-wavelet per-finger model.

Inner folds are rebuilt inside each outer-training scope.  Complete
target-finger movement groups (with surrounding rest) are balanced by event
duration, movement mass, and other-finger activity.  Every inner fold refits
the glove target, ICA, joint HHL/HHH CSP, and LARS initialization.  Training
uses a uniform event-group sampler and checkpoints are indexed by optimizer
updates, making the chosen schedule transferable to the full outer fit.

The decoder can use either the paper-style near-linear LARS initialization or
a nonlinear recurrent residual on the fixed split-local LARS trajectory.
Checkpoint selection uses one configured primary quantity; PCC and morphology
diagnostics remain separately visible.  Released competition test labels are
never loaded.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LassoLarsCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from single_wavelet_support import (
    FREQUENCY_ORDER,
    HISTORY,
    LITTLE,
    OFFSET,
    SAMPLES_PER_BIN,
    SOURCE_RATE,
    csp_candidate_union,
    extract_energy,
    finger_csp_bank,
    linear_gamma_leaf_signals,
    linear_lower_high_gamma_leaf_signals,
    pearson,
    resample_ecog,
)
from build_event_stratified_folds import (
    balanced_assignment,
    group_features,
    runs,
)
from ecog_decoding.models import WaveletPacketEnergy, fit_fastica_spatial_weights
from ecog_decoding.preprocessing import local_baseline_correct
from ecog_decoding.regression import lagged
from ecog_decoding.training import FINGER_NAMES
from gpu_lasso import fit_torch_lasso_cv
from train_event_grouped_lars_lstm import indices_from_intervals
from train_event_grouped_lars_lstm_nested import intervals_from_mask
from single_wavelet_model import (
    OvercompleteWaveletPacketEnergy,
    SingleWaveletDecoder,
    extract_all,
    predict_intervals,
)


FROZEN_UPDATES = (20, 40, 80, 100, 120, 200)
UNFREEZE_AFTER = 100
UNFROZEN_UPDATES = (20, 40, 80, 120, 200, 400)
TARGET_POLICIES = {1: (2.0, 0.10), 2: (1.0, 0.10), 3: (4.0, 0.20)}
CSP_MODES = {
    "ica_only": (),
    "movement_1": (-1,),
    "movement_2": (-1, -2),
    "movement_4": (-1, -2, -3, -4),
    "tails_2x2": (0, 1, -2, -1),
    "tails_4x4": (0, 1, 2, 3, -4, -3, -2, -1),
}
CSP_BAND_MODES = (
    "joint_hhl_hhh",
    "separate_hhl_hhh",
    "separate_50_100_hhl_hhh",
)


def resolve_seed_roles(
    model_seed: int, split_seed: int | None, sampler_seed: int | None
) -> tuple[int, int, int]:
    """Keep historical coupled behavior unless data seeds are explicit."""
    return (
        model_seed,
        model_seed if split_seed is None else split_seed,
        model_seed if sampler_seed is None else sampler_seed,
    )


def split_intervals(
    definition: dict[str, object], outer_fold: int
) -> dict[str, dict[str, object]]:
    """Return the requested outer split and its two nested training splits."""
    rows = int(definition["training_rows"])
    outer = definition["folds"][outer_fold]
    outer_training = np.zeros(rows, dtype=bool)
    outer_training[indices_from_intervals(outer["training_intervals_after_purge"])] = True
    result: dict[str, dict[str, object]] = {
        "outer": {
            "training_intervals": outer["training_intervals_after_purge"],
            "validation_intervals": outer["validation_intervals"],
        }
    }
    for inner_fold, inner in enumerate(definition["folds"]):
        if inner_fold == outer_fold:
            continue
        training = np.zeros(rows, dtype=bool)
        training[indices_from_intervals(inner["training_intervals_after_purge"])] = True
        training &= outer_training
        validation = np.zeros(rows, dtype=bool)
        validation[indices_from_intervals(inner["validation_intervals"])] = True
        validation &= outer_training
        result[f"inner{inner_fold}"] = {
            "training_intervals": intervals_from_mask(training),
            "validation_intervals": intervals_from_mask(validation),
        }
    if len(result) != 3:
        raise RuntimeError("expected one outer and two inner splits")
    return result


def scale_from_training(corrected: np.ndarray, training: np.ndarray) -> np.ndarray:
    scale = np.quantile(corrected[training], 0.995, axis=0)
    floor = max(float(np.max(scale)) * 0.01, np.finfo(np.float64).eps)
    return np.maximum(scale, floor)


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, values, allow_pickle=False)
    os.replace(temporary, path)


def training_ecog_samples_runtime(
    ecog: np.ndarray, training: np.ndarray, samples_per_bin: int
) -> np.ndarray:
    """Map 25 Hz target rows to samples at the configured frontend rate."""
    bins = np.asarray(training, dtype=np.int64) + OFFSET
    indices = (
        bins[:, None] * samples_per_bin
        + np.arange(samples_per_bin, dtype=np.int64)[None]
    ).reshape(-1)
    indices = indices[indices < ecog.shape[0]]
    return np.asarray(ecog[indices], dtype=np.float32)


def valid_leaf_cache(path: Path, rows: int) -> bool:
    if not path.exists():
        return False
    try:
        values = np.load(path, mmap_mode="r")
        return values.ndim == 2 and values.shape[0] == rows
    except (EOFError, OSError, ValueError):
        return False


def correct_local_intervals(
    raw: np.ndarray,
    intervals: list[list[int]],
    window_seconds: float,
    quantile: float,
) -> np.ndarray:
    corrected = np.full(raw.shape, np.nan, dtype=np.float64)
    for start, stop in intervals:
        if stop - start < 3:
            continue
        corrected[start:stop] = local_baseline_correct(
            raw[start:stop],
            sampling_rate_hz=25.0,
            window_seconds=window_seconds,
            quantile=quantile,
            smoothing_seconds=0.16,
        ).corrected
    return corrected


def make_subject_target(
    raw: np.ndarray,
    training_intervals: list[list[int]],
    validation_intervals: list[list[int]],
    subject: int,
    *,
    finger_index: int | None = None,
    little_event_decontamination: bool = False,
    little_event_ratio_low: float = 0.8,
    little_event_ratio_high: float = 1.2,
) -> np.ndarray:
    """Fit the established local glove baseline separately inside one split."""
    window_seconds, quantile = TARGET_POLICIES[subject]
    training = indices_from_intervals(training_intervals)
    validation = indices_from_intervals(validation_intervals)
    train_corrected = correct_local_intervals(
        raw, training_intervals, window_seconds, quantile
    )
    validation_corrected = correct_local_intervals(
        raw, validation_intervals, window_seconds, quantile
    )
    scale = scale_from_training(train_corrected, training)
    result = np.full(raw.shape, np.nan, dtype=np.float32)
    result[training] = np.clip(train_corrected[training] / scale, 0.0, 2.0)
    result[validation] = np.clip(validation_corrected[validation] / scale, 0.0, 2.0)
    if little_event_decontamination:
        if finger_index != LITTLE:
            raise ValueError("little-event decontamination is only valid for little-finger models")
        result = suppress_weak_little_events(
            result,
            training_intervals + validation_intervals,
            ratio_low=little_event_ratio_low,
            ratio_high=little_event_ratio_high,
        )
    return result


def suppress_weak_little_events(
    target: np.ndarray,
    intervals: list[list[int]],
    *,
    ratio_low: float = 0.8,
    ratio_high: float = 1.2,
    movement_threshold: float = 0.10,
) -> np.ndarray:
    """Attenuate only little-finger events dominated by another finger.

    The rule is evaluated independently inside each supplied interval.  It is
    intentionally not winner-take-all: events with comparable little-finger
    and other-finger peaks are preserved, with a smooth transition between
    ``ratio_low`` and ``ratio_high``.
    """
    if ratio_low < 0 or ratio_high <= ratio_low:
        raise ValueError("little-event ratio bounds must satisfy 0 <= low < high")
    result = np.asarray(target, dtype=np.float32).copy()
    for start, stop in intervals:
        if stop - start < 3:
            continue
        scoped = result[start:stop]
        smooth = ndimage.gaussian_filter1d(scoped, sigma=1.0, axis=0, mode="nearest")
        active = ndimage.binary_closing(
            smooth[:, LITTLE] >= movement_threshold,
            structure=np.ones(3, dtype=bool),
        )
        for event_start, event_stop in runs(active):
            if event_stop - event_start < 3:
                continue
            peaks = np.nanmax(smooth[event_start:event_stop], axis=0)
            other_peak = float(np.nanmax(peaks[:LITTLE]))
            ratio = float(peaks[LITTLE] / max(other_peak, np.finfo(np.float32).eps))
            phase = float(np.clip((ratio - ratio_low) / (ratio_high - ratio_low), 0.0, 1.0))
            weight = phase * phase * (3.0 - 2.0 * phase)
            result[start + event_start : start + event_stop, LITTLE] *= weight
    return result


def scoped_event_groups(
    target: np.ndarray,
    training_intervals: list[list[int]],
    threshold: float,
    merge_gap_bins: int,
    minimum_event_bins: int,
    maximum_rest_group_bins: int,
    finger_index: int = LITTLE,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Partition each outer-training interval into complete event/rest groups."""
    smooth = ndimage.gaussian_filter1d(
        np.nan_to_num(target, nan=0.0), sigma=1.0, axis=0, mode="nearest"
    )
    active = smooth >= threshold
    groups: list[tuple[int, int]] = []
    for scope_start, scope_stop in training_intervals:
        finger_active = ndimage.binary_closing(
            active[scope_start:scope_stop, finger_index],
            structure=np.ones(merge_gap_bins + 1),
        )
        events = [
            (scope_start + start, scope_start + stop)
            for start, stop in runs(finger_active)
            if stop - start >= minimum_event_bins
        ]
        if events:
            boundaries = [scope_start]
            boundaries.extend(
                (left[1] + right[0]) // 2
                for left, right in zip(events[:-1], events[1:])
            )
            boundaries.append(scope_stop)
            groups.extend(
                (int(start), int(stop))
                for start, stop in zip(boundaries[:-1], boundaries[1:])
                if stop - start >= minimum_event_bins
            )
        else:
            for start in range(scope_start, scope_stop, maximum_rest_group_bins):
                stop = min(scope_stop, start + maximum_rest_group_bins)
                if stop - start >= minimum_event_bins:
                    groups.append((int(start), int(stop)))
    if len(groups) < 5:
        raise RuntimeError(f"only {len(groups)} event/rest groups are available")
    return groups, smooth, active


def five_fold_assignment(
    groups: list[tuple[int, int]],
    smooth: np.ndarray,
    active: np.ndarray,
    threshold: float,
    seed: int,
    finger_index: int = LITTLE,
) -> tuple[np.ndarray, float]:
    features = group_features(groups, smooth, active, threshold)
    fingers = 5
    priority_columns = np.asarray(
        (
            1 + finger_index,
            1 + fingers + finger_index,
            1 + 2 * fingers + finger_index,
        )
    )
    priority = features[:, priority_columns]
    emphasized = np.concatenate((features, priority, priority, priority), axis=1)
    assignment, objective = balanced_assignment(
        emphasized, folds=5, seed=seed, trials=1000
    )
    return assignment, objective


def group_intervals_after_exclusion(
    groups: list[tuple[int, int]],
    selected: np.ndarray,
    scope_mask: np.ndarray,
    excluded: np.ndarray,
) -> list[list[int]]:
    result: list[list[int]] = []
    for index, (start, stop) in enumerate(groups):
        if selected[index]:
            continue
        mask = np.zeros(scope_mask.size, dtype=bool)
        mask[start:stop] = True
        mask &= scope_mask & ~excluded
        result.extend(intervals_from_mask(mask))
    return result


def balanced_inner_splits(
    *,
    rows: int,
    outer_training_intervals: list[list[int]],
    target: np.ndarray,
    threshold: float,
    merge_gap_bins: int,
    minimum_event_bins: int,
    maximum_rest_group_bins: int,
    purge_bins: int,
    seed: int,
    finger_index: int = LITTLE,
) -> tuple[list[dict[str, object]], list[tuple[int, int]], float]:
    groups, smooth, active = scoped_event_groups(
        target,
        outer_training_intervals,
        threshold,
        merge_gap_bins,
        minimum_event_bins,
        maximum_rest_group_bins,
        finger_index,
    )
    assignment, objective = five_fold_assignment(
        groups, smooth, active, threshold, seed, finger_index
    )
    scope = np.zeros(rows, dtype=bool)
    scope[indices_from_intervals(outer_training_intervals)] = True
    splits = []
    for fold in range(5):
        selected = assignment == fold
        validation = np.zeros(rows, dtype=bool)
        for index in np.flatnonzero(selected):
            start, stop = groups[int(index)]
            validation[start:stop] = True
        excluded = validation.copy()
        for start, stop in runs(validation):
            excluded[max(0, start - purge_bins) : min(rows, stop + purge_bins)] = True
        training = scope & ~excluded
        validation &= scope
        training_groups = group_intervals_after_exclusion(
            groups, selected, scope, excluded
        )
        training_intervals = [
            interval
            for interval in intervals_from_mask(training)
            if interval[1] - interval[0] >= 3
        ]
        training_groups = [
            interval
            for interval in training_groups
            if interval[1] - interval[0] >= 3
        ]
        validation_groups = [
            [int(groups[index][0]), int(groups[index][1])]
            for index in np.flatnonzero(selected)
        ]
        splits.append(
            {
                "fold": fold,
                "validation_group_ids": np.flatnonzero(selected).tolist(),
                "training_intervals": training_intervals,
                "training_groups": training_groups,
                "validation_intervals": validation_groups,
                "training_bins": int(training.sum()),
                "validation_bins": int(validation.sum()),
            }
        )
    return splits, groups, objective


def grouped_lars_subfolds(
    training: np.ndarray,
    training_groups: list[list[int]],
    rows: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    global_to_local = np.full(rows, -1, dtype=np.int64)
    global_to_local[training] = np.arange(training.size)
    result = []
    for parity in (0, 1):
        held = indices_from_intervals(training_groups[parity::2])
        held = global_to_local[held]
        held = held[held >= 0]
        keep = np.ones(training.size, dtype=bool)
        keep[held] = False
        fitted = np.flatnonzero(keep)
        if fitted.size and held.size:
            result.append((fitted, held))
    if len(result) != 2:
        raise RuntimeError("could not construct grouped LARS subfolds")
    return result


def fit_csp_band_rows(
    *,
    joint_bins: np.ndarray,
    hhl_bins: np.ndarray | None,
    hhh_bins: np.ndarray | None,
    target: np.ndarray,
    training: np.ndarray,
    finger_index: int,
    component_indices: tuple[int, ...],
    csp_band_mode: str,
    lower_high_gamma_bins: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit joint or leaf-specific gamma CSP rows for one spatial layer."""
    if csp_band_mode not in CSP_BAND_MODES:
        raise ValueError(f"unsupported CSP band mode {csp_band_mode!r}")
    if csp_band_mode == "joint_hhl_hhh":
        weights, audit = finger_csp_bank(
            joint_bins, target, training, finger_index, component_indices
        )
        return weights, {"band_mode": csp_band_mode, **audit}
    if hhl_bins is None or hhh_bins is None:
        raise ValueError("separate CSP modes require HHL and HHH bins")
    hhl_weights, hhl_audit = finger_csp_bank(
        hhl_bins, target, training, finger_index, component_indices
    )
    hhh_weights, hhh_audit = finger_csp_bank(
        hhh_bins, target, training, finger_index, component_indices
    )
    band_weights = []
    band_audits: dict[str, object] = {}
    if csp_band_mode == "separate_50_100_hhl_hhh":
        if lower_high_gamma_bins is None:
            raise ValueError(
                "separate_50_100_hhl_hhh requires aggregated 50--100 Hz bins"
            )
        lower_weights, lower_audit = finger_csp_bank(
            lower_high_gamma_bins,
            target,
            training,
            finger_index,
            component_indices,
        )
        band_weights.append(lower_weights)
        band_audits["wavelet_50_100_hz"] = lower_audit
    band_weights.extend((hhl_weights, hhh_weights))
    weights = np.concatenate(band_weights, axis=0)
    return weights, {
        "band_mode": csp_band_mode,
        "spatial_rows": int(weights.shape[0]),
        **band_audits,
        "hhl_100_125_hz": hhl_audit,
        "hhh_125_150_hz": hhh_audit,
    }


def fit_initialization(
    *,
    ecog: np.ndarray,
    joint_bins: np.ndarray,
    target: np.ndarray,
    training_intervals: list[list[int]],
    training_groups: list[list[int]],
    frontend: WaveletPacketEnergy,
    device: torch.device,
    ica_prescreen: int,
    component_chunk: int,
    finger_index: int = LITTLE,
    csp_mode: str = "movement_1",
    csp_band_mode: str = "joint_hhl_hhh",
    hhl_bins: np.ndarray | None = None,
    hhh_bins: np.ndarray | None = None,
    lower_high_gamma_bins: np.ndarray | None = None,
    ica_weights: np.ndarray | None = None,
    samples_per_bin: int = SAMPLES_PER_BIN,
    include_candidate_pool: bool = False,
    lasso_backend: str = "sklearn_lars",
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    training = indices_from_intervals(training_intervals)
    if csp_mode not in CSP_MODES:
        raise ValueError(f"unsupported CSP mode {csp_mode!r}")
    ica = (
        np.asarray(ica_weights, dtype=np.float32)
        if ica_weights is not None
        else fit_fastica_spatial_weights(
            training_ecog_samples_runtime(ecog, training, samples_per_bin),
            n_components=ecog.shape[1],
            max_samples=50_000,
            random_state=0,
            backend="torch",
            device=device,
        )
    )
    if ica.shape != (ecog.shape[1], ecog.shape[1]):
        raise ValueError("FastICA weights must contain one row per retained channel")
    joint_weights, csp_audit = fit_csp_band_rows(
        joint_bins=joint_bins,
        hhl_bins=hhl_bins,
        hhh_bins=hhh_bins,
        lower_high_gamma_bins=lower_high_gamma_bins,
        target=target,
        training=training,
        finger_index=finger_index,
        component_indices=CSP_MODES[csp_mode],
        csp_band_mode=csp_band_mode,
    )
    normalized_rows = []
    csp_stds = []
    training_samples = training_ecog_samples_runtime(ecog, training, samples_per_bin)
    for row in joint_weights:
        row_std = float(np.std(training_samples @ row, dtype=np.float64))
        if row_std < 1.0e-7:
            raise RuntimeError("CSP spatial projection has near-zero training variance")
        normalized = (row / row_std).astype(np.float32)
        normalized_rows.append(normalized)
        csp_stds.append(row_std)
    joint_weights = (
        np.stack(normalized_rows)
        if normalized_rows
        else np.empty((0, ecog.shape[1]), dtype=np.float32)
    )
    ica_energy = extract_energy(ecog, ica, frontend, device, component_chunk)
    streams = [ica_energy.reshape(ica_energy.shape[0], -1)]
    if joint_weights.shape[0]:
        joint_energy = extract_energy(
            ecog, joint_weights, frontend, device, component_chunk
        )
        streams.append(joint_energy.reshape(joint_energy.shape[0], -1))
    stream = np.concatenate(streams, axis=1)
    features = lagged(np.asarray(stream, dtype=np.float32), HISTORY)
    candidates, candidate_audit = csp_candidate_union(
        features,
        target,
        training,
        stream.shape[1],
        ica_energy.shape[1] * ica_energy.shape[2],
        ica_prescreen,
        finger_index,
    )
    scaler = StandardScaler()
    train_x = scaler.fit_transform(features[training][:, candidates]).astype(
        np.float64, copy=False
    )
    lars_splits = grouped_lars_subfolds(training, training_groups, target.shape[0])
    train_y = target[training, finger_index]
    lasso_audit: dict[str, object]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    lasso_started = time.perf_counter()
    if lasso_backend == "sklearn_lars":
        lasso = LassoLarsCV(
            cv=lars_splits,
            max_iter=500,
            n_jobs=1,
        )
        lasso.fit(train_x, train_y)
        lasso_coefficients = np.asarray(lasso.coef_, dtype=np.float32)
        lasso_intercept = float(lasso.intercept_)
        lasso_alpha = float(lasso.alpha_)
        lasso_audit = {"backend": "sklearn_lars"}
        selection_method = "lasso_lars_cv"
    elif lasso_backend == "torch_fista":
        lasso = fit_torch_lasso_cv(
            train_x,
            train_y,
            lars_splits,
            device=device,
        )
        lasso_coefficients = lasso.coef_
        lasso_intercept = float(lasso.intercept_)
        lasso_alpha = float(lasso.alpha_)
        lasso_audit = {
            "backend": "torch_fista",
            "alpha_count": int(lasso.alphas_.size),
            "iterations": int(lasso.iterations_),
            "minimum_mean_validation_mse": float(
                np.min(lasso.mean_validation_mse_)
            ),
        }
        selection_method = "torch_fista_lasso_cv"
    else:
        raise ValueError(f"unsupported Lasso backend {lasso_backend!r}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    lasso_audit["elapsed_seconds"] = float(time.perf_counter() - lasso_started)
    nonzero = np.flatnonzero(np.abs(lasso_coefficients) > 1.0e-7)
    if nonzero.size == 0:
        centered_target = train_y - np.mean(train_y)
        ranked = np.argsort(np.abs(train_x.T @ centered_target))[::-1]
        selected_nonzero = ranked[: min(64, ranked.size)]
        ridge = RidgeCV(
            alphas=np.logspace(-2, 4, 13),
            cv=lars_splits,
            scoring="neg_mean_squared_error",
        )
        ridge.fit(train_x[:, selected_nonzero], train_y)
        coefficients = np.asarray(ridge.coef_, dtype=np.float32)
        intercept = float(ridge.intercept_)
        fitted_alpha = float(ridge.alpha_)
        selection_method = f"ridge_fallback_after_null_{lasso_backend}"
    else:
        selected_nonzero = nonzero
        coefficients = lasso_coefficients[nonzero].astype(np.float32)
        intercept = lasso_intercept
        fitted_alpha = lasso_alpha
    selected = candidates[selected_nonzero]
    initialization = {
        "selected_indices": selected.astype(np.int64),
        "feature_mean": scaler.mean_[selected_nonzero].astype(np.float32),
        "feature_scale": scaler.scale_[selected_nonzero].astype(np.float32),
        "coefficients": coefficients,
        "intercept": np.asarray(intercept, dtype=np.float32),
        "selected_features": np.asarray(features[:, selected], dtype=np.float32),
    }
    if include_candidate_pool:
        current_candidate_positions = np.flatnonzero(
            candidates // stream.shape[1] == HISTORY - 1
        )
        if current_candidate_positions.size == 0:
            raise RuntimeError("candidate pool contains no current-bin features")
        causal_stream_positions = np.unique(candidates % stream.shape[1])
        causal_values = np.asarray(
            stream[HISTORY - 1 :, causal_stream_positions], dtype=np.float32
        )
        causal_scaler = StandardScaler().fit(causal_values[training])
        selected_causal_stream_positions = np.unique(selected % stream.shape[1])
        selected_causal_values = np.asarray(
            stream[HISTORY - 1 :, selected_causal_stream_positions],
            dtype=np.float32,
        )
        selected_causal_scaler = StandardScaler().fit(
            selected_causal_values[training]
        )
        initialization.update(
            {
                "candidate_indices": candidates.astype(np.int64),
                "candidate_feature_mean": scaler.mean_.astype(np.float32),
                "candidate_feature_scale": scaler.scale_.astype(np.float32),
                "selected_candidate_positions": selected_nonzero.astype(np.int64),
                "current_candidate_positions": current_candidate_positions.astype(
                    np.int64
                ),
                "causal_stream_positions": causal_stream_positions.astype(np.int64),
                "causal_feature_mean": causal_scaler.mean_.astype(np.float32),
                "causal_feature_scale": causal_scaler.scale_.astype(np.float32),
                "causal_features": causal_values,
                "selected_causal_stream_positions": (
                    selected_causal_stream_positions.astype(np.int64)
                ),
                "selected_causal_feature_mean": (
                    selected_causal_scaler.mean_.astype(np.float32)
                ),
                "selected_causal_feature_scale": (
                    selected_causal_scaler.scale_.astype(np.float32)
                ),
                "selected_causal_features": selected_causal_values,
                "candidate_features": np.asarray(
                    features[:, candidates], dtype=np.float32
                ),
            }
        )
    spatial = np.concatenate((ica, joint_weights)).astype(np.float32)
    audit = {
        "selected_features": int(selected_nonzero.size),
        "selection_method": selection_method,
        "selection_alpha": fitted_alpha,
        "lasso": lasso_audit,
        "csp_mode": csp_mode,
        "csp_band_mode": csp_band_mode,
        "csp": {**csp_audit, "pre_normalization_std": csp_stds},
        **candidate_audit,
    }
    return initialization, spatial, audit


def save_initialization(
    root: Path,
    initialization: dict[str, np.ndarray],
    spatial: np.ndarray,
    target: np.ndarray,
    split: dict[str, object],
    audit: dict[str, object],
) -> None:
    root.mkdir(parents=True, exist_ok=True)

    def atomic_numpy(path: Path, write) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=root, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    atomic_numpy(
        root / "spatial.npy",
        lambda handle: np.save(handle, spatial, allow_pickle=False),
    )
    atomic_numpy(
        root / "target.npy",
        lambda handle: np.save(handle, target, allow_pickle=False),
    )
    split_text = json.dumps({**split, "initialization_audit": audit}, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=root, prefix=".split.json.", delete=False
    ) as handle:
        split_temporary = Path(handle.name)
        handle.write(split_text)
        handle.flush()
        os.fsync(handle.fileno())
    split_temporary.replace(root / "split.json")

    # This is the file readers use as the completion marker, so publish it last.
    atomic_numpy(
        root / "initialization.npz",
        lambda handle: np.savez(handle, **initialization),
    )


def load_initialization(root: Path):
    saved = np.load(root / "initialization.npz")
    return (
        {name: saved[name] for name in saved.files},
        np.load(root / "spatial.npy"),
        np.load(root / "target.npy"),
        json.loads((root / "split.json").read_text()),
    )


@contextmanager
def initialization_cache_lock(root: Path):
    """Serialize creation and loading of one split-local initialization."""
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.with_name(f".{root.name}.lock")
    with lock_path.open("a+b") as handle:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        else:
            import msvcrt

            if lock_path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def load_or_create_initialization(root: Path, create):
    """Load a complete cache or create it once while concurrent jobs wait."""
    with initialization_cache_lock(root):
        if (root / "initialization.npz").exists():
            return load_initialization(root)
        initialization, spatial, target, split, audit = create()
        save_initialization(root, initialization, spatial, target, split, audit)
        return initialization, spatial, target, {
            **split,
            "initialization_audit": audit,
        }


def load_cached_ica(
    root: Path | None,
    subject: int,
    finger: str,
    outer_fold: int,
    split_name: str,
    channel_count: int,
) -> np.ndarray | None:
    if root is None:
        return None
    path = (
        root
        / f"sub{subject}"
        / finger
        / "cache"
        / f"outer{outer_fold}"
        / split_name
        / "spatial.npy"
    )
    spatial = np.load(path)
    return np.asarray(spatial[:channel_count], dtype=np.float32)


def sequence_pools(
    groups: list[list[int]], steps: int
) -> list[np.ndarray]:
    pools = []
    for start, stop in groups:
        if stop - start >= steps:
            pools.append(np.arange(start, stop - steps + 1, dtype=np.int64))
    if not pools:
        raise RuntimeError("no event group is long enough for a training sequence")
    return pools


class UniformGroupSampler:
    def __init__(
        self, groups: list[list[int]], steps: int, seed: int
    ) -> None:
        self.pools = sequence_pools(groups, steps)
        self.rng = np.random.default_rng(seed)
        self.order = np.empty(0, dtype=np.int64)
        self.position = 0

    def sample(self, count: int) -> np.ndarray:
        chosen = []
        while len(chosen) < count:
            if self.position >= self.order.size:
                self.order = self.rng.permutation(len(self.pools))
                self.position = 0
            group = int(self.order[self.position])
            self.position += 1
            chosen.append(int(self.rng.choice(self.pools[group])))
        return np.asarray(chosen, dtype=np.int64)


class DenseSequenceSampler:
    """Cycle through every strided sequence start, as in the earlier trainer."""

    def __init__(
        self, groups: list[list[int]], steps: int, stride: int, seed: int
    ) -> None:
        starts = [
            np.arange(start, stop - steps + 1, stride, dtype=np.int64)
            for start, stop in groups
            if stop - start >= steps
        ]
        if not starts:
            raise RuntimeError("no interval is long enough for a training sequence")
        self.starts = np.concatenate(starts)
        self.rng = np.random.default_rng(seed)
        self.order = np.empty(0, dtype=np.int64)
        self.position = 0

    def sample(self, count: int) -> np.ndarray:
        chosen: list[int] = []
        while len(chosen) < count:
            if self.position >= self.order.size:
                self.order = self.rng.permutation(self.starts.size)
                self.position = 0
            take = min(count - len(chosen), self.order.size - self.position)
            indices = self.order[self.position : self.position + take]
            chosen.extend(self.starts[indices].tolist())
            self.position += take
        return np.asarray(chosen, dtype=np.int64)


def make_sampler(
    args: argparse.Namespace, groups: list[list[int]], seed: int
) -> UniformGroupSampler | DenseSequenceSampler:
    if args.sampler_mode == "dense_sequences":
        return DenseSequenceSampler(
            groups, args.sequence_steps, args.sequence_stride, seed
        )
    return UniformGroupSampler(groups, args.sequence_steps, seed)


def optimizer(
    model: SingleWaveletDecoder,
    head_lr: float,
    spatial_lr: float,
    wavelet_lr: float,
    interlevel_lr: float,
    weight_decay: float,
    movement_gain_lr: float | None = None,
    signed_pooling_lr: float = 0.0,
) -> torch.optim.Optimizer:
    head_parameters = list(model.lstm.parameters()) + list(model.output.parameters())
    if model.movement_output is not None:
        head_parameters += list(model.movement_output.parameters())
    if model.velocity_output is not None:
        head_parameters += list(model.velocity_output.parameters())
    parameter_groups = [
        {
            "params": head_parameters,
            "lr": head_lr,
        },
        {"params": model.spatial.parameters(), "lr": spatial_lr},
        {"params": model.wavelet.layers.parameters(), "lr": wavelet_lr},
    ]
    interlevel_parameters = list(model.wavelet.skip_gates.parameters()) + list(
        model.wavelet.normalization_gates.parameters()
    )
    if interlevel_parameters:
        parameter_groups.append(
            {"params": interlevel_parameters, "lr": interlevel_lr}
        )
    if model.movement_gain is not None:
        parameter_groups.append(
            {
                "params": [model.movement_gain],
                "lr": head_lr if movement_gain_lr is None else movement_gain_lr,
            }
        )
    if model.signed_pooling_gates is not None:
        parameter_groups.append(
            {"params": [model.signed_pooling_gates], "lr": signed_pooling_lr}
        )
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


def make_model(
    spatial: np.ndarray,
    initialization: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> SingleWaveletDecoder:
    return SingleWaveletDecoder(
        spatial,
        initialization,
        args.hidden_size,
        lars_candidate_scale=args.lars_candidate_scale,
        near_zero_std=args.lars_near_zero_std,
        open_gate_bias=args.lars_open_gate_bias,
        forget_gate_bias=args.lars_forget_gate_bias,
        output_activation=args.output_activation,
        softplus_beta=args.softplus_beta,
        recurrent_cell=args.recurrent_cell,
        energy_window_samples=args.samples_per_bin,
        tap_resample_up=args.tap_resample_up,
        tap_resample_down=args.tap_resample_down,
        wavelet_frontend=args.wavelet_frontend,
        wavelet_interlevel_skip=args.wavelet_interlevel_skip,
        wavelet_interlevel_normalization=args.wavelet_interlevel_normalization,
        wavelet_final_normalization=args.wavelet_final_normalization,
        residual_input=args.residual_input,
        residual_history_bins=args.residual_history_bins,
        residual_input_width=args.residual_input_width,
        residual_include_direct=args.residual_include_direct,
        movement_head=args.movement_loss_weight > 0,
        velocity_head=args.velocity_loss_weight > 0,
        movement_modulation=args.movement_modulation,
        wavelet_signed_pooling=args.wavelet_signed_pooling,
        residual_output_init_std=args.residual_output_init_std,
    )


def cached_initialization_features(
    initialization: dict[str, np.ndarray], residual_input: str
) -> np.ndarray:
    """Return the raw feature matrix expected by the configured decoder."""
    if residual_input in ("causal_candidate", "selected_causal"):
        prefix = "candidate" if residual_input == "causal_candidate" else "selected"
        required = ("candidate_features", "causal_features")
        if prefix == "selected":
            required = ("selected_features", "selected_causal_features")
        missing = [name for name in required if name not in initialization]
        if missing:
            raise ValueError(
                "initialization cache lacks " + ", ".join(repr(name) for name in missing)
            )
        direct_name, causal_name = required
        return np.concatenate(
            (initialization[direct_name], initialization[causal_name]), axis=1
        )
    name = (
        "candidate_features"
        if residual_input in ("candidate", "current_candidate")
        else "selected_features"
    )
    if name not in initialization:
        raise ValueError(
            f"initialization cache lacks {name!r}; refit the split-local initialization"
        )
    return initialization[name]


def sequence_correlation_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return mean one-minus-Pearson correlation across sequence rows."""
    centered_prediction = prediction - prediction.mean(dim=1, keepdim=True)
    centered_target = target - target.mean(dim=1, keepdim=True)
    denominator = (
        torch.linalg.vector_norm(centered_prediction, dim=1)
        * torch.linalg.vector_norm(centered_target, dim=1)
    ).clamp_min(1.0e-8)
    correlation = torch.sum(
        centered_prediction * centered_target, dim=1
    ) / denominator
    return 1.0 - correlation.mean()


def masked_sequence_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return correlation loss over masked bins in each sufficiently observed row."""
    weights = mask.to(prediction.dtype)
    count = weights.sum(dim=1)
    valid = count >= 3
    if not torch.any(valid):
        return prediction.sum() * 0.0
    safe_count = count.clamp_min(1.0)[:, None]
    prediction_mean = (prediction * weights).sum(dim=1, keepdim=True) / safe_count
    target_mean = (target * weights).sum(dim=1, keepdim=True) / safe_count
    centered_prediction = (prediction - prediction_mean) * weights
    centered_target = (target - target_mean) * weights
    numerator = (centered_prediction * centered_target).sum(dim=1)
    denominator = (
        torch.linalg.vector_norm(centered_prediction, dim=1)
        * torch.linalg.vector_norm(centered_target, dim=1)
    ).clamp_min(1.0e-8)
    correlation = numerator / denominator
    return 1.0 - correlation[valid].mean()


def trajectory_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_scale: torch.Tensor,
    movement_threshold: float,
    movement_weight: float,
) -> torch.Tensor:
    """Return scale-normalized MSE with optional emphasis on movement bins."""
    squared_error = ((prediction - target) / target_scale).square()
    if movement_weight == 1.0:
        return squared_error.mean()
    weights = torch.where(
        target >= movement_threshold,
        torch.as_tensor(movement_weight, dtype=target.dtype, device=target.device),
        torch.ones((), dtype=target.dtype, device=target.device),
    )
    return torch.sum(weights * squared_error) / torch.sum(weights)


def train_updates(
    *,
    model: SingleWaveletDecoder,
    call,
    optimizer_instance: torch.optim.Optimizer,
    cached: torch.Tensor,
    padded_ecog: torch.Tensor,
    target: torch.Tensor,
    sampler: UniformGroupSampler,
    updates: int,
    steps: int,
    batch_size: int,
    target_scale: torch.Tensor,
    raw_stem: bool,
    movement_loss_weight: float = 0.0,
    movement_trajectory_weight: float = 1.0,
    movement_threshold: float = 0.10,
    movement_positive_weight: torch.Tensor | None = None,
    velocity_loss_weight: float = 0.0,
    velocity_scale: torch.Tensor | None = None,
    correlation_loss_weight: float = 0.0,
    derivative_correlation_weight: float = 0.0,
    raw_target: torch.Tensor | None = None,
    raw_movement_correlation_weight: float = 0.0,
    raw_movement_derivative_correlation_weight: float = 0.0,
) -> list[float]:
    offsets = torch.arange(steps, device=target.device)
    losses = []
    model.train()
    for _ in range(updates):
        starts = torch.as_tensor(
            sampler.sample(batch_size), dtype=torch.long, device=target.device
        )
        index = starts[:, None] + offsets[None]
        observed = target[index]
        optimizer_instance.zero_grad(set_to_none=True)
        result_or_pair = (
            call(padded_ecog, starts, steps)
            if raw_stem
            else call(cached[index])
        )
        if movement_loss_weight or velocity_loss_weight:
            auxiliary = result_or_pair
            result = auxiliary[0]
            auxiliary_index = 1
            if movement_loss_weight:
                movement_logit = auxiliary[auxiliary_index]
                auxiliary_index += 1
            if velocity_loss_weight:
                velocity_prediction = auxiliary[auxiliary_index]
        else:
            result = result_or_pair
        loss = trajectory_mse_loss(
            result,
            observed,
            target_scale,
            movement_threshold,
            movement_trajectory_weight,
        )
        if movement_loss_weight:
            movement_target = (observed >= movement_threshold).to(result.dtype)
            loss = loss + movement_loss_weight * F.binary_cross_entropy_with_logits(
                movement_logit,
                movement_target,
                pos_weight=movement_positive_weight,
            )
        if velocity_loss_weight:
            if velocity_scale is None:
                raise ValueError("velocity scale is required for auxiliary velocity loss")
            observed_velocity = torch.diff(observed, dim=1)
            predicted_velocity = velocity_prediction[:, 1:]
            loss = loss + velocity_loss_weight * (
                (predicted_velocity - observed_velocity) / velocity_scale
            ).square().mean()
        if correlation_loss_weight:
            loss = loss + correlation_loss_weight * sequence_correlation_loss(
                result, observed
            )
        if derivative_correlation_weight:
            loss = loss + derivative_correlation_weight * (
                sequence_correlation_loss(
                    torch.diff(result, dim=1), torch.diff(observed, dim=1)
                )
            )
        if raw_movement_correlation_weight or raw_movement_derivative_correlation_weight:
            if raw_target is None:
                raise ValueError("raw target is required for raw movement correlation")
            raw_observed = raw_target[index]
            moving = observed >= movement_threshold
            if raw_movement_correlation_weight:
                loss = loss + raw_movement_correlation_weight * (
                    masked_sequence_correlation_loss(result, raw_observed, moving)
                )
            if raw_movement_derivative_correlation_weight:
                adjacent_moving = moving[:, 1:] & moving[:, :-1]
                loss = loss + raw_movement_derivative_correlation_weight * (
                    masked_sequence_correlation_loss(
                        torch.diff(result, dim=1),
                        torch.diff(raw_observed, dim=1),
                        adjacent_moving,
                    )
                )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer_instance.step()
        losses.append(float(loss.detach()))
    return losses


def movement_positive_weight(
    target: torch.Tensor,
    rows: np.ndarray,
    finger_index: int,
    movement_threshold: float,
) -> torch.Tensor:
    """Balance target-finger movement and rest in the auxiliary BCE."""
    scoped = target[
        torch.as_tensor(rows, dtype=torch.long, device=target.device), finger_index
    ]
    positive = (scoped >= movement_threshold).sum().clamp_min(1)
    negative = (scoped < movement_threshold).sum().clamp_min(1)
    return (negative / positive).to(target.dtype)


def grouped_velocity_scale(
    target: torch.Tensor,
    groups: list[list[int]],
) -> torch.Tensor:
    """Estimate velocity scale without differencing across event boundaries."""
    differences = [
        torch.diff(target[start:stop])
        for start, stop in groups
        if stop - start >= 2
    ]
    if not differences:
        return torch.as_tensor(0.01, dtype=target.dtype, device=target.device)
    return torch.cat(differences).std().clamp_min(0.01)


def balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    positive = truth
    negative = ~truth
    sensitivity = float(np.mean(prediction[positive])) if positive.any() else np.nan
    specificity = float(np.mean(~prediction[negative])) if negative.any() else np.nan
    return float(np.nanmean((sensitivity, specificity)))


def validation_metrics(
    prediction: np.ndarray,
    cleaned: np.ndarray,
    raw: np.ndarray,
    intervals: list[list[int]],
    movement_threshold: float,
    rest_threshold: float,
) -> dict[str, float]:
    rows = indices_from_intervals(intervals)
    estimate = prediction[rows]
    target = cleaned[rows]
    raw_target = raw[rows]
    target_variance = max(float(np.var(target)), 1.0e-6)
    interval_nmse = []
    velocity_prediction = []
    velocity_target = []
    for start, stop in intervals:
        interval_nmse.append(
            float(np.mean((prediction[start:stop] - cleaned[start:stop]) ** 2))
            / target_variance
        )
        if stop - start > 1:
            velocity_prediction.append(np.diff(prediction[start:stop]))
            velocity_target.append(np.diff(cleaned[start:stop]))
    resting = target <= rest_threshold
    moving = target >= movement_threshold
    centered = target - target.mean()
    slope = float(centered @ (estimate - estimate.mean()) / max(centered @ centered, 1.0e-8))
    velocity_pcc = (
        pearson(np.concatenate(velocity_prediction), np.concatenate(velocity_target))
        if velocity_prediction
        else 0.0
    )
    return {
        "event_macro_nmse": float(np.mean(interval_nmse)),
        "normalized_rmse": float(
            np.sqrt(np.mean((estimate - target) ** 2) / target_variance)
        ),
        "raw_pcc": pearson(estimate, raw_target),
        "cleaned_pcc": pearson(estimate, target),
        "rest_false_positive_rms": float(
            np.sqrt(np.mean(estimate[resting] ** 2)) if resting.any() else np.nan
        ),
        "movement_rmse": float(
            np.sqrt(np.mean((estimate[moving] - target[moving]) ** 2))
            if moving.any()
            else np.nan
        ),
        "amplitude_slope": slope,
        "velocity_pcc": velocity_pcc,
        "movement_balanced_accuracy": balanced_accuracy(
            target >= movement_threshold, estimate >= movement_threshold
        ),
    }


@torch.inference_mode()
def cached_prediction(
    model: SingleWaveletDecoder,
    cached: torch.Tensor,
    intervals: list[list[int]],
    rows: int,
) -> np.ndarray:
    return predict_intervals(model, cached, intervals, rows)


@torch.inference_mode()
def raw_prediction(
    model: SingleWaveletDecoder,
    padded_ecog: torch.Tensor,
    intervals: list[list[int]],
    rows: int,
) -> np.ndarray:
    model.eval()
    prediction = np.full(rows, np.nan, dtype=np.float32)
    for start, stop in intervals:
        features = model.extract_sequences(
            padded_ecog,
            torch.as_tensor([start], device=padded_ecog.device),
            stop - start,
        )
        prediction[start:stop] = (
            model.decode_features(features)[0].float().cpu().numpy()
        )
    return prediction


def monitor_inner_fold(
    *,
    model: SingleWaveletDecoder,
    cached: torch.Tensor,
    padded_ecog: torch.Tensor,
    target: torch.Tensor,
    target_np: np.ndarray,
    raw: np.ndarray,
    training_groups: list[list[int]],
    validation_intervals: list[list[int]],
    args: argparse.Namespace,
    seed: int,
    finger_index: int = LITTLE,
) -> dict[str, dict[str, float]]:
    raw_target = torch.as_tensor(raw, dtype=target.dtype, device=target.device)
    training_rows = indices_from_intervals(training_groups)
    target_scale = target[
        torch.as_tensor(training_rows, device=target.device), finger_index
    ].std().clamp_min(0.1)
    positive_weight = movement_positive_weight(
        target,
        training_rows,
        finger_index,
        args.movement_threshold,
    )
    velocity_scale = grouped_velocity_scale(
        target[:, finger_index], training_groups
    )
    sampler = make_sampler(args, training_groups, seed)
    metrics = {}
    baseline = cached_prediction(model, cached, validation_intervals, raw.size)
    metrics["frozen_0"] = validation_metrics(
        baseline,
        target_np[:, finger_index],
        raw,
        validation_intervals,
        args.movement_threshold,
        args.rest_threshold,
    )
    opt = optimizer(
        model,
        args.head_learning_rate,
        0.0,
        0.0,
        0.0,
        args.weight_decay,
        args.movement_modulation_learning_rate,
    )
    decode = (
        model.decode_features_with_auxiliary
        if args.movement_loss_weight or args.velocity_loss_weight
        else model.decode_features
    )
    if args.compile:
        decode = torch.compile(decode, mode="reduce-overhead")
    completed = 0
    unfreeze_state = None
    for checkpoint in FROZEN_UPDATES:
        train_updates(
            model=model,
            call=decode,
            optimizer_instance=opt,
            cached=cached,
            padded_ecog=padded_ecog,
            target=target[:, finger_index],
            sampler=sampler,
            updates=checkpoint - completed,
            steps=args.sequence_steps,
            batch_size=args.batch_size,
            target_scale=target_scale,
            raw_stem=False,
            movement_loss_weight=args.movement_loss_weight,
            movement_trajectory_weight=args.movement_trajectory_weight,
            movement_threshold=args.movement_threshold,
            movement_positive_weight=positive_weight,
            velocity_loss_weight=args.velocity_loss_weight,
            velocity_scale=velocity_scale,
            correlation_loss_weight=args.correlation_loss_weight,
            derivative_correlation_weight=args.derivative_correlation_weight,
            raw_target=raw_target,
            raw_movement_correlation_weight=args.raw_movement_correlation_weight,
            raw_movement_derivative_correlation_weight=(
                args.raw_movement_derivative_correlation_weight
            ),
        )
        completed = checkpoint
        prediction = cached_prediction(model, cached, validation_intervals, raw.size)
        metrics[f"frozen_{checkpoint}"] = validation_metrics(
            prediction,
            target_np[:, finger_index],
            raw,
            validation_intervals,
            args.movement_threshold,
            args.rest_threshold,
        )
        if checkpoint == UNFREEZE_AFTER:
            unfreeze_state = copy.deepcopy(model.state_dict())
    if not UNFROZEN_UPDATES:
        return metrics
    if unfreeze_state is None:
        raise RuntimeError("unfreeze checkpoint was not captured")
    model.load_state_dict(unfreeze_state)
    sampler = make_sampler(args, training_groups, seed + 1000)
    opt = optimizer(
        model,
        args.head_learning_rate * 0.5,
        args.spatial_learning_rate,
        args.wavelet_learning_rate,
        args.interlevel_learning_rate,
        args.weight_decay,
        args.movement_modulation_learning_rate,
        args.signed_pooling_learning_rate,
    )
    forward = (
        model.forward_with_auxiliary
        if args.movement_loss_weight or args.velocity_loss_weight
        else model.forward
    )
    if args.compile:
        forward = torch.compile(forward, mode="reduce-overhead")
    completed = 0
    for checkpoint in UNFROZEN_UPDATES:
        train_updates(
            model=model,
            call=forward,
            optimizer_instance=opt,
            cached=cached,
            padded_ecog=padded_ecog,
            target=target[:, finger_index],
            sampler=sampler,
            updates=checkpoint - completed,
            steps=args.sequence_steps,
            batch_size=args.batch_size,
            target_scale=target_scale,
            raw_stem=True,
            movement_loss_weight=args.movement_loss_weight,
            movement_trajectory_weight=args.movement_trajectory_weight,
            movement_threshold=args.movement_threshold,
            movement_positive_weight=positive_weight,
            velocity_loss_weight=args.velocity_loss_weight,
            velocity_scale=velocity_scale,
            correlation_loss_weight=args.correlation_loss_weight,
            derivative_correlation_weight=args.derivative_correlation_weight,
            raw_target=raw_target,
            raw_movement_correlation_weight=args.raw_movement_correlation_weight,
            raw_movement_derivative_correlation_weight=(
                args.raw_movement_derivative_correlation_weight
            ),
        )
        completed = checkpoint
        prediction = raw_prediction(model, padded_ecog, validation_intervals, raw.size)
        metrics[f"unfrozen_{UNFREEZE_AFTER}_{checkpoint}"] = validation_metrics(
            prediction,
            target_np[:, finger_index],
            raw,
            validation_intervals,
            args.movement_threshold,
            args.rest_threshold,
        )
    return metrics


def schedule_order() -> list[str]:
    return (
        ["frozen_0"]
        + [f"frozen_{value}" for value in FROZEN_UPDATES]
        + [f"unfrozen_{UNFREEZE_AFTER}_{value}" for value in UNFROZEN_UPDATES]
    )


def one_standard_error_selection(
    inner_records: list[dict[str, object]],
    require_lstm_update: bool = False,
    selection_metric: str = "event_macro_nmse",
    selection_rule: str = "one_se",
) -> tuple[str, dict[str, dict[str, float]]]:
    if selection_metric not in ("event_macro_nmse", "raw_pcc"):
        raise ValueError(f"unsupported selection metric {selection_metric!r}")
    if selection_rule not in ("one_se", "best"):
        raise ValueError(f"unsupported selection rule {selection_rule!r}")
    summary = {}
    for schedule in schedule_order():
        nmse_values = np.asarray(
            [record["metrics"][schedule]["event_macro_nmse"] for record in inner_records],
            dtype=np.float64,
        )
        pcc_values = np.asarray(
            [record["metrics"][schedule]["raw_pcc"] for record in inner_records],
            dtype=np.float64,
        )
        summary[schedule] = {
            "mean_event_macro_nmse": float(nmse_values.mean()),
            "sem_event_macro_nmse": float(
                nmse_values.std(ddof=1) / np.sqrt(nmse_values.size)
            ),
            "mean_raw_pcc": float(pcc_values.mean()),
            "sem_raw_pcc": float(pcc_values.std(ddof=1) / np.sqrt(pcc_values.size)),
        }
    candidates = schedule_order()[1:] if require_lstm_update else schedule_order()
    if selection_metric == "raw_pcc":
        best = max(candidates, key=lambda name: summary[name]["mean_raw_pcc"])
        threshold = summary[best]["mean_raw_pcc"] - summary[best]["sem_raw_pcc"]
        selected = best if selection_rule == "best" else next(
            name for name in candidates if summary[name]["mean_raw_pcc"] >= threshold
        )
    else:
        best = min(
            candidates, key=lambda name: summary[name]["mean_event_macro_nmse"]
        )
        threshold = (
            summary[best]["mean_event_macro_nmse"]
            + summary[best]["sem_event_macro_nmse"]
        )
        selected = best if selection_rule == "best" else next(
            name
            for name in candidates
            if summary[name]["mean_event_macro_nmse"] <= threshold
        )
    summary[best]["one_se_threshold"] = threshold
    return selected, summary


def train_final_schedule(
    *,
    model: SingleWaveletDecoder,
    cached: torch.Tensor,
    padded_ecog: torch.Tensor,
    target: torch.Tensor,
    raw: np.ndarray,
    training_groups: list[list[int]],
    schedule: str,
    args: argparse.Namespace,
    seed: int,
    finger_index: int = LITTLE,
) -> None:
    raw_target = torch.as_tensor(raw, dtype=target.dtype, device=target.device)
    training_rows = indices_from_intervals(training_groups)
    scale = target[
        torch.as_tensor(training_rows, device=target.device), finger_index
    ].std().clamp_min(0.1)
    positive_weight = movement_positive_weight(
        target,
        training_rows,
        finger_index,
        args.movement_threshold,
    )
    velocity_scale = grouped_velocity_scale(
        target[:, finger_index], training_groups
    )
    parts = schedule.split("_")
    frozen_updates = int(parts[1])
    unfrozen_updates = 0 if parts[0] == "frozen" else int(parts[2])
    if frozen_updates:
        sampler = make_sampler(args, training_groups, seed)
        opt = optimizer(
            model,
            args.head_learning_rate,
            0.0,
            0.0,
            0.0,
            args.weight_decay,
            args.movement_modulation_learning_rate,
        )
        decode = (
            model.decode_features_with_auxiliary
            if args.movement_loss_weight or args.velocity_loss_weight
            else model.decode_features
        )
        if args.compile:
            decode = torch.compile(decode, mode="reduce-overhead")
        train_updates(
            model=model,
            call=decode,
            optimizer_instance=opt,
            cached=cached,
            padded_ecog=padded_ecog,
            target=target[:, finger_index],
            sampler=sampler,
            updates=frozen_updates,
            steps=args.sequence_steps,
            batch_size=args.batch_size,
            target_scale=scale,
            raw_stem=False,
            movement_loss_weight=args.movement_loss_weight,
            movement_trajectory_weight=args.movement_trajectory_weight,
            movement_threshold=args.movement_threshold,
            movement_positive_weight=positive_weight,
            velocity_loss_weight=args.velocity_loss_weight,
            velocity_scale=velocity_scale,
            correlation_loss_weight=args.correlation_loss_weight,
            derivative_correlation_weight=args.derivative_correlation_weight,
            raw_target=raw_target,
            raw_movement_correlation_weight=args.raw_movement_correlation_weight,
            raw_movement_derivative_correlation_weight=(
                args.raw_movement_derivative_correlation_weight
            ),
        )
    if unfrozen_updates:
        sampler = make_sampler(args, training_groups, seed + 1000)
        opt = optimizer(
            model,
            args.head_learning_rate * 0.5,
            args.spatial_learning_rate,
            args.wavelet_learning_rate,
            args.interlevel_learning_rate,
            args.weight_decay,
            args.movement_modulation_learning_rate,
            args.signed_pooling_learning_rate,
        )
        forward = (
            model.forward_with_auxiliary
            if args.movement_loss_weight or args.velocity_loss_weight
            else model.forward
        )
        if args.compile:
            forward = torch.compile(forward, mode="reduce-overhead")
        train_updates(
            model=model,
            call=forward,
            optimizer_instance=opt,
            cached=cached,
            padded_ecog=padded_ecog,
            target=target[:, finger_index],
            sampler=sampler,
            updates=unfrozen_updates,
            steps=args.sequence_steps,
            batch_size=args.batch_size,
            target_scale=scale,
            raw_stem=True,
            movement_loss_weight=args.movement_loss_weight,
            movement_trajectory_weight=args.movement_trajectory_weight,
            movement_threshold=args.movement_threshold,
            movement_positive_weight=positive_weight,
            velocity_loss_weight=args.velocity_loss_weight,
            velocity_scale=velocity_scale,
            correlation_loss_weight=args.correlation_loss_weight,
            derivative_correlation_weight=args.derivative_correlation_weight,
            raw_target=raw_target,
            raw_movement_correlation_weight=args.raw_movement_correlation_weight,
            raw_movement_derivative_correlation_weight=(
                args.raw_movement_derivative_correlation_weight
            ),
        )


def plot_oof(
    path: Path,
    target: np.ndarray,
    initialized: np.ndarray,
    tuned: np.ndarray,
    title: str = "five-fold event-balanced nested tuning",
) -> None:
    observed = np.isfinite(tuned)
    first = int(np.flatnonzero(observed)[0])
    stop = min(tuned.size, first + 1250)
    x = np.arange(first, stop) / 25.0
    figure, axis = plt.subplots(figsize=(14, 4.8), constrained_layout=True)
    axis.plot(x, target[first:stop], color="black", linewidth=1.0, label="cleaned target")
    axis.plot(x, initialized[first:stop], linewidth=0.8, alpha=0.7, label="LARS initialization")
    axis.plot(x, tuned[first:stop], linewidth=0.9, label="inner-selected checkpoint")
    axis.set_title(title)
    axis.set_xlabel("original recording time (s)")
    axis.set_ylabel("cleaned movement scale")
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--finger", choices=tuple(FINGER_NAMES), default="little")
    parser.add_argument(
        "--outer-folds", type=int, nargs="+", choices=(0, 1, 2), default=(0, 1, 2)
    )
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path("outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1"),
    )
    parser.add_argument("--resampled-cache", type=Path, default=None)
    parser.add_argument("--leaf-cache", type=Path, default=None)
    parser.add_argument("--model-rate", type=int, choices=(400, 1000), default=1000)
    parser.add_argument("--tap-resample-up", type=int, default=5)
    parser.add_argument("--tap-resample-down", type=int, default=2)
    parser.add_argument(
        "--wavelet-frontend",
        choices=("depth3", "overcomplete_depth3_depth4"),
        default="depth3",
        help=(
            "depth3 uses eight 0--200 Hz leaves; overcomplete_depth3_depth4 "
            "retains those leaves and their sixteen depth-4 children in one tree"
        ),
    )
    parser.add_argument(
        "--wavelet-interlevel-skip",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--wavelet-interlevel-normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--wavelet-final-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also learn normalization on the final leaves before energy pooling",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--initialization-cache-root",
        type=Path,
        default=None,
        help=(
            "optional prior balanced-run directory whose split-local target, "
            "ICA/CSP, and LARS caches should be reused for a paired decoder comparison"
        ),
    )
    parser.add_argument(
        "--reuse-inner-metrics-from",
        type=Path,
        default=None,
        help="reuse a completed same-learning-rate run's inner-fold curves",
    )
    parser.add_argument(
        "--require-lstm-update",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep zero-update LARS as a baseline but require the selected model to train the LSTM",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("event_macro_nmse", "raw_pcc"),
        default="raw_pcc",
    )
    parser.add_argument(
        "--selection-rule", choices=("one_se", "best"), default="best"
    )
    parser.add_argument(
        "--force-schedule",
        default=None,
        help="diagnostic outer-fold schedule; inner curves remain unchanged",
    )
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--movement-threshold", type=float, default=0.10)
    parser.add_argument("--rest-threshold", type=float, default=0.05)
    parser.add_argument(
        "--little-event-decontamination",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="softly suppress little-finger events dominated by another finger",
    )
    parser.add_argument("--little-event-ratio-low", type=float, default=0.8)
    parser.add_argument("--little-event-ratio-high", type=float, default=1.2)
    parser.add_argument("--merge-gap-bins", type=int, default=12)
    parser.add_argument("--minimum-event-bins", type=int, default=3)
    parser.add_argument("--maximum-rest-group-bins", type=int, default=250)
    parser.add_argument("--purge-bins", type=int, default=95)
    parser.add_argument("--ica-prescreen", type=int, default=512)
    parser.add_argument("--component-chunk", type=int, default=16)
    parser.add_argument(
        "--lasso-backend",
        choices=("sklearn_lars", "torch_fista"),
        default="sklearn_lars",
        help="split-local sparse initializer; torch_fista fits its alpha path on the selected device",
    )
    parser.add_argument("--csp-mode", choices=tuple(CSP_MODES), default="movement_1")
    parser.add_argument(
        "--csp-band-mode", choices=CSP_BAND_MODES, default="joint_hhl_hhh"
    )
    parser.add_argument(
        "--ica-cache-root",
        type=Path,
        default=None,
        help="reuse split-local full-rank ICA rows while refitting CSP and LARS",
    )
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument(
        "--head-initialization",
        choices=("lars_linear_regime",),
        default="lars_linear_regime",
        help="embed the split-local LARS predictor in one near-linear LSTM unit",
    )
    parser.add_argument("--lars-candidate-scale", type=float, default=1.0)
    parser.add_argument("--lars-near-zero-std", type=float, default=1.0e-3)
    parser.add_argument("--lars-open-gate-bias", type=float, default=5.0)
    parser.add_argument("--lars-forget-gate-bias", type=float, default=-5.0)
    parser.add_argument(
        "--recurrent-cell",
        choices=("standard", "paper_equations", "residual_lstm", "residual_gru"),
        default="standard",
        help=(
            "LARS-initialized standard/paper LSTM, or a zero-initialized "
            "LSTM/GRU residual on the fixed LARS logit"
        ),
    )
    parser.add_argument(
        "--residual-input",
        choices=(
            "selected",
            "candidate",
            "current_candidate",
            "causal_candidate",
            "selected_causal",
        ),
        default="selected",
        help=(
            "feed the residual recurrent cell the LARS-selected subset, the "
            "complete split-local pre-LARS candidate pool, only its newest-bin "
            "members, the current-time source stream for every candidate identity, "
            "or only current-time streams represented among nonzero LARS atoms; the "
            "full LARS direct path is retained in all residual modes"
        ),
    )
    parser.add_argument(
        "--output-activation", choices=("linear", "softplus"), default="softplus"
    )
    parser.add_argument(
        "--residual-history-bins",
        type=int,
        default=1,
        help=(
            "number of newest 25 Hz lag bins visible to current_candidate "
            "recurrent input; the direct LARS path always retains all 25 bins"
        ),
    )
    parser.add_argument(
        "--residual-input-width",
        type=int,
        default=None,
        help=(
            "optional fixed zero-padded width for current_candidate recurrent "
            "input, allowing torch.compile graph reuse across split-local pools"
        ),
    )
    parser.add_argument(
        "--residual-include-direct",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "append the fixed split-local LARS logit to the residual recurrent "
            "input so the same LSTM can learn state-dependent corrections"
        ),
    )
    parser.add_argument("--softplus-beta", type=float, default=10.0)
    parser.add_argument("--sequence-steps", type=int, default=100)
    parser.add_argument("--sequence-stride", type=int, default=25)
    parser.add_argument(
        "--sampler-mode",
        choices=("uniform_group", "dense_sequences"),
        default="dense_sequences",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--head-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--spatial-learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--wavelet-learning-rate", type=float, default=3.0e-6)
    parser.add_argument(
        "--interlevel-learning-rate",
        type=float,
        default=3.0e-4,
        help="learning rate for zero-gated interlevel skip/normalization paths",
    )
    parser.add_argument(
        "--frozen-update-grid",
        type=int,
        nargs="+",
        default=FROZEN_UPDATES,
        help="cumulative head-only update checkpoints",
    )
    parser.add_argument(
        "--unfreeze-after",
        type=int,
        default=UNFREEZE_AFTER,
        help="head-only checkpoint from which end-to-end tuning begins",
    )
    parser.add_argument(
        "--unfrozen-update-grid",
        type=int,
        nargs="+",
        default=UNFROZEN_UPDATES,
        help="cumulative end-to-end update checkpoints after unfreezing",
    )
    parser.add_argument(
        "--frozen-only",
        action="store_true",
        help="compare the LARS initialization with LSTM-only updates; do not tune the stem",
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--movement-loss-weight",
        type=float,
        default=0.0,
        help="weight of a balanced target-finger movement/rest BCE auxiliary head",
    )
    parser.add_argument(
        "--movement-trajectory-weight",
        type=float,
        default=1.0,
        help="relative trajectory-MSE weight for bins at or above the movement threshold",
    )
    parser.add_argument(
        "--velocity-loss-weight",
        type=float,
        default=0.0,
        help="weight of an auxiliary velocity-regression head on the shared LSTM state",
    )
    parser.add_argument(
        "--movement-modulation",
        action="store_true",
        help="learn a smooth state-conditioned trajectory gain initialized exactly to one",
    )
    parser.add_argument(
        "--movement-modulation-learning-rate",
        type=float,
        default=None,
        help="optional learning rate for the scalar movement modulation gain",
    )
    parser.add_argument(
        "--wavelet-signed-pooling",
        action="store_true",
        help="add a zero-gated signed mean to each existing leaf-energy statistic",
    )
    parser.add_argument(
        "--signed-pooling-learning-rate",
        type=float,
        default=3.0e-4,
        help="learning rate for the eight zero-initialized signed-pooling gates",
    )
    parser.add_argument(
        "--residual-output-init-std",
        type=float,
        default=0.0,
        help=(
            "near-zero random initialization for residual-head coefficients; "
            "zero preserves exact LARS output but delays recurrent gradients by one update"
        ),
    )
    parser.add_argument("--correlation-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--derivative-correlation-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--raw-movement-correlation-weight",
        type=float,
        default=0.0,
        help="movement-bin shape loss against the raw glove trajectory",
    )
    parser.add_argument(
        "--raw-movement-derivative-correlation-weight",
        type=float,
        default=0.0,
        help="movement-bin velocity-shape loss against the raw glove trajectory",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="fix event-fold construction while varying only model initialization",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="fix minibatch order while varying only model initialization",
    )
    parser.add_argument("--feature-chunk", type=int, default=256)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prepare-shared-only",
        action="store_true",
        help="prepare the subject-level 400 Hz and fixed-leaf caches, then exit",
    )
    args = parser.parse_args()
    if (
        args.movement_loss_weight < 0
        or args.movement_trajectory_weight <= 0
        or args.velocity_loss_weight < 0
        or args.residual_output_init_std < 0
        or args.correlation_loss_weight < 0
        or args.derivative_correlation_weight < 0
        or args.raw_movement_correlation_weight < 0
        or args.raw_movement_derivative_correlation_weight < 0
        or (
            args.movement_modulation_learning_rate is not None
            and args.movement_modulation_learning_rate <= 0
        )
        or args.signed_pooling_learning_rate <= 0
    ):
        raise ValueError("auxiliary loss weights must be nonnegative")
    if args.residual_input in (
        "candidate",
        "current_candidate",
        "causal_candidate",
        "selected_causal",
    ) and args.recurrent_cell not in ("residual_lstm", "residual_gru"):
        raise ValueError(
            "candidate residual inputs require --recurrent-cell residual_lstm or residual_gru"
        )
    if not 1 <= args.residual_history_bins <= HISTORY:
        raise ValueError(f"--residual-history-bins must be between 1 and {HISTORY}")
    if args.residual_input != "current_candidate" and args.residual_history_bins != 1:
        raise ValueError(
            "--residual-history-bins only applies to --residual-input current_candidate"
        )
    if args.residual_input_width is not None and args.residual_input_width <= 0:
        raise ValueError("--residual-input-width must be positive")
    if (
        args.residual_input
        not in ("current_candidate", "causal_candidate", "selected_causal")
        and args.residual_input_width is not None
    ):
        raise ValueError(
            "--residual-input-width only applies to current/causal candidate input"
        )
    if args.movement_modulation and args.movement_loss_weight <= 0:
        raise ValueError("--movement-modulation requires --movement-loss-weight")
    if args.frozen_only and (
        args.wavelet_interlevel_skip
        or args.wavelet_interlevel_normalization
        or args.wavelet_signed_pooling
    ):
        raise ValueError(
            "trainable wavelet paths require end-to-end raw-stem updates; "
            "they cannot affect --frozen-only cached-feature training"
        )
    if args.model_rate % 25:
        raise ValueError("model rate must be divisible by the 25 Hz target rate")
    args.samples_per_bin = args.model_rate // 25
    if args.model_rate == 400 and (args.tap_resample_up, args.tap_resample_down) != (1, 1):
        raise ValueError("400 Hz mode already spans 0--200 Hz; do not stretch its taps")

    frozen_grid = tuple(sorted(set(int(value) for value in args.frozen_update_grid)))
    unfrozen_grid = tuple(
        sorted(set(int(value) for value in args.unfrozen_update_grid))
    )
    if not frozen_grid or any(value <= 0 for value in frozen_grid):
        raise ValueError("frozen update checkpoints must be positive")
    if not args.frozen_only and (
        not unfrozen_grid or any(value <= 0 for value in unfrozen_grid)
    ):
        raise ValueError("unfrozen update checkpoints must be positive")
    if not args.frozen_only and args.unfreeze_after not in frozen_grid:
        raise ValueError("--unfreeze-after must appear in --frozen-update-grid")
    globals()["FROZEN_UPDATES"] = frozen_grid
    globals()["UNFREEZE_AFTER"] = int(args.unfreeze_after)
    globals()["UNFROZEN_UPDATES"] = () if args.frozen_only else unfrozen_grid

    finger_index = list(FINGER_NAMES).index(args.finger)
    if args.output is None:
        args.output = Path(
            f"outputs/single_wavelet_1000hz_tap5over2_nested_v2/"
            f"sub{args.subject}/{args.finger}"
        )
    if args.resampled_cache is None:
        args.resampled_cache = Path(
            f"/dev/shm/ecog_wavelet_{args.model_rate}hz/sub{args.subject}/train_ecog.npy"
        )
    if args.leaf_cache is None:
        args.leaf_cache = Path(
            f"/dev/shm/ecog_wavelet_{args.model_rate}hz_"
            f"taps{args.tap_resample_up}over{args.tap_resample_down}/sub{args.subject}"
        )

    started = time.perf_counter()
    model_seed, split_seed, sampler_seed = resolve_seed_roles(
        args.seed, args.split_seed, args.sampler_seed
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if args.compile:
        # Inner folds have different LARS feature counts; retain their compiled
        # graphs instead of falling back to eager execution after eight shapes.
        torch._dynamo.config.cache_size_limit = 64
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    reused_report = (
        json.loads((args.reuse_inner_metrics_from / "summary.json").read_text())
        if args.reuse_inner_metrics_from is not None
        else None
    )

    fold_definition = json.loads(
        (
            args.fold_root
            / f"sub{args.subject}"
            / args.finger
            / "folds.json"
        ).read_text()
    )
    rows = int(fold_definition["training_rows"])
    source_ecog = args.prepared_root / f"sub{args.subject}" / "train_ecog.npy"
    if args.model_rate == SOURCE_RATE:
        ecog = np.asarray(np.load(source_ecog, mmap_mode="r"))
    else:
        ecog = np.asarray(resample_ecog(source_ecog, args.resampled_cache))
    raw_full = np.load(
        args.prepared_root / f"sub{args.subject}" / "train_glove_25hz_raw.npy",
        mmap_mode="r",
    )
    raw_matrix = np.asarray(raw_full[OFFSET : OFFSET + rows], dtype=np.float64)
    raw = raw_matrix[:, finger_index].astype(np.float32)
    frontend_type = (
        OvercompleteWaveletPacketEnergy
        if args.wavelet_frontend == "overcomplete_depth3_depth4"
        else WaveletPacketEnergy
    )
    frontend = frontend_type(
        wavelet="bior6.8",
        levels=3,
        kernel_size=17,
        trainable=False,
        padding_mode="constant",
        energy_window_samples=args.samples_per_bin,
        energy_stride_samples=args.samples_per_bin,
        tap_resample_up=args.tap_resample_up,
        tap_resample_down=args.tap_resample_down,
    ).to(device).eval()
    csp_frontend = (
        frontend
        if args.wavelet_frontend == "depth3"
        else WaveletPacketEnergy(
            wavelet="bior6.8",
            levels=3,
            kernel_size=17,
            trainable=False,
            padding_mode="constant",
            energy_window_samples=args.samples_per_bin,
            energy_stride_samples=args.samples_per_bin,
            tap_resample_up=args.tap_resample_up,
            tap_resample_down=args.tap_resample_down,
        ).to(device).eval()
    )
    hhl_path = args.leaf_cache / "linear_hhl_100_125.npy"
    hhh_path = args.leaf_cache / "linear_hhh_125_150.npy"
    if not (
        valid_leaf_cache(hhl_path, ecog.shape[0])
        and valid_leaf_cache(hhh_path, ecog.shape[0])
    ):
        hhl_values, hhh_values = linear_gamma_leaf_signals(
            ecog, csp_frontend, device
        )
        atomic_save_npy(hhl_path, hhl_values)
        atomic_save_npy(hhh_path, hhh_values)
    lower_first_path = args.leaf_cache / "linear_50_75.npy"
    lower_second_path = args.leaf_cache / "linear_75_100.npy"
    use_lower_high_gamma = args.csp_band_mode == "separate_50_100_hhl_hhh"
    if use_lower_high_gamma and not (
        valid_leaf_cache(lower_first_path, ecog.shape[0])
        and valid_leaf_cache(lower_second_path, ecog.shape[0])
    ):
        lower_first, lower_second = linear_lower_high_gamma_leaf_signals(
            ecog, csp_frontend, device
        )
        atomic_save_npy(lower_first_path, lower_first)
        atomic_save_npy(lower_second_path, lower_second)
    hhl = np.load(hhl_path, mmap_mode="r")
    hhh = np.load(hhh_path, mmap_mode="r")
    if args.prepare_shared_only:
        print(
            json.dumps(
                {
                    "subject": args.subject,
                    "resampled_cache": str(args.resampled_cache),
                    "leaf_cache": str(args.leaf_cache),
                    "status": "prepared",
                }
            ),
            flush=True,
        )
        return
    bins = ecog.shape[0] // args.samples_per_bin
    hhl_bins = hhl[: bins * args.samples_per_bin].reshape(
        bins, args.samples_per_bin, -1
    )
    hhh_bins = hhh[: bins * args.samples_per_bin].reshape(
        bins, args.samples_per_bin, -1
    )
    joint_bins = np.concatenate((hhl_bins, hhh_bins), axis=1)
    lower_high_gamma_bins = None
    if use_lower_high_gamma:
        lower_first = np.load(lower_first_path, mmap_mode="r")
        lower_second = np.load(lower_second_path, mmap_mode="r")
        lower_first_bins = lower_first[: bins * args.samples_per_bin].reshape(
            bins, args.samples_per_bin, -1
        )
        lower_second_bins = lower_second[: bins * args.samples_per_bin].reshape(
            bins, args.samples_per_bin, -1
        )
        lower_high_gamma_bins = np.concatenate(
            (lower_first_bins, lower_second_bins), axis=1
        )
    ecog_t = torch.from_numpy(ecog.copy()).to(device)
    context = frontend.effective_kernel_size // 2
    padded_ecog = F.pad(ecog_t.T[None], (context, context)).squeeze(0).T

    initialized_oof = np.full(rows, np.nan, dtype=np.float32)
    tuned_oof = np.full(rows, np.nan, dtype=np.float32)
    target_oof = np.full(rows, np.nan, dtype=np.float32)
    outer_records = []

    for outer_fold in args.outer_folds:
        fold_started = time.perf_counter()
        outer_definition = split_intervals(fold_definition, outer_fold)["outer"]
        outer_training_intervals = [
            interval
            for interval in outer_definition["training_intervals"]
            if interval[1] - interval[0] >= 3
        ]
        outer_validation_intervals = outer_definition["validation_intervals"]
        outer_target = make_subject_target(
            raw_matrix,
            outer_training_intervals,
            outer_validation_intervals,
            args.subject,
            finger_index=finger_index,
            little_event_decontamination=args.little_event_decontamination,
            little_event_ratio_low=args.little_event_ratio_low,
            little_event_ratio_high=args.little_event_ratio_high,
        )
        splits, groups, assignment_objective = balanced_inner_splits(
            rows=rows,
            outer_training_intervals=outer_training_intervals,
            target=outer_target,
            threshold=args.threshold,
            merge_gap_bins=args.merge_gap_bins,
            minimum_event_bins=args.minimum_event_bins,
            maximum_rest_group_bins=args.maximum_rest_group_bins,
            purge_bins=args.purge_bins,
            seed=split_seed + outer_fold,
            finger_index=finger_index,
        )
        inner_records = []
        for split in ([] if reused_report is not None else splits):
            cache_root = args.initialization_cache_root or args.output
            cache = cache_root / "cache" / f"outer{outer_fold}" / f"inner{split['fold']}"
            requested_split = split

            def create_inner_initialization():
                target_np = make_subject_target(
                    raw_matrix,
                    requested_split["training_intervals"],
                    requested_split["validation_intervals"],
                    args.subject,
                    finger_index=finger_index,
                    little_event_decontamination=args.little_event_decontamination,
                    little_event_ratio_low=args.little_event_ratio_low,
                    little_event_ratio_high=args.little_event_ratio_high,
                )
                initialization, spatial, audit = fit_initialization(
                    ecog=ecog,
                    joint_bins=joint_bins,
                    target=target_np,
                    training_intervals=requested_split["training_intervals"],
                    training_groups=requested_split["training_groups"],
                    frontend=frontend,
                    device=device,
                    ica_prescreen=args.ica_prescreen,
                    component_chunk=args.component_chunk,
                    finger_index=finger_index,
                    csp_mode=args.csp_mode,
                    csp_band_mode=args.csp_band_mode,
                    hhl_bins=hhl_bins,
                    hhh_bins=hhh_bins,
                    lower_high_gamma_bins=lower_high_gamma_bins,
                    ica_weights=load_cached_ica(
                        args.ica_cache_root,
                        args.subject,
                        args.finger,
                        outer_fold,
                        f"inner{requested_split['fold']}",
                        ecog.shape[1],
                    ),
                    samples_per_bin=args.samples_per_bin,
                    lasso_backend=args.lasso_backend,
                    include_candidate_pool=args.residual_input
                    in (
                        "candidate",
                        "current_candidate",
                        "causal_candidate",
                        "selected_causal",
                    ),
                )
                return initialization, spatial, target_np, requested_split, audit

            initialization, spatial, target_np, split = load_or_create_initialization(
                cache, create_inner_initialization
            )
            torch.manual_seed(model_seed)
            model = make_model(spatial, initialization, args).to(device)
            cached = torch.from_numpy(
                cached_initialization_features(initialization, args.residual_input)
            ).to(device)
            target = torch.from_numpy(target_np.astype(np.float32)).to(device)
            metrics = monitor_inner_fold(
                model=model,
                cached=cached,
                padded_ecog=padded_ecog,
                target=target,
                target_np=target_np,
                raw=raw,
                training_groups=split["training_groups"],
                validation_intervals=split["validation_intervals"],
                args=args,
                seed=sampler_seed + int(split["fold"]),
                finger_index=finger_index,
            )
            inner_records.append(
                {
                    "fold": int(split["fold"]),
                    "training_bins": int(split["training_bins"]),
                    "validation_bins": int(split["validation_bins"]),
                    "training_group_count": len(split["training_groups"]),
                    "selected_feature_count": int(
                        initialization["selected_indices"].size
                    ),
                    "recurrent_input_feature_count": int(model.lstm.input_size),
                    "metrics": metrics,
                }
            )
            direct_feature_count = (
                int(model.candidate_indices.numel())
                if hasattr(model, "candidate_indices")
                else int(model.selected_indices.numel())
            )
            print(
                f"outer={outer_fold} inner={split['fold']} "
                f"recurrent_features={model.lstm.input_size} "
                f"direct_features={direct_feature_count}",
                flush=True,
            )
        if reused_report is not None:
            reused_outer = next(
                record
                for record in reused_report["outer_folds"]
                if int(record["outer_fold"]) == outer_fold
            )
            inner_records = reused_outer["inner_records"]
        selected_schedule, selection_summary = one_standard_error_selection(
            inner_records,
            require_lstm_update=args.require_lstm_update,
            selection_metric=args.selection_metric,
            selection_rule=args.selection_rule,
        )
        if args.force_schedule is not None:
            if args.force_schedule not in schedule_order():
                raise ValueError(
                    f"forced schedule {args.force_schedule!r} is not in the configured grid"
                )
            selected_schedule = args.force_schedule

        outer_cache_root = args.initialization_cache_root or args.output
        outer_cache = outer_cache_root / "cache" / f"outer{outer_fold}" / "outer"
        def create_outer_initialization():
            initialization, spatial, audit = fit_initialization(
                ecog=ecog,
                joint_bins=joint_bins,
                target=outer_target,
                training_intervals=outer_training_intervals,
                training_groups=[[int(start), int(stop)] for start, stop in groups],
                frontend=frontend,
                device=device,
                ica_prescreen=args.ica_prescreen,
                component_chunk=args.component_chunk,
                finger_index=finger_index,
                csp_mode=args.csp_mode,
                csp_band_mode=args.csp_band_mode,
                hhl_bins=hhl_bins,
                hhh_bins=hhh_bins,
                lower_high_gamma_bins=lower_high_gamma_bins,
                ica_weights=load_cached_ica(
                    args.ica_cache_root,
                    args.subject,
                    args.finger,
                    outer_fold,
                    "outer",
                    ecog.shape[1],
                ),
                samples_per_bin=args.samples_per_bin,
                lasso_backend=args.lasso_backend,
                include_candidate_pool=args.residual_input
                in (
                    "candidate",
                    "current_candidate",
                    "causal_candidate",
                    "selected_causal",
                ),
            )
            return initialization, spatial, outer_target, outer_definition, audit

        initialization, spatial, outer_target, _ = load_or_create_initialization(
            outer_cache, create_outer_initialization
        )
        torch.manual_seed(model_seed)
        model = make_model(spatial, initialization, args).to(device)
        cached = torch.from_numpy(
            cached_initialization_features(initialization, args.residual_input)
        ).to(device)
        target_np = outer_target.astype(np.float32)
        target = torch.from_numpy(target_np).to(device)
        with torch.inference_mode():
            initialized = model.direct_features(cached).float().cpu().numpy()
        train_final_schedule(
            model=model,
            cached=cached,
            padded_ecog=padded_ecog,
            target=target,
            raw=raw,
            training_groups=[[int(start), int(stop)] for start, stop in groups],
            schedule=selected_schedule,
            args=args,
            seed=sampler_seed,
            finger_index=finger_index,
        )
        if selected_schedule.startswith("unfrozen"):
            final_features = extract_all(model, padded_ecog, rows, args.feature_chunk)
        else:
            final_features = cached
        prediction = predict_intervals(
            model, final_features, outer_validation_intervals, rows
        )
        validation = indices_from_intervals(outer_validation_intervals)
        initialized_oof[validation] = initialized[validation]
        tuned_oof[validation] = prediction[validation]
        target_oof[validation] = target_np[validation, finger_index]
        initialized_metrics = validation_metrics(
            initialized,
            target_np[:, finger_index],
            raw,
            outer_validation_intervals,
            args.movement_threshold,
            args.rest_threshold,
        )
        selected_metrics = validation_metrics(
            prediction,
            target_np[:, finger_index],
            raw,
            outer_validation_intervals,
            args.movement_threshold,
            args.rest_threshold,
        )
        record = {
            "outer_fold": outer_fold,
            "event_group_count": len(groups),
            "fold_assignment_objective": assignment_objective,
            "selected_schedule": selected_schedule,
            "selection_summary": selection_summary,
            "inner_records": inner_records,
            "initialized_outer_metrics": initialized_metrics,
            "selected_outer_metrics": selected_metrics,
            "runtime_seconds": time.perf_counter() - fold_started,
        }
        outer_records.append(record)
        torch.save(model.state_dict(), args.output / f"outer{outer_fold}_model.pt")
        print(
            json.dumps(
                {
                    "outer_fold": outer_fold,
                    "selected_schedule": selected_schedule,
                    "initial_raw_pcc": initialized_metrics["raw_pcc"],
                    "selected_raw_pcc": selected_metrics["raw_pcc"],
                    "initial_nmse": initialized_metrics["event_macro_nmse"],
                    "selected_nmse": selected_metrics["event_macro_nmse"],
                }
            ),
            flush=True,
        )

    observed = np.isfinite(tuned_oof)
    report = {
        "protocol": (
            "one single-path subject/finger model; five event-balanced inner folds per "
            "outer-training scope; split-local target, "
            "ICA, configured HHL/HHH CSP and LARS refits; "
            f"{args.recurrent_cell} nonlinear gated LSTM decoder with "
            f"{args.residual_input} residual input "
            f"with {args.head_initialization}; {args.sampler_mode} minibatches; "
            f"optimizer-update checkpoints; {args.selection_rule} selection on "
            f"{args.selection_metric}; outer fold evaluated once; released test untouched"
        ),
        "frontend": {
            "name": (
                "single_wavelet_frequency_compressed_depth3_depth4_overcomplete"
                if args.wavelet_frontend == "overcomplete_depth3_depth4"
                else "single_wavelet_frequency_compressed_depth3"
            ),
            "source_rate_hz": SOURCE_RATE,
            "model_rate_hz": args.model_rate,
            "input_resampling": (
                "none" if args.model_rate == SOURCE_RATE else "polyphase_2_over_5_kaiser_8.6"
            ),
            "tap_interpolation": {
                "up": args.tap_resample_up,
                "down": args.tap_resample_down,
                "method": "polyphase_kaiser_8.6",
            },
            "wavelet": "bior6.8",
            "levels": [3, 4]
            if args.wavelet_frontend == "overcomplete_depth3_depth4"
            else [3],
            "leaf_count": 24
            if args.wavelet_frontend == "overcomplete_depth3_depth4"
            else 8,
            "nominal_frequency_edges_hz": {
                "depth3": list(range(0, 201, 25)),
                "depth4": [12.5 * index for index in range(17)],
            }
            if args.wavelet_frontend == "overcomplete_depth3_depth4"
            else list(range(0, 201, 25)),
            "auxiliary_temporal_branches": [],
            "lmp_branch": False,
            "zero_initialized_interlevel_skip": args.wavelet_interlevel_skip,
            "zero_initialized_interlevel_normalization": (
                args.wavelet_interlevel_normalization
            ),
            "final_leaf_normalization": args.wavelet_final_normalization,
            "zero_initialized_signed_leaf_pooling": args.wavelet_signed_pooling,
            "energy_pool_samples": args.samples_per_bin,
        },
        "primary_selection_metric": args.selection_metric,
        "monitored_diagnostic_metrics": [
            "raw_pcc",
            "cleaned_pcc",
            "normalized_rmse",
            "rest_false_positive_rms",
            "movement_rmse",
            "amplitude_slope",
            "velocity_pcc",
            "movement_balanced_accuracy",
        ],
        "released_test_touched": False,
        "subject": args.subject,
        "finger": args.finger,
        "target_policy": {
            "window_seconds": TARGET_POLICIES[args.subject][0],
            "baseline_quantile": TARGET_POLICIES[args.subject][1],
            "smoothing_seconds": 0.16,
            "little_event_decontamination": args.little_event_decontamination,
            "little_event_ratio_low": args.little_event_ratio_low,
            "little_event_ratio_high": args.little_event_ratio_high,
        },
        "outer_validation_used_for_checkpoint_selection": False,
        "require_lstm_update": args.require_lstm_update,
        "reused_inner_metrics_from": (
            str(args.reuse_inner_metrics_from)
            if args.reuse_inner_metrics_from is not None
            else None
        ),
        "decoder": (
            f"fixed split-local LARS direct path plus zero-initialized "
            f"{args.recurrent_cell} nonlinear residual; recurrent input="
            f"{args.residual_input}"
            if args.recurrent_cell in ("residual_lstm", "residual_gru")
            else f"LARS-initialized {args.recurrent_cell} nonlinear gated LSTM"
        ),
        "training_objective": {
            "trajectory": "normalized mean squared error",
            "movement_trajectory_weight": args.movement_trajectory_weight,
            "movement_state_bce_weight": args.movement_loss_weight,
            "movement_state_trajectory_modulation": args.movement_modulation,
            "auxiliary_velocity_mse_weight": args.velocity_loss_weight,
            "within_sequence_correlation_weight": args.correlation_loss_weight,
            "within_sequence_velocity_correlation_weight": (
                args.derivative_correlation_weight
            ),
            "raw_movement_level_correlation_weight": (
                args.raw_movement_correlation_weight
            ),
            "raw_movement_velocity_correlation_weight": (
                args.raw_movement_derivative_correlation_weight
            ),
            "model_outputs": ["trajectory"]
            + (
                ["target_finger_movement_logit"]
                if args.movement_loss_weight
                else []
            )
            + (["target_finger_velocity"] if args.velocity_loss_weight else []),
            "residual_recurrent_sees_direct_lars_logit": args.residual_include_direct,
            "residual_output_initialization_std": args.residual_output_init_std,
        },
        "learning_rates": {
            "head": args.head_learning_rate,
            "movement_modulation": args.movement_modulation_learning_rate,
            "spatial": args.spatial_learning_rate,
            "wavelet_taps": args.wavelet_learning_rate,
            "signed_leaf_pooling": args.signed_pooling_learning_rate,
            "interlevel_paths": args.interlevel_learning_rate,
        },
        "initialization_cache_root": (
            str(args.initialization_cache_root)
            if args.initialization_cache_root is not None
            else str(args.output)
        ),
        "lars_lstm_initialization": {
            "candidate_scale": args.lars_candidate_scale,
            "near_zero_std": args.lars_near_zero_std,
            "open_gate_bias": args.lars_open_gate_bias,
            "forget_gate_bias": args.lars_forget_gate_bias,
            "recurrent_cell": args.recurrent_cell,
        },
        "randomization_seeds": {
            "model_initialization": model_seed,
            "event_split": split_seed,
            "minibatch_sampler": sampler_seed,
        },
        "schedule_candidates": schedule_order(),
        "outer_folds": outer_records,
        "mean_initialized_outer_raw_pcc": float(
            np.mean([record["initialized_outer_metrics"]["raw_pcc"] for record in outer_records])
        ),
        "mean_selected_outer_raw_pcc": float(
            np.mean([record["selected_outer_metrics"]["raw_pcc"] for record in outer_records])
        ),
        "mean_initialized_outer_event_macro_nmse": float(
            np.mean([record["initialized_outer_metrics"]["event_macro_nmse"] for record in outer_records])
        ),
        "mean_selected_outer_event_macro_nmse": float(
            np.mean([record["selected_outer_metrics"]["event_macro_nmse"] for record in outer_records])
        ),
        "stitched_initialized_raw_pcc": pearson(initialized_oof[observed], raw[observed]),
        "stitched_selected_raw_pcc": pearson(tuned_oof[observed], raw[observed]),
        "runtime_seconds": time.perf_counter() - started,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    np.save(args.output / "initialized_oof.npy", initialized_oof, allow_pickle=False)
    np.save(args.output / "selected_oof.npy", tuned_oof, allow_pickle=False)
    np.save(args.output / "target_oof.npy", target_oof, allow_pickle=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_oof(
        args.output / "oof_trace.png",
        target_oof,
        initialized_oof,
        tuned_oof,
        f"S{args.subject} {args.finger}: single-path nested tuning",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
