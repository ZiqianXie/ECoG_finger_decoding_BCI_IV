#!/usr/bin/env python3
"""Measure how much target signal remains in a frozen single-wavelet feature bank.

This is a diagnostic, not a candidate decoder.  Every outer fold uses its own
target, FastICA/CSP features, candidate screen, and LARS fit.  Ridge strengths
are selected with interval-grouped folds inside the outer-training scope; the
outer interval is evaluated once.  Released competition test labels are never
loaded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecog_decoding.training import FINGER_NAMES
from gpu_ridge import fit_torch_ridge_cv


ALPHAS = tuple(float(value) for value in np.logspace(-2, 5, 8))
EMA_DECAYS = (0.5, 0.8, 0.92, 0.97)
POWER_CALIBRATION_GRID = (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)


def rows_from_intervals(intervals: list[list[int]]) -> np.ndarray:
    return np.concatenate(
        [np.arange(start, stop, dtype=np.int64) for start, stop in intervals]
    )


def finite_intervals(
    intervals: list[list[int]], target: np.ndarray
) -> list[list[int]]:
    """Intersect intervals with contiguous rows having a finite scalar target."""
    finite = np.isfinite(np.asarray(target))
    result: list[list[int]] = []
    for start, stop in intervals:
        observed = np.flatnonzero(finite[start:stop])
        if observed.size == 0:
            continue
        boundaries = np.flatnonzero(np.diff(observed) > 1) + 1
        for block in np.split(observed, boundaries):
            result.append([int(start + block[0]), int(start + block[-1] + 1)])
    return result


def grouped_rows(
    intervals: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    rows = rows_from_intervals(intervals)
    groups = np.concatenate(
        [
            np.full(stop - start, group, dtype=np.int64)
            for group, (start, stop) in enumerate(intervals)
        ]
    )
    return rows, groups


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


def softplus(values: np.ndarray, beta: float) -> np.ndarray:
    return np.logaddexp(0.0, beta * values) / beta


def causal_ema(
    values: np.ndarray,
    intervals: list[list[int]],
    decay: float,
) -> np.ndarray:
    """Apply a causal EMA independently inside each complete event interval."""
    result = np.zeros_like(values, dtype=np.float32)
    for start, stop in intervals:
        if stop <= start:
            continue
        state = np.asarray(values[start], dtype=np.float32).copy()
        result[start] = state
        for row in range(start + 1, stop):
            state = decay * state + (1.0 - decay) * values[row]
            result[row] = state
    return result


def temporal_summary(
    current: np.ndarray,
    intervals: list[list[int]],
    base: np.ndarray,
) -> np.ndarray:
    summaries = [current]
    summaries.extend(causal_ema(current, intervals, decay) for decay in EMA_DECAYS)
    summaries.append(base[:, None])
    return np.concatenate(summaries, axis=1)


def interval_temporal_calibration_features(
    base: np.ndarray,
    intervals: list[list[int]],
    radius: int = 12,
) -> np.ndarray:
    """Build edge-replicated two-sided trajectory lags without crossing groups."""
    result = np.zeros((base.size, 2 * radius + 4), dtype=np.float32)
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    for start, stop in intervals:
        rows = np.arange(start, stop, dtype=np.int64)
        source = np.clip(rows[:, None] + offsets[None], start, stop - 1)
        result[rows, : offsets.size] = base[source]
    nonnegative = np.maximum(base, 0.0)
    result[:, -3] = nonnegative**0.5
    result[:, -2] = nonnegative**2
    result[:, -1] = nonnegative**3
    return result


def select_training_power_calibration(
    prediction: np.ndarray,
    raw_target: np.ndarray,
    training: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Select a nonnegative power transform using outer-training rows only."""
    nonnegative = np.maximum(np.asarray(prediction), 0.0)
    scores = {
        str(power): pearson(nonnegative[training] ** power, raw_target[training])
        for power in POWER_CALIBRATION_GRID
    }
    selected = max(POWER_CALIBRATION_GRID, key=lambda power: scores[str(power)])
    return nonnegative**selected, float(selected), scores


def select_ridge(
    features: np.ndarray,
    target: np.ndarray,
    training_intervals: list[list[int]],
    *,
    backend: str,
    device: torch.device,
    selection_metric: str,
) -> tuple[object, float, dict[str, float]]:
    training, groups = grouped_rows(training_intervals)
    splitter = GroupKFold(n_splits=min(5, len(training_intervals)))
    splits = list(splitter.split(training, target[training], groups))
    if backend == "torch":
        model = fit_torch_ridge_cv(
            features[training],
            target[training],
            splits,
            alphas=np.asarray(ALPHAS, dtype=np.float32),
            device=device,
            selection_metric=selection_metric,
        )
        scores = (
            model.mean_validation_mse_
            if selection_metric == "mse"
            else model.mean_validation_pcc_
        )
        means = {
            str(float(alpha)): float(score)
            for alpha, score in zip(model.alphas_, scores)
        }
        return model, model.alpha_, means
    if backend != "sklearn":
        raise ValueError(f"unsupported ridge backend {backend!r}")
    scores = {alpha: [] for alpha in ALPHAS}
    for train_local, validation_local in splits:
        train_rows = training[train_local]
        validation_rows = training[validation_local]
        for alpha in ALPHAS:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(features[train_rows], target[train_rows])
            estimate = model.predict(features[validation_rows])
            if selection_metric == "mse":
                score = float(np.mean((estimate - target[validation_rows]) ** 2))
            else:
                score = pearson(estimate, target[validation_rows])
            scores[alpha].append(score)
    means = {alpha: float(np.mean(values)) for alpha, values in scores.items()}
    selected = (
        min(means, key=means.get)
        if selection_metric == "mse"
        else max(means, key=means.get)
    )
    model = make_pipeline(StandardScaler(), Ridge(alpha=selected))
    model.fit(features[training], target[training])
    return model, selected, {str(alpha): loss for alpha, loss in means.items()}


def direct_lars(
    initialization: dict[str, np.ndarray],
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = initialization["candidate_features"]
    standardized = (
        candidates - initialization["candidate_feature_mean"]
    ) / initialization["candidate_feature_scale"]
    selected = initialization["selected_candidate_positions"].astype(np.int64)
    logit = (
        standardized[:, selected] @ initialization["coefficients"]
        + float(initialization["intercept"])
    )
    return softplus(logit, beta), standardized, logit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--finger", choices=FINGER_NAMES, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--outer-folds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--softplus-beta", type=float, default=10.0)
    parser.add_argument(
        "--ridge-backend", choices=("sklearn", "torch"), default="torch"
    )
    parser.add_argument(
        "--ridge-selection-metric", choices=("mse", "pcc"), default="mse"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raw-power-calibration", action="store_true")
    parser.add_argument(
        "--probes",
        nargs="+",
        choices=(
            "ridge_all_lags",
            "ridge_all_lags_raw",
            "ridge_current_temporal",
            "ridge_current_temporal_raw",
            "ridge_base_temporal_raw",
            "hist_current_temporal",
            "hist_current_temporal_raw",
        ),
        default=(
            "ridge_all_lags",
            "ridge_all_lags_raw",
            "ridge_current_temporal",
            "ridge_current_temporal_raw",
            "hist_current_temporal",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    finger_index = list(FINGER_NAMES).index(args.finger)

    stitched: dict[str, np.ndarray] = {}
    stitched_target = None
    stitched_raw = None
    outer_records = []
    for outer_fold in args.outer_folds:
        outer_started = time.perf_counter()
        directory = args.cache_root / "cache" / f"outer{outer_fold}" / "outer"
        split = json.loads((directory / "split.json").read_text())
        with np.load(directory / "initialization.npz") as archive:
            initialization = {name: archive[name] for name in archive.files}
        target = np.load(directory / "target.npy")[:, finger_index]
        rows = target.size
        raw = np.asarray(
            np.load(
                args.prepared_root
                / f"sub{args.subject}"
                / "train_glove_25hz_raw.npy",
                mmap_mode="r",
            )[24 : 24 + rows, finger_index]
        )
        if stitched_target is None:
            stitched_target = np.full(rows, np.nan, dtype=np.float32)
            stitched_raw = raw

        # Older caches recorded the unfiltered outer-fold definition rather than
        # the exact intervals used by the fit.  Intersecting with finite target
        # runs makes those caches safe while newer caches store the fitted scope.
        training_intervals = finite_intervals(split["training_intervals"], target)
        validation_intervals = finite_intervals(split["validation_intervals"], target)
        validation = rows_from_intervals(validation_intervals)
        all_intervals = training_intervals + validation_intervals
        base, standardized, _ = direct_lars(
            initialization, args.softplus_beta
        )
        current_positions = initialization["current_candidate_positions"].astype(
            np.int64
        )
        current = standardized[:, current_positions]
        temporal = temporal_summary(current, all_intervals, base)
        base_temporal = interval_temporal_calibration_features(base, all_intervals)
        residual = target - base
        raw_residual = raw - base

        predictions = {"lars": base}
        probe_audit: dict[str, object] = {}
        if "ridge_all_lags" in args.probes:
            ridge_all, ridge_all_alpha, ridge_all_curve = select_ridge(
                standardized,
                target,
                training_intervals,
                backend=args.ridge_backend,
                device=device,
                selection_metric=args.ridge_selection_metric,
            )
            predictions["ridge_all_lags"] = np.maximum(
                ridge_all.predict(standardized), 0.0
            )
            probe_audit.update(
                {
                    "ridge_all_alpha": ridge_all_alpha,
                    "ridge_all_inner_selection_curve": ridge_all_curve,
                }
            )
        if "ridge_all_lags_raw" in args.probes:
            ridge_all_raw, ridge_all_raw_alpha, ridge_all_raw_curve = select_ridge(
                standardized,
                raw,
                training_intervals,
                backend=args.ridge_backend,
                device=device,
                selection_metric=args.ridge_selection_metric,
            )
            predictions["ridge_all_lags_raw"] = np.maximum(
                ridge_all_raw.predict(standardized), 0.0
            )
            probe_audit.update(
                {
                    "ridge_all_raw_alpha": ridge_all_raw_alpha,
                    "ridge_all_raw_inner_selection_curve": ridge_all_raw_curve,
                }
            )
        if "ridge_current_temporal" in args.probes:
            ridge_current, ridge_current_alpha, ridge_current_curve = select_ridge(
                temporal,
                residual,
                training_intervals,
                backend=args.ridge_backend,
                device=device,
                selection_metric=args.ridge_selection_metric,
            )
            predictions["ridge_current_temporal"] = np.maximum(
                base + ridge_current.predict(temporal), 0.0
            )
            probe_audit.update(
                {
                    "ridge_current_alpha": ridge_current_alpha,
                    "ridge_current_inner_selection_curve": ridge_current_curve,
                }
            )
        if "ridge_current_temporal_raw" in args.probes:
            ridge_current_raw, ridge_current_raw_alpha, ridge_current_raw_curve = (
                select_ridge(
                    temporal,
                    raw_residual,
                    training_intervals,
                    backend=args.ridge_backend,
                    device=device,
                    selection_metric=args.ridge_selection_metric,
                )
            )
            predictions["ridge_current_temporal_raw"] = np.maximum(
                base + ridge_current_raw.predict(temporal), 0.0
            )
            probe_audit.update(
                {
                    "ridge_current_raw_alpha": ridge_current_raw_alpha,
                    "ridge_current_raw_inner_selection_curve": ridge_current_raw_curve,
                }
            )
        if "ridge_base_temporal_raw" in args.probes:
            ridge_base_raw, ridge_base_raw_alpha, ridge_base_raw_curve = select_ridge(
                base_temporal,
                raw,
                training_intervals,
                backend=args.ridge_backend,
                device=device,
                selection_metric=args.ridge_selection_metric,
            )
            predictions["ridge_base_temporal_raw"] = np.maximum(
                ridge_base_raw.predict(base_temporal), 0.0
            )
            probe_audit.update(
                {
                    "ridge_base_raw_alpha": ridge_base_raw_alpha,
                    "ridge_base_raw_inner_selection_curve": ridge_base_raw_curve,
                }
            )
        if "hist_current_temporal" in args.probes:
            training = rows_from_intervals(training_intervals)
            nonlinear = make_pipeline(
                StandardScaler(),
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=200,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=2026,
                ),
            )
            nonlinear.fit(temporal[training], residual[training])
            predictions["hist_current_temporal"] = np.maximum(
                base + nonlinear.predict(temporal), 0.0
            )
        if "hist_current_temporal_raw" in args.probes:
            training = rows_from_intervals(training_intervals)
            nonlinear_raw = make_pipeline(
                StandardScaler(),
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=200,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=2026,
                ),
            )
            nonlinear_raw.fit(temporal[training], raw_residual[training])
            predictions["hist_current_temporal_raw"] = np.maximum(
                base + nonlinear_raw.predict(temporal), 0.0
            )
        if args.raw_power_calibration:
            training = rows_from_intervals(training_intervals)
            power_audit = {}
            for name, prediction in tuple(predictions.items()):
                calibrated, power, curve = select_training_power_calibration(
                    prediction, raw, training
                )
                predictions[name] = calibrated
                power_audit[name] = {
                    "selected_power": power,
                    "training_raw_pcc": curve,
                }
            probe_audit["raw_power_calibration"] = power_audit
        for name, prediction in predictions.items():
            stitched.setdefault(
                name, np.full(rows, np.nan, dtype=np.float32)
            )[validation] = prediction[validation]
        stitched_target[validation] = target[validation]
        cleaned_residual_pcc = {
            name: pearson(
                prediction[validation] - base[validation], residual[validation]
            )
            for name, prediction in predictions.items()
            if name in ("ridge_current_temporal", "hist_current_temporal")
        }
        record = {
            "outer_fold": outer_fold,
            **probe_audit,
            "raw_pcc": {
                name: pearson(prediction[validation], raw[validation])
                for name, prediction in predictions.items()
            },
            "cleaned_residual_pcc": cleaned_residual_pcc,
            "runtime_seconds": float(time.perf_counter() - outer_started),
        }
        outer_records.append(record)
        print(
            f"outer={outer_fold} runtime_seconds={record['runtime_seconds']:.3f}",
            flush=True,
        )

    assert stitched_target is not None and stitched_raw is not None
    observed = np.isfinite(stitched_target)
    report = {
        "protocol": (
            "outer-split residual predictability audit using split-local target, "
            "FastICA/CSP candidate bank and LARS; ridge alpha selected by grouped "
            "folds inside outer training; fixed nonlinear probe; released test untouched"
        ),
        "released_test_touched": False,
        "subject": args.subject,
        "finger": args.finger,
        "ridge_backend": args.ridge_backend,
        "ridge_selection_metric": args.ridge_selection_metric,
        "raw_power_calibration": args.raw_power_calibration,
        "probes": list(args.probes),
        "outer_records": outer_records,
        "stitched_raw_pcc": {
            name: pearson(prediction[observed], stitched_raw[observed])
            for name, prediction in stitched.items()
        },
        "stitched_cleaned_pcc": {
            name: pearson(prediction[observed], stitched_target[observed])
            for name, prediction in stitched.items()
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, prediction in stitched.items():
        np.save(args.output / f"{name}_oof.npy", prediction, allow_pickle=False)
    np.save(args.output / "target_oof.npy", stitched_target, allow_pickle=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
