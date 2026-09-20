#!/usr/bin/env python3
"""Benchmark CPU LARS and GPU FISTA on one saved split-local candidate matrix."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LassoLarsCV

from cross_validate_single_wavelet import grouped_lars_subfolds
from gpu_lasso import fit_torch_lasso_cv
from train_event_grouped_lars_lstm import indices_from_intervals


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) < 1.0e-10 or np.std(right) < 1.0e-10:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def metrics(
    coefficients: np.ndarray,
    intercept: float,
    features: np.ndarray,
    target: np.ndarray,
    validation: np.ndarray,
) -> dict[str, float | int]:
    prediction = features @ coefficients + intercept
    return {
        "selected_features": int(np.count_nonzero(np.abs(coefficients) > 1.0e-7)),
        "validation_pcc": pearson(prediction[validation], target[validation]),
        "validation_mse": float(
            np.mean((prediction[validation] - target[validation]) ** 2)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--finger-index", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    saved = np.load(args.cache / "initialization.npz")
    split = json.loads((args.cache / "split.json").read_text())
    target = np.load(args.cache / "target.npy")[:, args.finger_index]
    training = indices_from_intervals(split["training_intervals"])
    validation = indices_from_intervals(split["validation_intervals"])
    features = (
        saved["candidate_features"][training] - saved["candidate_feature_mean"]
    ) / saved["candidate_feature_scale"]
    all_features = (
        saved["candidate_features"] - saved["candidate_feature_mean"]
    ) / saved["candidate_feature_scale"]
    train_target = target[training]
    folds = grouped_lars_subfolds(training, split["training_groups"], target.size)

    cpu_started = time.perf_counter()
    cpu = LassoLarsCV(cv=folds, max_iter=500, n_jobs=1).fit(features, train_target)
    cpu_seconds = time.perf_counter() - cpu_started

    device = torch.device(args.device)
    if device.type == "cuda":
        _ = torch.ones(1, device=device).square()
        torch.cuda.synchronize(device)
    gpu_started = time.perf_counter()
    gpu = fit_torch_lasso_cv(features, train_target, folds, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gpu_seconds = time.perf_counter() - gpu_started

    cpu_metrics = metrics(
        np.asarray(cpu.coef_), float(cpu.intercept_), all_features, target, validation
    )
    gpu_metrics = metrics(gpu.coef_, gpu.intercept_, all_features, target, validation)
    cpu_prediction = all_features @ np.asarray(cpu.coef_) + float(cpu.intercept_)
    gpu_prediction = all_features @ gpu.coef_ + gpu.intercept_
    report = {
        "cache": str(args.cache),
        "rows": int(features.shape[0]),
        "features": int(features.shape[1]),
        "cpu_lars": {
            "elapsed_seconds": float(cpu_seconds),
            "alpha": float(cpu.alpha_),
            **cpu_metrics,
        },
        "gpu_fista": {
            "elapsed_seconds": float(gpu_seconds),
            "alpha": float(gpu.alpha_),
            "iterations": int(gpu.iterations_),
            **gpu_metrics,
        },
        "gpu_speedup": float(cpu_seconds / gpu_seconds),
        "validation_prediction_pcc_between_backends": pearson(
            cpu_prediction[validation], gpu_prediction[validation]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
