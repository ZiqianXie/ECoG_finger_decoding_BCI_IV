#!/usr/bin/env python3
"""Deep diagnostics for Subject 3 train/test failure and visualization data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import signal

from diagnose_preprocessing_and_training import (
    FINGERS,
    fit_gpu_ridge,
    initialized_energy,
    lagged,
    pearson,
    pearson_columns,
    ridge_predict,
)


HISTORICAL = (0, 1, 2, 4)


def fit_ridge_models(
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    top_features: int,
) -> list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray, float, np.ndarray], float]]:
    inner_stop = int(round(x.shape[0] * 0.8))
    alphas = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    models = []
    for finger in range(y.shape[1]):
        correlations = pearson_columns(x[:inner_stop], y[:inner_stop, finger])
        selected = np.argpartition(np.abs(correlations), -top_features)[-top_features:]
        best_alpha = alphas[0]
        best_score = -float("inf")
        for alpha in alphas:
            fit = fit_gpu_ridge(x[:inner_stop, selected], y[:inner_stop, finger], alpha, device)
            score = pearson(ridge_predict(x[inner_stop:, selected], fit), y[inner_stop:, finger])
            if score > best_score:
                best_score = score
                best_alpha = alpha
        models.append((selected, fit_gpu_ridge(x[:, selected], y[:, finger], best_alpha, device), best_alpha))
    return models


def predict_ridge(
    x: np.ndarray,
    models: list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray, float, np.ndarray], float]],
) -> np.ndarray:
    prediction = np.zeros((x.shape[0], len(models)), dtype=np.float32)
    for finger, (selected, fit, _) in enumerate(models):
        prediction[:, finger] = ridge_predict(x[:, selected], fit)
    return prediction


def correlations(prediction: np.ndarray, target: np.ndarray) -> list[float]:
    return [pearson(prediction[:, finger], target[:, finger]) for finger in range(target.shape[1])]


def lag_sweep(prediction: np.ndarray, target: np.ndarray, maximum_lag: int = 75) -> dict[str, object]:
    rows = []
    best_by_finger = [{"r": -float("inf"), "lag_bins": 0} for _ in FINGERS]
    best_macro = {"r": -float("inf"), "lag_bins": 0}
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag < 0:
            pred = prediction[-lag:]
            truth = target[:lag]
        elif lag > 0:
            pred = prediction[:-lag]
            truth = target[lag:]
        else:
            pred = prediction
            truth = target
        values = correlations(pred, truth)
        macro = float(np.mean([values[index] for index in HISTORICAL]))
        rows.append({"lag_seconds": lag / 25.0, "historical_four": macro, "fingers": values})
        if macro > best_macro["r"]:
            best_macro = {"r": macro, "lag_bins": lag}
        for finger, value in enumerate(values):
            if value > best_by_finger[finger]["r"]:
                best_by_finger[finger] = {"r": value, "lag_bins": lag}
    return {
        "rows": rows,
        "zero_lag": rows[maximum_lag],
        "best_historical_four": {
            "r": best_macro["r"],
            "lag_seconds": best_macro["lag_bins"] / 25.0,
        },
        "best_by_finger": [
            {"finger": FINGERS[i], "r": value["r"], "lag_seconds": value["lag_bins"] / 25.0}
            for i, value in enumerate(best_by_finger)
        ],
    }


def segment_stats(values: np.ndarray) -> dict[str, object]:
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q05": np.quantile(values, 0.05, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q95": np.quantile(values, 0.95, axis=0).tolist(),
        "active_fraction": np.mean(values > 0.08, axis=0).tolist(),
    }


def spectrum(values: np.ndarray, sampling_rate: float = 1000.0) -> dict[str, object]:
    frequency, power = signal.welch(values, fs=sampling_rate, nperseg=4096, axis=0)
    mask = frequency <= 250.0
    frequency = frequency[mask]
    median = np.median(power[mask], axis=1)
    keep = np.arange(0, frequency.size, 4)
    return {
        "frequency_hz": frequency[keep].tolist(),
        "median_channel_log10_power": np.log10(np.maximum(median[keep], 1e-14)).tolist(),
    }


def transfer_matrix(
    segments: dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    top_features: int = 256,
) -> dict[str, object]:
    names = list(segments)
    matrix = []
    per_finger: dict[str, list[list[float]]] = {name: [] for name in FINGERS}
    alphas: dict[str, list[float]] = {}
    for source in names:
        source_x, source_y = segments[source]
        models = fit_ridge_models(source_x, source_y, device, top_features)
        alphas[source] = [float(model[2]) for model in models]
        macro_row = []
        finger_rows = [[] for _ in FINGERS]
        for destination in names:
            destination_x, destination_y = segments[destination]
            prediction = predict_ridge(destination_x, models)
            values = correlations(prediction, destination_y)
            macro_row.append(float(np.mean([values[index] for index in HISTORICAL])))
            for finger, value in enumerate(values):
                finger_rows[finger].append(value)
        matrix.append(macro_row)
        for finger, name in enumerate(FINGERS):
            per_finger[name].append(finger_rows[finger])
    return {"segments": names, "historical_four": matrix, "per_finger": per_finger, "alphas": alphas}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed"))
    parser.add_argument("--training-root", type=Path, default=Path("outputs/training"))
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/subject3_shift.json"))
    parser.add_argument("--device", default="cuda:7")
    args = parser.parse_args()
    subject = 3
    config = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)
    root = args.prepared_root / "sub3"
    metadata = json.loads((root / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    history = int(config["decoder"]["cnn_history_bins"])
    offset = history - 1

    train_ecog = np.load(root / "train_ecog.npy", mmap_mode="r")
    test_ecog = np.load(root / "test_ecog.npy", mmap_mode="r")
    train_raw = np.load(root / "train_glove_25hz_raw.npy")
    test_raw = np.load(root / "test_glove_25hz_raw.npy")
    train_target = np.load(root / "train_glove_paper_baseline_only.npy")
    test_target = np.load(root / "test_glove_paper_baseline_only.npy")
    ica = np.load(args.training_root / "sub3" / "fastica_unmixing.npy")

    print("extracting initialized band energies", flush=True)
    train_energy = initialized_energy(train_ecog, ica, config, device)
    test_energy = initialized_energy(test_ecog, ica, config, device)
    train_x_all = lagged(train_energy, history)
    test_x = lagged(test_energy, history)
    train_count = split - offset
    train_x = train_x_all[:train_count]
    validation_x = train_x_all[train_count:]
    train_y = train_target[offset:split]
    validation_y = train_target[split:]
    test_y = test_target[offset:]

    print("fitting main diagnostic ridge", flush=True)
    ridge_models = fit_ridge_models(train_x, train_y, device, 512)
    validation_prediction = predict_ridge(validation_x, ridge_models)
    test_prediction = predict_ridge(test_x, ridge_models)
    ssm_prediction = np.load(args.training_root / "sub3" / "test_prediction.npy")

    print("building cross-period transfer matrix", flush=True)
    boundary = train_count // 2
    segments = {
        "early train": (train_x[:boundary], train_y[:boundary]),
        "late train": (train_x[boundary:], train_y[boundary:]),
        "validation": (validation_x, validation_y),
        "test": (test_x, test_y),
    }
    transfer = transfer_matrix(segments, device)

    print("measuring spectral and feature shift", flush=True)
    energy_segments = {
        "train": train_energy[offset:split],
        "validation": train_energy[split:],
        "test": test_energy[offset:],
    }
    train_mean = energy_segments["train"].mean(axis=0)
    train_std = energy_segments["train"].std(axis=0)
    train_std = np.maximum(train_std, 1e-6)
    test_mean_z = (energy_segments["test"].mean(axis=0) - train_mean) / train_std
    validation_mean_z = (energy_segments["validation"].mean(axis=0) - train_mean) / train_std
    component_count = ica.shape[0]
    band_count = 8

    raw_ranges = {
        "train": np.asarray(train_ecog[: split * 40]),
        "validation": np.asarray(train_ecog[split * 40:]),
        "test": np.asarray(test_ecog),
    }
    channel_shift = {}
    train_channel_mean = raw_ranges["train"].mean(axis=0)
    train_channel_std = np.maximum(raw_ranges["train"].std(axis=0), 1e-6)
    for name, values in raw_ranges.items():
        channel_shift[name] = {
            "mean_z": ((values.mean(axis=0) - train_channel_mean) / train_channel_std).tolist(),
            "std_ratio": (values.std(axis=0) / train_channel_std).tolist(),
        }

    window = 750
    activity = np.sum(test_y, axis=1)
    rolling = np.convolve(activity, np.ones(window), mode="valid")
    window_start = int(np.argmax(rolling))
    window_stop = window_start + window
    raw_aligned = test_raw[offset:]
    timeline = {
        "time_seconds": (np.arange(window) / 25.0).tolist(),
        "start_seconds_in_test": (window_start + offset) / 25.0,
        "raw": raw_aligned[window_start:window_stop].round(5).tolist(),
        "paper_baseline": test_y[window_start:window_stop].round(5).tolist(),
        "ridge": test_prediction[window_start:window_stop].round(5).tolist(),
        "ssm": ssm_prediction[window_start:window_stop].round(5).tolist(),
    }

    report = {
        "subject": 3,
        "finger_names": FINGERS,
        "split_seconds": {"model_train": split / 25.0, "validation": (train_target.shape[0] - split) / 25.0, "test": test_target.shape[0] / 25.0},
        "ridge_validation_lag_sweep": lag_sweep(validation_prediction, train_raw[split:]),
        "ridge_test_lag_sweep": lag_sweep(test_prediction, test_raw[offset:]),
        "ssm_test_lag_sweep": lag_sweep(ssm_prediction, test_raw[offset:]),
        "ridge_zero_lag_by_finger": dict(zip(FINGERS, correlations(test_prediction, test_raw[offset:]))),
        "ssm_zero_lag_by_finger": dict(zip(FINGERS, correlations(ssm_prediction, test_raw[offset:]))),
        "transfer": transfer,
        "targets": {
            "train_paper_baseline": segment_stats(train_target[offset:split]),
            "validation_paper_baseline": segment_stats(train_target[split:]),
            "test_paper_baseline": segment_stats(test_target[offset:]),
            "train_raw": segment_stats(train_raw[offset:split]),
            "validation_raw": segment_stats(train_raw[split:]),
            "test_raw": segment_stats(test_raw[offset:]),
        },
        "energy_shift": {
            "shape": [component_count, band_count],
            "band_names": ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"],
            "validation_mean_z": validation_mean_z.reshape(component_count, band_count).round(4).tolist(),
            "test_mean_z": test_mean_z.reshape(component_count, band_count).round(4).tolist(),
            "median_abs_validation_z": float(np.median(np.abs(validation_mean_z))),
            "median_abs_test_z": float(np.median(np.abs(test_mean_z))),
            "fraction_test_abs_z_gt_1": float(np.mean(np.abs(test_mean_z) > 1.0)),
        },
        "channel_shift": channel_shift,
        "spectra": {name: spectrum(values) for name, values in raw_ranges.items()},
        "timeline": timeline,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps({
        "ridge_test_best_lag": report["ridge_test_lag_sweep"]["best_historical_four"],
        "ssm_test_best_lag": report["ssm_test_lag_sweep"]["best_historical_four"],
        "median_abs_test_energy_z": report["energy_shift"]["median_abs_test_z"],
        "fraction_test_energy_abs_z_gt_1": report["energy_shift"]["fraction_test_abs_z_gt_1"],
        "transfer_historical_four": transfer["historical_four"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
