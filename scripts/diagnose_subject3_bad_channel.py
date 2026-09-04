#!/usr/bin/env python3
"""Test whether Subject 3's test-only channel-50 artifact causes decoding failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from diagnose_preprocessing_and_training import initialized_energy, lagged, make_model
from diagnose_subject3_shift import (
    HISTORICAL,
    correlations,
    fit_ridge_models,
    lag_sweep,
    predict_ridge,
)


def rms_per_second(values: np.ndarray) -> np.ndarray:
    usable = values[: values.size // 1000 * 1000]
    return np.sqrt(np.mean(usable.reshape(-1, 1000).astype(np.float64) ** 2, axis=1))


@torch.inference_mode()
def ssm_predict(
    ecog: np.ndarray,
    config: dict[str, object],
    checkpoint_path: Path,
    device: torch.device,
) -> np.ndarray:
    model = make_model(config, ecog.shape[1], device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    x = torch.from_numpy(np.asarray(ecog).T.copy()).unsqueeze(0).to(device)
    return model(x).squeeze(0).float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed"))
    parser.add_argument("--training-root", type=Path, default=Path("outputs/training"))
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/subject3_bad_channel.json"))
    parser.add_argument("--device", default="cuda:7")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)
    root = args.prepared_root / "sub3"
    metadata = json.loads((root / "metadata.json").read_text())
    retained = list(metadata["retained_channels_one_based"])
    suspect_original_channel = 50
    suspect = retained.index(suspect_original_channel)
    split = int(metadata["target_fit_samples_25hz"])
    history = int(config["decoder"]["cnn_history_bins"])
    offset = history - 1

    train_ecog = np.load(root / "train_ecog.npy", mmap_mode="r")
    test_ecog = np.asarray(np.load(root / "test_ecog.npy", mmap_mode="r"))
    train_target = np.load(root / "train_glove_paper_baseline_only.npy")
    test_target = np.load(root / "test_glove_paper_baseline_only.npy")
    test_raw = np.load(root / "test_glove_25hz_raw.npy")[offset:]
    ica = np.load(args.training_root / "sub3" / "fastica_unmixing.npy")

    print("fitting ridge from clean training period", flush=True)
    train_energy = initialized_energy(train_ecog, ica, config, device)
    train_x_all = lagged(train_energy, history)
    train_count = split - offset
    train_x = train_x_all[:train_count]
    train_y = train_target[offset:split]
    models = fit_ridge_models(train_x, train_y, device, 512)
    train_energy_mean = train_energy[offset:split].mean(axis=0)
    train_energy_std = np.maximum(train_energy[offset:split].std(axis=0), 1e-6)

    variants = {
        "original": test_ecog,
        "zero_channel_50": test_ecog.copy(),
        "clip_all_at_8_sd": np.clip(test_ecog, -8.0, 8.0),
        "zero_channel_50_and_clip": np.clip(test_ecog, -8.0, 8.0),
    }
    variants["zero_channel_50"][:, suspect] = 0.0
    variants["zero_channel_50_and_clip"][:, suspect] = 0.0
    window = 750
    activity = np.sum(test_target[offset:], axis=1)
    window_start = int(np.argmax(np.convolve(activity, np.ones(window), mode="valid")))
    window_stop = window_start + window
    results = {}
    timeline_predictions = {}
    for name, values in variants.items():
        print(f"evaluating {name}", flush=True)
        energy = initialized_energy(values, ica, config, device)
        x = lagged(energy, history)
        ridge_prediction = predict_ridge(x, models)
        ridge_values = correlations(ridge_prediction, test_raw)
        shift = (energy[offset:].mean(axis=0) - train_energy_mean) / train_energy_std
        ssm_prediction = ssm_predict(
            values, config, args.training_root / "sub3" / "best_model.pt", device
        )
        ssm_values = correlations(ssm_prediction, test_raw)
        results[name] = {
            "ridge_historical_four": float(np.mean([ridge_values[i] for i in HISTORICAL])),
            "ridge_by_finger": ridge_values,
            "ridge_best_lag": lag_sweep(ridge_prediction, test_raw)["best_historical_four"],
            "ssm_historical_four": float(np.mean([ssm_values[i] for i in HISTORICAL])),
            "ssm_by_finger": ssm_values,
            "ssm_best_lag": lag_sweep(ssm_prediction, test_raw)["best_historical_four"],
            "median_abs_energy_mean_z": float(np.median(np.abs(shift))),
            "fraction_energy_abs_z_gt_1": float(np.mean(np.abs(shift) > 1.0)),
        }
        if name in {"original", "zero_channel_50"}:
            timeline_predictions[name] = {
                "ridge": ridge_prediction[window_start:window_stop].round(5).tolist(),
                "ssm": ssm_prediction[window_start:window_stop].round(5).tolist(),
            }

    train_channel = np.asarray(train_ecog[:, suspect])
    test_channel = test_ecog[:, suspect]
    report = {
        "subject": 3,
        "suspect_original_channel": suspect_original_channel,
        "suspect_retained_index": suspect,
        "train_std": float(train_channel.std()),
        "test_std": float(test_channel.std()),
        "test_to_train_std_ratio": float(test_channel.std() / train_channel.std()),
        "train_abs_max": float(np.max(np.abs(train_channel))),
        "test_abs_max": float(np.max(np.abs(test_channel))),
        "train_rms_per_second": rms_per_second(train_channel).round(5).tolist(),
        "test_rms_per_second": rms_per_second(test_channel).round(5).tolist(),
        "variants": results,
        "timeline": {
            "time_seconds": (np.arange(window) / 25.0).tolist(),
            "start_seconds_in_test": (window_start + offset) / 25.0,
            "raw": test_raw[window_start:window_stop].round(5).tolist(),
            "paper_baseline": test_target[offset + window_start:offset + window_stop].round(5).tolist(),
            "predictions": timeline_predictions,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps(report["variants"], indent=2), flush=True)


if __name__ == "__main__":
    main()
