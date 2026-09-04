#!/usr/bin/env python3
"""Benchmark baseline targets with a fixed ICA/wavelet/ridge decoder.

All feature extraction is shared across candidates.  Candidate selection uses
only the chronological validation segment; released test labels are evaluated
after ranking and are never part of the selection score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ecog_decoding.models import WaveletPacketEnergy, fit_fastica_spatial_weights
from ecog_decoding.training import trajectory_metrics


FINGERS = ("thumb", "index", "middle", "ring", "little")


@torch.inference_mode()
def initialized_energy(
    ecog: np.ndarray,
    ica: np.ndarray,
    config: dict[str, object],
    device: torch.device,
) -> np.ndarray:
    front = config["wavelet_packet_frontend"]
    module = WaveletPacketEnergy(
        wavelet=str(front["wavelet"]),
        levels=int(front["levels"]),
        kernel_size=int(front["kernel_size"]),
        trainable=False,
        padding_mode=str(front["padding_mode"]),
        energy_window_samples=int(front["energy_window_samples"]),
        energy_stride_samples=int(front["energy_stride_samples"]),
    ).to(device).eval()
    values = torch.from_numpy(np.asarray(ecog).T.copy()).unsqueeze(0).to(device)
    spatial = torch.nn.functional.conv1d(
        values,
        torch.as_tensor(ica, dtype=values.dtype, device=device)[:, :, None],
    )
    energy = module(spatial).squeeze(0).permute(2, 0, 1).flatten(1)
    return energy.float().cpu().numpy()


def lagged(values: np.ndarray, history: int) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(values, history, axis=0)
    return np.ascontiguousarray(windows.transpose(0, 2, 1).reshape(windows.shape[0], -1))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
    return float(xc @ yc / denominator) if denominator > 0 else 0.0


def correlation_screen(x: np.ndarray, y: np.ndarray, count: int) -> np.ndarray:
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    denominator = np.sqrt(np.sum(xc * xc, axis=0) * np.sum(yc * yc))
    correlations = np.divide(
        xc.T @ yc,
        denominator,
        out=np.zeros(x.shape[1], dtype=np.float32),
        where=denominator > 0,
    )
    count = min(int(count), correlations.size)
    selected = np.argpartition(np.abs(correlations), -count)[-count:]
    return selected[np.argsort(np.abs(correlations[selected]))[::-1]]


def ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    xt = torch.from_numpy((x - mean) / scale).to(device)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    y_mean = float(yt.mean())
    yt = yt - y_mean
    gram = xt.T @ xt / xt.shape[0]
    rhs = xt.T @ yt / xt.shape[0]
    gram.diagonal().add_(alpha)
    weight = torch.linalg.solve(gram, rhs)
    return mean, scale, y_mean, weight.cpu().numpy()


def ridge_predict(
    x: np.ndarray,
    fit: tuple[np.ndarray, np.ndarray, float, np.ndarray],
) -> np.ndarray:
    mean, scale, y_mean, weight = fit
    return ((x - mean) / scale) @ weight + y_mean


def fit_candidate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_target: np.ndarray,
    validation_raw: np.ndarray,
    test_x: np.ndarray,
    test_raw: np.ndarray,
    top_features: int,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    inner_stop = int(round(0.8 * train_x.shape[0]))
    alphas = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    validation_prediction = np.zeros_like(validation_target, dtype=np.float32)
    test_prediction = np.zeros_like(test_raw, dtype=np.float32)
    per_finger: dict[str, object] = {}
    for finger, name in enumerate(FINGERS):
        selected = correlation_screen(
            train_x[:inner_stop], train_y[:inner_stop, finger], top_features
        )
        scores: dict[str, float] = {}
        best_alpha = alphas[0]
        best_score = -float("inf")
        for alpha in alphas:
            fit = ridge_fit(
                train_x[:inner_stop, selected],
                train_y[:inner_stop, finger],
                alpha,
                device,
            )
            score = pearson(
                ridge_predict(train_x[inner_stop:, selected], fit),
                train_y[inner_stop:, finger],
            )
            scores[str(alpha)] = score
            if score > best_score:
                best_score = score
                best_alpha = alpha
        fit = ridge_fit(train_x[:, selected], train_y[:, finger], best_alpha, device)
        validation_prediction[:, finger] = ridge_predict(validation_x[:, selected], fit)
        test_prediction[:, finger] = ridge_predict(test_x[:, selected], fit)
        per_finger[name] = {
            "best_alpha": best_alpha,
            "inner_validation_r": best_score,
            "alpha_scores": scores,
            "selected_feature_indices": selected.tolist(),
        }
    return (
        {
            "per_finger": per_finger,
            "validation_target_metrics": trajectory_metrics(validation_prediction, validation_target),
            "validation_raw_metrics": trajectory_metrics(validation_prediction, validation_raw),
            "test_raw_metrics": trajectory_metrics(test_prediction, test_raw),
        },
        validation_prediction,
        test_prediction,
    )


def candidate_names(root: Path) -> list[str]:
    names = ["paper_baseline_only"]
    names.extend(
        sorted(
            path.stem.removeprefix("train_glove_")
            for path in root.glob("train_glove_local_w*_q*.npy")
            if not path.stem.endswith("_baseline")
        )
    )
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ridge_target_benchmark_v2"))
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--targets", nargs="+")
    args = parser.parse_args()

    root = args.prepared_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    metadata = json.loads((root / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    history = int(config["decoder"]["cnn_history_bins"])
    offset = history - 1
    device = torch.device(args.device)

    train_ecog = np.load(root / "train_ecog.npy", mmap_mode="r")
    test_ecog = np.load(root / "test_ecog.npy", mmap_mode="r")
    raw_train = np.load(root / "train_glove_25hz_raw.npy")
    raw_test = np.load(root / "test_glove_25hz_raw.npy")
    ica_path = output / "fastica_unmixing.npy"
    if ica_path.exists():
        ica = np.load(ica_path)
    else:
        ica = fit_fastica_spatial_weights(
            np.asarray(train_ecog[: split * 40]),
            max_samples=int(config["decoder"]["fastica_max_samples"]),
            random_state=int(config["decoder"]["fastica_random_state"]),
            backend=str(config["decoder"]["fastica_backend"]),
            device=device,
        )
        np.save(ica_path, ica, allow_pickle=False)

    train_energy_path = output / "train_initialized_energy.npy"
    test_energy_path = output / "test_initialized_energy.npy"
    if train_energy_path.exists() and test_energy_path.exists():
        train_energy = np.load(train_energy_path)
        test_energy = np.load(test_energy_path)
    else:
        train_energy = initialized_energy(train_ecog, ica, config, device)
        test_energy = initialized_energy(test_ecog, ica, config, device)
        np.save(train_energy_path, train_energy, allow_pickle=False)
        np.save(test_energy_path, test_energy, allow_pickle=False)

    train_x_all = lagged(train_energy, history)
    test_x = lagged(test_energy, history)
    train_count = split - offset
    train_x = train_x_all[:train_count]
    validation_x = train_x_all[train_count:]
    validation_raw = raw_train[split:]
    test_raw = raw_test[offset:]

    summary_path = output / "summary.json"
    results: dict[str, object] = {}
    if summary_path.exists():
        results.update(json.loads(summary_path.read_text()).get("results", {}))
    names = args.targets if args.targets else candidate_names(root)
    for name in names:
        train_target = np.load(root / f"train_glove_{name}.npy")
        validation_target = train_target[split:]
        result, validation_prediction, test_prediction = fit_candidate(
            train_x,
            train_target[offset:split],
            validation_x,
            validation_target,
            validation_raw,
            test_x,
            test_raw,
            args.top_features,
            device,
        )
        results[name] = result
        candidate_output = output / name
        candidate_output.mkdir(exist_ok=True)
        np.save(candidate_output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
        np.save(candidate_output / "test_prediction.npy", test_prediction, allow_pickle=False)
        (candidate_output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        score = result["validation_raw_metrics"]["pearson_historical_four"]
        print(f"target={name} validation_raw_r4={score:.4f}", flush=True)

    ranking = sorted(
        results,
        key=lambda name: results[name]["validation_raw_metrics"]["pearson_historical_four"],
        reverse=True,
    )
    report = {
        "subject": args.subject,
        "selection_metric": "validation_raw_metrics.pearson_historical_four",
        "test_labels_used_for_selection": False,
        "top_features_per_finger": args.top_features,
        "candidate_ranking": ranking,
        "results": results,
    }
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"best_validation_target={ranking[0]}", flush=True)


if __name__ == "__main__":
    main()
