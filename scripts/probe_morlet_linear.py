#!/usr/bin/env python3
"""Training-only linear probes for continuous Morlet dictionary features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ecog_decoding.spectrotemporal import MorletAtomBank
from ecog_decoding.training import FINGER_NAMES


TARGETS = {1: "local_w2_q10", 2: "local_w1_q10", 3: "local_w4_q20"}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    y = np.asarray(y, dtype=np.float64) - np.mean(y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 0 else 0.0


def causal_mean(x: torch.Tensor, width: int) -> torch.Tensor:
    return F.avg_pool1d(
        F.pad(x.T[None], (width - 1, 0)), kernel_size=width, stride=1
    )[0].T


def correlation_prescreen(
    x: torch.Tensor, y: torch.Tensor, maximum_features: int
) -> torch.Tensor:
    if x.shape[1] <= maximum_features:
        return torch.arange(x.shape[1], device=x.device)
    centered_x = x - x.mean(dim=0)
    centered_y = y - y.mean()
    denominator = torch.linalg.vector_norm(centered_x, dim=0) * torch.linalg.vector_norm(
        centered_y
    )
    correlation = (centered_x.T @ centered_y).abs() / denominator.clamp_min(1e-8)
    return torch.topk(correlation, maximum_features).indices


def ridge_path_predict(
    fit_x: torch.Tensor,
    fit_y: torch.Tensor,
    validation_x: torch.Tensor,
    alphas: tuple[float, ...],
) -> dict[float, torch.Tensor]:
    mean = fit_x.mean(dim=0)
    scale = fit_x.std(dim=0, correction=0).clamp_min(1.0e-5)
    train = (fit_x - mean) / scale
    validation = (validation_x - mean) / scale
    y_mean = fit_y.mean()
    centered_y = fit_y - y_mean
    gram = train @ train.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    projected_target = eigenvectors.T @ centered_y
    cross = validation @ train.T @ eigenvectors
    return {
        alpha: cross @ (projected_target / (eigenvalues + alpha)) + y_mean
        for alpha in alphas
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--finger", choices=tuple(FINGER_NAMES), default="index")
    parser.add_argument("--fold", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/morlet_cache_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/morlet_linear_probe_v1"))
    parser.add_argument(
        "--reference-feature-root",
        type=Path,
        default=Path("outputs/windowed_ica_wavelet_asymmetric_v1"),
    )
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--spatial-components", type=int, default=32)
    parser.add_argument("--atoms", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=1001)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    prepared = args.prepared_root / f"sub{args.subject}"
    metadata = json.loads((prepared / "metadata.json").read_text())
    rows = int(metadata["target_fit_samples_25hz"]) - (args.history - 1)
    boundaries = np.rint(np.linspace(0.5 * rows, rows, 4)).astype(int)
    fit_end, validation_stop = map(int, boundaries[args.fold : args.fold + 2])
    offset = args.history - 1
    fit_raw_stop = (fit_end + offset) * 40
    validation_raw_stop = (validation_stop + offset) * 40
    cache = args.cache_root / (
        f"sub{args.subject}_fold{args.fold}_fit{fit_end}_"
        f"c{args.spatial_components}_seed{args.seed}.npz"
    )
    values = np.load(cache)
    weights = torch.from_numpy(values["weights"]).to(device)
    mean = torch.from_numpy(values["mean"]).to(device)
    ecog = np.load(prepared / "train_ecog.npy", mmap_mode="r")
    recording = torch.from_numpy(np.asarray(ecog[:validation_raw_stop]).copy()).to(device)
    spatial = weights @ (recording - mean).T
    spatial /= spatial[:, :fit_raw_stop].std(dim=1, correction=0).clamp_min(1e-6)[:, None]

    bank = MorletAtomBank(
        atom_count=args.atoms,
        kernel_size=args.kernel_size,
        trainable=False,
    ).to(device)
    with torch.inference_mode():
        representation, broadband = bank(spatial[None])
    representation = representation[0]
    broadband_bins = broadband[0].T
    energy_bins = representation[:, 1].permute(2, 0, 1).flatten(start_dim=1)
    signed_bins = representation[:, 0].permute(2, 0, 1).flatten(start_dim=1)
    cosine_bins = representation[:, 2].permute(2, 0, 1).flatten(start_dim=1)
    sine_bins = representation[:, 3].permute(2, 0, 1).flatten(start_dim=1)
    energy_window = energy_bins.unfold(0, args.history, 1)[:validation_stop].flatten(1)
    broadband_window = broadband_bins.unfold(0, args.history, 1)[:validation_stop].flatten(1)
    energy = energy_bins[offset : offset + validation_stop]
    signed = signed_bins[offset : offset + validation_stop]
    cosine = cosine_bins[offset : offset + validation_stop]
    sine = sine_bins[offset : offset + validation_stop]
    broadband = broadband_bins[offset : offset + validation_stop]
    feature_sets = {
        "energy_current": torch.cat((energy, broadband), dim=1),
        "energy_1s": torch.cat((causal_mean(energy, 25), causal_mean(broadband, 25)), dim=1),
        "energy_multiscale": torch.cat(
            (
                energy,
                causal_mean(energy, 5),
                causal_mean(energy, 25),
                broadband,
                causal_mean(broadband, 25),
            ),
            dim=1,
        ),
        "energy_exact_1s_window": torch.cat((energy_window, broadband_window), dim=1),
        "all_current": torch.cat((signed, energy, cosine, sine, broadband), dim=1),
    }
    reference_path = (
        args.reference_feature_root
        / f"sub{args.subject}"
        / "train_initialized_window_features.npy"
    )
    if reference_path.exists():
        reference = np.load(reference_path, mmap_mode="r")
        feature_sets["reference_asymmetric_wavelet"] = torch.from_numpy(
            np.asarray(reference[:validation_stop]).copy()
        ).to(device)
    finger = list(FINGER_NAMES).index(args.finger)
    cleaned = torch.from_numpy(
        np.asarray(
            np.load(prepared / f"train_glove_{TARGETS[args.subject]}.npy")[
                offset : offset + validation_stop, finger
            ],
            dtype=np.float32,
        )
    ).to(device)
    raw = np.asarray(
        np.load(prepared / "train_glove_25hz_raw.npy")[
            offset : offset + validation_stop, finger
        ],
        dtype=np.float32,
    )
    tune_start = int(round(0.8 * fit_end))
    alphas = (1.0, 10.0, 100.0, 1000.0, 10000.0)
    report: dict[str, object] = {
        "protocol": "alpha selected on tail of inner training; reported on untouched inner validation",
        "released_test_touched": False,
        "final_chronological_validation_touched": False,
        "fit_end_index": fit_end,
        "validation_end_index": validation_stop,
        "feature_sets": {},
    }
    for name, features in feature_sets.items():
        input_dimension = int(features.shape[1])
        tuning_selected = correlation_prescreen(
            features[:tune_start], cleaned[:tune_start], 4096
        )
        tuning_features = features[:, tuning_selected]
        tuning: dict[str, float] = {}
        tuning_predictions = ridge_path_predict(
            tuning_features[:tune_start],
            cleaned[:tune_start],
            tuning_features[tune_start:fit_end],
            alphas,
        )
        for alpha, prediction in tuning_predictions.items():
            tuning[str(alpha)] = pearson(
                prediction.float().cpu().numpy(), raw[tune_start:fit_end]
            )
        selected_alpha = max(alphas, key=lambda value: tuning[str(value)])
        final_selected = correlation_prescreen(features[:fit_end], cleaned[:fit_end], 4096)
        final_features = features[:, final_selected]
        prediction = ridge_path_predict(
            final_features[:fit_end],
            cleaned[:fit_end],
            final_features[fit_end:validation_stop],
            (selected_alpha,),
        )[selected_alpha].float().cpu().numpy()
        score = pearson(prediction, raw[fit_end:validation_stop])
        report["feature_sets"][name] = {
            "dimension": input_dimension,
            "fitted_dimension": int(final_selected.numel()),
            "selected_alpha": selected_alpha,
            "inner_tuning_raw_pcc": tuning,
            "inner_validation_raw_pcc": score,
        }
        print(name, json.dumps(report["feature_sets"][name]), flush=True)
        del features
        torch.cuda.empty_cache()

    output = (
        args.output_root
        / f"sub{args.subject}"
        / args.finger
        / f"fold{args.fold}"
        / f"seed{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
