#!/usr/bin/env python3
"""Permanent synthetic recovery suite for the learnable wavelet filter bank."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage, signal
from torch import nn

from ecog_decoding.models import WaveletPacketEnergy
from ecog_decoding.spectral_audit import (
    normalized_response_similarity,
    summarize_responses,
    wavelet_path_responses,
)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 0 else 0.0


def movement_control(bins: int, rng: np.random.Generator) -> np.ndarray:
    control = np.zeros(bins, dtype=np.float32)
    for _ in range(max(8, bins // 80)):
        width = int(rng.integers(8, 35))
        start = int(rng.integers(0, max(1, bins - width)))
        pulse = np.sin(np.linspace(0.0, np.pi, width, dtype=np.float32)) ** 2
        control[start : start + width] = np.maximum(control[start : start + width], pulse)
    return ndimage.gaussian_filter1d(control, sigma=1.0).astype(np.float32)


def band_limited_noise(
    samples: int,
    low: float,
    high: float,
    sampling_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    sos = signal.butter(4, (low, high), btype="bandpass", fs=sampling_rate, output="sos")
    return signal.sosfiltfilt(sos, rng.standard_normal(samples)).astype(np.float32)


def make_scenario(
    scenario: str,
    bins: int,
    sampling_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    rng = np.random.default_rng(seed)
    samples_per_bin = int(round(sampling_rate / 25.0))
    samples = bins * samples_per_bin
    target = movement_control(bins, rng)
    envelope = np.repeat(0.15 + target, samples_per_bin)
    time = np.arange(samples) / sampling_rate
    target_bands = [[78.0, 88.0]]
    source = envelope * np.sin(2.0 * np.pi * 83.0 * time + rng.uniform(0, 2 * np.pi))

    if scenario == "nearby_bands":
        distractor_control = np.clip(
            0.65 * target + 0.35 * movement_control(bins, rng), 0.0, 1.0
        )
        distractor = np.repeat(0.15 + distractor_control, samples_per_bin)
        source += distractor * np.sin(2.0 * np.pi * 96.0 * time + rng.uniform(0, 2 * np.pi))
    elif scenario == "broadband":
        source = envelope * band_limited_noise(samples, 65.0, 105.0, sampling_rate, rng)
        target_bands = [[65.0, 105.0]]
    elif scenario == "line_noise_drift":
        source += 1.2 * np.sin(2.0 * np.pi * 60.0 * time)
        source += 0.8 * np.sin(2.0 * np.pi * 120.0 * time)
        source += 0.8 * np.sin(2.0 * np.pi * 0.35 * time)
    elif scenario not in {"single_band", "spatial_mix", "correlated_fingers"}:
        raise ValueError(f"unknown scenario {scenario}")

    distractor_control = movement_control(bins, rng)
    if scenario == "correlated_fingers":
        distractor_control = np.clip(0.75 * target + 0.25 * distractor_control, 0.0, 1.0)
    distractor = np.repeat(0.15 + distractor_control, samples_per_bin)
    nuisance = distractor * np.sin(2.0 * np.pi * 145.0 * time + rng.uniform(0, 2 * np.pi))
    sources = np.column_stack(
        (
            source,
            nuisance,
            0.35 * rng.standard_normal(samples),
            0.20 * np.sin(2.0 * np.pi * 0.7 * time),
        )
    )
    if scenario == "spatial_mix":
        mixing = rng.normal(size=(4, 6))
    else:
        mixing = np.eye(4)
    values = sources @ mixing
    values += 0.20 * rng.standard_normal(values.shape)
    fit_stop = int(round(2.0 * samples / 3.0))
    mean = values[:fit_stop].mean(axis=0)
    scale = values[:fit_stop].std(axis=0)
    values = ((values - mean) / np.maximum(scale, 1.0e-6)).astype(np.float32)
    return values, target, target_bands


class RecoveryModel(nn.Module):
    def __init__(self, channels: int, components: int, levels: int = 3) -> None:
        super().__init__()
        self.spatial = nn.Conv1d(channels, components, 1, bias=False)
        nn.init.orthogonal_(self.spatial.weight)
        self.frontend = WaveletPacketEnergy(
            wavelet="bior6.8",
            levels=levels,
            kernel_size=17,
            trainable=True,
            padding_mode="reflect",
            energy_window_samples=40,
            energy_stride_samples=40,
        )
        path_count = 2**levels
        self.gate_logits = nn.Parameter(torch.zeros(components, path_count))
        self.head = nn.Linear(components * path_count, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(values)
        energy = self.frontend(spatial).permute(0, 3, 1, 2)
        gated = energy * torch.sigmoid(self.gate_logits)[None, None]
        # Softplus retains a gradient when an initialization undershoots zero;
        # ReLU made some valid recovery cases irreversibly collapse to rest.
        return torch.nn.functional.softplus(
            self.head(gated.flatten(2)), beta=5.0
        ).squeeze(-1)


def spectral_regularization(
    model: RecoveryModel, fft_size: int = 2048
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.zeros(2, 1, fft_size, device=model.gate_logits.device)
    values[1, ..., fft_size // 2] = 1.0e-4
    response = values
    for layer in model.frontend.layers:
        response = model.frontend._same_filter(response, layer)
        response = 1.7156 * torch.tanh((2.0 / 3.0) * response)
    local_response = (response[1] - response[0]) / 1.0e-4
    magnitude = torch.abs(torch.fft.rfft(local_response, dim=-1))
    magnitude = magnitude / magnitude.amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
    smoothness = torch.mean(torch.diff(magnitude, n=2, dim=-1).square())
    normalized = magnitude / torch.linalg.vector_norm(magnitude, dim=-1, keepdim=True).clamp_min(1.0e-8)
    similarity = normalized @ normalized.T
    diversity = (similarity - torch.eye(similarity.shape[0], device=similarity.device)).square().mean()
    return smoothness, diversity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=("single_band", "nearby_bands", "broadband", "spatial_mix", "correlated_fingers", "line_noise_drift"),
        default="single_band",
    )
    parser.add_argument("--bins", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--levels", type=int, choices=(3, 4), default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    values, target, target_bands = make_scenario(args.scenario, args.bins, 1000.0, args.seed)
    split = int(round(2.0 * args.bins / 3.0))
    tensor = torch.from_numpy(values.T.copy()).unsqueeze(0).to(device)
    observed = torch.from_numpy(target).unsqueeze(0).to(device)
    model = RecoveryModel(
        values.shape[1], min(values.shape[1], 6), levels=args.levels
    ).to(device)
    initial_frontend = copy.deepcopy(model.frontend).eval()
    train_model: nn.Module = model
    if args.compile:
        train_model = torch.compile(model, mode="reduce-overhead")
    optimizer = torch.optim.AdamW(
        [
            {"params": model.spatial.parameters(), "lr": 3.0e-4},
            {"params": model.frontend.parameters(), "lr": 2.0e-4},
            {"params": [model.gate_logits], "lr": 1.0e-3},
            {"params": model.head.parameters(), "lr": 2.0e-3},
        ],
        weight_decay=1.0e-5,
    )
    best_validation = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        prediction = train_model(tensor)
        fit_prediction = prediction[:, :split]
        fit_observed = observed[:, :split]
        weights = 1.0 + 4.0 * fit_observed
        regression = torch.mean(
            weights * (fit_prediction - fit_observed).square()
        ) / torch.mean(weights)
        centered_prediction = fit_prediction - fit_prediction.mean()
        centered_observed = fit_observed - fit_observed.mean()
        correlation = torch.sum(centered_prediction * centered_observed) / (
            torch.linalg.vector_norm(centered_prediction)
            * torch.linalg.vector_norm(centered_observed)
        ).clamp_min(1.0e-7)
        smoothness, diversity = spectral_regularization(model)
        sparsity = torch.sigmoid(model.gate_logits).mean()
        loss = (
            regression
            + 0.10 * (1.0 - correlation)
            + 1.0e-3 * smoothness
            + 2.0e-4 * diversity
            + 2.0e-5 * sparsity
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                estimate = model(tensor)
                validation = float(torch.mean((estimate[:, split:] - observed[:, split:]).square()))
            history.append(
                {
                    "epoch": epoch,
                    "train_mse": float(regression.detach()),
                    "validation_mse": validation,
                }
            )
            if validation < best_validation:
                best_validation = validation
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        prediction = model(tensor).squeeze(0).cpu().numpy()
    initial_frequency, initial_magnitude = wavelet_path_responses(initial_frontend)
    frequency, magnitude = wavelet_path_responses(model.frontend)
    gate = torch.sigmoid(model.gate_logits).detach().cpu().numpy()
    head = np.abs(model.head.weight.detach().cpu().numpy().reshape(gate.shape))
    importance = np.sum(gate * head, axis=0)
    composite_power = np.sum(importance[:, None] * np.square(magnitude), axis=0)
    target_mask = np.zeros_like(frequency, dtype=bool)
    for low, high in target_bands:
        target_mask |= (frequency >= low) & (frequency <= high)
    target_power_fraction = float(
        np.sum(composite_power[target_mask]) / max(float(np.sum(composite_power)), 1.0e-12)
    )
    result = {
        "scenario": args.scenario,
        "seed": args.seed,
        "levels": args.levels,
        "target_bands_hz": target_bands,
        "fit_bins": split,
        "validation_bins": args.bins - split,
        "validation_pcc": pearson(prediction[split:], target[split:]),
        "validation_mse": float(np.mean(np.square(prediction[split:] - target[split:]))),
        "head_weighted_target_band_power_fraction": target_power_fraction,
        "head_weighted_peak_hz": float(frequency[int(np.argmax(composite_power))]),
        "path_power_response_similarity_initial_to_learned": normalized_response_similarity(
            initial_magnitude, magnitude
        ).tolist(),
        "initial_response": summarize_responses(
            initial_frequency, initial_magnitude, model.frontend.band_names
        ),
        "learned_response": summarize_responses(
            frequency, magnitude, model.frontend.band_names
        ),
        "path_importance": {
            name: float(value) for name, value in zip(model.frontend.band_names, importance, strict=True)
        },
        "history": history,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state}, args.output / "model.pt")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    time = np.arange(args.bins - split) / 25.0
    axes[0].plot(time, target[split:], color="black", lw=2, label="control")
    axes[0].plot(time, prediction[split:], color="#2563eb", lw=1.3, label="prediction")
    axes[0].set(xlabel="validation time (s)", ylabel="amplitude", title=f"{args.scenario}: held-out recovery")
    axes[0].legend(frameon=False)
    normalized_power = composite_power / max(float(np.max(composite_power)), 1.0e-12)
    axes[1].plot(frequency, normalized_power, color="#dc2626", lw=1.5)
    for low, high in target_bands:
        axes[1].axvspan(low, high, color="#f59e0b", alpha=0.2)
    axes[1].set(xlim=(0, 250), xlabel="frequency (Hz)", ylabel="normalized weighted power", title="Head-weighted learned spectral response")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(args.output / "recovery.png", dpi=160)
    plt.close(figure)
    print(json.dumps({key: result[key] for key in ("scenario", "validation_pcc", "head_weighted_target_band_power_fraction", "head_weighted_peak_hz")}, indent=2))


if __name__ == "__main__":
    main()
