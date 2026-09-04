#!/usr/bin/env python3
"""Mechanistic counterfactual probes for a frozen per-finger decoder.

The script treats matched in-distribution interventions as primary evidence.
It reports direct/recurrent decomposition and explicit LSTM gate-state changes,
then adds integrated gradients as a secondary consistency diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ecog_decoding.probing import (
    decode_feature_groups,
    frequency_response_summary,
    linearized_wavelet_impulse_response,
    manual_lstm_trace,
    matched_rest_donors,
    phase_masks,
)
from ecog_decoding.training import FINGER_NAMES
from train_exact_window_end_to_end import ExactWindowFingerDecoder, make_windows


SCALAR_METRICS = ("prediction", "pre_relu", "direct_term", "recurrent_term")
STATE_METRICS = (
    "input_logit",
    "forget_logit",
    "candidate_logit",
    "output_logit",
    "cell_write",
    "retention",
    "cell_state",
    "hidden_state",
)
ALL_METRICS = SCALAR_METRICS + STATE_METRICS


def evenly_sample(indices: np.ndarray, maximum: int) -> np.ndarray:
    if indices.size <= maximum:
        return indices
    return indices[np.linspace(0, indices.size - 1, maximum, dtype=np.int64)]


def terminal_value(trace: dict[str, torch.Tensor], metric: str) -> torch.Tensor:
    value = trace[metric][:, -1]
    return value if metric in SCALAR_METRICS else value.mean(dim=-1)


def load_frozen_model(
    run_directory: Path,
    finger_name: str,
    device: torch.device,
) -> tuple[ExactWindowFingerDecoder, dict[str, object], np.ndarray]:
    summary = json.loads((run_directory / "summary.json").read_text())
    checkpoint = torch.load(
        run_directory / f"{finger_name}.pt", map_location="cpu", weights_only=True
    )
    state = checkpoint["model_state_dict"]
    selected = np.asarray(checkpoint["feature_indices"], dtype=np.int64)
    input_channels = int(state["spatial.weight"].shape[1])
    component_count = int(state["spatial.weight"].shape[0])
    hidden_size = int(state["lstm.weight_hh_l0"].shape[0] // 4)
    model = ExactWindowFingerDecoder(
        input_channels=input_channels,
        component_count=component_count,
        selected_indices=selected,
        feature_mean=state["feature_mean"].cpu().numpy(),
        feature_scale=state["feature_scale"].cpu().numpy(),
        hidden_size=hidden_size,
        wavelet_levels=int(summary.get("wavelet_levels", 3)),
        frontend=str(summary["frontend"]),
        head_initialization=str(summary.get("head_initialization", "residual_ridge")),
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model, summary, selected


@torch.inference_mode()
def extract_standardized(
    model: ExactWindowFingerDecoder,
    windows: np.ndarray,
    device: torch.device,
    chunk_steps: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, windows.shape[0], chunk_steps):
        batch = torch.from_numpy(np.ascontiguousarray(windows[start : start + chunk_steps]))
        features = model.extract(batch[None].to(device)).squeeze(0)
        standardized = (features - model.feature_mean) / model.feature_scale
        chunks.append(standardized.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def build_contexts(
    standardized: np.ndarray,
    indices: np.ndarray,
    context_steps: int,
) -> np.ndarray:
    return np.stack(
        [standardized[index - context_steps + 1 : index + 1] for index in indices]
    ).astype(np.float32, copy=False)


def intervention_maps(
    model: ExactWindowFingerDecoder,
    standardized: np.ndarray,
    targets: np.ndarray,
    finger: int,
    phase_indices: dict[str, np.ndarray],
    groups: list,
    context_steps: int,
    max_lag: int,
    threshold: float,
    group_batch: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    phase_names = tuple(phase_indices)
    signed = np.full(
        (len(phase_names), len(groups), max_lag + 1, len(ALL_METRICS)),
        np.nan,
        dtype=np.float32,
    )
    absolute = np.full_like(signed, np.nan)
    flips = np.full((len(phase_names), len(groups), max_lag + 1, 2), np.nan, dtype=np.float32)
    counts: dict[str, int] = {}
    for phase_number, phase_name in enumerate(phase_names):
        output_indices = phase_indices[phase_name]
        counts[phase_name] = int(output_indices.size)
        if output_indices.size == 0:
            continue
        contexts_np = build_contexts(standardized, output_indices, context_steps)
        contexts = torch.from_numpy(contexts_np).to(device)
        with torch.inference_mode():
            baseline = manual_lstm_trace(model.lstm, model.temporal, model.direct, contexts)
        baseline_terminal = {
            metric: terminal_value(baseline, metric) for metric in ALL_METRICS
        }
        baseline_prediction = baseline_terminal["prediction"]
        baseline_active = baseline_terminal["pre_relu"] > 0
        for lag in range(max_lag + 1):
            query_indices = output_indices - lag
            donor_indices = matched_rest_donors(
                standardized, targets, finger, query_indices, threshold=threshold
            )
            donor = torch.from_numpy(standardized[donor_indices]).to(device)
            position = context_steps - 1 - lag
            for begin in range(0, len(groups), group_batch):
                selected_groups = groups[begin : begin + group_batch]
                copies = contexts.unsqueeze(0).repeat(len(selected_groups), 1, 1, 1)
                for local_group, group in enumerate(selected_groups):
                    columns = torch.as_tensor(
                        group.selected_columns, dtype=torch.long, device=device
                    )
                    copies[local_group, :, position].index_copy_(
                        1, columns, donor.index_select(1, columns)
                    )
                flat = copies.flatten(0, 1)
                with torch.inference_mode():
                    perturbed = manual_lstm_trace(
                        model.lstm, model.temporal, model.direct, flat
                    )
                for metric_number, metric in enumerate(ALL_METRICS):
                    perturbed_value = terminal_value(perturbed, metric).reshape(
                        len(selected_groups), output_indices.size
                    )
                    difference = baseline_terminal[metric][None] - perturbed_value
                    signed[
                        phase_number,
                        begin : begin + len(selected_groups),
                        lag,
                        metric_number,
                    ] = difference.mean(dim=1).cpu().numpy()
                    absolute[
                        phase_number,
                        begin : begin + len(selected_groups),
                        lag,
                        metric_number,
                    ] = difference.abs().mean(dim=1).cpu().numpy()
                perturbed_prediction = terminal_value(
                    perturbed, "prediction"
                ).reshape(len(selected_groups), output_indices.size)
                perturbed_active = terminal_value(
                    perturbed, "pre_relu"
                ).reshape(len(selected_groups), output_indices.size) > 0
                flips[
                    phase_number,
                    begin : begin + len(selected_groups),
                    lag,
                    0,
                ] = (
                    baseline_active[None] != perturbed_active
                ).float().mean(dim=1).cpu().numpy()
                flips[
                    phase_number,
                    begin : begin + len(selected_groups),
                    lag,
                    1,
                ] = (
                    (baseline_prediction[None] >= threshold)
                    != (perturbed_prediction >= threshold)
                ).float().mean(dim=1).cpu().numpy()
    return signed, absolute, flips, counts


def interaction_maps(
    model: ExactWindowFingerDecoder,
    standardized: np.ndarray,
    targets: np.ndarray,
    finger: int,
    phase_indices: dict[str, np.ndarray],
    beta_columns: np.ndarray,
    gamma_columns: np.ndarray,
    context_steps: int,
    max_lag: int,
    threshold: float,
    pair_batch: int,
    device: torch.device,
) -> np.ndarray:
    phase_names = tuple(phase_indices)
    interaction = np.full(
        (len(phase_names), max_lag + 1, max_lag + 1, len(SCALAR_METRICS)),
        np.nan,
        dtype=np.float32,
    )
    beta_t = torch.as_tensor(beta_columns, dtype=torch.long, device=device)
    gamma_t = torch.as_tensor(gamma_columns, dtype=torch.long, device=device)
    pairs = [(beta_lag, gamma_lag) for beta_lag in range(max_lag + 1) for gamma_lag in range(max_lag + 1)]
    for phase_number, phase_name in enumerate(phase_names):
        output_indices = phase_indices[phase_name]
        if output_indices.size == 0:
            continue
        contexts = torch.from_numpy(
            build_contexts(standardized, output_indices, context_steps)
        ).to(device)
        with torch.inference_mode():
            baseline = manual_lstm_trace(model.lstm, model.temporal, model.direct, contexts)
        base = {metric: terminal_value(baseline, metric) for metric in SCALAR_METRICS}
        donor_cache: dict[int, torch.Tensor] = {}
        for lag in range(max_lag + 1):
            donors = matched_rest_donors(
                standardized, targets, finger, output_indices - lag, threshold=threshold
            )
            donor_cache[lag] = torch.from_numpy(standardized[donors]).to(device)
        for begin in range(0, len(pairs), pair_batch):
            selected_pairs = pairs[begin : begin + pair_batch]
            # Three counterfactuals per pair: beta removed, gamma removed, both.
            copies = contexts.unsqueeze(0).repeat(3 * len(selected_pairs), 1, 1, 1)
            for local_pair, (beta_lag, gamma_lag) in enumerate(selected_pairs):
                beta_position = context_steps - 1 - beta_lag
                gamma_position = context_steps - 1 - gamma_lag
                beta_only = 3 * local_pair
                gamma_only = beta_only + 1
                both = beta_only + 2
                beta_donor = donor_cache[beta_lag].index_select(1, beta_t)
                gamma_donor = donor_cache[gamma_lag].index_select(1, gamma_t)
                for variant in (beta_only, both):
                    copies[variant, :, beta_position].index_copy_(
                        1, beta_t, beta_donor
                    )
                for variant in (gamma_only, both):
                    copies[variant, :, gamma_position].index_copy_(
                        1, gamma_t, gamma_donor
                    )
            with torch.inference_mode():
                trace = manual_lstm_trace(
                    model.lstm,
                    model.temporal,
                    model.direct,
                    copies.flatten(0, 1),
                )
            for metric_number, metric in enumerate(SCALAR_METRICS):
                values = terminal_value(trace, metric).reshape(
                    len(selected_pairs), 3, output_indices.size
                )
                second_difference = (
                    base[metric][None] - values[:, 0] - values[:, 1] + values[:, 2]
                ).mean(dim=1)
                for local_pair, (beta_lag, gamma_lag) in enumerate(selected_pairs):
                    interaction[
                        phase_number, beta_lag, gamma_lag, metric_number
                    ] = float(second_difference[local_pair])
    return interaction


def integrated_gradients(
    model: ExactWindowFingerDecoder,
    standardized: np.ndarray,
    targets: np.ndarray,
    finger: int,
    phase_indices: dict[str, np.ndarray],
    groups: list,
    context_steps: int,
    max_lag: int,
    threshold: float,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    result = np.full(
        (len(phase_indices), len(groups), max_lag + 1), np.nan, dtype=np.float32
    )
    for phase_number, output_indices in enumerate(phase_indices.values()):
        if output_indices.size == 0:
            continue
        observed = torch.from_numpy(
            build_contexts(standardized, output_indices, context_steps)
        ).to(device)
        background = observed.clone()
        for lag in range(max_lag + 1):
            donors = matched_rest_donors(
                standardized, targets, finger, output_indices - lag, threshold=threshold
            )
            background[:, context_steps - 1 - lag] = torch.from_numpy(
                standardized[donors]
            ).to(device)
        gradient_sum = torch.zeros_like(observed)
        for alpha in torch.linspace(0.0, 1.0, steps, device=device):
            interpolated = (background + alpha * (observed - background)).requires_grad_(True)
            prediction = model.decode(interpolated)[:, -1].sum()
            gradient_sum += torch.autograd.grad(prediction, interpolated)[0]
        attribution = (observed - background) * gradient_sum / steps
        for group_number, group in enumerate(groups):
            columns = torch.as_tensor(group.selected_columns, dtype=torch.long, device=device)
            values = attribution.index_select(2, columns).sum(dim=2)
            result[phase_number, group_number] = (
                values[:, -(max_lag + 1) :].mean(dim=0).flip(0).detach().cpu().numpy()
            )
    return result


def effective_kernel_changes(
    initial_spatial: np.ndarray,
    trained_spatial: np.ndarray,
    initial_impulse: np.ndarray,
    trained_impulse: np.ndarray,
) -> dict[str, np.ndarray]:
    def distance(a: np.ndarray, h: np.ndarray, b: np.ndarray, g: np.ndarray) -> np.ndarray:
        a2 = np.sum(a * a, axis=1)[:, None]
        b2 = np.sum(b * b, axis=1)[:, None]
        h2 = np.sum(h * h, axis=1)[None]
        g2 = np.sum(g * g, axis=1)[None]
        cross = (np.sum(a * b, axis=1)[:, None] * np.sum(h * g, axis=1)[None])
        return np.sqrt(np.maximum(a2 * h2 + b2 * g2 - 2.0 * cross, 0.0))

    baseline_norm = np.linalg.norm(initial_spatial, axis=1)[:, None] * np.linalg.norm(
        initial_impulse, axis=1
    )[None]
    denominator = np.maximum(baseline_norm, np.finfo(np.float64).eps)
    return {
        "spatial_only_relative_change": distance(
            initial_spatial, initial_impulse, trained_spatial, initial_impulse
        )
        / denominator,
        "temporal_only_relative_change": distance(
            initial_spatial, initial_impulse, initial_spatial, trained_impulse
        )
        / denominator,
        "full_relative_change": distance(
            initial_spatial, initial_impulse, trained_spatial, trained_impulse
        )
        / denominator,
    }


@torch.inference_mode()
def data_weighted_band_outputs(
    model: ExactWindowFingerDecoder,
    initial_model: ExactWindowFingerDecoder,
    initial_spatial: np.ndarray,
    windows: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    batch = torch.from_numpy(np.ascontiguousarray(windows)).to(device)
    initial_spatial_t = torch.from_numpy(initial_spatial).to(device)
    trained_spatial_t = model.spatial.weight[:, :, 0]
    combinations = {
        "initial": (initial_spatial_t, initial_model.wavelet),
        "spatial_only": (trained_spatial_t, initial_model.wavelet),
        "temporal_only": (initial_spatial_t, model.wavelet),
        "trained": (trained_spatial_t, model.wavelet),
    }
    output: dict[str, np.ndarray] = {}
    raw = batch.squeeze(0) if batch.ndim == 4 and batch.shape[0] == 1 else batch
    if raw.ndim != 3:
        raise ValueError("sample windows must have shape (window, channel, sample)")
    for name, (spatial_weight, frontend) in combinations.items():
        spatial = torch.einsum("oc,nct->not", spatial_weight, raw)
        values = frontend(spatial).mean(dim=-1)
        output[f"{name}_mean"] = values.mean(dim=0).float().cpu().numpy()
        output[f"{name}_rms"] = values.square().mean(dim=0).sqrt().float().cpu().numpy()
    return output


def plot_band_lag(
    signed: np.ndarray,
    groups: list,
    band_names: tuple[str, ...],
    phase_names: tuple[str, ...],
    output: Path,
) -> None:
    metric = ALL_METRICS.index("prediction")
    figure, axes = plt.subplots(len(phase_names), 1, figsize=(11, 2.8 * len(phase_names)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    limit = float(np.nanmax(np.abs(signed[..., metric]))) or 1.0
    for phase, (axis, phase_name) in enumerate(zip(axes, phase_names, strict=True)):
        by_band = np.full((len(band_names), signed.shape[2]), np.nan)
        for band in range(len(band_names)):
            selected = [index for index, group in enumerate(groups) if group.band == band]
            if selected:
                by_band[band] = np.nanmean(signed[phase, selected, :, metric], axis=0)
        image = axis.imshow(by_band, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit, origin="lower")
        axis.set_yticks(np.arange(len(band_names)), band_names)
        axis.set_title(phase_name.replace("_", " "))
        axis.set_xlabel("causal history lag (40 ms bins)")
    figure.colorbar(image, ax=axes, label="prediction(full) - prediction(rest intervention)")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_interaction(interaction: np.ndarray, phase_names: tuple[str, ...], output: Path) -> None:
    metric = SCALAR_METRICS.index("prediction")
    figure, axes = plt.subplots(1, len(phase_names), figsize=(4 * len(phase_names), 3.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    limit = float(np.nanmax(np.abs(interaction[..., metric]))) or 1.0
    for phase, (axis, phase_name) in enumerate(zip(axes, phase_names, strict=True)):
        image = axis.imshow(interaction[phase, :, :, metric], origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_title(phase_name.replace("_", " "))
        axis.set_xlabel("gamma lag")
        axis.set_ylabel("beta lag")
    figure.colorbar(image, ax=axes, label="paired removal interaction")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_filter_response(
    frequencies: np.ndarray,
    initial_magnitude: np.ndarray,
    trained_magnitude: np.ndarray,
    band_names: tuple[str, ...],
    output: Path,
) -> None:
    columns = 3
    rows = int(np.ceil(len(band_names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(13, 3.0 * rows), constrained_layout=True)
    for band, (axis, name) in enumerate(zip(np.ravel(axes), band_names, strict=False)):
        initial = initial_magnitude[band] / max(initial_magnitude[band].max(), 1.0e-12)
        trained = trained_magnitude[band] / max(trained_magnitude[band].max(), 1.0e-12)
        keep = frequencies <= 250
        axis.plot(frequencies[keep], 20 * np.log10(np.maximum(initial[keep], 1.0e-5)), label="initial")
        axis.plot(frequencies[keep], 20 * np.log10(np.maximum(trained[keep], 1.0e-5)), label="trained")
        axis.set_ylim(-80, 5)
        axis.set_title(name)
        axis.grid(alpha=0.2)
    for axis in np.ravel(axes)[len(band_names) :]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False)
    figure.supxlabel("frequency (Hz)")
    figure.supylabel("normalized magnitude (dB)")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--finger", choices=FINGER_NAMES, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--ica-root", type=Path, default=Path("outputs/paper_ica_lars_v1"))
    parser.add_argument("--target", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--transition-bins", type=int, default=5)
    parser.add_argument("--max-lag", type=int, default=24)
    parser.add_argument("--maximum-contexts", type=int, default=32)
    parser.add_argument("--group-batch", type=int, default=12)
    parser.add_argument("--pair-batch", type=int, default=12)
    parser.add_argument("--ig-steps", type=int, default=12)
    parser.add_argument("--chunk-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    finger = list(FINGER_NAMES).index(args.finger)
    model, training_summary, selected = load_frozen_model(
        args.run_directory, args.finger, device
    )
    initial_model = ExactWindowFingerDecoder(
        input_channels=model.spatial.in_channels,
        component_count=model.spatial.out_channels,
        selected_indices=selected,
        feature_mean=model.feature_mean.cpu().numpy(),
        feature_scale=model.feature_scale.cpu().numpy(),
        hidden_size=model.lstm.hidden_size,
        wavelet_levels=int(training_summary.get("wavelet_levels", 3)),
        frontend=str(training_summary["frontend"]),
        head_initialization=str(training_summary.get("head_initialization", "residual_ridge")),
    ).to(device).eval()
    prepared = args.prepared_root / f"sub{args.subject}"
    metadata = json.loads((prepared / "metadata.json").read_text())
    split_index = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    if args.split == "validation":
        ecog = np.load(prepared / "train_ecog.npy", mmap_mode="r")
        all_windows = make_windows(ecog, 1000, 40)
        first = split_index - offset
        windows = all_windows[first:]
        cleaned_target = np.load(prepared / f"train_glove_{args.target}.npy")[split_index:]
        prediction_file = args.run_directory / "validation_prediction.npy"
    else:
        ecog = np.load(prepared / "test_ecog.npy", mmap_mode="r")
        windows = make_windows(ecog, 1000, 40)
        cleaned_target = np.load(prepared / f"test_glove_{args.target}.npy")[offset:]
        prediction_file = args.run_directory / "test_prediction.npy"
    saved_prediction = np.load(prediction_file)[:, finger]
    standardized = extract_standardized(model, windows, device, args.chunk_steps)
    if standardized.shape[0] != cleaned_target.shape[0]:
        raise ValueError("feature and target lengths do not match")
    with torch.inference_mode():
        reconstructed = model.decode(torch.from_numpy(standardized[None]).to(device)).squeeze(0).cpu().numpy()
    reconstruction_error = float(np.max(np.abs(reconstructed - saved_prediction)))
    if reconstruction_error > 2.0e-5:
        raise RuntimeError(f"saved prediction reconstruction error is {reconstruction_error:.3g}")

    band_names = tuple(model.wavelet.band_names)
    full_feature_count = model.spatial.out_channels * len(band_names)
    if int(selected.max()) >= full_feature_count:
        energy_bins = int((int(selected.max()) + 1 + full_feature_count - 1) // full_feature_count)
    else:
        energy_bins = 25
    # The frontend always pools a 1 s, 1000 Hz window into 25 non-overlapping bins.
    energy_bins = 25
    groups = decode_feature_groups(
        selected, model.spatial.out_channels, band_names, energy_bins
    )
    context_steps = int(training_summary["optimization"]["sequence_steps"])
    if args.max_lag >= context_steps:
        parser.error("--max-lag must be smaller than the training sequence length")
    masks = phase_masks(
        cleaned_target[:, finger], reconstructed, args.threshold, args.transition_bins
    )
    phase_indices = {
        name: evenly_sample(
            np.flatnonzero(mask & (np.arange(mask.size) >= context_steps - 1)),
            args.maximum_contexts,
        )
        for name, mask in masks.items()
    }
    signed, absolute, flips, counts = intervention_maps(
        model,
        standardized,
        cleaned_target,
        finger,
        phase_indices,
        groups,
        context_steps,
        args.max_lag,
        args.threshold,
        args.group_batch,
        device,
    )

    initial_impulse = linearized_wavelet_impulse_response(initial_model.wavelet)
    trained_impulse = linearized_wavelet_impulse_response(model.wavelet)
    frequencies, initial_magnitude, initial_filter_summary = frequency_response_summary(initial_impulse)
    _, trained_magnitude, trained_filter_summary = frequency_response_summary(trained_impulse)
    initial_phase = np.unwrap(
        np.angle(
            np.fft.rfft(
                np.roll(initial_impulse, -(initial_impulse.shape[1] // 2), axis=1),
                axis=1,
            )
        ),
        axis=1,
    )
    trained_phase = np.unwrap(
        np.angle(
            np.fft.rfft(
                np.roll(trained_impulse, -(trained_impulse.shape[1] // 2), axis=1),
                axis=1,
            )
        ),
        axis=1,
    )
    power = trained_magnitude**2
    beta_mask = (frequencies >= 13.0) & (frequencies <= 30.0)
    gamma_mask = (frequencies >= 70.0) & (frequencies <= 200.0)
    beta_fraction = power[:, beta_mask].sum(axis=1) / np.maximum(power.sum(axis=1), 1.0e-12)
    gamma_fraction = power[:, gamma_mask].sum(axis=1) / np.maximum(power.sum(axis=1), 1.0e-12)
    present_bands = np.unique([group.band for group in groups])
    beta_threshold = max(0.15, 0.5 * beta_fraction[present_bands].max())
    gamma_threshold = max(0.15, 0.5 * gamma_fraction[present_bands].max())
    beta_bands = present_bands[beta_fraction[present_bands] >= beta_threshold]
    gamma_bands = present_bands[gamma_fraction[present_bands] >= gamma_threshold]
    if beta_bands.size == 0:
        beta_bands = np.asarray([present_bands[int(np.argmax(beta_fraction[present_bands]))]])
    if gamma_bands.size == 0:
        gamma_bands = np.asarray([present_bands[int(np.argmax(gamma_fraction[present_bands]))]])
    beta_columns = np.unique(
        np.concatenate([group.selected_columns for group in groups if group.band in beta_bands])
    )
    gamma_columns = np.unique(
        np.concatenate([group.selected_columns for group in groups if group.band in gamma_bands])
    )
    interaction = interaction_maps(
        model,
        standardized,
        cleaned_target,
        finger,
        phase_indices,
        beta_columns,
        gamma_columns,
        context_steps,
        args.max_lag,
        args.threshold,
        args.pair_batch,
        device,
    )
    ig = integrated_gradients(
        model,
        standardized,
        cleaned_target,
        finger,
        phase_indices,
        groups,
        context_steps,
        args.max_lag,
        args.threshold,
        args.ig_steps,
        device,
    )

    initial_spatial = np.load(args.ica_root / f"sub{args.subject}" / "fastica_unmixing.npy")
    trained_spatial = model.spatial.weight[:, :, 0].detach().cpu().numpy()
    kernel_changes = effective_kernel_changes(
        initial_spatial, trained_spatial, initial_impulse, trained_impulse
    )
    sample_windows = windows[
        np.linspace(0, windows.shape[0] - 1, min(256, windows.shape[0]), dtype=np.int64)
    ]
    weighted_outputs = data_weighted_band_outputs(
        model, initial_model, initial_spatial, sample_windows, device
    )

    args.output.mkdir(parents=True, exist_ok=True)
    group_components = np.asarray([group.component for group in groups], dtype=np.int16)
    group_bands = np.asarray([group.band for group in groups], dtype=np.int16)
    np.savez_compressed(
        args.output / "probe_arrays.npz",
        signed_intervention=signed,
        absolute_intervention=absolute,
        threshold_flips=flips,
        beta_gamma_interaction=interaction,
        integrated_gradients=ig,
        group_components=group_components,
        group_bands=group_bands,
        frequencies_hz=frequencies,
        initial_impulse=initial_impulse,
        trained_impulse=trained_impulse,
        initial_magnitude=initial_magnitude,
        trained_magnitude=trained_magnitude,
        initial_phase=initial_phase,
        trained_phase=trained_phase,
        **kernel_changes,
        **{f"band_output_{name}": value for name, value in weighted_outputs.items()},
    )
    prediction_metric = ALL_METRICS.index("prediction")
    top_effects: dict[str, list[dict[str, object]]] = {}
    for phase, phase_name in enumerate(phase_indices):
        strength = np.nanmax(np.abs(signed[phase, :, :, prediction_metric]), axis=1)
        ranking = np.argsort(np.nan_to_num(strength, nan=-1.0))[::-1][:10]
        top_effects[phase_name] = [
            {
                "component": groups[index].component,
                "band": groups[index].band_name,
                "peak_lag_bins": int(
                    np.nanargmax(np.abs(signed[phase, index, :, prediction_metric]))
                ),
                "filter_group_delay_ms": float(
                    trained_filter_summary[groups[index].band]["group_delay_ms"]
                ),
                "group_delay_corrected_lag_ms": float(
                    40.0
                    * np.nanargmax(
                        np.abs(signed[phase, index, :, prediction_metric])
                    )
                    - trained_filter_summary[groups[index].band]["group_delay_ms"]
                ),
                "signed_prediction_effect": float(
                    signed[
                        phase,
                        index,
                        np.nanargmax(np.abs(signed[phase, index, :, prediction_metric])),
                        prediction_metric,
                    ]
                ),
            }
            for index in ranking
            if np.isfinite(strength[index])
        ]
    report = {
        "subject": args.subject,
        "finger": args.finger,
        "split": args.split,
        "frozen_run_directory": str(args.run_directory),
        "primary_evidence": "matched conditional rest-state component-band interventions",
        "secondary_evidence": "integrated gradients",
        "released_test_used_for_selection": False,
        "prediction_reconstruction_max_abs_error": reconstruction_error,
        "phase_context_counts": counts,
        "context_steps": context_steps,
        "maximum_lag_bins": args.max_lag,
        "lag_bin_ms": 40,
        "metrics": list(ALL_METRICS),
        "flip_metrics": ["relu_activation_flip", "movement_threshold_flip"],
        "band_names": list(band_names),
        "beta_bands": [band_names[index] for index in beta_bands],
        "high_gamma_bands": [band_names[index] for index in gamma_bands],
        "beta_power_fraction": beta_fraction.tolist(),
        "high_gamma_power_fraction": gamma_fraction.tolist(),
        "initial_filter_response": initial_filter_summary,
        "trained_filter_response": trained_filter_summary,
        "effective_kernel_change_mean": {
            name: float(np.mean(value)) for name, value in kernel_changes.items()
        },
        "data_weighted_band_output_mean": {
            name: value.tolist() for name, value in weighted_outputs.items()
        },
        "top_component_band_effects": top_effects,
        "interpretation_note": (
            "Prediction interaction is the single-context inclusion-exclusion second "
            "difference. Pre-ReLU, direct, and recurrent terms are saved separately so "
            "ReLU threshold crossing is not mislabeled as recurrent interaction."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    phase_names = tuple(phase_indices)
    plot_band_lag(signed, groups, band_names, phase_names, args.output / "band_lag_interventions.png")
    plot_interaction(interaction, phase_names, args.output / "beta_gamma_interaction.png")
    plot_filter_response(
        frequencies,
        initial_magnitude,
        trained_magnitude,
        band_names,
        args.output / "filter_response.png",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
