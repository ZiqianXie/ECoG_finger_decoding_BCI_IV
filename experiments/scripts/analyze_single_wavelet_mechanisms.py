#!/usr/bin/env python3
"""Interpret one subject's frozen single-wavelet ensembles in physical space.

The analysis uses the route manifest, all six seeds per finger, the exact
registered channel map, and development ECoG covariance. It reports both
decoder spatial filters and covariance-transformed forward patterns, plus the
small-signal frequency response of every trained wavelet path.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from ecog_decoding.models import WaveletPacketEnergy
from ecog_decoding.training import FINGER_NAMES


HISTORY_BINS = 25
PATH_COUNT = 8
TARGET_RATE_HZ = 25.0
NOMINAL_BAND_NAMES = tuple(f"{low}-{low + 25} Hz" for low in range(0, 200, 25))
REFERENCE_CODES = {1: "zt", 2: "bp", 3: "cc"}
EXCLUDED_INFERRED_CHANNELS = {1: [55], 2: [21, 38], 3: [50]}


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("model_state_dict", payload)


def load_geometry(path: Path) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            channel = int(row["data_channel_1based"])
            result[channel] = {
                "grid": np.asarray(
                    [int(row["grid_row_1based"]), int(row["grid_col_1based"])],
                    dtype=np.float64,
                ),
                "xyz": np.asarray([float(row[axis]) for axis in ("x", "y", "z")]),
                "coordinate_status": row["coordinate_status"],
                "confidence": row["confidence"],
            }
    return result


def streaming_covariance(path: Path, chunk_rows: int = 25000) -> np.ndarray:
    values = np.load(path, mmap_mode="r")
    count = 0
    total = np.zeros(values.shape[1], dtype=np.float64)
    cross = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    for start in range(0, values.shape[0], chunk_rows):
        chunk = np.asarray(values[start : start + chunk_rows], dtype=np.float64)
        keep = np.all(np.isfinite(chunk), axis=1)
        chunk = chunk[keep]
        count += chunk.shape[0]
        total += chunk.sum(axis=0)
        cross += chunk.T @ chunk
    if count < 2:
        raise RuntimeError("not enough finite ECoG rows for covariance")
    mean = total / count
    return (cross - count * np.outer(mean, mean)) / (count - 1)


def forward_patterns(spatial: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Return one covariance-transformed forward pattern per spatial row."""
    projected = covariance @ spatial.T
    variance = np.einsum("rc,cr->r", spatial, projected)
    scale = np.maximum(np.abs(variance), 1.0e-12)
    return (projected / scale[None, :]).T


def structural_feature_importance(
    state: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = state["selected_indices"].detach().cpu().numpy().astype(np.int64)
    component_count = int(state["spatial.weight"].shape[0])
    features_per_lag = component_count * PATH_COUNT
    full_count = HISTORY_BINS * features_per_lag
    if selected.size == 0 or selected.min() < 0 or selected.max() >= full_count:
        raise ValueError(
            f"selected feature range is incompatible with {component_count} spatial rows"
        )
    lag = selected // features_per_lag
    within_lag = selected % features_per_lag
    component = within_lag // PATH_COUNT
    nominal_band = within_lag % PATH_COUNT

    if "lstm.weight_ih_l0" in state:
        input_key = "lstm.weight_ih_l0"
        recurrent_key = "lstm.weight_hh_l0"
        output_key = "output.weight"
    else:
        input_key = "residual_lstm.weight_ih_l0"
        recurrent_key = "residual_lstm.weight_hh_l0"
        output_key = "residual_head.weight"
    input_weights = state[input_key].detach().cpu().numpy()
    hidden_size = int(state[recurrent_key].shape[1])
    input_weights = input_weights.reshape(4, hidden_size, selected.size)
    output_weights = np.abs(state[output_key].detach().cpu().numpy()[0])
    importance = np.sqrt(
        np.sum((input_weights * output_weights[None, :, None]) ** 2, axis=(0, 1))
    )
    if not np.any(importance > 0):
        importance = np.sqrt(np.sum(input_weights**2, axis=(0, 1)))
    importance = np.maximum(importance, 0.0)
    importance /= importance.sum()
    return component, nominal_band, lag, importance


def row_and_channel_importance(
    patterns: np.ndarray,
    filters: np.ndarray,
    component: np.ndarray,
    feature_importance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row = np.bincount(component, weights=feature_importance, minlength=patterns.shape[0])

    def project(values: np.ndarray) -> np.ndarray:
        power = np.square(values.astype(np.float64))
        power /= np.maximum(power.sum(axis=1, keepdims=True), 1.0e-12)
        channel = row @ power
        channel /= channel.sum()
        return channel

    return row, project(patterns), project(filters)


def weighted_pair_distance(weights: np.ndarray, coordinates: np.ndarray) -> float:
    probability = np.asarray(weights, dtype=np.float64)
    probability /= probability.sum()
    distance = np.linalg.norm(coordinates[:, None] - coordinates[None], axis=-1)
    return float(0.5 * np.sum(probability[:, None] * probability[None] * distance))


def localization_metrics(
    weights: np.ndarray,
    grid: np.ndarray,
    xyz: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
) -> dict[str, float]:
    probability = np.asarray(weights, dtype=np.float64)
    probability /= probability.sum()
    observed = weighted_pair_distance(probability, grid)
    observed_mm = weighted_pair_distance(probability, xyz)
    null = np.asarray(
        [weighted_pair_distance(rng.permutation(probability), grid) for _ in range(permutations)]
    )
    null_mm = np.asarray(
        [weighted_pair_distance(rng.permutation(probability), xyz) for _ in range(permutations)]
    )
    null_sd = float(null.std(ddof=1))
    null_mm_sd = float(null_mm.std(ddof=1))
    z = (observed - float(null.mean())) / null_sd if null_sd > 0 else 0.0
    z_mm = (
        (observed_mm - float(null_mm.mean())) / null_mm_sd
        if null_mm_sd > 0
        else 0.0
    )
    p_local = (1.0 + float(np.sum(null <= observed))) / (permutations + 1.0)
    p_local_mm = (
        1.0 + float(np.sum(null_mm <= observed_mm))
    ) / (permutations + 1.0)
    distances = np.linalg.norm(grid[:, None] - grid[None], axis=-1)
    distances_mm = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    medoid = int(np.argmin(np.sum(probability[None, :] * distances, axis=1)))
    medoid_mm = int(
        np.argmin(np.sum(probability[None, :] * distances_mm, axis=1))
    )
    from_medoid = distances[medoid]
    from_medoid_mm = distances_mm[medoid_mm]
    order = np.argsort(from_medoid)
    order_mm = np.argsort(from_medoid_mm)
    cumulative = np.cumsum(probability[order])
    cumulative_mm = np.cumsum(probability[order_mm])

    def radius(mass: float) -> float:
        index = min(int(np.searchsorted(cumulative, mass)), len(order) - 1)
        return float(from_medoid[order[index]])

    def radius_mm(mass: float) -> float:
        index = min(int(np.searchsorted(cumulative_mm, mass)), len(order_mm) - 1)
        return float(from_medoid_mm[order_mm[index]])

    centroid = probability @ xyz
    rms_radius = float(
        np.sqrt(np.sum(probability * np.sum((xyz - centroid[None]) ** 2, axis=1)))
    )

    return {
        "effective_channel_count": float(1.0 / np.sum(probability**2)),
        "top_5_channel_mass": float(np.sort(probability)[-5:].sum()),
        "weighted_pair_distance_grid_steps": observed,
        "weighted_pair_distance_mm": observed_mm,
        "permutation_null_mean_grid_steps": float(null.mean()),
        "permutation_localization_z": float(z),
        "permutation_one_sided_p_local": float(p_local),
        "permutation_null_mean_mm": float(null_mm.mean()),
        "permutation_localization_z_mm": float(z_mm),
        "permutation_one_sided_p_local_mm": float(p_local_mm),
        "weighted_centroid_xyz": centroid.tolist(),
        "weighted_rms_radius_mm": rms_radius,
        "radius_50_mass_grid_steps": radius(0.50),
        "radius_80_mass_grid_steps": radius(0.80),
        "radius_50_mass_mm": radius_mm(0.50),
        "radius_80_mass_mm": radius_mm(0.80),
        "medoid_index": medoid,
        "physical_medoid_index": medoid_mm,
    }


def same_filter(
    values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    total = dilation * (weight.shape[-1] - 1)
    return F.conv1d(
        F.pad(values, (total // 2, total - total // 2)),
        weight,
        bias,
        dilation=dilation,
    )


def frequency_response(
    state: dict[str, torch.Tensor],
    frequency_order: np.ndarray,
    sampling_rate_hz: float,
    fft_size: int,
    dilations: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = 1.0e-4
    signals = torch.zeros(2, 1, fft_size, dtype=torch.float32)
    signals[0, 0, fft_size // 2] = amplitude
    bands = signals
    for level, dilation in enumerate(dilations):
        bands = same_filter(
            bands,
            state[f"wavelet.layers.{level}.weight"].detach().cpu(),
            state[f"wavelet.layers.{level}.bias"].detach().cpu(),
            dilation,
        )
        bands = 1.7156 * torch.tanh((2.0 / 3.0) * bands)
    response = ((bands[0] - bands[1]) / amplitude).double().numpy()
    response = response[frequency_order]
    frequency = np.fft.rfftfreq(fft_size, d=1.0 / sampling_rate_hz)
    power = np.abs(np.fft.rfft(response, axis=1)) ** 2
    power /= np.maximum(power.sum(axis=1, keepdims=True), 1.0e-30)
    return frequency, power


def frequency_summary(frequency: np.ndarray, power: np.ndarray) -> dict[str, object]:
    probability = np.asarray(power, dtype=np.float64)
    probability /= probability.sum()
    cumulative = np.cumsum(probability)
    lower = int(np.searchsorted(cumulative, 0.05))
    upper = int(np.searchsorted(cumulative, 0.95))
    return {
        "peak_hz": float(frequency[int(np.argmax(probability))]),
        "centroid_hz": float(np.sum(frequency * probability)),
        "central_90_percent_hz": [float(frequency[lower]), float(frequency[upper])],
        "mass_above_200_hz": float(probability[frequency > 200].sum()),
    }


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 0 or np.std(right) <= 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def plot_grid(
    axis: plt.Axes,
    values: np.ndarray,
    retained_channels: list[int],
    geometry: dict[int, dict[str, object]],
    *,
    signed: bool,
    annotate: int = 3,
    limit: float | None = None,
) -> object:
    grid = np.full((8, 8), np.nan)
    for channel, value in zip(retained_channels, values, strict=True):
        row, col = geometry[channel]["grid"].astype(int)
        grid[row - 1, col - 1] = value
    if signed:
        scale = limit or max(float(np.nanmax(np.abs(grid))), 1.0e-12)
        image = axis.imshow(grid, cmap="coolwarm", vmin=-scale, vmax=scale)
        top = np.argsort(np.abs(values))[-annotate:][::-1]
    else:
        image = axis.imshow(grid, cmap="magma", vmin=0.0, vmax=limit)
        top = np.argsort(values)[-annotate:][::-1]
    for index in top:
        channel = retained_channels[index]
        row, col = geometry[channel]["grid"].astype(int)
        axis.text(
            col - 1,
            row - 1,
            str(channel),
            color="white" if signed or values[index] > 0.35 * np.max(values) else "black",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
        )
    axis.set_xticks(range(8), range(1, 9), fontsize=7)
    axis.set_yticks(range(8), range(1, 9), fontsize=7)
    return image


def render_figures(
    records: dict[str, dict[str, object]],
    geometry: dict[int, dict[str, object]],
    output_root: Path,
    subject: int,
) -> None:
    figure, axes = plt.subplots(1, 5, figsize=(18, 4.2), constrained_layout=True)
    spatial_maps = [
        100.0 * np.asarray(records[finger]["forward_pattern_channel_importance"])
        for finger in FINGER_NAMES
    ]
    spatial_limit = max(float(np.max(np.concatenate(spatial_maps))), 1.0e-12)
    image = None
    for axis, finger, values in zip(axes, FINGER_NAMES, spatial_maps, strict=True):
        record = records[finger]
        image = plot_grid(
            axis,
            values,
            record["retained_channels"],
            geometry,
            signed=False,
            limit=spatial_limit,
        )
        loc = record["forward_pattern_localization"]
        axis.set_title(
            f"{finger}\nN_eff={loc['effective_channel_count']:.1f}, "
            f"z={loc['permutation_localization_z']:.1f}",
            fontsize=10,
        )
        axis.set_xlabel("grid column")
    axes[0].set_ylabel("grid row")
    figure.colorbar(image, ax=axes, label="decoder-weighted forward-pattern mass (%)", shrink=0.82)
    figure.suptitle(
        f"Subject {subject}: spatial source footprints of the five frozen decoders"
    )
    figure.savefig(output_root / f"s{subject}_spatial_forward_patterns.png", dpi=200)
    plt.close(figure)

    retained = records[FINGER_NAMES[0]]["retained_channels"]
    xyz = np.asarray([geometry[channel]["xyz"] for channel in retained])
    physical_maps = [
        100.0 * np.asarray(records[finger]["forward_pattern_channel_importance"])
        for finger in FINGER_NAMES
    ]
    physical_limit = max(float(np.max(np.concatenate(physical_maps))), 1.0e-12)
    figure, axes = plt.subplots(2, 5, figsize=(18, 8.2), constrained_layout=True)
    image = None
    for column, (finger, values) in enumerate(
        zip(FINGER_NAMES, physical_maps, strict=True)
    ):
        top = np.argsort(values)[-5:][::-1]
        for row, (horizontal, vertical, horizontal_label, vertical_label) in enumerate(
            ((1, 2, "registered y (mm)", "registered z (mm)"),
             (0, 1, "registered x (mm)", "registered y (mm)"))
        ):
            axis = axes[row, column]
            image = axis.scatter(
                xyz[:, horizontal],
                xyz[:, vertical],
                c=values,
                s=28.0 + 48.0 * values,
                cmap="magma",
                vmin=0.0,
                vmax=physical_limit,
                edgecolors="#334155",
                linewidths=0.35,
            )
            for index in top:
                axis.text(
                    xyz[index, horizontal],
                    xyz[index, vertical],
                    str(retained[index]),
                    color="white" if values[index] > 0.35 * physical_limit else "black",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                )
            axis.set_xlabel(horizontal_label, fontsize=8)
            if column == 0:
                axis.set_ylabel(vertical_label, fontsize=8)
            axis.set_aspect("equal", adjustable="datalim")
            axis.tick_params(labelsize=7)
            if row == 0:
                loc = records[finger]["forward_pattern_localization"]
                axis.set_title(
                    f"{finger}\n3-D pair distance={loc['weighted_pair_distance_mm']:.1f} mm, "
                    f"p={loc['permutation_one_sided_p_local_mm']:.3f}",
                    fontsize=9,
                )
    figure.colorbar(
        image,
        ax=axes,
        label="decoder-weighted forward-pattern mass (%)",
        shrink=0.82,
    )
    figure.suptitle(
        f"Subject {subject}: source footprints at exact registered physical coordinates"
    )
    figure.savefig(
        output_root / f"s{subject}_spatial_forward_patterns_physical_views.png",
        dpi=200,
    )
    plt.close(figure)

    figure = plt.figure(figsize=(18, 4.3), constrained_layout=True)
    image = None
    for column, (finger, values) in enumerate(
        zip(FINGER_NAMES, physical_maps, strict=True), start=1
    ):
        axis = figure.add_subplot(1, 5, column, projection="3d")
        image = axis.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=values,
            s=20.0 + 32.0 * values,
            cmap="magma",
            vmin=0.0,
            vmax=physical_limit,
            depthshade=False,
        )
        top = np.argsort(values)[-5:][::-1]
        for index in top:
            axis.text(
                xyz[index, 0],
                xyz[index, 1],
                xyz[index, 2],
                str(retained[index]),
                fontsize=7,
            )
        axis.set_title(finger, fontsize=10)
        axis.set_xlabel("x", fontsize=7)
        axis.set_ylabel("y", fontsize=7)
        axis.set_zlabel("z", fontsize=7)
        axis.tick_params(labelsize=6)
        axis.view_init(elev=20, azim=-55)
    figure.colorbar(image, ax=figure.axes, label="forward-pattern mass (%)", shrink=0.72)
    figure.suptitle(
        f"Exact registered S{subject} electrode geometry (millimetres)"
    )
    figure.savefig(
        output_root / f"s{subject}_spatial_forward_patterns_physical_3d.png",
        dpi=200,
    )
    plt.close(figure)

    figure, axes = plt.subplots(5, 3, figsize=(10.5, 16), constrained_layout=True)
    for row_index, finger in enumerate(FINGER_NAMES):
        record = records[finger]
        retained = record["retained_channels"]
        for column, row_record in enumerate(record["top_spatial_rows"]):
            axis = axes[row_index, column]
            pattern = np.asarray(row_record["signed_forward_pattern"])
            plot_grid(axis, pattern / max(np.max(np.abs(pattern)), 1.0e-12), retained, geometry, signed=True)
            axis.set_title(
                f"{finger}: row {row_record['row']}\n"
                f"use={100 * row_record['importance']:.1f}%, "
                f"N_eff={row_record['effective_channel_count']:.1f}",
                fontsize=9,
            )
    figure.suptitle("Three most-used spatial rows per finger (signed covariance forward patterns)")
    figure.savefig(output_root / f"s{subject}_top_spatial_rows_signed.png", dpi=190)
    plt.close(figure)

    figure, axes = plt.subplots(5, 3, figsize=(10.8, 16), constrained_layout=True)
    image = None
    for row_index, finger in enumerate(FINGER_NAMES):
        record = records[finger]
        for column, row_record in enumerate(record["top_spatial_rows"]):
            axis = axes[row_index, column]
            pattern = np.asarray(row_record["signed_forward_pattern"])
            normalized = pattern / max(float(np.max(np.abs(pattern))), 1.0e-12)
            image = axis.scatter(
                xyz[:, 1],
                xyz[:, 2],
                c=normalized,
                s=22.0 + 150.0 * np.abs(normalized),
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                edgecolors="#334155",
                linewidths=0.3,
            )
            for index in np.argsort(np.abs(normalized))[-4:][::-1]:
                axis.text(
                    xyz[index, 1],
                    xyz[index, 2],
                    str(retained[index]),
                    ha="center",
                    va="center",
                    color="white" if abs(normalized[index]) > 0.55 else "black",
                    fontsize=7,
                    fontweight="bold",
                )
            axis.set_aspect("equal", adjustable="datalim")
            axis.tick_params(labelsize=7)
            axis.set_title(
                f"{finger}: row {row_record['row']}\n"
                f"use={100 * row_record['importance']:.1f}%",
                fontsize=9,
            )
            if row_index == len(FINGER_NAMES) - 1:
                axis.set_xlabel("registered y (mm)")
            if column == 0:
                axis.set_ylabel("registered z (mm)")
    figure.colorbar(image, ax=axes, label="signed forward pattern (row-normalized)", shrink=0.82)
    figure.suptitle(
        "Most-used spatial rows at exact registered physical positions (side view)"
    )
    figure.savefig(
        output_root / f"s{subject}_top_spatial_rows_physical_yz.png", dpi=190
    )
    plt.close(figure)

    band_matrix = np.stack([records[finger]["band_importance"] for finger in FINGER_NAMES])
    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    image = axis.imshow(100.0 * band_matrix, aspect="auto", cmap="viridis", vmin=0.0)
    axis.set_yticks(range(5), FINGER_NAMES)
    axis.set_xticks(range(8), NOMINAL_BAND_NAMES, rotation=35, ha="right")
    axis.set_xlabel("nominal frequency path")
    axis.set_ylabel("finger decoder")
    axis.set_title(
        f"Subject {subject}: trained LSTM structural use of wavelet paths"
    )
    figure.colorbar(image, ax=axis, label="importance (%)")
    figure.savefig(output_root / f"s{subject}_spectral_path_importance.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharex=True, sharey=True, constrained_layout=True)
    for axis, finger in zip(axes, FINGER_NAMES, strict=True):
        record = records[finger]
        frequency = np.asarray(record["frequency_hz"])
        density = np.asarray(record["effective_spectral_density"])
        density /= density.max()
        axis.plot(frequency, density, color="#2563eb", linewidth=1.1)
        axis.fill_between(frequency, 0, density, color="#93c5fd", alpha=0.55)
        axis.set_xlim(0, 250)
        axis.set_ylim(0, 1.03)
        axis.set_title(
            f"{finger}\ncentroid={record['effective_frequency_summary']['centroid_hz']:.0f} Hz",
            fontsize=9,
        )
        axis.set_xlabel("Hz")
    axes[0].set_ylabel("normalized decoder-weighted support")
    figure.suptitle("Actual trained filter support (0–250 Hz view)")
    figure.savefig(
        output_root / f"s{subject}_effective_frequency_support.png", dpi=200
    )
    plt.close(figure)

    overlap = np.empty((5, 5), dtype=np.float64)
    for left, left_finger in enumerate(FINGER_NAMES):
        for right, right_finger in enumerate(FINGER_NAMES):
            overlap[left, right] = safe_correlation(
                np.asarray(records[left_finger]["forward_pattern_channel_importance"]),
                np.asarray(records[right_finger]["forward_pattern_channel_importance"]),
            )
    figure, axis = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    image = axis.imshow(overlap, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(5), FINGER_NAMES, rotation=35, ha="right")
    axis.set_yticks(range(5), FINGER_NAMES)
    for row in range(5):
        for col in range(5):
            axis.text(col, row, f"{overlap[row, col]:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title(
        f"Across-finger similarity of S{subject} spatial source footprints"
    )
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.savefig(output_root / f"s{subject}_spatial_overlap.png", dpi=200)
    plt.close(figure)


def write_report(
    records: dict[str, dict[str, object]], output: Path, subject: int
) -> None:
    maximum_center_shift = max(
        float(record["maximum_path_centroid_shift_from_initial_hz"])
        for record in records.values()
    )
    maximum_kernel_change = max(
        float(record["maximum_relative_wavelet_kernel_change"])
        for record in records.values()
    )
    lines = [
        f"# Subject {subject} mechanistic interpretation: trained single-wavelet ensemble",
        "",
        f"This audit uses all six frozen seeds for each S{subject} finger from `outputs/final_single_wavelet_routes_v1/aggregate_summary.json`. It does not retrain or select models.",
        "",
        f"All retained S{subject} model inputs have exact sample-level correspondence to the registered `{REFERENCE_CODES[subject]}` recording. The uncertain inferred channels {EXCLUDED_INFERRED_CHANNELS[subject]} were excluded during preprocessing and therefore cannot affect these maps.",
        "",
        "The principal spatial map is a covariance-transformed forward pattern (the covariance between each learned component and each electrode), weighted by the final recurrent decoder's structural use of that component. This is preferable to reading a discriminative spatial-filter coefficient as though it were a source amplitude. The raw squared-filter result is retained in JSON as a sensitivity comparison.",
        "",
        "| Finger | Route | Top channels | Effective channels | Top-5 mass | Grid z | Grid p | Top bands | Spectral centroid | Seed-map PCC |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for finger in FINGER_NAMES:
        record = records[finger]
        loc = record["forward_pattern_localization"]
        route = record["route"]
        band_importance = np.asarray(record["band_importance"])
        top_bands = np.argsort(band_importance)[-3:][::-1]
        top_band_text = ", ".join(
            f"{NOMINAL_BAND_NAMES[index]} ({100 * band_importance[index]:.0f}%)"
            for index in top_bands
        )
        lines.append(
            f"| {finger} | {route['csp_mode']}, {route['recurrent_cell']} | "
            f"{', '.join(map(str, record['top_channels'][:5]))} | "
            f"{loc['effective_channel_count']:.1f} | {100 * loc['top_5_channel_mass']:.1f}% | "
            f"{loc['permutation_localization_z']:.2f} | {loc['permutation_one_sided_p_local']:.4f} | "
            f"{top_band_text} | "
            f"{record['effective_frequency_summary']['centroid_hz']:.1f} Hz | "
            f"{record['mean_pairwise_seed_map_pcc']:.3f} |"
        )
    lines += [
        "",
        "## Exact registered physical positions",
        "",
        f"The following statistics and coordinates use the registered 3-D `{REFERENCE_CODES[subject]}` electrode locations directly. Distances and permutation tests are in millimetres, not grid cells. Coordinate values are preserved in the reference coordinate system; no anatomical parcel name is inferred from the coordinates alone.",
        "",
        "| Finger | Top-contact coordinates `(channel: x, y, z mm)` | Weighted centroid `(x, y, z mm)` | Pair distance | RMS radius | 80% radius | Physical medoid | 3-D z | 3-D p |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for finger in FINGER_NAMES:
        record = records[finger]
        loc = record["forward_pattern_localization"]
        coordinates = "; ".join(
            f"{entry['channel']}: {entry['xyz'][0]:.1f}, {entry['xyz'][1]:.1f}, {entry['xyz'][2]:.1f}"
            for entry in record["top_channel_registered_coordinates"][:5]
        )
        centroid = loc["weighted_centroid_xyz"]
        lines.append(
            f"| {finger} | {coordinates} | "
            f"{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f} | "
            f"{loc['weighted_pair_distance_mm']:.1f} mm | "
            f"{loc['weighted_rms_radius_mm']:.1f} mm | "
            f"{loc['radius_80_mass_mm']:.1f} mm | "
            f"{loc['physical_medoid_channel']} | "
            f"{loc['permutation_localization_z_mm']:.2f} | "
            f"{loc['permutation_one_sided_p_local_mm']:.4f} |"
        )
    lines += [
        "",
        f"A negative spatial z means the observed map is more compact than permutations of the same weights over the retained contacts. Effective-channel count is an inverse participation ratio: lower values indicate concentration, while values approaching {len(records[FINGER_NAMES[0]]['retained_channels'])} indicate a dispersed footprint.",
        "",
        f"The trained wavelet kernels remain extremely close to their fixed biorthogonal initialization: across all 30 S{subject} checkpoints, the largest path-centroid displacement is {maximum_center_shift:.3f} Hz and the largest relative kernel change is {100 * maximum_kernel_change:.3f}%. The important learned choice is therefore mainly which paths the sparse/recurrent readout uses, not a material relocation of the wavelet passbands.",
        "",
        "The eight nominal labels are ordering aids, not assumptions. Each curve and centroid is computed from the actual learned three-layer filters by finite-difference impulse response with zero-input subtraction. The model squares and pools each path afterward, so these are carrier-frequency supports; phase sign is not preserved by the decoder.",
        "",
        "The signed top-row figure should be read for opponent structure: adjacent positive and negative lobes suggest a local spatial derivative, while far-separated lobes or broad same-sign patterns indicate distributed mixing/common-source suppression.",
    ]
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument(
        "--route-summary",
        type=Path,
        default=Path("outputs/final_single_wavelet_routes_v1/aggregate_summary.json"),
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=Path("outputs/channel_geometry_inference_v1"),
    )
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mechanistic_single_wavelet_all_subjects_v1"),
    )
    parser.add_argument("--sampling-rate-hz", type=float, default=1000.0)
    parser.add_argument("--fft-size", type=int, default=8192)
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()

    subject = args.subject
    output_root = args.output_root / f"sub{subject}"
    output_root.mkdir(parents=True, exist_ok=True)
    route_summary = json.loads(args.route_summary.read_text())
    geometry = load_geometry(
        args.geometry_root / f"subject_{subject}_channel_map.csv"
    )
    metadata = json.loads(
        (args.prepared_root / f"sub{subject}" / "metadata.json").read_text()
    )
    retained_channels = [int(value) for value in metadata["retained_channels_one_based"]]
    if not retained_channels:
        raise RuntimeError(f"S{subject} has no retained channels")
    if not all(geometry[channel]["coordinate_status"] == "registered" for channel in retained_channels):
        raise RuntimeError(f"an S{subject} model input lacks a registered coordinate")
    grid = np.asarray([geometry[channel]["grid"] for channel in retained_channels])
    xyz = np.asarray([geometry[channel]["xyz"] for channel in retained_channels])
    covariance = streaming_covariance(
        args.prepared_root / f"sub{subject}" / "train_ecog.npy"
    )
    reference_wavelet = WaveletPacketEnergy(
        wavelet="bior6.8",
        levels=3,
        kernel_size=17,
        trainable=True,
        padding_mode="constant",
        energy_window_samples=40,
        energy_stride_samples=40,
        tap_resample_up=5,
        tap_resample_down=2,
    )
    reference_state = {
        f"wavelet.{key}": value.detach().cpu()
        for key, value in reference_wavelet.state_dict().items()
    }
    rng = np.random.default_rng(20260909 + subject)
    records: dict[str, dict[str, object]] = {}

    for finger in FINGER_NAMES:
        pair = route_summary["pairs"][f"S{subject}_{finger}"]
        route = pair["route"]
        root = Path(pair["ensemble_root"])
        seeds = [int(value) for value in route["seeds"]]
        scale = float(route["tap_resample_up"]) / float(route["tap_resample_down"])
        dilations = (1, int(round(2 * scale)), int(round(4 * scale)))
        seed_patterns = []
        seed_filters = []
        seed_bands = []
        seed_lags = []
        seed_spectra = []
        seed_kernel_changes = []
        seed_centroid_shifts = []
        seed_row_records: list[dict[str, object]] = []
        frequency = None
        reference_path_summaries = None

        for seed in seeds:
            checkpoint = root / f"sub{subject}" / finger / f"seed{seed}" / "model.pt"
            state = load_state(checkpoint)
            filters = state["spatial.weight"].detach().cpu().numpy()[:, :, 0]
            if filters.shape[1] != len(retained_channels):
                raise RuntimeError(f"{checkpoint} has an unexpected channel count")
            patterns = forward_patterns(filters, covariance)
            component, band, lag, feature_importance = structural_feature_importance(state)
            row_importance, channel_pattern, channel_filter = row_and_channel_importance(
                patterns, filters, component, feature_importance
            )
            band_importance = np.bincount(
                band, weights=feature_importance, minlength=PATH_COUNT
            )
            lag_importance = np.bincount(
                lag, weights=feature_importance, minlength=HISTORY_BINS
            )
            order = state["frequency_order"].detach().cpu().numpy().astype(np.int64)
            this_frequency, path_power = frequency_response(
                state,
                order,
                args.sampling_rate_hz,
                args.fft_size,
                dilations,
            )
            reference_frequency, reference_power = frequency_response(
                reference_state,
                order,
                args.sampling_rate_hz,
                args.fft_size,
                dilations,
            )
            if not np.array_equal(this_frequency, reference_frequency):
                raise RuntimeError("trained and initial wavelet frequency grids differ")
            current_path_summaries = [
                frequency_summary(this_frequency, path_power[index])
                for index in range(PATH_COUNT)
            ]
            reference_path_summaries = [
                frequency_summary(reference_frequency, reference_power[index])
                for index in range(PATH_COUNT)
            ]
            seed_centroid_shifts.append(
                [
                    current_path_summaries[index]["centroid_hz"]
                    - reference_path_summaries[index]["centroid_hz"]
                    for index in range(PATH_COUNT)
                ]
            )
            seed_kernel_changes.append(
                [
                    float(
                        torch.linalg.vector_norm(
                            state[f"wavelet.layers.{level}.weight"].detach().cpu()
                            - reference_state[f"wavelet.layers.{level}.weight"]
                        )
                        / torch.linalg.vector_norm(
                            reference_state[f"wavelet.layers.{level}.weight"]
                        ).clamp_min(1.0e-12)
                    )
                    for level in range(3)
                ]
            )
            effective_spectrum = band_importance @ path_power
            effective_spectrum /= effective_spectrum.sum()
            frequency = this_frequency
            seed_patterns.append(channel_pattern)
            seed_filters.append(channel_filter)
            seed_bands.append(band_importance)
            seed_lags.append(lag_importance)
            seed_spectra.append(effective_spectrum)
            seed_row_records.append(
                {
                    "patterns": patterns,
                    "filters": filters,
                    "importance": row_importance,
                    "path_frequency_summaries": current_path_summaries,
                }
            )

        pattern_mean = np.mean(seed_patterns, axis=0)
        pattern_mean /= pattern_mean.sum()
        filter_mean = np.mean(seed_filters, axis=0)
        filter_mean /= filter_mean.sum()
        band_mean = np.mean(seed_bands, axis=0)
        band_mean /= band_mean.sum()
        lag_mean = np.mean(seed_lags, axis=0)
        lag_mean /= lag_mean.sum()
        spectrum_mean = np.mean(seed_spectra, axis=0)
        spectrum_mean /= spectrum_mean.sum()
        seed_correlations = [
            safe_correlation(seed_patterns[left], seed_patterns[right])
            for left in range(len(seeds))
            for right in range(left + 1, len(seeds))
        ]
        localization = localization_metrics(
            pattern_mean, grid, xyz, rng, args.permutations
        )
        filter_localization = localization_metrics(
            filter_mean, grid, xyz, rng, args.permutations
        )
        localization["medoid_channel"] = retained_channels[
            int(localization.pop("medoid_index"))
        ]
        localization["physical_medoid_channel"] = retained_channels[
            int(localization.pop("physical_medoid_index"))
        ]
        filter_localization["medoid_channel"] = retained_channels[
            int(filter_localization.pop("medoid_index"))
        ]
        filter_localization["physical_medoid_channel"] = retained_channels[
            int(filter_localization.pop("physical_medoid_index"))
        ]
        top_channels = np.argsort(pattern_mean)[::-1]

        # Spatial row identities are stable because each seed starts from the
        # same deterministic ICA/CSP/LARS solution. Average signed patterns and
        # row use across seeds before selecting rows for display.
        row_count = seed_row_records[0]["patterns"].shape[0]
        if not all(record["patterns"].shape[0] == row_count for record in seed_row_records):
            raise RuntimeError(
                f"S{subject} {finger} has inconsistent spatial-row counts across seeds"
            )
        mean_rows = np.mean([record["patterns"] for record in seed_row_records], axis=0)
        mean_row_importance = np.mean(
            [record["importance"] for record in seed_row_records], axis=0
        )
        top_rows = np.argsort(mean_row_importance)[-3:][::-1]
        top_row_records = []
        for row_index in top_rows:
            signed = mean_rows[row_index]
            power = signed**2
            power /= max(power.sum(), 1.0e-12)
            top_row_records.append(
                {
                    "row": int(row_index),
                    "importance": float(mean_row_importance[row_index]),
                    "effective_channel_count": float(1.0 / np.sum(power**2)),
                    "top_absolute_channels": [
                        retained_channels[index]
                        for index in np.argsort(np.abs(signed))[-6:][::-1]
                    ],
                    "signed_forward_pattern": signed.tolist(),
                }
            )

        record = {
            "subject": subject,
            "finger": finger,
            "ensemble_root": str(root),
            "seeds": seeds,
            "route": route,
            "ensemble_released_test_pcc_descriptive": pair.get(
                "mean_prediction_pcc", pair.get("ensemble_pcc")
            ),
            "retained_channels": retained_channels,
            "all_retained_channels_exactly_registered": True,
            "excluded_inferred_channels": EXCLUDED_INFERRED_CHANNELS[subject],
            "forward_pattern_channel_importance": pattern_mean.tolist(),
            "spatial_filter_channel_importance": filter_mean.tolist(),
            "forward_pattern_vs_filter_map_pcc": safe_correlation(pattern_mean, filter_mean),
            "top_channels": [retained_channels[index] for index in top_channels[:10]],
            "top_channel_mass": [float(pattern_mean[index]) for index in top_channels[:10]],
            "top_channel_registered_coordinates": [
                {
                    "channel": retained_channels[index],
                    "grid_row_col": geometry[retained_channels[index]]["grid"]
                    .astype(int)
                    .tolist(),
                    "xyz": geometry[retained_channels[index]]["xyz"].tolist(),
                }
                for index in top_channels[:10]
            ],
            "forward_pattern_localization": localization,
            "spatial_filter_localization": filter_localization,
            "mean_pairwise_seed_map_pcc": float(np.mean(seed_correlations)),
            "minimum_pairwise_seed_map_pcc": float(np.min(seed_correlations)),
            "top_spatial_rows": top_row_records,
            "band_names": list(NOMINAL_BAND_NAMES),
            "band_importance": band_mean.tolist(),
            "history_time_seconds": (
                (np.arange(HISTORY_BINS) - (HISTORY_BINS - 1)) / TARGET_RATE_HZ
            ).tolist(),
            "history_importance": lag_mean.tolist(),
            "frequency_hz": frequency.tolist(),
            "effective_spectral_density": spectrum_mean.tolist(),
            "effective_frequency_summary": frequency_summary(frequency, spectrum_mean),
            "path_frequency_summary_by_seed": [
                record["path_frequency_summaries"] for record in seed_row_records
            ],
            "initial_path_frequency_summaries": reference_path_summaries,
            "path_centroid_shift_from_initial_hz_by_seed": seed_centroid_shifts,
            "relative_wavelet_kernel_change_by_seed": seed_kernel_changes,
            "maximum_path_centroid_shift_from_initial_hz": float(
                np.max(np.abs(seed_centroid_shifts))
            ),
            "maximum_relative_wavelet_kernel_change": float(
                np.max(seed_kernel_changes)
            ),
            "definitions": {
                "feature_importance": (
                    "Frobenius magnitude of all four recurrent input-gate blocks, weighted "
                    "by the trained output connection of each hidden unit"
                ),
                "forward_pattern": (
                    "development ECoG covariance times each spatial filter, normalized by "
                    "that component's variance; squared within row and weighted by feature use"
                ),
                "frequency_response": (
                    "finite-difference impulse response of the actual trained filters after "
                    "zero-input subtraction; packet paths reordered to nominal low-to-high order"
                ),
            },
        }
        records[finger] = record
        destination = output_root / finger
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"S{subject} {finger}: top={record['top_channels'][:5]} "
            f"N_eff={localization['effective_channel_count']:.2f} "
            f"z={localization['permutation_localization_z']:.2f}",
            flush=True,
        )

    (output_root / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    render_figures(records, geometry, output_root, subject)
    write_report(records, output_root / "report.md", subject)


if __name__ == "__main__":
    main()
