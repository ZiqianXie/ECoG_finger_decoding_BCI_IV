#!/usr/bin/env python3
"""Combine per-subject single-wavelet audits into matched 15-pair figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecog_decoding.training import FINGER_NAMES


def load_geometry(path: Path) -> dict[int, np.ndarray]:
    with path.open(newline="") as handle:
        return {
            int(row["data_channel_1based"]): np.asarray(
                [float(row[axis]) for axis in ("x", "y", "z")],
                dtype=np.float64,
            )
            for row in csv.DictReader(handle)
        }


def load_records(root: Path) -> dict[tuple[int, str], dict[str, object]]:
    records: dict[tuple[int, str], dict[str, object]] = {}
    for subject in (1, 2, 3):
        payload = json.loads((root / f"sub{subject}" / "summary.json").read_text())
        for finger in FINGER_NAMES:
            records[(subject, finger)] = payload[finger]
    return records


def format_p(value: float) -> str:
    return "p<0.001" if value < 0.001 else f"p={value:.3f}"


def plot_physical_topography(
    records: dict[tuple[int, str], dict[str, object]],
    geometry_root: Path,
    output: Path,
) -> None:
    geometry = {
        subject: load_geometry(
            geometry_root / f"subject_{subject}_channel_map.csv"
        )
        for subject in (1, 2, 3)
    }
    maps = [
        100.0 * np.asarray(record["forward_pattern_channel_importance"])
        for record in records.values()
    ]
    limit = max(float(np.max(np.concatenate(maps))), 1.0e-12)
    figure = plt.figure(figsize=(18, 11), constrained_layout=True)
    image = None
    axes = []
    for subject in (1, 2, 3):
        for finger_index, finger in enumerate(FINGER_NAMES):
            record = records[(subject, finger)]
            retained = [int(value) for value in record["retained_channels"]]
            xyz = np.asarray([geometry[subject][channel] for channel in retained])
            values = 100.0 * np.asarray(
                record["forward_pattern_channel_importance"]
            )
            axis = figure.add_subplot(
                3, 5, (subject - 1) * 5 + finger_index + 1, projection="3d"
            )
            axes.append(axis)
            image = axis.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                c=values,
                s=13.0 + 24.0 * values,
                cmap="magma",
                vmin=0.0,
                vmax=limit,
                depthshade=False,
            )
            for index in np.argsort(values)[-3:][::-1]:
                axis.text(
                    xyz[index, 0],
                    xyz[index, 1],
                    xyz[index, 2],
                    str(retained[index]),
                    fontsize=6,
                )
            localization = record["forward_pattern_localization"]
            axis.set_title(
                f"S{subject} {finger}\n"
                f"distance={localization['weighted_pair_distance_mm']:.1f} mm, "
                f"{format_p(localization['permutation_one_sided_p_local_mm'])}",
                fontsize=8,
            )
            axis.set_xlabel("x", fontsize=6)
            axis.set_ylabel("y", fontsize=6)
            axis.set_zlabel("z", fontsize=6)
            axis.tick_params(labelsize=5)
            axis.view_init(elev=20, azim=-55)
    if image is not None:
        figure.colorbar(
            image,
            ax=axes,
            label="decoder-weighted forward-pattern mass (%)",
            shrink=0.68,
        )
    figure.suptitle(
        "Frozen single-wavelet reference decoders at exact registered electrode coordinates",
        fontsize=14,
    )
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_spectral_importance(
    records: dict[tuple[int, str], dict[str, object]], output: Path
) -> None:
    matrices = {
        subject: np.stack(
            [
                np.asarray(records[(subject, finger)]["band_importance"])
                for finger in FINGER_NAMES
            ]
        )
        * 100.0
        for subject in (1, 2, 3)
    }
    limit = max(float(np.max(matrix)) for matrix in matrices.values())
    names = records[(1, FINGER_NAMES[0])]["band_names"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    image = None
    for subject, axis in zip((1, 2, 3), axes, strict=True):
        image = axis.imshow(
            matrices[subject], aspect="auto", cmap="viridis", vmin=0.0, vmax=limit
        )
        axis.set_title(f"Subject {subject}")
        axis.set_yticks(range(5), FINGER_NAMES)
        axis.set_xticks(range(8), names, rotation=45, ha="right", fontsize=8)
        axis.set_xlabel("nominal frequency path")
    axes[0].set_ylabel("finger decoder")
    if image is not None:
        figure.colorbar(
            image, ax=axes, label="structural decoder importance (%)", shrink=0.82
        )
    figure.suptitle("Structural use of wavelet paths across all three subjects")
    figure.savefig(output, dpi=200)
    plt.close(figure)


def compact_summary(
    records: dict[tuple[int, str], dict[str, object]]
) -> dict[str, object]:
    pairs: dict[str, object] = {}
    for subject in (1, 2, 3):
        for finger in FINGER_NAMES:
            record = records[(subject, finger)]
            localization = record["forward_pattern_localization"]
            route = record["route"]
            pairs[f"S{subject}_{finger}"] = {
                "route": {
                    "csp_mode": route["csp_mode"],
                    "recurrent_cell": route["recurrent_cell"],
                },
                "seed_count": len(record["seeds"]),
                "released_test_pcc_descriptive": record[
                    "ensemble_released_test_pcc_descriptive"
                ],
                "top_channels": record["top_channels"][:5],
                "effective_channel_count": localization["effective_channel_count"],
                "top_5_channel_mass": localization["top_5_channel_mass"],
                "weighted_pair_distance_mm": localization[
                    "weighted_pair_distance_mm"
                ],
                "permutation_localization_z_mm": localization[
                    "permutation_localization_z_mm"
                ],
                "permutation_one_sided_p_local_mm": localization[
                    "permutation_one_sided_p_local_mm"
                ],
                "spectral_centroid_hz": record["effective_frequency_summary"][
                    "centroid_hz"
                ],
                "mean_pairwise_seed_map_pcc": record[
                    "mean_pairwise_seed_map_pcc"
                ],
                "maximum_path_centroid_shift_from_initial_hz": record[
                    "maximum_path_centroid_shift_from_initial_hz"
                ],
                "maximum_relative_wavelet_kernel_change": record[
                    "maximum_relative_wavelet_kernel_change"
                ],
            }
    return {
        "analysis": "frozen six-seed single-wavelet reference family",
        "scope": "15 subject-finger pairs; 90 checkpoints; no retraining or model selection",
        "spatial_method": (
            "covariance-transformed forward patterns weighted by downstream recurrent-decoder use"
        ),
        "localization_null": (
            "permutation of the same channel weights over exact registered 3-D coordinates"
        ),
        "pairs": pairs,
        "overall": {
            "maximum_path_centroid_shift_from_initial_hz": max(
                value["maximum_path_centroid_shift_from_initial_hz"]
                for value in pairs.values()
            ),
            "maximum_relative_wavelet_kernel_change": max(
                value["maximum_relative_wavelet_kernel_change"]
                for value in pairs.values()
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("outputs/mechanistic_single_wavelet_all_subjects_v1"),
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=Path("outputs/channel_geometry_inference_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mechanistic_single_wavelet_all_subjects_v1/combined"),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = load_records(args.summary_root)
    plot_physical_topography(
        records,
        args.geometry_root,
        args.output_root / "all_subjects_physical_forward_patterns_3d.png",
    )
    plot_spectral_importance(
        records,
        args.output_root / "all_subjects_spectral_path_importance.png",
    )
    summary = compact_summary(records)
    (args.output_root / "compact_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
