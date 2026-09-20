"""Infer BCI Competition IV Dataset 4 channel-to-grid correspondence.

The competition release deliberately scrambled channel order.  This script
matches the released recordings to the anatomically registered recordings in
Kai Miller's public finger-flexion library, then uses spatial correlation to
place any channels that were excluded from the registered library copy.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata


SUBJECT_TO_REFERENCE = {1: "zt", 2: "bp", 3: "cc"}
GRID_SHAPES = {"zt": (8, 8), "bp": (6, 8), "cc": (8, 8)}

# Zero-based grid locations in the same order as rows of each reference file.
# The source locations preserve acquisition-grid order after bad channels were
# removed.  Gaps below are also visible in the 3-D coordinate spacing.
REFERENCE_GRID_CELLS = {
    "bp": (
        [(0, column) for column in range(8)]
        + [(1, 0), (1, 1)]
        + [(1, column) for column in range(3, 8)]
        + [(row, column) for row in range(2, 5) for column in range(8)]
        + [(5, column) for column in range(1, 8)]
    ),
    "cc": (
        [(row, column) for row in range(2) for column in range(8)]
        + [(2, column) for column in range(1, 8)]
        + [(row, column) for row in range(3, 8) for column in range(8)]
    ),
    "zt": (
        [(0, column) for column in range(2, 8)]
        + [(row, column) for row in (1, 2) for column in range(8)]
        + [(3, column) for column in range(8) if column != 6]
        + [(row, column) for row in range(4, 8) for column in range(8)]
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--competition-root",
        type=Path,
        default=Path("data/raw/bci_competition_iv_ds4/mat"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("data/reference/fingerflex/fingerflex/data"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/channel_geometry_inference_v1"),
    )
    parser.add_argument("--reference-offset", type=int, default=5000)
    parser.add_argument("--match-stride", type=int, default=100)
    parser.add_argument("--correlation-stride", type=int, default=10)
    parser.add_argument("--bootstrap-blocks", type=int, default=12)
    return parser.parse_args()


def normalized_columns(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    output = output - output.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(output, axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    return output / scale


def load_competition(path: Path) -> np.ndarray:
    payload = loadmat(path, variable_names=["train_data", "test_data"])
    return np.concatenate([payload["train_data"], payload["test_data"]], axis=0)


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = loadmat(path, variable_names=["data", "locs"])
    return payload["data"], np.asarray(payload["locs"], dtype=np.float64)


def match_channels(
    competition: np.ndarray,
    reference: np.ndarray,
    reference_offset: int,
    stride: int,
) -> tuple[dict[int, int], np.ndarray]:
    stop = reference_offset + competition.shape[0]
    if stop > reference.shape[0]:
        raise ValueError("reference recording is too short for requested alignment")
    left = normalized_columns(competition[::stride])
    right = normalized_columns(reference[reference_offset:stop:stride])
    correlation = left.T @ right
    rows, columns = linear_sum_assignment(-np.abs(correlation))
    mapping = {int(row): int(column) for row, column in zip(rows, columns)}
    return mapping, correlation


def grid_neighbors(cell: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    row, column = cell
    output = []
    for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        candidate = row + delta_row, column + delta_column
        if 0 <= candidate[0] < shape[0] and 0 <= candidate[1] < shape[1]:
            output.append(candidate)
    return output


def car_correlation(values: np.ndarray, stride: int) -> np.ndarray:
    sampled = np.asarray(values[::stride], dtype=np.float64)
    sampled -= sampled.mean(axis=1, keepdims=True)
    sampled -= sampled.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(sampled, axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    correlation = (sampled / scale).T @ (sampled / scale)
    np.fill_diagonal(correlation, 1.0)
    return np.nan_to_num(correlation)


def edge_zscore(
    correlation: np.ndarray,
    unknown_channel: int,
    neighbor_channel: int,
    mapped_channel_to_cell: dict[int, tuple[int, int]],
    candidate_neighbors: set[tuple[int, int]],
) -> float:
    baseline_channels = [
        channel
        for channel, cell in mapped_channel_to_cell.items()
        if channel != neighbor_channel and cell not in candidate_neighbors
    ]
    baseline = correlation[neighbor_channel, baseline_channels]
    center = float(np.median(baseline))
    scale = float(1.4826 * np.median(np.abs(baseline - center)))
    if scale < 1.0e-6:
        scale = float(np.std(baseline))
    if scale < 1.0e-6:
        scale = 1.0
    return float((correlation[unknown_channel, neighbor_channel] - center) / scale)


def candidate_scores(
    correlation: np.ndarray,
    unknown_channels: list[int],
    candidate_cells: list[tuple[int, int]],
    mapped_channel_to_cell: dict[int, tuple[int, int]],
    shape: tuple[int, int],
) -> np.ndarray:
    channel_by_cell = {cell: channel for channel, cell in mapped_channel_to_cell.items()}
    scores = np.full((len(unknown_channels), len(candidate_cells)), -np.inf)
    for row, unknown in enumerate(unknown_channels):
        for column, candidate in enumerate(candidate_cells):
            neighbors = set(grid_neighbors(candidate, shape))
            neighbor_channels = [channel_by_cell[cell] for cell in neighbors if cell in channel_by_cell]
            if not neighbor_channels:
                continue
            scores[row, column] = float(
                np.mean(
                    [
                        edge_zscore(
                            correlation,
                            unknown,
                            neighbor,
                            mapped_channel_to_cell,
                            neighbors,
                        )
                        for neighbor in neighbor_channels
                    ]
                )
            )
    return scores


def best_assignment(scores: np.ndarray) -> dict[int, int]:
    rows, columns = linear_sum_assignment(-scores)
    return {int(row): int(column) for row, column in zip(rows, columns)}


def bootstrap_assignments(
    competition: np.ndarray,
    unknown_channels: list[int],
    candidate_cells: list[tuple[int, int]],
    mapped_channel_to_cell: dict[int, tuple[int, int]],
    shape: tuple[int, int],
    stride: int,
    blocks: int,
) -> tuple[np.ndarray, dict[int, Counter[int]]]:
    full_correlation = car_correlation(competition, stride)
    full_scores = candidate_scores(
        full_correlation,
        unknown_channels,
        candidate_cells,
        mapped_channel_to_cell,
        shape,
    )
    votes: dict[int, Counter[int]] = defaultdict(Counter)
    boundaries = np.linspace(0, competition.shape[0], blocks + 1, dtype=int)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        block_correlation = car_correlation(competition[start:stop], stride)
        block_scores = candidate_scores(
            block_correlation,
            unknown_channels,
            candidate_cells,
            mapped_channel_to_cell,
            shape,
        )
        for unknown_row, candidate_column in best_assignment(block_scores).items():
            votes[unknown_row][candidate_column] += 1
    return full_scores, votes


def adjacency_statistics(
    correlation: np.ndarray,
    mapped_channel_to_cell: dict[int, tuple[int, int]],
) -> dict[str, float | int]:
    adjacent = []
    nonadjacent = []
    items = sorted(mapped_channel_to_cell.items())
    for left_index, (left_channel, left_cell) in enumerate(items):
        for right_channel, right_cell in items[left_index + 1 :]:
            distance = abs(left_cell[0] - right_cell[0]) + abs(left_cell[1] - right_cell[1])
            value = float(correlation[left_channel, right_channel])
            if distance == 1:
                adjacent.append(value)
            elif distance > 1:
                nonadjacent.append(value)
    labels = np.r_[np.ones(len(adjacent)), np.zeros(len(nonadjacent))]
    values = np.r_[adjacent, nonadjacent]
    ranks = rankdata(values)
    positive_rank_sum = float(ranks[labels == 1].sum())
    n_positive = len(adjacent)
    n_negative = len(nonadjacent)
    auc = (
        positive_rank_sum - n_positive * (n_positive + 1) / 2
    ) / (n_positive * n_negative)
    return {
        "adjacent_pair_count": n_positive,
        "nonadjacent_pair_count": n_negative,
        "median_adjacent_correlation": float(np.median(adjacent)),
        "median_nonadjacent_correlation": float(np.median(nonadjacent)),
        "adjacency_auc_from_correlation": float(auc),
    }


def coordinate_model(cells: Iterable[tuple[int, int]], locs: np.ndarray) -> np.ndarray:
    design = []
    for row, column in cells:
        design.append([1.0, row, column, row * column, row * row, column * column])
    coefficients, *_ = np.linalg.lstsq(np.asarray(design), locs, rcond=None)
    return coefficients


def predict_coordinate(coefficients: np.ndarray, cell: tuple[int, int]) -> np.ndarray:
    row, column = cell
    design = np.asarray([1.0, row, column, row * column, row * row, column * column])
    return design @ coefficients


def confidence_label(
    probability: float,
    selected_score: float,
    structurally_unique: bool = False,
) -> str:
    if structurally_unique:
        return "high"
    if probability >= 0.90 and selected_score > 0:
        return "high"
    if probability >= 0.90:
        return "medium"
    if probability >= 0.70:
        return "medium"
    return "low"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_grid_maps(subject_payloads: dict[int, dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    for axis, (subject, payload) in zip(axes, subject_payloads.items()):
        shape = tuple(payload["grid_shape"])
        axis.set_xlim(-0.6, shape[1] - 0.4)
        axis.set_ylim(shape[0] - 0.4, -0.6)
        axis.set_aspect("equal")
        axis.set_xticks(range(shape[1]), labels=range(1, shape[1] + 1))
        axis.set_yticks(range(shape[0]), labels=range(1, shape[0] + 1))
        axis.set_xlabel("grid column")
        axis.set_ylabel("grid row")
        axis.set_title(
            f"Competition Subject {subject} ({payload['reference_code']})\n"
            f"label = released data channel"
        )
        for row in range(shape[0]):
            for column in range(shape[1]):
                cell = f"{row},{column}"
                entry = payload["by_cell"].get(cell)
                if entry is None:
                    face = "#e5e7eb"
                    label = "-"
                    text_color = "#6b7280"
                else:
                    face = "#d97706" if entry["match_method"] == "correlation_inference" else "#2563eb"
                    label = str(entry["data_channel_1based"])
                    text_color = "white"
                circle = plt.Circle((column, row), 0.36, facecolor=face, edgecolor="white", linewidth=1.2)
                axis.add_patch(circle)
                axis.text(column, row, label, ha="center", va="center", fontsize=8, color=text_color)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle("Recovered physical-grid correspondence", fontsize=15, fontweight="bold")
    figure.legend(
        handles=[
            Patch(facecolor="#2563eb", label="exact sample match"),
            Patch(facecolor="#d97706", label="inferred grid cell"),
            Patch(facecolor="#e5e7eb", label="not present in release"),
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    figure.savefig(path, dpi=220)
    plt.close(figure)


def render_correlation_reordering(subject_payloads: dict[int, dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(11, 15), constrained_layout=True)
    for row, (subject, payload) in enumerate(subject_payloads.items()):
        correlation = payload["correlation_matrix"]
        released = np.abs(correlation)
        order = payload["physical_channel_order_zero_based"]
        physical = released[np.ix_(order, order)]
        for column, (matrix, title) in enumerate(
            ((released, "released channel order"), (physical, "recovered physical-grid order"))
        ):
            axis = axes[row, column]
            image = axis.imshow(matrix, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            axis.set_title(f"Subject {subject}: {title}")
            axis.set_xlabel("channel index")
            axis.set_ylabel("channel index")
    figure.colorbar(image, ax=axes, shrink=0.75, label="absolute CAR correlation")
    figure.suptitle("Spatial structure appears after channel reordering", fontsize=15, fontweight="bold")
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    subject_payloads: dict[int, dict[str, object]] = {}
    summary_subjects: dict[str, object] = {}

    for subject, reference_code in SUBJECT_TO_REFERENCE.items():
        competition = load_competition(args.competition_root / f"sub{subject}_comp.mat")
        reference, locs = load_reference(
            args.reference_root / reference_code / f"{reference_code}_fingerflex.mat"
        )
        reference_cells = list(REFERENCE_GRID_CELLS[reference_code])
        shape = GRID_SHAPES[reference_code]
        if len(reference_cells) != reference.shape[1] or len(locs) != reference.shape[1]:
            raise ValueError(f"reference geometry size mismatch for {reference_code}")

        mapping, cross_correlation = match_channels(
            competition,
            reference,
            args.reference_offset,
            args.match_stride,
        )
        match_values = {
            channel: float(cross_correlation[channel, reference_channel])
            for channel, reference_channel in mapping.items()
        }
        if min(abs(value) for value in match_values.values()) < 0.999999:
            raise RuntimeError(f"non-exact reference match for subject {subject}")

        mapped_channel_to_cell = {
            channel: reference_cells[reference_channel]
            for channel, reference_channel in mapping.items()
        }
        unknown_channels = sorted(set(range(competition.shape[1])) - set(mapping))
        all_cells = [(row, column) for row in range(shape[0]) for column in range(shape[1])]
        candidate_cells = sorted(set(all_cells) - set(reference_cells))

        full_scores, votes = bootstrap_assignments(
            competition,
            unknown_channels,
            candidate_cells,
            mapped_channel_to_cell,
            shape,
            args.correlation_stride,
            args.bootstrap_blocks,
        )
        assignment_columns = best_assignment(full_scores)
        inferred_channel_to_cell = {
            unknown_channels[row]: candidate_cells[column]
            for row, column in assignment_columns.items()
        }
        correlation = car_correlation(competition, args.correlation_stride)
        spatial_statistics = adjacency_statistics(correlation, mapped_channel_to_cell)
        coordinate_coefficients = coordinate_model(reference_cells, locs)

        rows: list[dict[str, object]] = []
        by_cell: dict[str, dict[str, object]] = {}
        for channel in range(competition.shape[1]):
            if channel in mapping:
                reference_channel = mapping[channel]
                cell = reference_cells[reference_channel]
                coordinate = locs[reference_channel]
                row = {
                    "competition_subject": subject,
                    "data_channel_1based": channel + 1,
                    "reference_patient_code": reference_code,
                    "reference_channel_1based": reference_channel + 1,
                    "grid_row_1based": cell[0] + 1,
                    "grid_col_1based": cell[1] + 1,
                    "nominal_row_major_contact_1based": cell[0] * shape[1] + cell[1] + 1,
                    "x": float(coordinate[0]),
                    "y": float(coordinate[1]),
                    "z": float(coordinate[2]),
                    "coordinate_status": "registered",
                    "match_method": "exact_sample_correlation",
                    "match_correlation": match_values[channel],
                    "bootstrap_selection_probability": 1.0,
                    "confidence": "exact",
                    "inference_basis": "exact_sample_match",
                }
            else:
                cell = inferred_channel_to_cell[channel]
                unknown_row = unknown_channels.index(channel)
                candidate_column = candidate_cells.index(cell)
                probability = votes[unknown_row][candidate_column] / args.bootstrap_blocks
                selected_score = float(full_scores[unknown_row, candidate_column])
                structurally_unique = len(candidate_cells) == 1
                coordinate = predict_coordinate(coordinate_coefficients, cell)
                row = {
                    "competition_subject": subject,
                    "data_channel_1based": channel + 1,
                    "reference_patient_code": reference_code,
                    "reference_channel_1based": "",
                    "grid_row_1based": cell[0] + 1,
                    "grid_col_1based": cell[1] + 1,
                    "nominal_row_major_contact_1based": cell[0] * shape[1] + cell[1] + 1,
                    "x": float(coordinate[0]),
                    "y": float(coordinate[1]),
                    "z": float(coordinate[2]),
                    "coordinate_status": "grid_polynomial_imputation",
                    "match_method": "correlation_inference",
                    "match_correlation": "",
                    "bootstrap_selection_probability": probability,
                    "confidence": confidence_label(
                        probability,
                        selected_score,
                        structurally_unique,
                    ),
                    "inference_basis": (
                        "structural_exclusion"
                        if structurally_unique
                        else "local_correlation"
                    ),
                }
            rows.append(row)
            by_cell[f"{cell[0]},{cell[1]}"] = row

        unobserved_cells = sorted(set(candidate_cells) - set(inferred_channel_to_cell.values()))
        physical_order = [
            int(entry["data_channel_1based"]) - 1
            for _, entry in sorted(
                by_cell.items(),
                key=lambda item: tuple(map(int, item[0].split(","))),
            )
        ]
        output_csv = args.output_root / f"subject_{subject}_channel_map.csv"
        write_csv(output_csv, rows)

        score_payload = {
            str(channel + 1): {
                f"r{cell[0] + 1}c{cell[1] + 1}": float(full_scores[row_index, column_index])
                for column_index, cell in enumerate(candidate_cells)
            }
            for row_index, channel in enumerate(unknown_channels)
        }
        vote_payload = {
            str(channel + 1): {
                f"r{candidate_cells[column][0] + 1}c{candidate_cells[column][1] + 1}": count
                for column, count in sorted(votes[row_index].items())
            }
            for row_index, channel in enumerate(unknown_channels)
        }
        summary_subjects[str(subject)] = {
            "reference_patient_code": reference_code,
            "competition_channel_count": int(competition.shape[1]),
            "registered_reference_channel_count": int(reference.shape[1]),
            "reference_alignment_offset_samples": args.reference_offset,
            "exact_match_count": len(mapping),
            "minimum_absolute_exact_match_correlation": float(
                min(abs(value) for value in match_values.values())
            ),
            "unknown_competition_channels_1based": [channel + 1 for channel in unknown_channels],
            "candidate_missing_grid_cells_1based": [
                [cell[0] + 1, cell[1] + 1] for cell in candidate_cells
            ],
            "inferred_unknown_channel_cells_1based": {
                str(channel + 1): [cell[0] + 1, cell[1] + 1]
                for channel, cell in inferred_channel_to_cell.items()
            },
            "inferred_channel_evidence": {
                str(row["data_channel_1based"]): {
                    "confidence": row["confidence"],
                    "inference_basis": row.get("inference_basis", "exact_sample_match"),
                    "bootstrap_selection_probability": row["bootstrap_selection_probability"],
                }
                for row in rows
                if row["match_method"] == "correlation_inference"
            },
            "unobserved_grid_cells_1based": [
                [cell[0] + 1, cell[1] + 1] for cell in unobserved_cells
            ],
            "full_data_candidate_scores": score_payload,
            "bootstrap_assignment_votes": vote_payload,
            "spatial_validation": spatial_statistics,
        }
        subject_payloads[subject] = {
            "reference_code": reference_code,
            "grid_shape": shape,
            "by_cell": by_cell,
            "correlation_matrix": correlation,
            "physical_channel_order_zero_based": physical_order,
        }

    summary = {
        "analysis": "BCI Competition IV Dataset 4 physical channel correspondence",
        "reference_source": "Kai Miller fingerflex archive, anatomically registered recordings",
        "subject_to_reference": SUBJECT_TO_REFERENCE,
        "reference_offset_samples": args.reference_offset,
        "matching_definition": (
            "Hungarian one-to-one assignment maximizing absolute sample correlation between "
            "competition data and the anatomically registered reference recording"
        ),
        "unknown_localization_definition": (
            "Assignment to reference-missing grid cells by robustly normalized CAR correlation "
            "with the cell's physical four-neighbors; stability is measured over contiguous blocks"
        ),
        "subjects": summary_subjects,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render_grid_maps(subject_payloads, args.output_root / "channel_maps.png")
    render_correlation_reordering(subject_payloads, args.output_root / "correlation_reordering.png")

    report_lines = [
        "# BCI Competition IV Dataset 4 channel geometry inference",
        "",
        "The competition release scrambled the data-channel order. The registered Stanford "
        "fingerflex archive contains the same recordings with 3-D electrode coordinates. "
        "At a 5,000-sample offset, the recordings match exactly after a one-to-one channel "
        "assignment.",
        "",
        "## Results",
        "",
    ]
    for subject, payload in summary_subjects.items():
        assignments = ", ".join(
            f"channel {channel} -> grid r{cell[0]}c{cell[1]} "
            f"({payload['inferred_channel_evidence'][channel]['confidence']}; "
            f"{payload['inferred_channel_evidence'][channel]['inference_basis'].replace('_', ' ')})"
            for channel, cell in payload["inferred_unknown_channel_cells_1based"].items()
        )
        report_lines.append(
            f"- Subject {subject} matches patient `{payload['reference_patient_code']}`: "
            f"{payload['exact_match_count']}/{payload['competition_channel_count']} channels "
            f"have exact sample correlation. Remaining {assignments}."
        )
    report_lines.extend(
        [
            "",
            "Exact matches inherit the registered 3-D coordinate. Coordinates for the few "
            "reference-excluded channels are polynomial grid imputations and must not be treated "
            "as imaging-derived locations. `nominal_row_major_contact_1based` is a grid-cell label, "
            "not a verified hospital hardware contact number.",
            "",
            "Confidence is intentionally conservative: Subject 1's remaining channel is stable "
            "across blocks but has weak absolute neighborhood fit; Subject 2's two-channel "
            "assignment is ambiguous and could be swapped; Subject 3 has only one available grid "
            "cell, so its cell is fixed by exclusion even though its coordinate is imputed.",
            "",
            "## Files",
            "",
            "- `subject_1_channel_map.csv`, `subject_2_channel_map.csv`, and "
            "  `subject_3_channel_map.csv`: released-channel correspondence.",
            "- `channel_maps.png`: physical grids labeled with released data channels.",
            "- `correlation_reordering.png`: correlation matrices before and after physical ordering.",
            "- `summary.json`: matching evidence, candidate scores, and bootstrap votes.",
            "",
            "## Sources",
            "",
            "- Tangermann et al. (2012), *Review of the BCI Competition IV*, "
            "  https://doi.org/10.3389/fnins.2012.00055",
            "- Kubanek et al. (2009), *Decoding Flexion of Individual Fingers Using "
            "  Electrocorticographic Signals in Humans*, https://doi.org/10.1088/1741-2560/6/6/066001",
            "- Miller (2019), *A library of human electrocorticographic data and analyses*, "
            "  https://doi.org/10.1038/s41562-019-0678-3",
        ]
    )
    (args.output_root / "report.md").write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
