#!/usr/bin/env python3
"""Plot a candidate OOF prediction against an explicitly fixed reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cross_validate_single_wavelet import validation_metrics


def load_vector(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.float64).squeeze()
    if values.ndim != 1:
        raise ValueError(f"{path} must contain one prediction vector")
    return values


def contiguous_intervals(mask: np.ndarray) -> list[list[int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [[int(start), int(stop)] for start, stop in edges.reshape(-1, 2)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--raw-target",
        type=Path,
        help="optional raw glove vector or matrix used for paper-comparable PCC",
    )
    parser.add_argument("--raw-offset", type=int, default=0)
    parser.add_argument("--raw-column", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="JSON destination; defaults to the plot path with a .json suffix",
    )
    parser.add_argument("--title", default="fixed-reference OOF comparison")
    parser.add_argument("--sampling-rate", type=float, default=25.0)
    parser.add_argument("--window-bins", type=int, default=1250)
    args = parser.parse_args()

    target = load_vector(args.target)
    reference = load_vector(args.reference)
    candidate = load_vector(args.candidate)
    if target.shape != reference.shape or target.shape != candidate.shape:
        raise ValueError("target, reference, and candidate shapes must match")
    observed = np.isfinite(target) & np.isfinite(reference) & np.isfinite(candidate)
    if not observed.any():
        raise ValueError("comparison arrays have no jointly observed bins")
    start = int(np.flatnonzero(observed)[0])
    stop = min(target.size, start + args.window_bins)
    x = np.arange(start, stop) / args.sampling_rate

    if args.raw_target is None:
        raw = target
    else:
        raw_values = np.asarray(np.load(args.raw_target), dtype=np.float64)
        raw = (
            raw_values[:, args.raw_column]
            if raw_values.ndim == 2
            else raw_values.squeeze()
        )
        raw = raw[args.raw_offset : args.raw_offset + target.size]
        if raw.shape != target.shape:
            raise ValueError("raw target does not cover the comparison vectors")
    intervals = contiguous_intervals(observed)
    metrics = {
        "reference": validation_metrics(
            reference, target, raw, intervals, movement_threshold=0.1, rest_threshold=0.05
        ),
        "candidate": validation_metrics(
            candidate, target, raw, intervals, movement_threshold=0.1, rest_threshold=0.05
        ),
    }

    figure, axis = plt.subplots(figsize=(14, 4.8), constrained_layout=True)
    axis.plot(x, target[start:stop], color="black", linewidth=1.0, label="target")
    axis.plot(
        x,
        reference[start:stop],
        linewidth=0.8,
        alpha=0.75,
        label="fixed old baseline",
    )
    axis.plot(x, candidate[start:stop], linewidth=0.9, label="candidate")
    axis.set_title(args.title)
    axis.set_xlabel("original recording time (s)")
    axis.set_ylabel("movement scale")
    axis.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    metrics_output = args.metrics_output or args.output.with_suffix(".json")
    metrics_output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
