#!/usr/bin/env python3
"""Audit CSP decoder alignment and feature-count choices per finger.

The feature window ending at bin ``e`` predicts target bin ``e + shift``.
Positive shifts are causal forecasts; negative shifts use ECoG recorded after
the target and are included only as a diagnostic for protocol mismatches.
All model choices are selected on the chronological validation segment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_ridge_target_variants import (
    correlation_screen,
    lagged,
    pearson,
    ridge_fit,
    ridge_predict,
)
from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def aligned_rows(
    features: np.ndarray,
    target: np.ndarray,
    history: int,
    shift: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lagged features, targets, and target-bin indices for one shift."""
    windows = lagged(features, history)
    feature_end = np.arange(history - 1, features.shape[0])
    target_index = feature_end + shift
    valid = (target_index >= 0) & (target_index < target.shape[0])
    return windows[valid], target[target_index[valid]], target_index[valid]


def fill_edges(values: np.ndarray) -> np.ndarray:
    """Nearest-fill the few alignment edge bins left without a prediction."""
    result = values.copy()
    for finger in range(result.shape[1]):
        known = np.flatnonzero(np.isfinite(result[:, finger]))
        if known.size == 0:
            result[:, finger] = 0.0
            continue
        result[: known[0], finger] = result[known[0], finger]
        result[known[-1] + 1 :, finger] = result[known[-1], finger]
        missing = np.flatnonzero(~np.isfinite(result[:, finger]))
        if missing.size:
            result[missing, finger] = np.interp(
                missing,
                known,
                result[known, finger],
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--csp-root", type=Path, default=Path("outputs/csp_ridge_besttarget_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/csp_alignment_v1"))
    parser.add_argument("--target", required=True)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--min-shift", type=int, default=-20)
    parser.add_argument("--max-shift", type=int, default=20)
    parser.add_argument("--shift-step", type=int, default=1)
    parser.add_argument("--top-features", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    prepared = args.prepared_root / f"sub{args.subject}"
    csp = args.csp_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((prepared / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    train_energy = np.load(csp / "train_csp_energy.npy")
    test_energy = np.load(csp / "test_csp_energy.npy")
    cleaned = np.load(prepared / f"train_glove_{args.target}.npy")
    raw_train = np.load(prepared / "train_glove_25hz_raw.npy")
    raw_test = np.load(prepared / "test_glove_25hz_raw.npy")
    alphas = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

    best: list[dict[str, object] | None] = [None] * cleaned.shape[1]
    candidates: list[dict[str, object]] = []
    if args.shift_step < 1:
        raise ValueError("shift-step must be positive")
    for shift in range(args.min_shift, args.max_shift + 1, args.shift_step):
        x_all, y_all, target_index = aligned_rows(
            train_energy,
            cleaned,
            args.history,
            shift,
        )
        feature_end = target_index - shift
        train_mask = (target_index < split) & (feature_end < split)
        validation_mask = target_index >= split
        train_x = x_all[train_mask]
        train_y = y_all[train_mask]
        validation_x = x_all[validation_mask]
        validation_index = target_index[validation_mask]
        test_x, _, test_index = aligned_rows(
            test_energy,
            raw_test,
            args.history,
            shift,
        )
        inner_stop = int(round(0.8 * train_x.shape[0]))

        for top_count in args.top_features:
            per_finger: dict[str, object] = {}
            for finger, name in enumerate(FINGER_NAMES):
                selected = correlation_screen(
                    train_x[:inner_stop],
                    train_y[:inner_stop, finger],
                    min(top_count, train_x.shape[1]),
                )
                best_alpha = alphas[0]
                best_inner = -float("inf")
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
                    if score > best_inner:
                        best_inner = score
                        best_alpha = alpha
                fit = ridge_fit(
                    train_x[:, selected],
                    train_y[:, finger],
                    best_alpha,
                    device,
                )
                validation_prediction = ridge_predict(validation_x[:, selected], fit)
                test_prediction = ridge_predict(test_x[:, selected], fit)
                validation_raw_r = pearson(
                    validation_prediction,
                    raw_train[validation_index, finger],
                )
                test_raw_r = pearson(test_prediction, raw_test[test_index, finger])
                record: dict[str, object] = {
                    "shift_bins": shift,
                    "shift_seconds": shift / 25.0,
                    "causal": shift >= 0,
                    "top_features": int(min(top_count, train_x.shape[1])),
                    "alpha": best_alpha,
                    "inner_cleaned_r": best_inner,
                    "validation_raw_r": validation_raw_r,
                    "test_raw_r": test_raw_r,
                }
                per_finger[name] = record
                if best[finger] is None or validation_raw_r > float(best[finger]["validation_raw_r"]):
                    best[finger] = {
                        **record,
                        "validation_index": validation_index.copy(),
                        "validation_prediction": validation_prediction.astype(np.float32),
                        "test_index": test_index.copy(),
                        "test_prediction": test_prediction.astype(np.float32),
                    }
            candidates.append(
                {
                    "shift_bins": shift,
                    "top_features": int(min(top_count, train_x.shape[1])),
                    "per_finger": per_finger,
                }
            )
        print(f"subject={args.subject} shift={shift:+d} complete", flush=True)

    validation_aligned = np.full((raw_train.shape[0] - split, cleaned.shape[1]), np.nan, dtype=np.float32)
    test_aligned = np.full_like(raw_test, np.nan, dtype=np.float32)
    compact_best: dict[str, object] = {}
    for finger, name in enumerate(FINGER_NAMES):
        chosen = best[finger]
        assert chosen is not None
        validation_aligned[
            np.asarray(chosen.pop("validation_index"), dtype=np.int64) - split,
            finger,
        ] = np.asarray(chosen.pop("validation_prediction"), dtype=np.float32)
        test_aligned[np.asarray(chosen.pop("test_index"), dtype=np.int64), finger] = np.asarray(
            chosen.pop("test_prediction"), dtype=np.float32
        )
        compact_best[name] = chosen
    validation_aligned = fill_edges(validation_aligned)
    test_aligned = fill_edges(test_aligned)

    report = {
        "subject": args.subject,
        "target": args.target,
        "selection": "per-finger shift and feature count selected on chronological validation raw PCC",
        "shift_definition": "feature window ending at e predicts target e + shift; negative is noncausal diagnostic",
        "best_per_finger": compact_best,
        "validation_raw_metrics": trajectory_metrics(validation_aligned, raw_train[split:]),
        "test_raw_metrics": trajectory_metrics(test_aligned, raw_test),
        "candidates": candidates,
    }
    np.save(output / "validation_prediction.npy", validation_aligned, allow_pickle=False)
    np.save(output / "test_prediction.npy", test_aligned, allow_pickle=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["best_per_finger"], indent=2), flush=True)
    print(json.dumps(report["test_raw_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
