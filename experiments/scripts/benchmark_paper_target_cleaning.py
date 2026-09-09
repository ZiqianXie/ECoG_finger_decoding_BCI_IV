#!/usr/bin/env python3
"""Benchmark the paper's threshold and winner-take-all target cleaning.

The released test trajectories are used only for final reporting. Threshold
and winner-take-all choices are ranked on the chronological validation data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_ridge_target_variants import fit_candidate, lagged


def cleaned_target(values: np.ndarray, threshold: float, winner_take_all: bool) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    result[result < threshold] = 0.0
    if winner_take_all:
        winner = np.argmax(result, axis=1)
        amplitude = result[np.arange(result.shape[0]), winner]
        result.fill(0.0)
        active = amplitude > 0.0
        result[np.flatnonzero(active), winner[active]] = amplitude[active]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--csp-root", type=Path, default=Path("outputs/csp_ridge_besttarget_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_target_cleaning_v1"))
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25],
    )
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    prepared = args.prepared_root / f"sub{args.subject}"
    csp = args.csp_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    split = int(json.loads((prepared / "metadata.json").read_text())["target_fit_samples_25hz"])
    train_energy = np.load(csp / "train_csp_energy.npy")
    test_energy = np.load(csp / "test_csp_energy.npy")
    baseline_only = np.load(prepared / "train_glove_paper_baseline_only.npy")
    raw_train = np.load(prepared / "train_glove_25hz_raw.npy")
    raw_test = np.load(prepared / "test_glove_25hz_raw.npy")
    train_x_all = lagged(train_energy, args.history)
    test_x = lagged(test_energy, args.history)
    offset = args.history - 1
    train_count = split - offset
    device = torch.device(args.device)

    results: dict[str, object] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for winner_take_all in (False, True):
        for threshold in args.thresholds:
            name = f"{'wta' if winner_take_all else 'threshold'}_{threshold:g}"
            target = cleaned_target(baseline_only, threshold, winner_take_all)
            result, validation_prediction, test_prediction = fit_candidate(
                train_x_all[:train_count],
                target[offset:split],
                train_x_all[train_count:],
                target[split:],
                raw_train[split:],
                test_x,
                raw_test[offset:],
                args.top_features,
                device,
            )
            results[name] = result
            predictions[name] = (validation_prediction, test_prediction)
            score = result["validation_raw_metrics"]["pearson_historical_four"]
            print(f"subject={args.subject} candidate={name} validation_r4={score:.4f}", flush=True)

    ranking = sorted(
        results,
        key=lambda name: results[name]["validation_raw_metrics"]["pearson_historical_four"],
        reverse=True,
    )
    best = ranking[0]
    np.save(output / "validation_prediction.npy", predictions[best][0], allow_pickle=False)
    np.save(output / "test_prediction.npy", predictions[best][1], allow_pickle=False)
    report = {
        "subject": args.subject,
        "selection": "paper baseline plus threshold/WTA selected on chronological validation raw PCC",
        "best_candidate": best,
        "ranking": ranking,
        "results": results,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"best": best, "test": results[best]["test_raw_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
