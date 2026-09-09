#!/usr/bin/env python3
"""Rank glove targets with exact 1 s ICA/wavelet windows and GPU ridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_ridge_target_variants import fit_candidate


def discover(root: Path) -> list[str]:
    names = ["raw_25hz", "paper_baseline_only"]
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
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/windowed_ica_sklearn_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/windowed_target_benchmark_v1"))
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    prepared = args.prepared_root / f"sub{args.subject}"
    feature_root = args.feature_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    split = int(json.loads((prepared / "metadata.json").read_text())["target_fit_samples_25hz"])
    offset = args.history - 1
    train_x_all = np.load(feature_root / "train_initialized_window_features.npy", mmap_mode="r")
    test_x = np.load(feature_root / "test_initialized_window_features.npy", mmap_mode="r")
    train_count = split - offset
    train_x = np.asarray(train_x_all[:train_count])
    validation_x = np.asarray(train_x_all[train_count:])
    raw_train = np.load(prepared / "train_glove_25hz_raw.npy")
    raw_test = np.load(prepared / "test_glove_25hz_raw.npy")[offset:]
    names = args.targets or discover(prepared)
    results: dict[str, object] = {}
    for name in names:
        path = (
            prepared / "train_glove_25hz_raw.npy"
            if name == "raw_25hz"
            else prepared / f"train_glove_{name}.npy"
        )
        target = np.load(path)
        result, validation_prediction, test_prediction = fit_candidate(
            train_x,
            target[offset:split],
            validation_x,
            target[split:],
            raw_train[split:],
            np.asarray(test_x),
            raw_test,
            args.top_features,
            torch.device(args.device),
        )
        results[name] = result
        candidate = output / name
        candidate.mkdir(exist_ok=True)
        np.save(candidate / "validation_prediction.npy", validation_prediction, allow_pickle=False)
        np.save(candidate / "test_prediction.npy", test_prediction, allow_pickle=False)
        print(
            f"target={name} validation_r4={result['validation_raw_metrics']['pearson_historical_four']:.4f}",
            flush=True,
        )
    ranking = sorted(
        results,
        key=lambda name: results[name]["validation_raw_metrics"]["pearson_historical_four"],
        reverse=True,
    )
    report = {
        "subject": args.subject,
        "method": "exact independent 1 s sklearn-ICA/wavelet windows plus screened ridge",
        "selection_metric": "validation raw PCC",
        "top_features": args.top_features,
        "candidate_ranking": ranking,
        "results": results,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"best={ranking[0]}", flush=True)


if __name__ == "__main__":
    main()
