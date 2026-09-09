#!/usr/bin/env python3
"""Rank glove-baseline targets using one fixed CSP feature representation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_ridge_target_variants import fit_candidate, lagged


def discover_targets(root: Path) -> list[str]:
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
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--csp-root", type=Path, default=Path("outputs/csp_ridge_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/csp_target_benchmark_v2"))
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = args.prepared_root / f"sub{args.subject}"
    csp_root = args.csp_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    train_energy = np.load(csp_root / "train_csp_energy.npy")
    test_energy = np.load(csp_root / "test_csp_energy.npy")
    train_x_all = lagged(train_energy, args.history)
    test_x = lagged(test_energy, args.history)
    train_count = split - offset
    train_x = train_x_all[:train_count]
    validation_x = train_x_all[train_count:]
    train_raw = np.load(root / "train_glove_25hz_raw.npy")
    test_raw = np.load(root / "test_glove_25hz_raw.npy")[offset:]

    summary_path = output / "summary.json"
    results: dict[str, object] = {}
    if summary_path.exists():
        results.update(json.loads(summary_path.read_text()).get("results", {}))
    names = args.targets if args.targets else discover_targets(root)
    for name in names:
        target_path = (
            root / "train_glove_25hz_raw.npy"
            if name == "raw_25hz"
            else root / f"train_glove_{name}.npy"
        )
        target = np.load(target_path)
        result, validation_prediction, test_prediction = fit_candidate(
            train_x,
            target[offset:split],
            validation_x,
            target[split:],
            train_raw[split:],
            test_x,
            test_raw,
            args.top_features,
            torch.device(args.device),
        )
        results[name] = result
        candidate_output = output / name
        candidate_output.mkdir(exist_ok=True)
        np.save(candidate_output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
        np.save(candidate_output / "test_prediction.npy", test_prediction, allow_pickle=False)
        score = result["validation_raw_metrics"]["pearson_historical_four"]
        print(f"target={name} validation_raw_r4={score:.4f}", flush=True)

    ranking = sorted(
        results,
        key=lambda name: results[name]["validation_raw_metrics"]["pearson_historical_four"],
        reverse=True,
    )
    report = {
        "subject": args.subject,
        "method": "fixed train-only CSP representation plus screened GPU ridge",
        "selection_metric": "validation_raw_metrics.pearson_historical_four",
        "test_labels_used_for_selection": False,
        "candidate_ranking": ranking,
        "results": results,
    }
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"best_validation_target={ranking[0]}", flush=True)


if __name__ == "__main__":
    main()
