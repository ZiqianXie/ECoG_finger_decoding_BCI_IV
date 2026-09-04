#!/usr/bin/env python3
"""Benchmark complementary exact-window ICA/wavelet and CSP-band features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_ridge_target_variants import fit_candidate, lagged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--window-root", type=Path, default=Path("outputs/windowed_ica_sklearn_v1"))
    parser.add_argument("--csp-root", type=Path, default=Path("outputs/csp_ridge_besttarget_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/hybrid_fixed_ridge_v1"))
    parser.add_argument("--target", required=True)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    prepared = args.prepared_root / f"sub{args.subject}"
    windows = args.window_root / f"sub{args.subject}"
    csp = args.csp_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    split = int(json.loads((prepared / "metadata.json").read_text())["target_fit_samples_25hz"])
    offset = args.history - 1
    window_train = np.load(windows / "train_initialized_window_features.npy", mmap_mode="r")
    window_test = np.load(windows / "test_initialized_window_features.npy", mmap_mode="r")
    csp_train = lagged(np.load(csp / "train_csp_energy.npy"), args.history)
    csp_test = lagged(np.load(csp / "test_csp_energy.npy"), args.history)
    train_count = split - offset
    train_x = np.concatenate((np.asarray(window_train[:train_count]), csp_train[:train_count]), axis=1)
    validation_x = np.concatenate((np.asarray(window_train[train_count:]), csp_train[train_count:]), axis=1)
    test_x = np.concatenate((np.asarray(window_test), csp_test), axis=1)
    target = np.load(prepared / f"train_glove_{args.target}.npy")
    raw_train = np.load(prepared / "train_glove_25hz_raw.npy")
    raw_test = np.load(prepared / "test_glove_25hz_raw.npy")
    result, validation_prediction, test_prediction = fit_candidate(
        train_x,
        target[offset:split],
        validation_x,
        target[split:],
        raw_train[split:],
        test_x,
        raw_test[offset:],
        args.top_features,
        torch.device(args.device),
    )
    np.save(output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
    np.save(output / "test_prediction.npy", test_prediction, allow_pickle=False)
    report = {
        "subject": args.subject,
        "method": "concatenated exact-window sklearn-ICA/wavelet and CSP-band features",
        "target": args.target,
        "top_features": args.top_features,
        "window_feature_count": int(window_train.shape[1]),
        "csp_feature_count": int(csp_train.shape[1]),
        "result": result,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"validation": result["validation_raw_metrics"], "test": result["test_raw_metrics"]}), flush=True)


if __name__ == "__main__":
    main()
