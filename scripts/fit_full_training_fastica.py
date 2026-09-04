#!/usr/bin/env python3
"""Fit label-free FastICA on the complete released training recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.models import fit_fastica_spatial_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/full_training_fastica_v1")
    )
    parser.add_argument("--backend", choices=("torch", "sklearn"), default="torch")
    parser.add_argument("--max-samples", type=int, default=50_000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    prepared = args.prepared_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    ecog = np.load(prepared / "train_ecog.npy", mmap_mode="r")
    unmixing = fit_fastica_spatial_weights(
        np.asarray(ecog),
        max_samples=args.max_samples,
        random_state=args.random_state,
        backend=args.backend,
        device=args.device,
    )
    np.save(output / "fastica_unmixing.npy", unmixing, allow_pickle=False)
    report = {
        "subject": args.subject,
        "method": "label-free FastICA on the complete released training recording",
        "backend": args.backend,
        "available_training_samples": int(ecog.shape[0]),
        "channels": int(ecog.shape[1]),
        "max_evenly_spaced_fit_samples": args.max_samples,
        "random_state": args.random_state,
        "test_data_used": False,
        "test_labels_used": False,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
