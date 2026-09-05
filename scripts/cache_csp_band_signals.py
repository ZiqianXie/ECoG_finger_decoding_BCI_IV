#!/usr/bin/env python3
"""Cache the continuous CSP carrier bands once for fast end-to-end training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal

from benchmark_csp_ridge import BAND_PROFILES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("/dev/shm/ecog_csp_band_cache"))
    parser.add_argument("--band-profile", choices=tuple(BAND_PROFILES), default="standard")
    parser.add_argument("--sampling-rate", type=float, default=1000.0)
    args = parser.parse_args()

    prepared = args.prepared_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    train = np.load(prepared / "train_ecog.npy", mmap_mode="r")
    test = np.load(prepared / "test_ecog.npy", mmap_mode="r")
    bands = BAND_PROFILES[args.band_profile]
    train_output = np.lib.format.open_memmap(
        output / "train_filtered_bands.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(bands), *train.shape),
    )
    test_output = np.lib.format.open_memmap(
        output / "test_filtered_bands.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(bands), *test.shape),
    )
    joined = np.concatenate((np.asarray(train), np.asarray(test)), axis=0)
    for band, (low, high) in enumerate(bands):
        sos = signal.butter(
            4,
            (low, high),
            btype="bandpass",
            fs=args.sampling_rate,
            output="sos",
        )
        filtered = signal.sosfiltfilt(sos, joined, axis=0).astype(np.float32)
        train_output[band] = filtered[: train.shape[0]]
        test_output[band] = filtered[train.shape[0] :]
        print(f"subject={args.subject} cached_band={low:g}-{high:g}Hz", flush=True)
    train_output.flush()
    test_output.flush()
    summary = {
        "subject": args.subject,
        "band_profile": args.band_profile,
        "bands_hz": bands,
        "sampling_rate": args.sampling_rate,
        "train_shape": list(train_output.shape),
        "test_shape": list(test_output.shape),
        "source": str(prepared),
        "note": "Butterworth carrier bands are cached once; trainable FIR corrections define the effective learned spectral filters.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
