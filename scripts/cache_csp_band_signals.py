#!/usr/bin/env python3
"""Cache the continuous CSP carrier bands once for fast end-to-end training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal

from ecog_decoding.spatial import CSP_BAND_PROFILES


def main() -> None:
    parser = argparse.ArgumentParser()
    subject_group = parser.add_mutually_exclusive_group(required=True)
    subject_group.add_argument("--subject", type=int)
    subject_group.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("/dev/shm/ecog_csp_band_cache"))
    parser.add_argument(
        "--band-profile", choices=tuple(CSP_BAND_PROFILES), default="standard"
    )
    parser.add_argument("--sampling-rate", type=float, default=1000.0)
    args = parser.parse_args()

    subjects = args.subjects if args.subjects is not None else [args.subject]
    bands = CSP_BAND_PROFILES[args.band_profile]
    for subject in subjects:
        prepared = args.prepared_root / f"sub{subject}"
        output = args.output_root / f"sub{subject}"
        output.mkdir(parents=True, exist_ok=True)
        train = np.load(prepared / "train_ecog.npy", mmap_mode="r")
        train_output = np.lib.format.open_memmap(
            output / "train_filtered_bands.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(bands), *train.shape),
        )
        for band, (low, high) in enumerate(bands):
            sos = signal.butter(
                4,
                (low, high),
                btype="bandpass",
                fs=args.sampling_rate,
                output="sos",
            )
            train_output[band] = signal.sosfiltfilt(
                sos, np.asarray(train), axis=0
            ).astype(np.float32)
            print(f"subject={subject} cached_band={low:g}-{high:g}Hz", flush=True)
        train_output.flush()
        summary = {
            "subject": subject,
            "band_profile": args.band_profile,
            "bands_hz": bands,
            "sampling_rate": args.sampling_rate,
            "train_shape": list(train_output.shape),
            "source": str(prepared),
            "test_data_loaded": False,
            "note": "Butterworth bands are cached only to fit CSP rows; the decoder consumes broadband ECoG.",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
