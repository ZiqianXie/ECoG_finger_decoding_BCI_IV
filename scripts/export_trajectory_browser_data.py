#!/usr/bin/env python3
"""Export compact JSON used by the local all-trajectory browser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Little")


def rounded_rows(value: np.ndarray, decimals: int) -> list[list[float]]:
    return np.round(value.astype(np.float64), decimals=decimals).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--input-root", type=Path, default=Path("outputs/preprocessed"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/browser_data"))
    parser.add_argument("--decimals", type=int, default=5)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "sampling_rate_hz": 25,
        "finger_names": FINGER_NAMES,
        "subjects": [],
    }
    for subject in args.subjects:
        source = args.input_root / f"sub{subject}"
        raw = np.load(source / "train_glove_25hz_raw.npy", allow_pickle=False)
        baseline = np.load(source / "train_glove_baseline.npy", allow_pickle=False)
        detrended = np.load(
            source / "train_glove_detrended_multifinger.npy", allow_pickle=False
        )
        intended = np.load(source / "train_target_intended.npy", allow_pickle=False)
        states = np.load(source / "train_active_finger.npy", allow_pickle=False)
        coupling = np.load(source / "finger_coupling_matrix.npy", allow_pickle=False)
        if not (raw.shape == baseline.shape == detrended.shape == intended.shape):
            raise ValueError(f"subject {subject} trajectory shapes do not match")
        if states.shape != (raw.shape[0],):
            raise ValueError(f"subject {subject} state shape does not match")

        payload = {
            "subject": subject,
            "sampling_rate_hz": 25,
            "finger_names": FINGER_NAMES,
            "sample_count": int(raw.shape[0]),
            "raw": rounded_rows(raw, args.decimals),
            "baseline": rounded_rows(baseline, args.decimals),
            "detrended": rounded_rows(detrended, args.decimals),
            "intended": rounded_rows(intended, args.decimals),
            "active_finger": states.astype(int).tolist(),
            "coupling_matrix": rounded_rows(coupling, args.decimals),
        }
        output_path = args.output_root / f"subject-{subject}.json"
        output_path.write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        manifest["subjects"].append(
            {
                "id": subject,
                "file": output_path.name,
                "samples": int(raw.shape[0]),
                "duration_seconds": float(raw.shape[0] / 25.0),
            }
        )
        print(f"exported subject {subject}: {output_path}")

    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
