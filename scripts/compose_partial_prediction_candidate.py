#!/usr/bin/env python3
"""Compose a complete saved-decoder candidate from per-finger replacements.

The validation-only ensemble utilities require five-column prediction arrays.
This helper keeps the untouched fingers from an existing complete candidate and
replaces only explicitly named fingers from another prediction directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--fingers", nargs="+", required=True, choices=tuple(FINGER_NAMES))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    base = args.base_root / f"sub{args.subject}"
    replacement = args.replacement_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    audit: dict[str, object] = {
        "subject": args.subject,
        "base_root": str(args.base_root),
        "replacement_root": str(args.replacement_root),
        "replaced_fingers": list(args.fingers),
        "arrays": {},
    }
    for partition in ("validation", "test"):
        base_values = np.load(base / f"{partition}_prediction.npy")
        replacement_values = np.load(replacement / f"{partition}_prediction.npy")
        if base_values.shape != replacement_values.shape or base_values.ndim != 2:
            raise ValueError(
                f"{partition} shape mismatch: {base_values.shape} vs "
                f"{replacement_values.shape}"
            )
        values = np.array(base_values, dtype=np.float32, copy=True)
        for name in args.fingers:
            index = list(FINGER_NAMES).index(name)
            values[:, index] = replacement_values[:, index]
        if not np.isfinite(values).all():
            raise ValueError(f"{partition} prediction contains non-finite values")
        np.save(output / f"{partition}_prediction.npy", values)
        audit["arrays"][partition] = {
            "shape": list(values.shape),
            "per_finger_sd": np.std(values, axis=0).tolist(),
        }
    (args.output_root / "summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
