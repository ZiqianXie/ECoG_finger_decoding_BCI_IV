#!/usr/bin/env python3
"""Create a common-average-referenced variant from the notched ECoG arrays."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/preprocessed_car_v1"))
    args = parser.parse_args()

    source = args.source_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / "metadata.json").read_text())
    fit_stop = int(metadata["normalization_fit_samples_1khz"])
    arrays = {}
    for prefix in ("train", "test"):
        values = np.asarray(np.load(source / f"{prefix}_ecog.npy", mmap_mode="r"), dtype=np.float32)
        arrays[prefix] = values - values.mean(axis=1, keepdims=True)
    mean = arrays["train"][:fit_stop].mean(axis=0, dtype=np.float64)
    scale = arrays["train"][:fit_stop].std(axis=0, dtype=np.float64)
    scale[scale < 1.0e-8] = 1.0
    for prefix in ("train", "test"):
        normalized = ((arrays[prefix] - mean) / scale).astype(np.float32)
        np.save(output / f"{prefix}_ecog.npy", normalized, allow_pickle=False)
    for path in source.glob("*.npy"):
        if path.name not in {"train_ecog.npy", "test_ecog.npy"}:
            shutil.copyfile(path, output / path.name)
    metadata["ecog_reference_variant"] = "instantaneous common average after initial channel standardization, followed by train-only restandardization"
    metadata["normalization_mean"] = mean.tolist()
    metadata["normalization_scale"] = scale.tolist()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"subject={args.subject} channels={arrays['train'].shape[1]}", flush=True)


if __name__ == "__main__":
    main()
