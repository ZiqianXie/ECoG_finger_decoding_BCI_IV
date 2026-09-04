#!/usr/bin/env python3
"""Create leakage-safe CAR, bipolar, or local-Laplacian ECoG variants."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from ecog_decoding.referencing import fit_standardize, rereference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument(
        "--source-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reference", choices=("car", "bipolar", "laplacian"), required=True
    )
    parser.add_argument("--grid-columns", type=int, default=8)
    args = parser.parse_args()

    source = args.source_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / "metadata.json").read_text())
    retained = metadata["retained_channels_one_based"]
    original_mean = np.asarray(metadata["normalization_mean"], dtype=np.float64)
    original_scale = np.asarray(metadata["normalization_scale"], dtype=np.float64)
    fit_stop = int(metadata["normalization_fit_samples_1khz"])

    referenced: dict[str, np.ndarray] = {}
    transform_metadata: dict[str, object] | None = None
    for prefix in ("train", "test"):
        standardized = np.load(source / f"{prefix}_ecog.npy", mmap_mode="r")
        # Undo the original channelwise standardization so the reference is
        # computed in the common notched-voltage scale.
        voltage = np.asarray(standardized, dtype=np.float64) * original_scale + original_mean
        referenced[prefix], details = rereference(
            voltage,
            retained,
            method=args.reference,
            grid_columns=args.grid_columns,
        )
        if transform_metadata is None:
            transform_metadata = details
        elif details != transform_metadata:
            raise RuntimeError("train and test reference transforms disagree")
    train, test, mean, scale = fit_standardize(
        referenced["train"], referenced["test"], fit_stop
    )
    np.save(output / "train_ecog.npy", train, allow_pickle=False)
    np.save(output / "test_ecog.npy", test, allow_pickle=False)
    for path in source.glob("*.npy"):
        if path.name not in {"train_ecog.npy", "test_ecog.npy"}:
            shutil.copyfile(path, output / path.name)

    metadata["output_shapes"]["train_ecog"] = list(train.shape)
    metadata["output_shapes"]["test_ecog"] = list(test.shape)
    metadata["ecog_reference_variant"] = args.reference
    metadata["reference_grid_columns"] = args.grid_columns
    metadata["reference_transform"] = transform_metadata
    metadata["normalization_mean"] = mean.tolist()
    metadata["normalization_scale"] = scale.tolist()
    metadata["reference_fit_samples_1khz"] = fit_stop
    metadata["reference_test_data_used_for_fit"] = False
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"subject={args.subject} reference={args.reference} "
        f"channels={train.shape[1]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
