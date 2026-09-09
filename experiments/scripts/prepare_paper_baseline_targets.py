#!/usr/bin/env python3
"""Create baseline-only targets with the constrained objective from the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.preprocessing import paper_baseline_correct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed"))
    parser.add_argument("--smoothness", type=float, default=1.0e5)
    args = parser.parse_args()
    for subject in args.subjects:
        root = args.prepared_root / f"sub{subject}"
        metadata = json.loads((root / "metadata.json").read_text())
        split = int(metadata["target_fit_samples_25hz"])
        train = np.load(root / "train_glove_25hz_raw.npy")
        test = np.load(root / "test_glove_25hz_raw.npy")
        print(f"subject={subject} fitting published constrained train baseline", flush=True)
        train_result = paper_baseline_correct(train, smoothness=args.smoothness)
        print(f"subject={subject} fitting published constrained test baseline", flush=True)
        test_result = paper_baseline_correct(test, smoothness=args.smoothness)
        scale = np.quantile(train_result.corrected[:split], 0.995, axis=0)
        scale_floor = max(float(np.max(scale)) * 0.01, np.finfo(np.float64).eps)
        scale = np.maximum(scale, scale_floor)
        train_normalized = np.clip(train_result.corrected / scale, 0.0, 2.0).astype(np.float32)
        test_normalized = np.clip(test_result.corrected / scale, 0.0, 2.0).astype(np.float32)
        np.save(root / "train_glove_paper_baseline_only.npy", train_normalized)
        np.save(root / "test_glove_paper_baseline_only.npy", test_normalized)
        np.save(root / "train_glove_paper_baseline.npy", train_result.baseline.astype(np.float32))
        np.save(root / "test_glove_paper_baseline.npy", test_result.baseline.astype(np.float32))
        audit = {
            "subject": subject,
            "objective": "L1 distance plus lambda times squared first differences, constrained baseline <= trajectory",
            "smoothness": args.smoothness,
            "normalization_fit_stop": split,
            "scale": scale.tolist(),
            "train_constraint_max_violation": float(np.max(train_result.baseline - train)),
            "test_constraint_max_violation": float(np.max(test_result.baseline - test)),
        }
        (root / "paper_baseline_metadata.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
