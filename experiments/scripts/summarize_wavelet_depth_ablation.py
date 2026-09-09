#!/usr/bin/env python3
"""Print compact depth-3/depth-4 fixed-front-end comparison metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    roots = {
        "depth3": "fixed_lars_windowed_ica_screen512_v1",
        "depth4": "fixed_lars_windowed_ica_depth4_screen512_v1",
        "asymmetric_depth4_lmp": (
            "fixed_lars_windowed_ica_asymmetric_screen512_v1"
        ),
    }
    rows: list[dict[str, object]] = []
    for subject in (1, 2):
        for depth, root in roots.items():
            path = args.output_root / root / f"sub{subject}" / "summary.json"
            payload = json.loads(path.read_text())
            rows.append(
                {
                    "subject": subject,
                    "depth": depth,
                    "validation_raw": payload["validation_raw_metrics"],
                    "test_raw": payload["test_raw_metrics"],
                }
            )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
