#!/usr/bin/env python3
"""Print a compact table from the spatial-reference ridge ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--repair-summaries", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for subject in (1, 2):
        for reference in ("car", "bipolar", "laplacian"):
            path = (
                args.output_root
                / f"reference_ridge_{reference}_v1"
                / f"sub{subject}"
                / "summary.json"
            )
            target = "local_w2_q10" if subject == 1 else "local_w1_q10"
            if path.is_file():
                payload = json.loads(path.read_text())
                candidate = next(iter(payload["results"].values()))
            else:
                metrics_path = path.parent / target / "metrics.json"
                candidate = json.loads(metrics_path.read_text())
                if args.repair_summaries:
                    if path.is_dir():
                        # Only remove the empty directory created by a mistaken
                        # file-level sync; rmdir refuses nonempty directories.
                        path.rmdir()
                    payload = {
                        "subject": subject,
                        "selection_metric": (
                            "validation_raw_metrics.pearson_historical_four"
                        ),
                        "test_labels_used_for_selection": False,
                        "top_features_per_finger": 512,
                        "candidate_ranking": [target],
                        "results": {target: candidate},
                    }
                    path.write_text(json.dumps(payload, indent=2) + "\n")
            rows.append(
                {
                    "subject": subject,
                    "reference": reference,
                    "validation_macro_five": candidate["validation_raw_metrics"][
                        "pearson_macro_five"
                    ],
                    "validation_historical_four": candidate["validation_raw_metrics"][
                        "pearson_historical_four"
                    ],
                    "test_macro_five": candidate["test_raw_metrics"][
                        "pearson_macro_five"
                    ],
                    "test_historical_four": candidate["test_raw_metrics"][
                        "pearson_historical_four"
                    ],
                    "test_by_finger": candidate["test_raw_metrics"][
                        "pearson_by_finger"
                    ],
                }
            )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
