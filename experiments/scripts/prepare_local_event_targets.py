#!/usr/bin/env python3
"""Create event-consistent finger targets from locally detrended glove traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from ecog_decoding.training import FINGER_NAMES


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--source-target", default="local_w1_q30")
    parser.add_argument("--output-name", default="local_event_w1_q30")
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--merge-gap-bins", type=int, default=5)
    parser.add_argument("--minimum-event-bins", type=int, default=3)
    args = parser.parse_args()

    root = args.prepared_root / f"sub{args.subject}"
    source = np.load(root / f"train_glove_{args.source_target}.npy")
    smooth = ndimage.gaussian_filter1d(source, sigma=1.0, axis=0, mode="nearest")
    active = np.max(smooth, axis=1) >= args.threshold
    active = ndimage.binary_closing(active, structure=np.ones(args.merge_gap_bins + 1))
    event_target = np.zeros_like(source, dtype=np.float32)
    events = []
    for start, stop in runs(active):
        if stop - start < args.minimum_event_bins:
            continue
        segment = smooth[start:stop]
        # Integrate only evidence above the activity threshold; choosing once
        # per event prevents samplewise identity flicker at flexion troughs.
        score = np.sum(np.maximum(segment - args.threshold, 0.0), axis=0)
        winner = int(np.argmax(score))
        ordered = np.sort(score)
        margin = float((ordered[-1] - ordered[-2]) / max(ordered[-1], 1.0e-8))
        event_target[start:stop, winner] = source[start:stop, winner]
        events.append(
            {
                "start": int(start),
                "stop": int(stop),
                "finger": FINGER_NAMES[winner],
                "score": score.tolist(),
                "relative_margin": margin,
            }
        )
    np.save(root / f"train_glove_{args.output_name}.npy", event_target, allow_pickle=False)
    counts = {name: sum(event["finger"] == name for event in events) for name in FINGER_NAMES}
    report = {
        "subject": args.subject,
        "source_target": args.source_target,
        "output_name": args.output_name,
        "threshold": args.threshold,
        "merge_gap_bins": args.merge_gap_bins,
        "minimum_event_bins": args.minimum_event_bins,
        "event_count": len(events),
        "event_counts_by_finger": counts,
        "median_relative_margin": float(np.median([event["relative_margin"] for event in events])),
        "events": events,
    }
    (root / f"{args.output_name}_metadata.json").write_text(json.dumps(report, indent=2) + "\n")

    disagreement = np.sum(np.abs(source - event_target), axis=1)
    width = 250
    center = int(np.argmax(np.convolve(disagreement, np.ones(width), mode="same")))
    start = max(0, center - width // 2)
    stop = min(source.shape[0], start + width)
    time = np.arange(start, stop) / 25.0
    figure, axes = plt.subplots(5, 1, figsize=(14, 11), sharex=True)
    for finger, name in enumerate(FINGER_NAMES):
        axes[finger].plot(time, source[start:stop, finger], color="#94a3b8", lw=1.0, label="local residual")
        axes[finger].plot(time, event_target[start:stop, finger], color="#2563eb", lw=1.2, label="event-consistent")
        axes[finger].set_ylabel(name)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("seconds")
    figure.suptitle(f"Subject {args.subject}: local residual versus event-consistent target")
    figure.tight_layout()
    figure.savefig(root / f"{args.output_name}_audit.png", dpi=160)
    plt.close(figure)
    print(json.dumps({key: report[key] for key in ("event_count", "event_counts_by_finger", "median_relative_margin")}), flush=True)


if __name__ == "__main__":
    main()
