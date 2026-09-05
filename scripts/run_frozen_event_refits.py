#!/usr/bin/env python3
"""Coordinate frozen full-data refits across GPUs and shared lab hosts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from ecog_decoding.training import FINGER_NAMES
from refit_frozen_event_model import resolve_options
from run_event_lars_e2e_nested_cv import acquire_claim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=tuple(FINGER_NAMES))
    parser.add_argument("--gpus", nargs="+", default=("0", "1", "2", "3"))
    parser.add_argument("--ensemble-map", type=Path, default=Path("configs/final_event_ensemble.yaml"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/frozen_event_full_refit_v1"))
    parser.add_argument("--selection-cache-root", type=Path, default=Path("outputs/frozen_event_full_refit_lars_v1"))
    parser.add_argument("--claim-timeout-hours", type=float, default=6.0)
    args = parser.parse_args()
    ensemble_map = yaml.safe_load(args.ensemble_map.read_text())

    tasks: list[tuple[str, list[str], Path]] = []
    for subject in args.subjects:
        for finger in args.fingers:
            options = resolve_options(ensemble_map, subject, finger)
            for seed in options["seeds"]:
                destination = args.output_root / f"sub{subject}" / finger / f"seed{seed}" / "summary.json"
                if destination.exists():
                    continue
                name = f"s{subject}_{finger}_seed{seed}"
                command = [
                    sys.executable,
                    "scripts/refit_frozen_event_model.py",
                    "--subject", str(subject),
                    "--finger", finger,
                    "--seed", str(seed),
                    "--ensemble-map", str(args.ensemble_map),
                    "--target-map", str(args.target_map),
                    "--output-root", str(args.output_root),
                    "--selection-cache-root", str(args.selection_cache_root),
                ]
                tasks.append((name, command, destination))

    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    pending = list(tasks)
    active: list[tuple[subprocess.Popen[bytes], object, str, str, Path]] = []
    failures: list[str] = []
    while pending or active:
        busy = {gpu for _, _, _, gpu, _ in active}
        for gpu in args.gpus:
            if not pending or gpu in busy:
                continue
            name, command, destination = pending.pop(0)
            if destination.exists():
                continue
            claim = acquire_claim(destination, args.claim_timeout_hours)
            if claim is None:
                continue
            log = open(log_root / f"{name}.log", "wb")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = "scripts:src"
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            active.append((process, log, name, gpu, claim))
            print(f"started {name} pid={process.pid} gpu={gpu}", flush=True)
        time.sleep(1.0)
        remaining = []
        for process, log, name, gpu, claim in active:
            code = process.poll()
            if code is None:
                remaining.append((process, log, name, gpu, claim))
                continue
            log.close()
            claim.unlink(missing_ok=True)
            print(f"finished {name} exit={code}", flush=True)
            if code:
                failures.append(name)
        active = remaining
    if failures:
        raise RuntimeError("failed tasks: " + ", ".join(failures))


if __name__ == "__main__":
    main()
