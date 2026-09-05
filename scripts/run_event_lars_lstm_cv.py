#!/usr/bin/env python3
"""Schedule all per-finger event-fold LARS-LSTM seed runs on available GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from ecog_decoding.training import FINGER_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=tuple(FINGER_NAMES))
    parser.add_argument("--folds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--gpus", nargs="+", default=("2", "3"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/event_lars_lstm_wavelet_v1"))
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/windowed_ica_wavelet_asymmetric_v1"))
    parser.add_argument("--selection-cache-root", type=Path, default=Path("outputs/event_lars_selection_v1"))
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    tasks: list[tuple[str, list[str], Path]] = []
    for subject in args.subjects:
        for finger in args.fingers:
            for fold in args.folds:
                for seed in args.seeds:
                    destination = (
                        args.output_root
                        / f"sub{subject}"
                        / finger
                        / f"fold{fold}"
                        / f"seed{seed}"
                        / "summary.json"
                    )
                    if destination.exists():
                        print(f"skip completed {destination}", flush=True)
                        continue
                    name = f"s{subject}_{finger}_f{fold}_seed{seed}"
                    command = [
                        sys.executable,
                        "scripts/train_event_grouped_lars_lstm.py",
                        "--subject", str(subject),
                        "--finger", finger,
                        "--fold", str(fold),
                        "--seed", str(seed),
                        "--epochs", str(args.epochs),
                        "--feature-root", str(args.feature_root),
                        "--selection-cache-root", str(args.selection_cache_root),
                        "--output-root", str(args.output_root),
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
                print(f"skip completed {destination}", flush=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            claim = destination.with_suffix(".claim")
            try:
                descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                print(f"skip claimed {destination}", flush=True)
                continue
            with os.fdopen(descriptor, "w") as stream:
                stream.write(f"host={os.uname().nodename} pid={os.getpid()}\n")
            log = open(log_root / f"{name}.log", "wb")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = "scripts:src"
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=environment
            )
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
