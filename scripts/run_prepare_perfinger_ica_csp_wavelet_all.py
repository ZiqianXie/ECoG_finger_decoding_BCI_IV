#!/usr/bin/env python3
"""Build compact split-local and full-refit ICA+CSP wavelet caches in parallel."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from ecog_decoding.training import FINGER_NAMES
from run_event_lars_e2e_nested_cv import acquire_claim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument(
        "--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=tuple(FINGER_NAMES)
    )
    parser.add_argument("--gpus", nargs="+", default=("0", "1", "2", "3"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/perfinger_ica_csp_wavelet_1000hz_all_tails2x2_hg_v1"),
    )
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path("outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1"),
    )
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--ica-root", type=Path, default=Path("outputs/paper_ica_lars_v1"))
    parser.add_argument("--chunk-steps", type=int, default=128)
    parser.add_argument(
        "--csp-source",
        choices=("high_gamma", "broadband", "seven_band"),
        default="high_gamma",
    )
    parser.add_argument(
        "--csp-mode",
        choices=("movement_1", "movement_2", "tails_2x2", "tails_4x4"),
        default="tails_2x2",
    )
    parser.add_argument(
        "--include-any-movement-csp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--band-cache-root",
        type=Path,
        default=Path("/dev/shm/ecog_csp_band_cache"),
    )
    parser.add_argument("--claim-timeout-hours", type=float, default=6.0)
    args = parser.parse_args()

    tasks: list[tuple[str, list[str], Path]] = []
    for subject in args.subjects:
        for finger in args.fingers:
            cache = args.output_root / f"sub{subject}" / finger
            common = [
                sys.executable,
                "scripts/prepare_perfinger_ica_csp_wavelet_features.py",
                "--subject",
                str(subject),
                "--finger",
                finger,
                "--csp-mode",
                args.csp_mode,
                "--csp-source",
                args.csp_source,
                "--prepared-root",
                str(args.prepared_root),
                "--ica-root",
                str(args.ica_root),
                "--fold-root",
                str(args.fold_root),
                "--output-root",
                str(cache),
                "--chunk-steps",
                str(args.chunk_steps),
                "--band-cache-root",
                str(args.band_cache_root),
                "--compact-csp-only",
            ]
            if args.include_any_movement_csp:
                common.append("--include-any-movement-csp")
            if not (cache / "split_summary.json").exists():
                tasks.append((f"s{subject}_{finger}_splits", common, cache / "split_summary.json"))
            if not (cache / "full" / "summary.json").exists():
                tasks.append(
                    (
                        f"s{subject}_{finger}_full",
                        [*common, "--full-refit"],
                        cache / "full" / "summary.json",
                    )
                )

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
            print(f"finished {name} exit={code} gpu={gpu}", flush=True)
            if code:
                failures.append(name)
        active = remaining

    if failures:
        raise SystemExit(f"failed cache jobs: {', '.join(failures)}")


if __name__ == "__main__":
    main()
