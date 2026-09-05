#!/usr/bin/env python3
"""Run leakage-controlled blocked-CV selection and final diverse ensembles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ecog_decoding.postprocessing import fit_nonnegative_gain, smooth_nonnegative
from ecog_decoding.training import FINGER_NAMES


TARGETS = {1: "local_w2_q10", 2: "local_w1_q10", 3: "local_w4_q20"}
CONFIGS = {
    "wavelet_lars": {
        "route": "wavelet",
        "head": "lars_linear_regime",
        "activation": "linear",
        "frontend_lr": "1e-5",
        "sequence": "100",
        "seed_offset": 10,
        "lars_candidate_scale": "1.0",
        "lars_near_zero_std": "1e-3",
    },
    "wavelet_softplus": {
        "route": "wavelet",
        "head": "residual_ridge",
        "activation": "softplus",
        "frontend_lr": "3e-5",
        "sequence": "50",
        "seed_offset": 20,
    },
    "csp_lars": {
        "route": "csp",
        "head": "lars_linear_regime",
        "activation": "linear",
        "frontend_lr": "1e-5",
        "sequence": "100",
        "seed_offset": 30,
        "lars_candidate_scale": "0.1",
        "lars_near_zero_std": "1e-4",
    },
    "csp_softplus": {
        "route": "csp",
        "head": "residual_ridge",
        "activation": "softplus",
        "frontend_lr": "3e-5",
        "sequence": "50",
        "seed_offset": 40,
    },
}
FINGER_INDEX = {name: index for index, name in enumerate(FINGER_NAMES)}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 0 else 0.0


def folds_for_subject(subject: int, prepared_root: Path, history: int) -> list[tuple[int, int]]:
    metadata = json.loads(
        (prepared_root / f"sub{subject}" / "metadata.json").read_text()
    )
    rows = int(metadata["target_fit_samples_25hz"]) - (history - 1)
    boundaries = np.rint(np.linspace(0.5 * rows, rows, 4)).astype(int)
    return [(int(boundaries[index]), int(boundaries[index + 1])) for index in range(3)]


def selection_root(route: str, subject: int, fold: int | None) -> Path:
    if fold is None:
        return Path(
            "outputs/fixed_lars_windowed_ica_asymmetric_screen512_v1"
            if route == "wavelet"
            else "outputs/fixed_lars_csp_repro_v1"
        )
    return Path(f"outputs/nested_cv_selection_{route}_s{subject}_fold{fold}_v1")


def route_arguments(route: str) -> list[str]:
    if route == "wavelet":
        return [
            "--feature-root", "outputs/windowed_ica_wavelet_asymmetric_v1",
            "--ica-root", "outputs/paper_ica_lars_v1",
            "--frontend", "asymmetric",
        ]
    return [
        "--feature-root", "outputs/csp_ridge_v2",
        "--csp-root", "outputs/csp_ridge_v2",
        "--band-cache-root", "/dev/shm/ecog_csp_band_cache",
        "--frontend", "csp_band",
    ]


def initialization_arguments(config: dict[str, object]) -> list[str]:
    if config["head"] != "lars_linear_regime":
        return []
    return [
        "--lars-candidate-scale", str(config["lars_candidate_scale"]),
        "--lars-near-zero-std", str(config["lars_near_zero_std"]),
    ]


def run_tasks(
    tasks: list[tuple[str, list[str], str | None]],
    concurrency: int,
    log_root: Path,
) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    active: list[tuple[subprocess.Popen[bytes], object, str, str | None]] = []
    failures: list[str] = []
    pending = list(tasks)
    gpu_pool = list(dict.fromkeys(gpu for _, _, gpu in tasks if gpu is not None))
    while active or pending:
        while pending and len(active) < concurrency:
            if gpu_pool:
                busy_gpus = {gpu for _, _, _, gpu in active}
                free_gpus = [gpu for gpu in gpu_pool if gpu not in busy_gpus]
                if not free_gpus:
                    break
                gpu = free_gpus[0]
            else:
                gpu = None
            name, command, _ = pending.pop(0)
            log = open(log_root / f"{name}.log", "wb")
            env = os.environ.copy()
            env["PYTHONPATH"] = "scripts:src"
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            active.append((process, log, name, gpu))
            print(f"started {name} pid={process.pid} gpu={gpu}", flush=True)
        time.sleep(1.0)
        remaining: list[tuple[subprocess.Popen[bytes], object, str, str | None]] = []
        for process, log, name, gpu in active:
            code = process.poll()
            if code is None:
                remaining.append((process, log, name, gpu))
                continue
            log.close()
            print(f"finished {name} exit={code}", flush=True)
            if code:
                failures.append(name)
        active = remaining
    if failures:
        raise RuntimeError("failed tasks: " + ", ".join(failures))


def selection_tasks(prepared_root: Path, history: int) -> list[tuple[str, list[str], None]]:
    tasks: list[tuple[str, list[str], None]] = []
    for subject in TARGETS:
        folds = folds_for_subject(subject, prepared_root, history)
        for route in ("wavelet", "csp"):
            energy_root = (
                "outputs/windowed_ica_wavelet_asymmetric_v1"
                if route == "wavelet"
                else "outputs/csp_ridge_v2"
            )
            for fold, (fit_end, validation_end) in enumerate(folds):
                name = f"select_{route}_s{subject}_fold{fold}"
                command = [
                    sys.executable,
                    "scripts/benchmark_fixed_lars.py",
                    "--subject", str(subject),
                    "--energy-root", energy_root,
                    "--output-root", str(selection_root(route, subject, fold)),
                    "--target", TARGETS[subject],
                    "--history", str(history),
                    "--fingers", *FINGER_NAMES,
                    "--fit-end-index", str(fit_end),
                    "--validation-end-index", str(validation_end),
                    "--max-features", "512",
                    "--max-iter", "500",
                    "--n-jobs", "1",
                    "--skip-test-evaluation",
                ]
                if route == "wavelet":
                    command.append("--prewindowed")
                tasks.append((name, command, None))
        # The full CSP selection is fitted on the training partition only and
        # is used solely to initialize the fixed-epoch final refits.
        tasks.append(
            (
                f"select_csp_s{subject}_full",
                [
                    sys.executable,
                    "scripts/benchmark_fixed_lars.py",
                    "--subject", str(subject),
                    "--energy-root", "outputs/csp_ridge_v2",
                    "--output-root", str(selection_root("csp", subject, None)),
                    "--target", TARGETS[subject],
                    "--history", str(history),
                    "--max-features", "512",
                    "--max-iter", "500",
                    "--n-jobs", "1",
                    "--skip-test-evaluation",
                    "--skip-validation-evaluation",
                ],
                None,
            )
        )
    return tasks


def cv_training_tasks(
    prepared_root: Path, history: int, gpus: list[str]
) -> list[tuple[str, list[str], str]]:
    tasks: list[tuple[str, list[str], str]] = []
    gpu_index = 0
    for subject in TARGETS:
        for config_name, config in CONFIGS.items():
            route = str(config["route"])
            for fold, (fit_end, validation_end) in enumerate(
                folds_for_subject(subject, prepared_root, history)
            ):
                name = f"cv_{config_name}_s{subject}_fold{fold}"
                output = Path(f"outputs/nested_cv_{config_name}_s{subject}_fold{fold}_v1")
                summary_path = output / f"sub{subject}" / "summary.json"
                if summary_path.exists():
                    print(f"skipping completed {name}: {summary_path}", flush=True)
                    continue
                command = [
                    sys.executable,
                    "scripts/train_exact_window_end_to_end.py",
                    "--subject", str(subject),
                    "--prepared-root", str(prepared_root),
                    "--selection-root", str(selection_root(route, subject, fold)),
                    "--output-root", str(output),
                    "--target", TARGETS[subject],
                    "--history", str(history),
                    "--fingers", *FINGER_NAMES,
                    "--fit-end-index", str(fit_end),
                    "--validation-end-index", str(validation_end),
                    "--head-initialization", str(config["head"]),
                    "--output-activation", str(config["activation"]),
                    "--frontend-warmup-epochs", "5",
                    "--batch-size", "16",
                    "--unfrozen-batch-size", "1",
                    "--sequence-steps", str(config["sequence"]),
                    "--sequence-stride", str(config["sequence"]),
                    "--epochs", "40",
                    "--validation-interval", "2",
                    "--patience", "8",
                    "--learning-rate", "1e-3",
                    "--frontend-learning-rate", str(config["frontend_lr"]),
                    "--direct-learning-rate", "1e-5",
                    "--seed", str(subject * 100 + int(config["seed_offset"])),
                    "--skip-test-evaluation",
                    *initialization_arguments(config),
                    *route_arguments(route),
                ]
                tasks.append((name, command, gpus[gpu_index % len(gpus)]))
                gpu_index += 1
    return tasks


def positive_affine(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    source_std = float(np.std(prediction))
    scale = float(np.std(target)) / source_std if source_std > 1.0e-8 else 1.0
    offset = float(np.mean(target) - scale * np.mean(prediction))
    return scale * prediction + offset, {"scale": scale, "offset": offset}


def summarize_cv(prepared_root: Path, history: int, output: Path) -> dict[str, object]:
    report: dict[str, object] = {"protocol": "three rolling blocked folds within training", "subjects": {}}
    for subject in TARGETS:
        aligned_raw = np.load(
            prepared_root / f"sub{subject}" / "train_glove_25hz_raw.npy"
        )[history - 1 :]
        aligned_cleaned = np.load(
            prepared_root / f"sub{subject}" / f"train_glove_{TARGETS[subject]}.npy"
        )[history - 1 :]
        folds = folds_for_subject(subject, prepared_root, history)
        subject_report: dict[str, object] = {"folds": folds, "per_finger": {}}
        predictions: dict[str, np.ndarray] = {}
        audits: dict[str, dict[str, object]] = {}
        target = np.concatenate([aligned_raw[start:stop] for start, stop in folds])
        cleaned_target = np.concatenate(
            [aligned_cleaned[start:stop] for start, stop in folds]
        )
        for config_name in CONFIGS:
            pieces = []
            summaries = []
            for fold in range(len(folds)):
                root = Path(f"outputs/nested_cv_{config_name}_s{subject}_fold{fold}_v1") / f"sub{subject}"
                pieces.append(np.load(root / "validation_prediction.npy"))
                summaries.append(json.loads((root / "summary.json").read_text()))
            predictions[config_name] = np.concatenate(pieces)
            audits[config_name] = {"summaries": summaries}
        for finger, finger_name in enumerate(FINGER_NAMES):
            candidates: list[dict[str, object]] = []
            for config_name in CONFIGS:
                values = predictions[config_name][:, finger]
                epochs = [
                    int(summary["per_finger"][finger_name]["best_epoch"])
                    for summary in audits[config_name]["summaries"]
                ]
                # Epoch zero is a valid non-collapsed fixed initializer.  A
                # collapsed member is defined by invalid or effectively
                # constant predictions, not merely by lack of fine-tuning gain.
                eligible = bool(
                    np.isfinite(values).all() and np.std(values) > 1.0e-8
                )
                calibrated, affine = positive_affine(values, target[:, finger])
                cleaned_values = smooth_nonnegative(
                    values,
                    already_nonnegative=CONFIGS[config_name]["activation"] == "softplus",
                )
                cleaned_gain = fit_nonnegative_gain(
                    cleaned_values, cleaned_target[:, finger]
                )
                fold_scores = []
                cursor = 0
                for start, stop in folds:
                    count = stop - start
                    fold_scores.append(
                        pearson(calibrated[cursor : cursor + count], target[cursor : cursor + count, finger])
                    )
                    cursor += count
                candidates.append(
                    {
                        "config": config_name,
                        "eligible": eligible,
                        "best_epochs": epochs,
                        "refit_epochs": max(0, int(np.median(epochs))),
                        "affine": affine,
                        "cleaned_gain": cleaned_gain,
                        "oof_prediction": calibrated,
                        "oof_validation_r": pearson(calibrated, target[:, finger]),
                        "mean_blocked_validation_r": float(np.mean(fold_scores)),
                        "fold_validation_r": fold_scores,
                    }
                )
            eligible = [item for item in candidates if bool(item["eligible"])]
            if not eligible:
                raise RuntimeError(f"no eligible candidates for S{subject} {finger_name}")
            eligible.sort(key=lambda item: float(item["mean_blocked_validation_r"]), reverse=True)
            selected = [eligible[0]]
            current = np.asarray(eligible[0]["oof_prediction"])
            current_block = float(eligible[0]["mean_blocked_validation_r"])
            trace = []
            remaining = eligible[1:]
            while remaining:
                proposals = []
                residual = current - target[:, finger]
                for item in remaining:
                    proposal = np.mean(
                        np.stack([*(np.asarray(x["oof_prediction"]) for x in selected), np.asarray(item["oof_prediction"])]),
                        axis=0,
                    )
                    cursor = 0
                    scores = []
                    for start, stop in folds:
                        count = stop - start
                        scores.append(pearson(proposal[cursor : cursor + count], target[cursor : cursor + count, finger]))
                        cursor += count
                    block = float(np.mean(scores))
                    residual_correlation = pearson(
                        np.asarray(item["oof_prediction"]) - target[:, finger], residual
                    )
                    proposals.append((block - current_block, -residual_correlation, item, proposal, block, residual_correlation))
                proposals.sort(key=lambda value: (value[0], value[1]), reverse=True)
                gain, _, item, proposal, block, residual_correlation = proposals[0]
                accepted = bool(gain >= 5.0e-4 and residual_correlation <= 0.995)
                trace.append(
                    {
                        "config": item["config"],
                        "blocked_gain": gain,
                        "residual_error_correlation": residual_correlation,
                        "accepted": accepted,
                    }
                )
                if not accepted:
                    break
                selected.append(item)
                remaining.remove(item)
                current = proposal
                current_block = block
            subject_report["per_finger"][finger_name] = {
                "selected_configs": [item["config"] for item in selected],
                "selected_refit_epochs": {
                    str(item["config"]): item["refit_epochs"] for item in selected
                },
                "selected_affine": {
                    str(item["config"]): item["affine"] for item in selected
                },
                "selected_cleaned_gain": {
                    str(item["config"]): item["cleaned_gain"] for item in selected
                },
                "selected_oof_r": pearson(current, target[:, finger]),
                "selected_mean_blocked_r": current_block,
                "selection_trace": trace,
                "candidate_audit": [
                    {key: value for key, value in item.items() if key != "oof_prediction"}
                    for item in candidates
                ],
            }
        report["subjects"][str(subject)] = subject_report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def refit_tasks(
    report: dict[str, object], prepared_root: Path, history: int, gpus: list[str]
) -> list[tuple[str, list[str], str]]:
    tasks: list[tuple[str, list[str], str]] = []
    gpu_index = 0
    for subject in TARGETS:
        per_finger = report["subjects"][str(subject)]["per_finger"]
        for finger_name in FINGER_NAMES:
            finger_report = per_finger[finger_name]
            for config_name in finger_report["selected_configs"]:
                config = CONFIGS[config_name]
                route = str(config["route"])
                epochs = int(finger_report["selected_refit_epochs"][config_name])
                name = f"refit_{config_name}_s{subject}_{finger_name}"
                output = Path(f"outputs/nested_refit_{config_name}_s{subject}_{finger_name}_v1")
                command = [
                    sys.executable,
                    "scripts/train_exact_window_end_to_end.py",
                    "--subject", str(subject),
                    "--prepared-root", str(prepared_root),
                    "--selection-root", str(selection_root(route, subject, None)),
                    "--output-root", str(output),
                    "--target", TARGETS[subject],
                    "--history", str(history),
                    "--fingers", finger_name,
                    "--no-validation-selection",
                    "--head-initialization", str(config["head"]),
                    "--output-activation", str(config["activation"]),
                    "--frontend-warmup-epochs", "5",
                    "--batch-size", "16",
                    "--unfrozen-batch-size", "1",
                    "--sequence-steps", str(config["sequence"]),
                    "--sequence-stride", str(config["sequence"]),
                    "--epochs", str(epochs),
                    "--validation-interval", "2",
                    "--learning-rate", "1e-3",
                    "--frontend-learning-rate", str(config["frontend_lr"]),
                    "--direct-learning-rate", "1e-5",
                    "--seed", str(subject * 100 + int(config["seed_offset"])),
                    *initialization_arguments(config),
                    *route_arguments(route),
                ]
                tasks.append((name, command, gpus[gpu_index % len(gpus)]))
                gpu_index += 1
    return tasks


def assemble_final(
    report: dict[str, object], prepared_root: Path, history: int, output_root: Path
) -> None:
    for subject in TARGETS:
        metadata = json.loads(
            (prepared_root / f"sub{subject}" / "metadata.json").read_text()
        )
        split = int(metadata["target_fit_samples_25hz"])
        validation_raw_target = np.load(
            prepared_root / f"sub{subject}" / "train_glove_25hz_raw.npy"
        )[split:]
        test_raw_target = np.load(
            prepared_root / f"sub{subject}" / "test_glove_25hz_raw.npy"
        )[history - 1 :]
        validation_cleaned_target = np.load(
            prepared_root / f"sub{subject}" / f"train_glove_{TARGETS[subject]}.npy"
        )[split:]
        test_cleaned_target = np.load(
            prepared_root / f"sub{subject}" / f"test_glove_{TARGETS[subject]}.npy"
        )[history - 1 :]
        validation_raw_coordinate = np.zeros_like(validation_raw_target, dtype=np.float32)
        test_raw_coordinate = np.zeros_like(test_raw_target, dtype=np.float32)
        validation_cleaned = np.zeros_like(validation_cleaned_target, dtype=np.float32)
        test_cleaned = np.zeros_like(test_cleaned_target, dtype=np.float32)
        per_finger = report["subjects"][str(subject)]["per_finger"]
        for finger, finger_name in enumerate(FINGER_NAMES):
            finger_report = per_finger[finger_name]
            validation_members = []
            test_members = []
            validation_cleaned_members = []
            test_cleaned_members = []
            for config_name in finger_report["selected_configs"]:
                root = Path(f"outputs/nested_refit_{config_name}_s{subject}_{finger_name}_v1") / f"sub{subject}"
                affine = finger_report["selected_affine"][config_name]
                scale = float(affine["scale"])
                offset = float(affine["offset"])
                validation_member = np.load(root / "validation_prediction.npy")[:, finger]
                test_member = np.load(root / "test_prediction.npy")[:, finger]
                validation_members.append(scale * validation_member + offset)
                test_members.append(scale * test_member + offset)
                gain = float(finger_report["selected_cleaned_gain"][config_name])
                activation = str(CONFIGS[config_name]["activation"])
                validation_cleaned_members.append(
                    gain
                    * smooth_nonnegative(
                        validation_member,
                        already_nonnegative=activation == "softplus",
                    )
                )
                test_cleaned_members.append(
                    gain
                    * smooth_nonnegative(
                        test_member,
                        already_nonnegative=activation == "softplus",
                    )
                )
            validation_raw_coordinate[:, finger] = np.mean(
                np.stack(validation_members), axis=0
            )
            test_raw_coordinate[:, finger] = np.mean(np.stack(test_members), axis=0)
            validation_cleaned[:, finger] = np.mean(
                np.stack(validation_cleaned_members), axis=0
            )
            test_cleaned[:, finger] = np.mean(
                np.stack(test_cleaned_members), axis=0
            )
            finger_report["final_validation_r"] = pearson(
                validation_raw_coordinate[:, finger], validation_raw_target[:, finger]
            )
            finger_report["test_r_descriptive_only"] = pearson(
                test_raw_coordinate[:, finger], test_raw_target[:, finger]
            )
            finger_report["final_validation_cleaned_r"] = pearson(
                validation_cleaned[:, finger], validation_cleaned_target[:, finger]
            )
            finger_report["test_cleaned_r_descriptive_only"] = pearson(
                test_cleaned[:, finger], test_cleaned_target[:, finger]
            )
        destination = output_root / f"sub{subject}"
        destination.mkdir(parents=True, exist_ok=True)
        # Backward-compatible filenames remain the raw-glove coordinate output
        # used for paper-comparable PCC.  Explicit names prevent visualization
        # code from confusing it with the cleaned nonnegative trajectory.
        np.save(
            destination / "validation_prediction.npy",
            validation_raw_coordinate,
            allow_pickle=False,
        )
        np.save(
            destination / "test_prediction.npy", test_raw_coordinate, allow_pickle=False
        )
        np.save(
            destination / "validation_prediction_raw_coordinate.npy",
            validation_raw_coordinate,
            allow_pickle=False,
        )
        np.save(
            destination / "test_prediction_raw_coordinate.npy",
            test_raw_coordinate,
            allow_pickle=False,
        )
        np.save(
            destination / "validation_prediction_cleaned.npy",
            validation_cleaned,
            allow_pickle=False,
        )
        np.save(
            destination / "test_prediction_cleaned.npy",
            test_cleaned,
            allow_pickle=False,
        )
        scores = [per_finger[name]["final_validation_r"] for name in FINGER_NAMES]
        test_scores = [per_finger[name]["test_r_descriptive_only"] for name in FINGER_NAMES]
        cleaned_scores = [
            per_finger[name]["final_validation_cleaned_r"] for name in FINGER_NAMES
        ]
        cleaned_test_scores = [
            per_finger[name]["test_cleaned_r_descriptive_only"] for name in FINGER_NAMES
        ]
        final_report = {
            "subject": subject,
            "protocol": "member and epoch selection used only three rolling blocked folds inside training; final validation evaluated once after fixed-epoch refit",
            "per_finger": per_finger,
            "validation_raw_macro_five": float(np.mean(scores)),
            "test_raw_macro_five_descriptive_only": float(np.mean(test_scores)),
            "validation_cleaned_macro_five": float(np.mean(cleaned_scores)),
            "test_cleaned_macro_five_descriptive_only": float(
                np.mean(cleaned_test_scores)
            ),
            "prediction_files": {
                "paper_comparable_raw_coordinate": "validation_prediction_raw_coordinate.npy and test_prediction_raw_coordinate.npy",
                "nonnegative_cleaned_flexion": "validation_prediction_cleaned.npy and test_prediction_cleaned.npy",
                "backward_compatible_default": "validation_prediction.npy and test_prediction.npy are raw-coordinate",
            },
            "released_test_used_for_selection": False,
        }
        (destination / "summary.json").write_text(json.dumps(final_report, indent=2) + "\n")
        print(json.dumps({"subject": subject, "validation": scores, "macro": np.mean(scores)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("selections", "cv", "summarize", "refit", "assemble"),
    )
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "3", "4", "5", "6", "7"])
    parser.add_argument("--concurrency", type=int, default=7)
    parser.add_argument("--report", type=Path, default=Path("outputs/nested_cv_selection_v1/summary.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/nested_cv_diverse_ensemble_v1"))
    args = parser.parse_args()
    logs = Path("outputs/nested_cv_logs_v1")
    if args.stage == "selections":
        run_tasks(selection_tasks(args.prepared_root, args.history), min(args.concurrency, 7), logs)
    elif args.stage == "cv":
        run_tasks(cv_training_tasks(args.prepared_root, args.history, args.gpus), args.concurrency, logs)
    elif args.stage == "summarize":
        summarize_cv(args.prepared_root, args.history, args.report)
    elif args.stage == "refit":
        report = json.loads(args.report.read_text())
        run_tasks(refit_tasks(report, args.prepared_root, args.history, args.gpus), args.concurrency, logs)
    else:
        report = json.loads(args.report.read_text())
        assemble_final(report, args.prepared_root, args.history, args.output_root)


if __name__ == "__main__":
    main()
