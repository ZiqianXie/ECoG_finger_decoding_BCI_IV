#!/usr/bin/env python3
"""Cache original-rate ICA plus finger-specific CSP wavelet features.

This is a matched spatial-capacity ablation for the established 1000 Hz
ICA/asymmetric-wavelet model.  The original FastICA rows are retained and a
small CSP bank is appended.  CSP is refit for every subject/finger and every
outer/inner training split; released-test data are never loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy import signal

from ecog_decoding.spatial import CSP_BANDS_HZ, fit_finger_csp_rows
from ecog_decoding.training import FINGER_NAMES
from train_event_grouped_lars_lstm import indices_from_intervals
from train_event_grouped_lars_lstm_nested import intervals_from_mask
from train_exact_window_end_to_end import ExactWindowFingerDecoder


SOURCE_RATE = 1000
TARGET_RATE = 25
SAMPLES_PER_BIN = SOURCE_RATE // TARGET_RATE
HISTORY_OFFSET = 24
CSP_MODES = {
    "movement_1": (-1,),
    "movement_2": (-1, -2),
    "tails_2x2": (0, 1, -2, -1),
    "tails_4x4": (0, 1, 2, 3, -4, -3, -2, -1),
}
HIGH_GAMMA_BANDS = ((65.0, 95.0), (105.0, 145.0))


def atomic_save(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, values, allow_pickle=False)
    os.replace(temporary, path)


def split_masks(
    definition: dict[str, object], outer_fold: int
) -> dict[str, tuple[list[list[int]], list[list[int]]]]:
    """Return the exact outer and nested-inner interval scopes used by training."""
    rows = int(definition["training_rows"])
    folds = definition["folds"]
    outer = folds[outer_fold]
    outer_training_mask = np.zeros(rows, dtype=bool)
    outer_training_mask[
        indices_from_intervals(outer["training_intervals_after_purge"])
    ] = True
    result = {
        "outer": (
            outer["training_intervals_after_purge"],
            outer["validation_intervals"],
        )
    }
    for inner_fold, inner in enumerate(folds):
        if inner_fold == outer_fold:
            continue
        training_mask = np.zeros(rows, dtype=bool)
        training_mask[
            indices_from_intervals(inner["training_intervals_after_purge"])
        ] = True
        training_mask &= outer_training_mask
        validation_mask = np.zeros(rows, dtype=bool)
        validation_mask[indices_from_intervals(inner["validation_intervals"])] = True
        validation_mask &= outer_training_mask
        result[f"inner{inner_fold}"] = (
            intervals_from_mask(training_mask),
            intervals_from_mask(validation_mask),
        )
    return result


def high_gamma_bins(ecog: np.ndarray) -> np.ndarray:
    """Return the two notch-separated high-gamma bands as one CSP sample set."""
    parts = []
    for low, high in HIGH_GAMMA_BANDS:
        sos = signal.butter(
            4, (low, high), btype="bandpass", fs=SOURCE_RATE, output="sos"
        )
        filtered = signal.sosfiltfilt(sos, np.asarray(ecog), axis=0).astype(
            np.float32
        )
        bins = filtered.shape[0] // SAMPLES_PER_BIN
        parts.append(
            filtered[: bins * SAMPLES_PER_BIN].reshape(
                bins, SAMPLES_PER_BIN, filtered.shape[1]
            )
        )
    return np.concatenate(parts, axis=1)


def band_cache_bins(
    cache_root: Path, subject: int
) -> tuple[np.ndarray, dict[str, object]]:
    """Load the shared seven-band filtering cache without copying it into RAM."""
    path = cache_root / f"sub{subject}" / "train_filtered_bands.npy"
    filtered = np.load(path, mmap_mode="r")
    if filtered.ndim != 3 or filtered.shape[0] != len(CSP_BANDS_HZ):
        raise ValueError(
            f"expected {len(CSP_BANDS_HZ)} cached bands with shape "
            f"(band, time, channel); got {filtered.shape}"
        )
    usable = filtered.shape[1] // SAMPLES_PER_BIN * SAMPLES_PER_BIN
    bins = filtered[:, :usable].reshape(
        len(CSP_BANDS_HZ), -1, SAMPLES_PER_BIN, filtered.shape[-1]
    )
    return bins, {
        "kind": "seven-band CSP rows projected through one shared wavelet path",
        "bands_hz": CSP_BANDS_HZ,
        "cache": str(path),
    }


def fit_csp_rows(
    csp_bins: np.ndarray,
    target: np.ndarray,
    training: np.ndarray,
    finger_index: int,
    component_indices: tuple[int, ...],
    *,
    include_any_movement: bool,
) -> tuple[np.ndarray, object]:
    """Fit band-conditioned spatial rows that all enter one wavelet path."""
    if csp_bins.ndim == 3:
        return fit_finger_csp_rows(
            csp_bins,
            target,
            training,
            finger_index,
            component_indices,
            history_offset=HISTORY_OFFSET,
        )
    if csp_bins.ndim != 4 or csp_bins.shape[0] != len(CSP_BANDS_HZ):
        raise ValueError(
            "multi-band CSP input must have shape (band, bin, sample, channel)"
        )

    shared_target = np.max(target, axis=1, keepdims=True)
    rows: list[np.ndarray] = []
    audit: list[dict[str, object]] = []
    for band_index, (low, high) in enumerate(CSP_BANDS_HZ):
        own, own_audit = fit_finger_csp_rows(
            csp_bins[band_index],
            target,
            training,
            finger_index,
            component_indices,
            history_offset=HISTORY_OFFSET,
        )
        rows.append(own)
        record: dict[str, object] = {
            "band_hz": [low, high],
            "decoded_finger": own_audit,
        }
        if include_any_movement:
            shared, shared_audit = fit_finger_csp_rows(
                csp_bins[band_index],
                shared_target,
                training,
                0,
                component_indices,
                history_offset=HISTORY_OFFSET,
            )
            rows.append(shared)
            record["any_movement"] = shared_audit
        audit.append(record)
    return np.concatenate(rows, axis=0).astype(np.float32), audit


def normalize_spatial_rows(
    weights: np.ndarray, ecog: np.ndarray, training: np.ndarray
) -> tuple[np.ndarray, list[float]]:
    bin_indices = training + HISTORY_OFFSET
    sample_indices = (
        bin_indices[:, None] * SAMPLES_PER_BIN
        + np.arange(SAMPLES_PER_BIN, dtype=np.int64)[None]
    ).reshape(-1)
    samples = np.asarray(ecog[sample_indices])
    projected = samples @ weights.T
    scales = np.std(projected, axis=0, dtype=np.float64)
    if np.any(scales < 1.0e-7):
        raise RuntimeError("CSP spatial projection has near-zero training variance")
    return (weights / scales[:, None]).astype(np.float32), scales.tolist()


@torch.inference_mode()
def extract_features(
    ecog: np.ndarray,
    spatial: np.ndarray,
    *,
    chunk_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, tuple[str, ...]]:
    ecog_tensor = torch.from_numpy(np.array(ecog, copy=True)).to(device)
    windows = ecog_tensor.unfold(0, SOURCE_RATE, SAMPLES_PER_BIN)
    model = ExactWindowFingerDecoder(
        input_channels=ecog.shape[1],
        component_count=spatial.shape[0],
        selected_indices=np.arange(1, dtype=np.int64),
        feature_mean=np.zeros(1, dtype=np.float32),
        feature_scale=np.ones(1, dtype=np.float32),
        hidden_size=1,
        frontend="asymmetric",
        head_initialization="lars_linear_regime",
        output_activation="linear",
    ).to(device).eval()
    model.spatial.weight[:, :, 0].copy_(torch.as_tensor(spatial, device=device))
    parts: list[np.ndarray] = []
    for begin in range(0, windows.shape[0], chunk_steps):
        stop = min(windows.shape[0], begin + chunk_steps)
        raw = windows[begin:stop]
        projected = model.spatial(raw)
        energy = model.wavelet(projected).flatten(1)
        parts.append(energy.float().cpu().numpy())
    return np.concatenate(parts), model.wavelet.band_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--finger", required=True, choices=tuple(FINGER_NAMES))
    parser.add_argument("--outer-folds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--full-refit",
        action="store_true",
        help="fit one CSP bank on all development rows and cache train/test features",
    )
    parser.add_argument("--csp-mode", choices=tuple(CSP_MODES), default="tails_2x2")
    parser.add_argument(
        "--csp-source",
        choices=("high_gamma", "broadband", "seven_band"),
        default="high_gamma",
    )
    parser.add_argument(
        "--include-any-movement-csp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="append any-finger-movement versus rest CSP rows in every source band",
    )
    parser.add_argument(
        "--band-cache-root",
        type=Path,
        default=Path("/dev/shm/ecog_csp_band_cache"),
    )
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument(
        "--ica-root", type=Path, default=Path("outputs/paper_ica_lars_v1")
    )
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path("outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/perfinger_ica_csp_wavelet_1000hz_cache_v1"),
    )
    parser.add_argument("--chunk-steps", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compact-csp-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "cache only the new CSP-wavelet columns; training joins them with "
            "the shared initialized ICA-wavelet cache"
        ),
    )
    args = parser.parse_args()

    started = time.perf_counter()
    definition_path = (
        args.fold_root / f"sub{args.subject}" / args.finger / "folds.json"
    )
    definition = json.loads(definition_path.read_text())
    rows = int(definition["training_rows"])
    target_name = str(definition["target"])
    prepared = args.prepared_root / f"sub{args.subject}"
    ecog = np.load(prepared / "train_ecog.npy", mmap_mode="r")
    complete_target = np.load(
        prepared / f"train_glove_{target_name}.npy", mmap_mode="r"
    )
    target = np.asarray(
        complete_target[HISTORY_OFFSET : HISTORY_OFFSET + rows], dtype=np.float32
    )
    ica = np.asarray(
        np.load(args.ica_root / f"sub{args.subject}" / "fastica_unmixing.npy"),
        dtype=np.float32,
    )
    if args.csp_source == "high_gamma":
        csp_bins = high_gamma_bins(ecog)
        csp_source = {"kind": "notch-separated high gamma", "bands_hz": HIGH_GAMMA_BANDS}
    elif args.csp_source == "broadband":
        bin_count = ecog.shape[0] // SAMPLES_PER_BIN
        csp_bins = np.asarray(ecog[: bin_count * SAMPLES_PER_BIN]).reshape(
            bin_count, SAMPLES_PER_BIN, ecog.shape[1]
        )
        csp_source = {"kind": "notched broadband", "bands_hz": None}
    else:
        csp_bins, csp_source = band_cache_bins(args.band_cache_root, args.subject)

    finger_index = list(FINGER_NAMES).index(args.finger)
    device = torch.device(args.device)
    if args.full_refit:
        full_target_name = target_name.removesuffix("_split_safe")
        full_target = np.load(
            prepared / f"train_glove_{full_target_name}.npy", mmap_mode="r"
        )
        train_rows = (ecog.shape[0] - SOURCE_RATE) // SAMPLES_PER_BIN + 1
        full_target = np.asarray(
            full_target[HISTORY_OFFSET : HISTORY_OFFSET + train_rows],
            dtype=np.float32,
        )
        training = np.arange(train_rows, dtype=np.int64)
        weights, csp_audit = fit_csp_rows(
            csp_bins,
            full_target,
            training,
            finger_index,
            CSP_MODES[args.csp_mode],
            include_any_movement=args.include_any_movement_csp,
        )
        weights, csp_scales = normalize_spatial_rows(weights, ecog, training)
        spatial = np.concatenate((ica, weights), axis=0).astype(np.float32)
        extraction_weights = weights if args.compact_csp_only else spatial
        train_features, band_names = extract_features(
            ecog, extraction_weights, chunk_steps=args.chunk_steps, device=device
        )
        test_ecog = np.load(prepared / "test_ecog.npy", mmap_mode="r")
        test_features, test_band_names = extract_features(
            test_ecog, extraction_weights, chunk_steps=args.chunk_steps, device=device
        )
        if test_band_names != band_names:
            raise RuntimeError("train and test wavelet feature orders disagree")
        destination = args.output_root / "full"
        destination.mkdir(parents=True, exist_ok=True)
        train_name = (
            "train_csp_features.npy" if args.compact_csp_only else "train_features.npy"
        )
        test_name = (
            "test_csp_features.npy" if args.compact_csp_only else "test_features.npy"
        )
        atomic_save(destination / train_name, train_features)
        atomic_save(destination / test_name, test_features)
        atomic_save(destination / "spatial_weights.npy", spatial)
        report = {
            "protocol": (
                "original-rate FastICA plus full-development finger-specific CSP; "
                "shared asymmetric wavelet tree"
            ),
            "subject": args.subject,
            "finger": args.finger,
            "target": full_target_name,
            "training_bins": int(training.size),
            "csp_mode": args.csp_mode,
            "csp_source": csp_source,
            "include_any_movement_csp": args.include_any_movement_csp,
            "ica_rows": int(ica.shape[0]),
            "csp_rows": int(weights.shape[0]),
            "spatial_shape": list(spatial.shape),
            "train_feature_shape": list(train_features.shape),
            "test_feature_shape": list(test_features.shape),
            "feature_order": "spatial row, asymmetric-wavelet band, 40-ms bin",
            "storage": "csp_only" if args.compact_csp_only else "combined_ica_csp",
            "band_names": list(band_names),
            "csp_projected_broadband_std_before_normalization": csp_scales,
            "csp_audit": csp_audit,
            "released_test_ecog_transformed": True,
            "released_test_labels_touched": False,
            "runtime_seconds": time.perf_counter() - started,
        }
        (destination / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)
        return
    records = []
    for outer_fold in args.outer_folds:
        for split_name, (training_intervals, validation_intervals) in split_masks(
            definition, outer_fold
        ).items():
            destination = args.output_root / f"outer{outer_fold}" / split_name
            summary_path = destination / "summary.json"
            features_path = destination / "features.npy"
            stored_features_path = (
                destination / "csp_features.npy"
                if args.compact_csp_only
                else features_path
            )
            spatial_path = destination / "spatial_weights.npy"
            if (
                summary_path.exists()
                and stored_features_path.exists()
                and spatial_path.exists()
            ):
                print(f"cached outer={outer_fold} split={split_name}", flush=True)
                continue
            split_started = time.perf_counter()
            training = indices_from_intervals(training_intervals)
            weights, csp_audit = fit_csp_rows(
                csp_bins,
                target,
                training,
                finger_index,
                CSP_MODES[args.csp_mode],
                include_any_movement=args.include_any_movement_csp,
            )
            weights, csp_scales = normalize_spatial_rows(weights, ecog, training)
            spatial = np.concatenate((ica, weights), axis=0).astype(np.float32)
            extraction_weights = weights if args.compact_csp_only else spatial
            features, band_names = extract_features(
                ecog, extraction_weights, chunk_steps=args.chunk_steps, device=device
            )
            features = features[:rows]
            destination.mkdir(parents=True, exist_ok=True)
            atomic_save(stored_features_path, features)
            atomic_save(spatial_path, spatial)
            report = {
                "protocol": (
                    "original-rate FastICA plus split-local finger-specific CSP; "
                    "shared asymmetric wavelet tree"
                ),
                "subject": args.subject,
                "finger": args.finger,
                "target": target_name,
                "outer_fold": outer_fold,
                "split": split_name,
                "training_intervals": training_intervals,
                "validation_intervals": validation_intervals,
                "training_bins": int(training.size),
                "csp_mode": args.csp_mode,
                "csp_source": csp_source,
                "include_any_movement_csp": args.include_any_movement_csp,
                "ica_rows": int(ica.shape[0]),
                "csp_rows": int(weights.shape[0]),
                "spatial_shape": list(spatial.shape),
                "feature_shape": list(features.shape),
                "feature_order": "spatial row, asymmetric-wavelet band, 40-ms bin",
                "storage": "csp_only" if args.compact_csp_only else "combined_ica_csp",
                "band_names": list(band_names),
                "csp_projected_broadband_std_before_normalization": csp_scales,
                "csp_audit": csp_audit,
                "released_test_touched": False,
                "runtime_seconds": time.perf_counter() - split_started,
            }
            summary_path.write_text(json.dumps(report, indent=2) + "\n")
            records.append(report)
            print(
                json.dumps(
                    {
                        "outer_fold": outer_fold,
                        "split": split_name,
                        "spatial_shape": report["spatial_shape"],
                        "feature_shape": report["feature_shape"],
                        "runtime_seconds": report["runtime_seconds"],
                    }
                ),
                flush=True,
            )
    split_report = {
        "subject": args.subject,
        "finger": args.finger,
        "generated_splits": len(records),
        "completed_splits": sum(
            1
            for outer_fold in args.outer_folds
            for split_name in split_masks(definition, outer_fold)
            if (
                args.output_root
                / f"outer{outer_fold}"
                / split_name
                / "summary.json"
            ).exists()
        ),
        "storage": "csp_only" if args.compact_csp_only else "combined_ica_csp",
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_root / "split_summary.json").write_text(
        json.dumps(split_report, indent=2) + "\n"
    )
    print(json.dumps(split_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
