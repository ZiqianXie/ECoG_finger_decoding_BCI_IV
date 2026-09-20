#!/usr/bin/env python3
"""Audit whether a trained all-finger trajectory head contains target signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cross_validate_single_wavelet import cached_initialization_features
from ecog_decoding.training import FINGER_NAMES
from single_wavelet_model import SingleWaveletDecoder
from single_wavelet_support import OFFSET, pearson
from train_event_grouped_lars_lstm import indices_from_intervals


def predict_intervals(model, features, intervals, rows, finger_index):
    primary = np.full(rows, np.nan, dtype=np.float32)
    auxiliary = np.full(rows, np.nan, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start, stop in intervals:
            decoded, all_fingers = model.decode_features_with_auxiliary(
                features[start:stop][None]
            )
            primary[start:stop] = decoded[0].float().cpu().numpy()
            auxiliary[start:stop] = (
                all_fingers[0, :, finger_index].float().cpu().numpy()
            )
    return primary, auxiliary


def make_model(initialization, spatial, settings):
    return SingleWaveletDecoder(
        spatial,
        initialization,
        int(settings["hidden_size"]),
        output_activation=settings["output_activation"],
        softplus_beta=float(settings["softplus_beta"]),
        recurrent_cell=settings["recurrent_cell"],
        energy_window_samples=int(settings["samples_per_bin"]),
        tap_resample_up=int(settings["tap_resample_up"]),
        tap_resample_down=int(settings["tap_resample_down"]),
        wavelet_frontend=settings["wavelet_frontend"],
        wavelet_interlevel_skip=bool(settings["wavelet_interlevel_skip"]),
        wavelet_interlevel_normalization=bool(
            settings["wavelet_interlevel_normalization"]
        ),
        wavelet_final_normalization=bool(settings["wavelet_final_normalization"]),
        residual_input=settings["residual_input"],
        residual_history_bins=int(settings["residual_history_bins"]),
        residual_input_width=int(settings["residual_input_width"]),
        residual_include_direct=bool(settings["residual_include_direct"]),
        movement_head=True,
        movement_head_outputs=5,
        residual_output_init_std=float(settings["residual_output_init_std"]),
        residual_dynamics=settings["residual_dynamics"],
        residual_decay=float(settings["residual_decay"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--finger", choices=tuple(FINGER_NAMES), required=True)
    parser.add_argument("--fixed-reference-pcc", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    finger_index = list(FINGER_NAMES).index(args.finger)
    part_root = args.root / "parts" / f"sub{args.subject}" / args.finger
    if not part_root.is_dir():
        part_root = args.root / f"sub{args.subject}" / args.finger
    first_report = json.loads((part_root / "outer0" / "summary.json").read_text())
    prepared_root = Path(first_report["settings"]["prepared_root"])
    raw_full = np.load(
        prepared_root / f"sub{args.subject}" / "train_glove_25hz_raw.npy",
        mmap_mode="r",
    )
    rows = np.load(part_root / "outer0" / "selected_oof.npy").size
    raw = np.asarray(raw_full[OFFSET : OFFSET + rows, finger_index])
    primary_oof = np.full(rows, np.nan, dtype=np.float32)
    auxiliary_oof = np.full(rows, np.nan, dtype=np.float32)

    for outer_fold in range(3):
        output = part_root / f"outer{outer_fold}"
        report = json.loads((output / "summary.json").read_text())
        settings = report["settings"]
        cache = (
            Path(settings["initialization_cache_root"])
            / "cache"
            / f"outer{outer_fold}"
            / "outer"
        )
        saved = np.load(cache / "initialization.npz")
        initialization = {name: saved[name] for name in saved.files}
        spatial = np.load(cache / "spatial.npy")
        split = json.loads((cache / "split.json").read_text())
        target = np.load(cache / "target.npy")
        device = torch.device(args.device)
        model = make_model(initialization, spatial, settings).to(device)
        model.load_state_dict(
            torch.load(
                output / f"outer{outer_fold}_model.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        features = torch.from_numpy(
            cached_initialization_features(initialization, settings["residual_input"])
        ).to(device)
        training_intervals = split["training_intervals"]
        validation_intervals = split["validation_intervals"]
        train_primary, train_auxiliary = predict_intervals(
            model, features, training_intervals, rows, finger_index
        )
        validation_primary, validation_auxiliary = predict_intervals(
            model, features, validation_intervals, rows, finger_index
        )
        training = indices_from_intervals(training_intervals)
        centered_auxiliary = train_auxiliary[training] - train_auxiliary[training].mean()
        centered_target = target[training, finger_index] - target[training, finger_index].mean()
        slope = float(
            centered_auxiliary @ centered_target
            / max(centered_auxiliary @ centered_auxiliary, 1.0e-8)
        )
        intercept = float(
            target[training, finger_index].mean()
            - slope * train_auxiliary[training].mean()
        )
        validation = indices_from_intervals(validation_intervals)
        primary_oof[validation] = validation_primary[validation]
        auxiliary_oof[validation] = (
            intercept + slope * validation_auxiliary[validation]
        )

    observed = np.isfinite(primary_oof) & np.isfinite(auxiliary_oof)
    result = {
        "primary_pcc": pearson(primary_oof[observed], raw[observed]),
        "calibrated_auxiliary_pcc": pearson(auxiliary_oof[observed], raw[observed]),
        "fixed_blends": {},
    }
    for auxiliary_weight in (0.25, 0.5, 0.75):
        blended = (1.0 - auxiliary_weight) * primary_oof + auxiliary_weight * auxiliary_oof
        pcc = pearson(blended[observed], raw[observed])
        result["fixed_blends"][str(auxiliary_weight)] = {
            "pcc": pcc,
            "gain_over_fixed_reference": pcc - args.fixed_reference_pcc,
        }
    text = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
