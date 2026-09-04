#!/usr/bin/env python3
"""Prepare notched ECoG and both trajectory target definitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ecog_decoding.io import BAD_CHANNELS_ONE_BASED, load_subject
from ecog_decoding.preprocessing import (
    apply_movement_corrector,
    downsample_glove,
    fit_movement_corrector,
    preprocess_ecog,
)


def save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/bci_competition_iv_ds4"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/preprocessed"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/preprocessing.yaml")
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    for subject in args.subjects:
        raw = load_subject(args.data_root, subject)
        ecog = preprocess_ecog(
            raw.train_ecog,
            raw.test_ecog,
            subject=subject,
            fs=float(config["sampling_rate_hz"]),
            notch_frequencies=config["notch_frequencies_hz"],
            notch_quality_factor=float(config["notch_quality_factor"]),
            normalization_fit_fraction=float(config["normalization_fit_fraction"]),
        )
        train_glove = downsample_glove(
            raw.train_glove,
            source_rate_hz=int(config["sampling_rate_hz"]),
            target_rate_hz=int(config["glove_rate_hz"]),
        )
        test_glove = downsample_glove(
            raw.test_glove,
            source_rate_hz=int(config["sampling_rate_hz"]),
            target_rate_hz=int(config["glove_rate_hz"]),
        )
        target_fit_stop = int(
            round(train_glove.shape[0] * float(config["normalization_fit_fraction"]))
        )
        movement_model = fit_movement_corrector(
            train_glove[:target_fit_stop],
            sampling_rate_hz=float(config["glove_rate_hz"]),
            baseline_smoothness=float(config["baseline_smoothness"]),
            baseline_asymmetry=float(config["baseline_asymmetry"]),
            baseline_iterations=int(config["baseline_iterations"]),
            activation_threshold=float(config["movement_activation_threshold"]),
            state_transition_penalty=float(config["movement_state_transition_penalty"]),
            coupling_minimum_activation=float(
                config["coupling_minimum_activation"]
            ),
            coupling_em_iterations=int(config["coupling_em_iterations"]),
        )
        movement = apply_movement_corrector(train_glove, movement_model)

        output = args.output_root / f"sub{subject}"
        save_array(output / "train_ecog.npy", ecog.train)
        save_array(output / "test_ecog.npy", ecog.test)
        save_array(output / "train_glove_25hz_raw.npy", train_glove.astype(np.float32))
        save_array(output / "test_glove_25hz_raw.npy", test_glove.astype(np.float32))
        save_array(output / "train_target_intended.npy", movement.intended)
        save_array(
            output / "train_glove_detrended_multifinger.npy",
            movement.detrended_multifinger,
        )
        save_array(output / "train_glove_baseline.npy", movement.baseline)
        save_array(output / "train_active_finger.npy", movement.active_finger)
        save_array(
            output / "finger_coupling_matrix.npy",
            movement_model.coupling_matrix.astype(np.float32),
        )

        fit_samples_1khz = int(
            round(raw.train_ecog.shape[0] * float(config["normalization_fit_fraction"]))
        )
        metadata = {
            "subject": subject,
            "input_shapes": {
                "train_ecog": list(raw.train_ecog.shape),
                "test_ecog": list(raw.test_ecog.shape),
                "train_glove": list(raw.train_glove.shape),
                "test_glove": list(raw.test_glove.shape),
            },
            "output_shapes": {
                "train_ecog": list(ecog.train.shape),
                "test_ecog": list(ecog.test.shape),
                "train_glove_25hz": list(train_glove.shape),
                "test_glove_25hz": list(test_glove.shape),
            },
            "bad_channels_one_based": list(BAD_CHANNELS_ONE_BASED[subject]),
            "retained_channels_one_based": list(ecog.retained_channels_one_based),
            "normalization_fit_samples_1khz": fit_samples_1khz,
            "normalization_mean": ecog.mean.tolist(),
            "normalization_scale": ecog.scale.tolist(),
            "target_fit_samples_25hz": target_fit_stop,
            "target_scale": movement_model.scale.tolist(),
            "finger_coupling_matrix": movement_model.coupling_matrix.tolist(),
            "movement_event_count": len(movement.events),
            "config": config,
            "notes": {
                "target_policy": (
                    "export uncoupled and event-corrected variants; select only "
                    "by chronological validation"
                ),
                "recommended_primary_target": (
                    "baseline-subtracted multifinger trajectory without "
                    "winner-take-all reassignment"
                ),
                "experimental_target": (
                    "event-level coupling correction retained for audit only"
                ),
                "test_metric": "use raw released test glove trajectories",
            },
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"prepared subject {subject}: {output}")


if __name__ == "__main__":
    main()
