#!/usr/bin/env python3
"""Measure which CSP frequency bands carry each finger's decodable signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark_csp_ridge import BANDS_HZ
from benchmark_ridge_target_variants import fit_candidate, lagged
from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--csp-root", type=Path, default=Path("outputs/csp_ridge_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/csp_band_diagnostic_v2"))
    parser.add_argument("--target", default="local_w4_q20")
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    prepared = args.prepared_root / f"sub{args.subject}"
    csp = args.csp_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((prepared / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    energy_train = np.load(csp / "train_csp_energy.npy")
    energy_test = np.load(csp / "test_csp_energy.npy")
    target = np.load(prepared / f"train_glove_{args.target}.npy")
    raw_train = np.load(prepared / "train_glove_25hz_raw.npy")
    raw_test = np.load(prepared / "test_glove_25hz_raw.npy")
    offset = args.history - 1
    cleaned_test = np.load(prepared / f"test_glove_{args.target}.npy")[offset:]
    train_count = split - offset
    per_band = energy_train.shape[1] // len(BANDS_HZ)
    reports: dict[str, object] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    matrix = []

    band_specs = [(f"{low:g}-{high:g}", [idx]) for idx, (low, high) in enumerate(BANDS_HZ)]
    band_specs += [("all", list(range(len(BANDS_HZ))))]
    for name, band_ids in band_specs:
        columns = np.concatenate([
            np.arange(idx * per_band, (idx + 1) * per_band) for idx in band_ids
        ])
        train_x_all = lagged(energy_train[:, columns], args.history)
        test_x = lagged(energy_test[:, columns], args.history)
        result, validation_prediction, test_prediction = fit_candidate(
            train_x_all[:train_count],
            target[offset:split],
            train_x_all[train_count:],
            target[split:],
            raw_train[split:],
            test_x,
            raw_test[offset:],
            args.top_features,
            torch.device(args.device),
        )
        reports[name] = result
        validation_predictions[name] = validation_prediction
        test_predictions[name] = test_prediction
        np.save(output / f"validation_prediction_{name.replace('-', '_')}.npy", validation_prediction)
        np.save(output / f"test_prediction_{name.replace('-', '_')}.npy", test_prediction)
        matrix.append(list(result["test_raw_metrics"]["pearson_by_finger"].values()))
        print(name, json.dumps(result["test_raw_metrics"]), flush=True)

    selected_bands: dict[str, str] = {}
    ensemble_validation = np.zeros_like(next(iter(validation_predictions.values())))
    ensemble_test = np.zeros_like(next(iter(test_predictions.values())))
    for finger, finger_name in enumerate(FINGER_NAMES):
        selected = max(
            validation_predictions,
            key=lambda name: reports[name]["validation_target_metrics"]["pearson_by_finger"][finger_name],
        )
        selected_bands[finger_name] = selected
        ensemble_validation[:, finger] = validation_predictions[selected][:, finger]
        ensemble_test[:, finger] = test_predictions[selected][:, finger]
    ensemble_report = {
        "selected_by_validation_cleaned_pcc": selected_bands,
        "validation_cleaned_metrics": trajectory_metrics(ensemble_validation, target[split:]),
        "validation_raw_metrics": trajectory_metrics(ensemble_validation, raw_train[split:]),
        "test_cleaned_metrics": trajectory_metrics(ensemble_test, cleaned_test),
        "test_raw_metrics": trajectory_metrics(ensemble_test, raw_test[offset:]),
    }
    np.save(output / "validation_prediction_selected.npy", ensemble_validation)
    np.save(output / "test_prediction_selected.npy", ensemble_test)
    print("selected", json.dumps(ensemble_report), flush=True)

    labels = [name for name, _ in band_specs]
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    image = axis.imshow(np.asarray(matrix), vmin=0.0, vmax=0.75, cmap="viridis", aspect="auto")
    axis.set_xticks(range(5), ["thumb", "index", "middle", "ring", "little"])
    axis.set_yticks(range(len(labels)), labels)
    axis.set_ylabel("frequency band (Hz)")
    axis.set_title(f"Subject {args.subject}: held-out PCC by CSP band and finger")
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", color="white" if value < .35 else "black")
    figure.colorbar(image, ax=axis, label="PCC against released target")
    figure.savefig(output / "band_by_finger.png", dpi=160)
    plt.close(figure)
    (output / "summary.json").write_text(
        json.dumps({"subject": args.subject, "bands": reports, "selected_ensemble": ensemble_report}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
