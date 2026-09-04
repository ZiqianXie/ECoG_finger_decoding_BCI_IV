#!/usr/bin/env python3
"""Generate an asymmetric depth-4 wavelet bank plus a signed LMP branch.

The shallow diagnostic retains depth-3 paths outside 60--200 Hz and replaces
only parents carrying substantial power in that range with their two depth-4
children.  A signed 0--5 Hz low-passed mean is appended for each ICA component.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ecog_decoding.models import WaveletPacketEnergy
from ecog_decoding.spectral_audit import wavelet_path_responses


def choose_split_parents(
    frequency: np.ndarray,
    magnitude: np.ndarray,
    low_hz: float = 60.0,
    high_hz: float = 200.0,
    minimum_power_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Select depth-3 paths with enough initialized power in the target range."""
    power = np.square(np.asarray(magnitude, dtype=np.float64))
    mask = (frequency >= low_hz) & (frequency <= high_hz)
    fraction = power[:, mask].sum(axis=1) / np.maximum(power.sum(axis=1), 1.0e-12)
    selected = np.flatnonzero(fraction >= minimum_power_fraction)
    if selected.size == 0:
        selected = np.asarray([int(np.argmax(fraction))])
    return selected.astype(np.int64), fraction


def lowpass_fir(
    sampling_rate_hz: float = 1000.0,
    cutoff_hz: float = 5.0,
    taps: int = 201,
) -> np.ndarray:
    """Return a symmetric unit-DC-gain windowed-sinc low-pass kernel."""
    if taps < 3 or taps % 2 == 0:
        raise ValueError("taps must be an odd integer of at least 3")
    normalized = cutoff_hz / sampling_rate_hz
    offsets = np.arange(taps, dtype=np.float64) - (taps - 1) / 2
    kernel = 2.0 * normalized * np.sinc(2.0 * normalized * offsets)
    kernel *= np.hamming(taps)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


@torch.inference_mode()
def generate(
    ecog: np.ndarray,
    unmixing: np.ndarray,
    depth3: WaveletPacketEnergy,
    depth4: WaveletPacketEnergy,
    split_parents: np.ndarray,
    lmp_kernel: torch.Tensor,
    output: Path,
    window_samples: int,
    stride_samples: int,
    batch_size: int,
    device: torch.device,
) -> tuple[int, int, list[str]]:
    windows = np.lib.stride_tricks.sliding_window_view(
        ecog, window_shape=window_samples, axis=0
    )[::stride_samples]
    retained = np.asarray(
        [index for index in range(8) if index not in set(split_parents.tolist())],
        dtype=np.int64,
    )
    children = np.asarray(
        [child for parent in split_parents for child in (2 * parent, 2 * parent + 1)],
        dtype=np.int64,
    )
    names = (
        [f"D3_{depth3.band_names[index]}" for index in retained]
        + [f"D4_{depth4.band_names[index]}" for index in children]
        + ["LMP_0_5_HZ_SIGNED"]
    )
    bins = window_samples // depth3.energy_stride_samples
    feature_count = unmixing.shape[0] * len(names) * bins
    features = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(windows.shape[0], feature_count)
    )
    weights = torch.as_tensor(unmixing, dtype=torch.float32, device=device)[:, :, None]
    retained_t = torch.as_tensor(retained, dtype=torch.long, device=device)
    children_t = torch.as_tensor(children, dtype=torch.long, device=device)
    for start in range(0, windows.shape[0], batch_size):
        stop = min(start + batch_size, windows.shape[0])
        batch = torch.from_numpy(np.ascontiguousarray(windows[start:stop])).to(device)
        spatial = F.conv1d(batch, weights)
        depth3_energy = depth3(spatial).index_select(2, retained_t)
        depth4_energy = depth4(spatial).index_select(2, children_t)
        flat_spatial = spatial.flatten(0, 1).unsqueeze(1)
        half = lmp_kernel.shape[-1] // 2
        lmp = F.conv1d(F.pad(flat_spatial, (half, half), mode="reflect"), lmp_kernel)
        lmp = F.avg_pool1d(
            lmp,
            kernel_size=depth3.energy_window_samples,
            stride=depth3.energy_stride_samples,
        ).reshape(spatial.shape[0], spatial.shape[1], 1, bins)
        combined = torch.cat((depth3_energy, depth4_energy, lmp), dim=2)
        features[start:stop] = combined.flatten(1).cpu().numpy()
        if start == 0 or stop == windows.shape[0] or stop % (batch_size * 20) == 0:
            print(f"{output.name}: {stop}/{windows.shape[0]}", flush=True)
    features.flush()
    return windows.shape[0], feature_count, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--ica-root", type=Path, default=Path("outputs/paper_ica_lars_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/windowed_ica_wavelet_asymmetric_v1"))
    parser.add_argument("--window-samples", type=int, default=1000)
    parser.add_argument("--stride-samples", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--minimum-split-power", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    prepared = args.prepared_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    depth3 = WaveletPacketEnergy(
        levels=3, trainable=False, padding_mode="constant"
    ).to(device).eval()
    depth4 = WaveletPacketEnergy(
        levels=4, trainable=False, padding_mode="constant"
    ).to(device).eval()
    frequency, magnitude = wavelet_path_responses(depth3)
    split_parents, power_fraction = choose_split_parents(
        frequency, magnitude, minimum_power_fraction=args.minimum_split_power
    )
    lmp_kernel = torch.from_numpy(lowpass_fir()).to(device)[None, None]
    unmixing = np.load(
        args.ica_root / f"sub{args.subject}" / "fastica_unmixing.npy"
    )
    result: dict[str, object] = {
        "subject": args.subject,
        "split_range_hz": [60.0, 200.0],
        "minimum_split_power_fraction": args.minimum_split_power,
        "depth3_path_power_fraction_60_200_hz": {
            name: float(value)
            for name, value in zip(depth3.band_names, power_fraction, strict=True)
        },
        "split_depth3_paths": [depth3.band_names[index] for index in split_parents],
        "lmp": {"cutoff_hz": 5.0, "signed": True, "fir_taps": 201},
    }
    for prefix in ("train", "test"):
        ecog = np.load(prepared / f"{prefix}_ecog.npy", mmap_mode="r")
        rows, columns, names = generate(
            ecog,
            unmixing,
            depth3,
            depth4,
            split_parents,
            lmp_kernel,
            output / f"{prefix}_initialized_window_features.npy",
            args.window_samples,
            args.stride_samples,
            args.batch_size,
            device,
        )
        result[prefix] = {"rows": rows, "features": columns}
        result["feature_band_names"] = names
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
