#!/usr/bin/env python3
"""Generate the paper-style CNN features independently in each 1 s window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ecog_decoding.models import WaveletPacketEnergy


@torch.inference_mode()
def generate(
    ecog: np.ndarray,
    unmixing: np.ndarray,
    module: WaveletPacketEnergy,
    output: Path,
    window_samples: int,
    stride_samples: int,
    batch_size: int,
    device: torch.device,
) -> tuple[int, int]:
    windows = np.lib.stride_tricks.sliding_window_view(
        ecog, window_shape=window_samples, axis=0
    )[::stride_samples]
    row_count = windows.shape[0]
    component_count = unmixing.shape[0]
    feature_count = component_count * (2**module.levels) * (
        window_samples // module.energy_stride_samples
    )
    features = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.float32,
        shape=(row_count, feature_count),
    )
    weights = torch.as_tensor(unmixing, dtype=torch.float32, device=device)[:, :, None]
    for start in range(0, row_count, batch_size):
        stop = min(start + batch_size, row_count)
        # sliding_window_view is (window, channel, sample). Copying only the
        # current batch avoids materializing the full overlapping array.
        batch = torch.from_numpy(np.ascontiguousarray(windows[start:stop])).to(device)
        spatial = torch.nn.functional.conv1d(batch, weights)
        energy = module(spatial).flatten(1)
        features[start:stop] = energy.cpu().numpy()
        if start == 0 or stop == row_count or stop % (batch_size * 20) == 0:
            print(f"{output.name}: {stop}/{row_count}", flush=True)
    features.flush()
    return row_count, feature_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--ica-root", type=Path, default=Path("outputs/paper_ica_lars_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/windowed_ica_wavelet_v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--window-samples", type=int, default=1000)
    parser.add_argument("--stride-samples", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--padding-mode", choices=("constant", "reflect"), default="constant")
    parser.add_argument(
        "--spatial-scale",
        type=float,
        default=1.0,
        help="multiply FastICA weights before the scaled-tanh wavelet tree",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    prepared = args.prepared_root / f"sub{args.subject}"
    source = args.ica_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    front = config["wavelet_packet_frontend"]
    device = torch.device(args.device)
    module = WaveletPacketEnergy(
        wavelet=str(front["wavelet"]),
        levels=int(front["levels"]),
        kernel_size=int(front["kernel_size"]),
        trainable=False,
        padding_mode=args.padding_mode,
        energy_window_samples=int(front["energy_window_samples"]),
        energy_stride_samples=int(front["energy_stride_samples"]),
    ).to(device).eval()
    unmixing = np.load(source / "fastica_unmixing.npy") * float(args.spatial_scale)
    result: dict[str, object] = {
        "subject": args.subject,
        "ica_root": str(args.ica_root),
        "window_samples": args.window_samples,
        "stride_samples": args.stride_samples,
        "padding_mode": args.padding_mode,
        "spatial_scale": args.spatial_scale,
    }
    for prefix in ("train", "test"):
        ecog = np.load(prepared / f"{prefix}_ecog.npy", mmap_mode="r")
        rows, columns = generate(
            ecog,
            unmixing,
            module,
            output / f"{prefix}_initialized_window_features.npy",
            args.window_samples,
            args.stride_samples,
            args.batch_size,
            device,
        )
        result[prefix] = {"rows": rows, "features": columns}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
