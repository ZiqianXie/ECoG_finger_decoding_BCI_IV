import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "experiments" / "scripts"))

from aggregate_canonical_refits import aggregate  # noqa: E402


def write_member(root: Path, seed: int, width: int = 4) -> Path:
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps(
            {
                "subject": 1,
                "finger": "thumb",
                "seed": seed,
                "released_test_used_for_selection": False,
                "structure_settings": {
                    "hidden_size": width,
                    "cell": "lstm",
                    "sampler_seed": seed,
                },
            }
        )
    )
    np.save(root / "test_raw_target.npy", np.arange(5, dtype=np.float32))
    np.save(
        root / "test_prediction_raw_scale.npy",
        np.arange(5, dtype=np.float32) + seed / 100.0,
    )
    return root


def test_equal_structure_members_are_aggregated(tmp_path: Path) -> None:
    members = [write_member(tmp_path / f"seed{seed}", seed) for seed in (1, 2)]
    report = aggregate(members, tmp_path / "ensemble", "same_lstm")

    assert report["member_count"] == 2
    assert report["member_structure_count"] == 1
    assert report["heterogeneous_model_soup"] is False


def test_mixed_structures_are_rejected(tmp_path: Path) -> None:
    members = [
        write_member(tmp_path / "seed1", 1, width=4),
        write_member(tmp_path / "seed2", 2, width=8),
    ]

    with pytest.raises(ValueError, match="heterogeneous model structures"):
        aggregate(members, tmp_path / "ensemble", "invalid")
