import json
from pathlib import Path

import yaml

from summarize_single_wavelet_ensemble import resolve_route


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_pair_route_overrides_default_whole_model() -> None:
    route_map = {
        "default": {"ensemble_root": "outputs/default", "seeds": [0, 1]},
        "subjects": {
            1: {
                "little": {
                    "ensemble_root": "outputs/little",
                    "seeds": [2, 3, 4],
                    "csp_mode": "tails_4x4",
                }
            }
        },
    }

    default_root, default_seeds, default_options = resolve_route(
        route_map, 1, "thumb", Path("unused"), (9,)
    )
    little_root, little_seeds, little_options = resolve_route(
        route_map, 1, "little", Path("unused"), (9,)
    )

    assert default_root == Path("outputs/default")
    assert default_seeds == (0, 1)
    assert default_options["ensemble_root"] == "outputs/default"
    assert little_root == Path("outputs/little")
    assert little_seeds == (2, 3, 4)
    assert little_options["csp_mode"] == "tails_4x4"


def test_final_route_and_published_artifact_cover_fifteen_single_paths() -> None:
    route_path = REPOSITORY_ROOT / "configs/final_single_wavelet_routes.yaml"
    route_map = yaml.safe_load(route_path.read_text())
    roots = {}
    for subject in (1, 2, 3):
        for finger in ("thumb", "index", "middle", "ring", "little"):
            root, seeds, options = resolve_route(
                route_map, subject, finger, Path("unused"), (9,)
            )
            roots[f"S{subject}_{finger}"] = root
            assert seeds == (0, 1, 2, 3, 4, 5)
            assert options["model_rate_hz"] == 1000

    assert len(roots) == 15
    assert roots["S1_thumb"].name.endswith("spatial_grid_ensemble_v1")
    assert roots["S1_middle"].name.endswith("middle_tails2x2_ensemble_v1")
    assert roots["S1_little"].name.endswith("little_tails4x4_ensemble_v1")

    result = json.loads(
        (
            REPOSITORY_ROOT
            / "docs/results/final-single-branch-six-seed.json"
        ).read_text()
    )
    assert len(result["pairs"]) == 15
    assert result["subjects"]["S1"]["macro_5_pcc"] > 0.556
    assert result["subjects"]["S2"]["macro_5_pcc"] > 0.408
    assert result["subjects"]["S3"]["macro_5_pcc"] > 0.582
    assert all(
        pair["configuration"]["frontend"]["leaf_count"] == 8
        and pair["configuration"]["frontend"]["auxiliary_temporal_branches"] == []
        and pair["configuration"]["frontend"]["lmp_branch"] is False
        and pair["configuration"]["training_objective"]["model_outputs"]
        == ["trajectory"]
        for pair in result["pairs"].values()
    )
