import json
import sys
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "experiments" / "scripts"))

from audit_canonical_models import audit_manifest  # noqa: E402


FINGERS = ("thumb", "index", "middle", "ring", "little")


def write_manifest(tmp_path: Path) -> Path:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "released_test_touched": False,
                "initialized": 0.25,
                "selected": 0.30,
            }
        )
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    models = {}
    artifact_pairs = {}
    for subject in (1, 2, 3):
        for finger in FINGERS:
            pair = f"S{subject}_{finger}"
            artifact_pairs[pair] = {
                "included_seed_count": 2,
                "ensemble_pcc": 0.35,
                "members": [{"seed": 0}, {"seed": 1}],
            }
            models[pair] = {
                "structure_id": "one_structure",
                "artifact": {
                    "kind": "same_structure_seed_ensemble",
                    "root": str(artifact),
                    "status": "complete",
                    "members": [0, 1],
                    "member_structure_ids": ["one_structure"],
                },
                "development_evidence": {
                    "initialized": {
                        "summary": str(summary),
                        "key": "initialized",
                    },
                    "selected": {
                        "summary": str(summary),
                        "key": "selected",
                    },
                },
            }
    (artifact / "aggregate_summary.json").write_text(
        json.dumps({"pairs": artifact_pairs})
    )
    manifest = tmp_path / "canonical.yaml"
    manifest.write_text(yaml.safe_dump({"version": 1, "models": models}))
    return manifest


def test_complete_positive_same_structure_registry_passes(tmp_path: Path) -> None:
    report = audit_manifest(write_manifest(tmp_path))

    assert report["observed_pair_count"] == 15
    assert report["all_development_rules_pass"] is True
    assert report["all_release_rules_pass"] is True


def test_negative_tuning_and_mixed_structure_ensemble_fail(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    summary_path = Path(
        manifest["models"]["S1_thumb"]["development_evidence"]["selected"][
            "summary"
        ]
    )
    summary = json.loads(summary_path.read_text())
    summary["selected"] = 0.20
    summary_path.write_text(json.dumps(summary))
    manifest["models"]["S2_ring"]["artifact"]["member_structure_ids"] = [
        "one_structure",
        "another_structure",
    ]
    manifest_path.write_text(yaml.safe_dump(manifest))

    report = audit_manifest(manifest_path)

    assert report["all_development_rules_pass"] is False
    assert set(report["development_failing_pairs"]) == set(report["pairs"])
    assert report["pairs"]["S2_ring"]["no_model_soup"] is False
