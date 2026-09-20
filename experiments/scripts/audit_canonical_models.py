#!/usr/bin/env python3
"""Audit the model-family-neutral canonical ECoG registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


FINGERS = ("thumb", "index", "middle", "ring", "little")
EXPECTED_PAIRS = {
    f"S{subject}_{finger}" for subject in (1, 2, 3) for finger in FINGERS
}
ALLOWED_ARTIFACT_KINDS = {"single_model", "same_structure_seed_ensemble"}


def nested_value(record: dict[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"missing key {dotted_key!r} at component {part!r}")
        value = value[part]
    return value


def evidence_value(spec: dict[str, str]) -> tuple[float, dict[str, Any]]:
    summary_path = Path(spec["summary"])
    summary = json.loads(summary_path.read_text())
    return float(nested_value(summary, spec["key"])), summary


def released_test_used(summary: dict[str, Any]) -> bool:
    for key in (
        "released_test_touched",
        "released_test_used_for_selection",
        "released_test_labels_used",
    ):
        if key in summary:
            return bool(summary[key])
    return False


def validate_artifact(
    pair: str,
    structure_id: str,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Validate the on-disk release and extract its descriptive test PCC."""
    root = Path(str(artifact["root"]))
    summary_path = root / "summary.json"
    if not summary_path.exists():
        summary_path = root / "aggregate_summary.json"
    result: dict[str, Any] = {
        "artifact_root_exists": root.exists(),
        "artifact_summary": str(summary_path),
        "artifact_summary_exists": summary_path.exists(),
        "artifact_summary_valid": False,
        "released_test_pcc_descriptive_only": None,
        "full_development_refit_gain_min": None,
        "artifact_validation_error": None,
    }
    if not root.exists() or not summary_path.exists():
        return result

    try:
        summary = json.loads(summary_path.read_text())
        members = list(artifact.get("members", []))
        if "pairs" in summary:
            pair_summary = summary["pairs"][pair]
            summary_members = list(pair_summary["members"])
            summary_seeds = [member["seed"] for member in summary_members]
            if summary_seeds != members:
                raise ValueError(
                    f"manifest members {members} != artifact members {summary_seeds}"
                )
            if int(pair_summary["included_seed_count"]) != len(members):
                raise ValueError("included seed count does not match manifest")
            pcc = pair_summary.get(
                "ensemble_pcc", pair_summary.get("mean_prediction_pcc")
            )
        else:
            subject_text, finger = pair.split("_", 1)
            if int(summary["subject"]) != int(subject_text[1:]):
                raise ValueError("artifact subject does not match manifest pair")
            if str(summary["finger"]) != finger:
                raise ValueError("artifact finger does not match manifest pair")
            if str(summary["structure_id"]) != structure_id:
                raise ValueError("artifact structure_id does not match manifest")
            if bool(summary.get("heterogeneous_model_soup", False)):
                raise ValueError("artifact reports a heterogeneous model soup")
            if "member_count" in summary:
                if int(summary["member_count"]) != len(members):
                    raise ValueError("artifact member count does not match manifest")
                summary_seeds = list(summary.get("member_seeds", []))
                if summary_seeds and summary_seeds != members:
                    raise ValueError("artifact member seeds do not match manifest")
                member_paths = [Path(path) for path in summary.get("member_paths", [])]
                if len(member_paths) != len(members) or not all(
                    (path / "summary.json").exists() for path in member_paths
                ):
                    raise ValueError("one or more aggregate member artifacts are missing")
                full_refit_gains = []
                for path in member_paths:
                    member_summary = json.loads((path / "summary.json").read_text())
                    full_refit_gains.append(
                        float(member_summary["development_refit_pcc"])
                        - float(member_summary["development_initialization_pcc"])
                    )
                if min(full_refit_gains) <= 0.0:
                    raise ValueError("one or more full-development refits lost PCC")
                result["full_development_refit_gain_min"] = min(full_refit_gains)
                pcc = summary["released_test_ensemble_pcc_descriptive_only"]
            else:
                if len(members) != 1 or not (root / "ridge_model.npz").exists():
                    raise ValueError("single ridge artifact is incomplete")
                full_refit_gain = float(summary["development_refit_pcc"]) - float(
                    summary["development_initialization_pcc"]
                )
                if full_refit_gain <= 0.0:
                    raise ValueError("full-development ridge refit lost PCC")
                result["full_development_refit_gain_min"] = full_refit_gain
                pcc = summary["released_test_raw_pcc_descriptive_only"]
        if released_test_used(summary):
            raise ValueError("artifact reports released test use for selection")
        result["released_test_pcc_descriptive_only"] = float(pcc)
        result["artifact_summary_valid"] = True
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        result["artifact_validation_error"] = str(error)
    return result


def audit_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text())
    models = manifest["models"]
    observed_pairs = set(models)
    missing_pairs = sorted(EXPECTED_PAIRS - observed_pairs)
    unexpected_pairs = sorted(observed_pairs - EXPECTED_PAIRS)
    pair_reports: dict[str, Any] = {}

    for pair in sorted(EXPECTED_PAIRS & observed_pairs):
        model = models[pair]
        structure_id = str(model["structure_id"])
        artifact = model["artifact"]
        kind = str(artifact["kind"])
        members = list(artifact.get("members", []))
        member_structures = set(artifact.get("member_structure_ids", []))
        no_model_soup = (
            kind in ALLOWED_ARTIFACT_KINDS
            and bool(members)
            and member_structures == {structure_id}
            and (kind != "single_model" or len(members) == 1)
            and (kind != "same_structure_seed_ensemble" or len(members) >= 2)
        )

        initialized, initialized_summary = evidence_value(
            model["development_evidence"]["initialized"]
        )
        selected, selected_summary = evidence_value(
            model["development_evidence"]["selected"]
        )
        tuning_gain = selected - initialized
        tuning_pass = tuning_gain > 0.0
        test_selection_clean = not (
            released_test_used(initialized_summary)
            or released_test_used(selected_summary)
        )
        artifact_root = Path(str(artifact["root"]))
        artifact_validation = validate_artifact(
            pair, structure_id, artifact
        )
        artifact_complete = (
            artifact.get("status") == "complete"
            and artifact_validation["artifact_summary_valid"]
        )
        pair_reports[pair] = {
            "structure_id": structure_id,
            "artifact_kind": kind,
            "member_count": len(members),
            "no_model_soup": no_model_soup,
            "initialized_development_oof_pcc": initialized,
            "selected_development_oof_pcc": selected,
            "same_model_tuning_gain": tuning_gain,
            "tuning_strictly_improves": tuning_pass,
            "released_test_unused_for_selection": test_selection_clean,
            "artifact_root": str(artifact_root),
            "artifact_status": artifact.get("status"),
            "artifact_complete": artifact_complete,
            **artifact_validation,
            "development_pass": no_model_soup and tuning_pass and test_selection_clean,
            "release_pass": (
                no_model_soup
                and tuning_pass
                and test_selection_clean
                and artifact_complete
            ),
        }

    development_failures = sorted(
        pair for pair, report in pair_reports.items() if not report["development_pass"]
    )
    release_failures = sorted(
        pair for pair, report in pair_reports.items() if not report["release_pass"]
    )
    complete_pair_set = not missing_pairs and not unexpected_pairs
    return {
        "manifest": str(manifest_path),
        "expected_pair_count": len(EXPECTED_PAIRS),
        "observed_pair_count": len(observed_pairs),
        "missing_pairs": missing_pairs,
        "unexpected_pairs": unexpected_pairs,
        "complete_pair_set": complete_pair_set,
        "development_passing_pair_count": len(pair_reports) - len(development_failures),
        "development_failing_pairs": development_failures,
        "all_development_rules_pass": complete_pair_set and not development_failures,
        "release_passing_pair_count": len(pair_reports) - len(release_failures),
        "release_failing_pairs": release_failures,
        "all_release_rules_pass": complete_pair_set and not release_failures,
        "pairs": pair_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/canonical_models.yaml")
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-development-pass", action="store_true")
    parser.add_argument("--require-release-pass", action="store_true")
    args = parser.parse_args()
    report = audit_manifest(args.manifest)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if args.require_development_pass and not report["all_development_rules_pass"]:
        raise SystemExit(1)
    if args.require_release_pass and not report["all_release_rules_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
