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
        artifact_complete = (
            artifact.get("status") == "complete" and artifact_root.exists()
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
