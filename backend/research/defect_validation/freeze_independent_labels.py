"""Freeze manual label hashes for the independent defect final-test subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.defect_validation.independent_final_test import (  # noqa: E402
    label_file_records,
    normalize_path,
    repo_root,
    write_label_freeze_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check or freeze manually created independent final-test labels. "
            "This never generates labels and never runs model predictions."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    root = repo_root()
    records, issues = label_file_records(root)
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "pass" if not issues else "pending_or_invalid",
                    "label_file_count": len(records),
                    "valid_label_file_count": sum(
                        row["status"] == "present_valid" for row in records
                    ),
                    "issue_count": len(issues),
                    "issues": issues,
                    "no_model_predictions_used": True,
                },
                indent=2,
            )
        )
        return 0
    if issues:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "labels are incomplete or invalid",
                    "issue_count": len(issues),
                    "issues": issues,
                    "no_model_predictions_used": True,
                },
                indent=2,
            )
        )
        return 2
    payload = write_label_freeze_manifest(root)
    print(
        json.dumps(
            {
                "status": "frozen",
                "path": normalize_path(
                    root
                    / "final_research_evidence"
                    / "defect_detection"
                    / "final_test"
                    / "LABEL_FREEZE_MANIFEST.json",
                    root,
                ),
                "label_file_count": payload["label_file_count"],
                "label_manifest_sha256": payload["label_manifest_sha256"],
                "no_model_predictions_used": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
