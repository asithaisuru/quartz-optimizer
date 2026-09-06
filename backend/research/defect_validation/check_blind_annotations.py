"""Report blind annotation progress without freezing or running inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.defect_validation.independent_final_test import (  # noqa: E402
    annotation_progress_report,
    repo_root,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check manual-label completeness and format for the frozen blind "
            "independent defect test set. This does not freeze labels and does "
            "not run model inference."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full machine-readable report.",
    )
    args = parser.parse_args(argv)

    report = annotation_progress_report(repo_root())
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"completed labels: {report['completed_labels']}/{report['image_count']}")
        print(f"missing labels: {report['missing_labels']}")
        print(f"malformed labels: {report['malformed_labels']}")
        print(f"45/45 ready: {report['ready_45_of_45']}")
        print(f"membership hash: {report['membership_sha256']}")
        print("model inference: NOT RUN")
    return 0 if report["ready_45_of_45"] == "YES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
