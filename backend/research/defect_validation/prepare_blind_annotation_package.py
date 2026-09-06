"""Create a blind annotation package for the frozen defect final-test subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.defect_validation.independent_final_test import (  # noqa: E402
    default_package_path,
    repo_root,
    write_blind_annotation_package,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate blind manual-annotation materials for the frozen "
            "independent defect final-test subset. No model predictions are run "
            "or included."
        )
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional package output directory.",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    output = Path(args.output).resolve() if args.output else default_package_path(root)
    report, files = write_blind_annotation_package(output=output, root=root)
    summary = {
        "status": report["final_test_manifest"]["status"],
        "image_count": report["final_test_manifest"]["image_count"],
        "specimen_count": report["final_test_manifest"]["specimen_count"],
        "membership_sha256": report["final_test_manifest"]["membership_sha256"],
        "manual_annotation_status": report["manual_annotation_status"]["status"],
        "no_model_predictions_included": report["no_model_predictions_included"],
        "output_directory": str(output),
        "files": {key: str(value) for key, value in files.items()},
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
