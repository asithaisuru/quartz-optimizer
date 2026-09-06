"""Validate exported independent expert Form A responses."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    EXPERT_RESPONSE_FIELDS,
    FINAL_COMPARISON_SPECIMENS,
    FORM_A_VERSION,
    OUTPUT_DIR,
    SYSTEM_YIELD_RESULTS,
    is_yes,
    load_primary_system_results,
    read_csv_rows,
    rel,
    to_float,
    to_int,
    write_json,
)


REQUIRED_TEXT_FIELDS = (
    "recommended_cut_shape",
    "orientation_description",
    "defects_or_areas_to_avoid",
    "rationale",
)

MANUFACTURABILITY_VALUES = {"YES", "NO", "UNCERTAIN"}


def validate_rows(rows: list[dict[str, str]]) -> dict:
    primary = load_primary_system_results()
    errors = []
    warnings = []
    by_specimen: dict[str, set[str]] = defaultdict(set)

    if not rows:
        warnings.append("No expert response rows present; comparison remains pending.")

    for index, row in enumerate(rows, start=2):
        missing_columns = [field for field in EXPERT_RESPONSE_FIELDS if field not in row]
        if missing_columns:
            errors.append({
                "row": index,
                "field": "csv_header",
                "message": "Missing expected columns: " + ", ".join(missing_columns),
            })
            continue

        specimen_id = row["specimen_id"].strip()
        expert_id = row["expert_id"].strip()
        by_specimen[specimen_id].add(expert_id)

        if not is_yes(row["consent_independent_blind"]):
            errors.append({
                "row": index,
                "field": "consent_independent_blind",
                "message": "Consent/independence confirmation must be YES.",
            })
        if row["form_version"].strip() not in {"", FORM_A_VERSION}:
            errors.append({
                "row": index,
                "field": "form_version",
                "message": f"Unexpected form version; expected {FORM_A_VERSION}.",
            })
        if specimen_id not in FINAL_COMPARISON_SPECIMENS:
            errors.append({
                "row": index,
                "field": "specimen_id",
                "message": "Specimen is not in the final Objective-3 comparison set.",
            })
            continue
        if specimen_id not in primary:
            errors.append({
                "row": index,
                "field": "specimen_id",
                "message": "No primary manufacturing-complete system result exists.",
            })
            continue

        known_weight = to_float(primary[specimen_id]["known_weight_ct"])
        retained = to_float(row["retained_weight_ct"])
        gem_count = to_int(row["gem_count"])
        confidence = to_int(row["confidence_1_to_5"])
        years = to_float(row["years_experience"])

        if not expert_id:
            errors.append({"row": index, "field": "expert_id", "message": "Required."})
        for field in REQUIRED_TEXT_FIELDS:
            if not row[field].strip():
                errors.append({"row": index, "field": field, "message": "Required."})
        if years is None or years < 0:
            errors.append({
                "row": index,
                "field": "years_experience",
                "message": "Must be a non-negative number.",
            })
        if gem_count is None or gem_count < 0:
            errors.append({
                "row": index,
                "field": "gem_count",
                "message": "Must be an integer >= 0.",
            })
        if retained is None or retained < 0 or (known_weight is not None and retained > known_weight):
            errors.append({
                "row": index,
                "field": "retained_weight_ct",
                "message": "Must be between 0 and the known specimen weight.",
            })
        if confidence is None or confidence < 1 or confidence > 5:
            errors.append({
                "row": index,
                "field": "confidence_1_to_5",
                "message": "Must be an integer from 1 to 5.",
            })
        manufacturable = row["manufacturable_with_standard_saw"].strip().upper()
        if manufacturable not in MANUFACTURABILITY_VALUES:
            errors.append({
                "row": index,
                "field": "manufacturable_with_standard_saw",
                "message": "Must be YES, NO, or UNCERTAIN.",
            })
        if gem_count == 0 and retained not in {0, 0.0}:
            errors.append({
                "row": index,
                "field": "retained_weight_ct",
                "message": "A zero-gem recommendation must retain 0 ct.",
            })

    counts = Counter(row.get("specimen_id", "").strip() for row in rows)
    missing_specimens = [
        specimen_id for specimen_id in FINAL_COMPARISON_SPECIMENS
        if counts.get(specimen_id, 0) == 0
    ]
    if missing_specimens:
        warnings.append(
            "Missing expert responses for specimens: " + ", ".join(missing_specimens)
        )

    consensus_ready = {
        specimen_id: len(by_specimen.get(specimen_id, set())) >= 3
        for specimen_id in FINAL_COMPARISON_SPECIMENS
    }
    descriptive_ready = {
        specimen_id: len(by_specimen.get(specimen_id, set())) >= 1
        for specimen_id in FINAL_COMPARISON_SPECIMENS
    }

    return {
        "input_rows": len(rows),
        "valid_for_comparison": bool(rows) and not errors,
        "errors": errors,
        "warnings": warnings,
        "response_count_by_specimen": dict(counts),
        "descriptive_ready_by_specimen": descriptive_ready,
        "consensus_ready_by_specimen": consensus_ready,
        "all_specimens_descriptive_ready": all(descriptive_ready.values()),
        "all_specimens_consensus_ready": all(consensus_ready.values()),
        "system_yield_results": rel(SYSTEM_YIELD_RESULTS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses_csv", type=Path)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=OUTPUT_DIR / "expert_response_validation.json",
    )
    args = parser.parse_args()

    rows = read_csv_rows(args.responses_csv)
    summary = validate_rows(rows)
    write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
