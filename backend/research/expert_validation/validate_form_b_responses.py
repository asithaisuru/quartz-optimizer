"""Validate post-system Form B expert review responses.

Form B is collected only after matching Form A recommendations are completed
and frozen. It is supplemental manufacturability review, not independent expert
ground truth for the Objective-3 comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:  # Package import for tests, script import for direct CLI execution.
    from . import common  # type: ignore
except ImportError:  # pragma: no cover
    import common  # type: ignore


YES_NO_VALUES = {"YES", "NO"}
YES_NO_UNCERTAIN_VALUES = {"YES", "NO", "UNCERTAIN"}
FORM_A_REQUIRED_TEXT_FIELDS = (
    "recommended_cut_shape",
    "orientation_description",
    "defects_or_areas_to_avoid",
    "rationale",
)
FORM_B_REQUIRED_TEXT_FIELDS = (
    "manufacturing_risks_or_practical_concerns",
    "changes_expert_would_make_to_system_plan",
)


def normalized_choice(value: str | None) -> str:
    return str(value or "").strip().upper()


def parse_optional_float(value: str | None) -> tuple[float | None, bool]:
    text = str(value or "").strip()
    if not text:
        return None, True
    try:
        return float(text), True
    except ValueError:
        return None, False


def parse_optional_int(value: str | None) -> tuple[int | None, bool]:
    text = str(value or "").strip()
    if not text:
        return None, True
    try:
        numeric = float(text)
    except ValueError:
        return None, False
    if not numeric.is_integer():
        return None, False
    return int(numeric), True


def parse_required_int(value: str | None) -> tuple[int | None, bool]:
    parsed, ok = parse_optional_int(value)
    return parsed, ok and parsed is not None


def parse_required_float(value: str | None) -> tuple[float | None, bool]:
    parsed, ok = parse_optional_float(value)
    return parsed, ok and parsed is not None


def completed_form_a_indexes(
    form_a_rows: list[dict[str, str]],
) -> dict[str, Any]:
    by_exact_response: set[tuple[str, str, str]] = set()
    by_expert_specimen: set[tuple[str, str]] = set()
    complete_count = 0

    for row in form_a_rows:
        missing_columns = [
            field for field in common.EXPERT_RESPONSE_FIELDS if field not in row
        ]
        if missing_columns:
            continue

        response_id = row["response_id"].strip()
        expert_id = row["expert_id"].strip()
        specimen_id = row["specimen_id"].strip()
        years, years_ok = parse_required_float(row.get("years_experience"))
        gem_count, gems_ok = parse_required_int(row.get("gem_count"))
        retained, retained_ok = parse_required_float(row.get("retained_weight_ct"))
        confidence, confidence_ok = parse_required_int(row.get("confidence_1_to_5"))
        manufacturable = normalized_choice(row.get("manufacturable_with_standard_saw"))

        if not response_id or not expert_id or not specimen_id:
            continue
        if specimen_id not in common.FINAL_COMPARISON_SPECIMENS:
            continue
        if row["form_version"].strip() not in {"", common.FORM_A_VERSION}:
            continue
        if not common.is_yes(row.get("consent_independent_blind")):
            continue
        if any(not row[field].strip() for field in FORM_A_REQUIRED_TEXT_FIELDS):
            continue
        if not years_ok or years is None or years < 0:
            continue
        if not gems_ok or gem_count is None or gem_count < 0:
            continue
        if not retained_ok or retained is None or retained < 0:
            continue
        if not confidence_ok or confidence is None or not 1 <= confidence <= 5:
            continue
        if manufacturable not in YES_NO_UNCERTAIN_VALUES:
            continue

        complete_count += 1
        by_exact_response.add((response_id, expert_id, specimen_id))
        by_expert_specimen.add((expert_id, specimen_id))

    return {
        "complete_count": complete_count,
        "by_exact_response": by_exact_response,
        "by_expert_specimen": by_expert_specimen,
    }


def has_completed_form_a_link(
    row: dict[str, str],
    form_a_index: dict[str, Any],
) -> bool:
    expert_id = row["expert_id"].strip()
    specimen_id = row["specimen_id"].strip()
    form_a_response_id = row["form_a_response_id"].strip()
    if form_a_response_id:
        return (
            form_a_response_id,
            expert_id,
            specimen_id,
        ) in form_a_index["by_exact_response"]
    return (expert_id, specimen_id) in form_a_index["by_expert_specimen"]


def validate_rows(
    form_b_rows: list[dict[str, str]],
    form_a_rows: list[dict[str, str]],
) -> dict[str, Any]:
    form_a_index = completed_form_a_indexes(form_a_rows)
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    linked_rows = 0
    by_specimen: dict[str, set[str]] = defaultdict(set)

    if not form_b_rows:
        warnings.append("No Form B response rows present; post-system review remains pending.")

    for index, row in enumerate(form_b_rows, start=2):
        missing_columns = [
            field for field in common.FORM_B_RESPONSE_FIELDS if field not in row
        ]
        if missing_columns:
            errors.append({
                "row": index,
                "field": "csv_header",
                "message": "Missing expected Form B columns: " + ", ".join(missing_columns),
            })
            continue

        expert_id = row["expert_id"].strip()
        specimen_id = row["specimen_id"].strip()
        form_a_response_id = row["form_a_response_id"].strip()
        by_specimen[specimen_id].add(expert_id)

        if not row["form_b_response_id"].strip():
            errors.append({"row": index, "field": "form_b_response_id", "message": "Required."})
        if not row["submitted_at"].strip():
            errors.append({"row": index, "field": "submitted_at", "message": "Required."})
        if not expert_id:
            errors.append({"row": index, "field": "expert_id", "message": "Required."})
        if not row["expert_name_or_code"].strip():
            errors.append({"row": index, "field": "expert_name_or_code", "message": "Required."})
        if row["form_version"].strip() not in {"", common.FORM_B_VERSION}:
            errors.append({
                "row": index,
                "field": "form_version",
                "message": f"Unexpected Form B version; expected {common.FORM_B_VERSION}.",
            })
        if specimen_id not in common.FINAL_COMPARISON_SPECIMENS:
            errors.append({
                "row": index,
                "field": "specimen_id",
                "message": "Specimen is not in the final Form B specimen set.",
            })

        if not common.is_yes(row["form_a_completed_before_system_shown"]):
            errors.append({
                "row": index,
                "field": "form_a_completed_before_system_shown",
                "message": "Must be YES before Form B can be used as post-system review.",
            })
        if not common.is_yes(row["consent_post_system_comparison"]):
            errors.append({
                "row": index,
                "field": "consent_post_system_comparison",
                "message": "Consent for supplemental post-system comparison must be YES.",
            })

        system_plan_reviewed = normalized_choice(row["system_plan_reviewed"])
        if system_plan_reviewed not in YES_NO_VALUES:
            errors.append({
                "row": index,
                "field": "system_plan_reviewed",
                "message": "Must be YES or NO.",
            })
        elif system_plan_reviewed != "YES":
            errors.append({
                "row": index,
                "field": "system_plan_reviewed",
                "message": "A valid Form B post-system review requires system_plan_reviewed=YES.",
            })

        manufacturability = normalized_choice(row["system_plan_manufacturability"])
        if manufacturability not in YES_NO_UNCERTAIN_VALUES:
            errors.append({
                "row": index,
                "field": "system_plan_manufacturability",
                "message": "Must be YES, NO, or UNCERTAIN.",
            })

        orientation_acceptable = normalized_choice(row["system_orientation_acceptable"])
        if orientation_acceptable not in YES_NO_UNCERTAIN_VALUES:
            errors.append({
                "row": index,
                "field": "system_orientation_acceptable",
                "message": "Must be YES, NO, or UNCERTAIN.",
            })
        elif orientation_acceptable == "NO" and not row[
            "orientation_disagreement_explanation"
        ].strip():
            errors.append({
                "row": index,
                "field": "orientation_disagreement_explanation",
                "message": "Required when system_orientation_acceptable=NO.",
            })

        for field in FORM_B_REQUIRED_TEXT_FIELDS:
            if not row[field].strip():
                errors.append({"row": index, "field": field, "message": "Required; enter none if no concern/change."})

        revised_retained, retained_ok = parse_optional_float(
            row["revised_retained_weight_estimate_ct"]
        )
        if not retained_ok or (revised_retained is not None and revised_retained < 0):
            errors.append({
                "row": index,
                "field": "revised_retained_weight_estimate_ct",
                "message": "Must be a non-negative number or blank.",
            })
        revised_gem_count, gem_count_ok = parse_optional_int(row["revised_gem_count"])
        if not gem_count_ok or (revised_gem_count is not None and revised_gem_count < 0):
            errors.append({
                "row": index,
                "field": "revised_gem_count",
                "message": "Must be an integer >= 0 or blank.",
            })
        confidence, confidence_ok = parse_required_int(
            row["feasibility_confidence_1_to_5"]
        )
        if not confidence_ok or confidence is None or not 1 <= confidence <= 5:
            errors.append({
                "row": index,
                "field": "feasibility_confidence_1_to_5",
                "message": "Must be an integer from 1 to 5.",
            })

        if not form_a_response_id:
            warnings.append(
                f"Row {index}: form_a_response_id is blank; linkage falls back to expert_id/specimen_id."
            )
        if has_completed_form_a_link(row, form_a_index):
            linked_rows += 1
        else:
            errors.append({
                "row": index,
                "field": "form_a_response_id",
                "message": (
                    "No matching completed Form A row found for expert_id, specimen_id, "
                    "and form_a_response_id when provided."
                ),
            })

    counts = Counter(row.get("specimen_id", "").strip() for row in form_b_rows)
    missing_specimens = [
        specimen_id for specimen_id in common.FINAL_COMPARISON_SPECIMENS
        if counts.get(specimen_id, 0) == 0
    ]
    if missing_specimens:
        warnings.append(
            "Missing Form B responses for specimens: " + ", ".join(missing_specimens)
        )

    linkage_ready_by_specimen = {
        specimen_id: bool(by_specimen.get(specimen_id, set()))
        for specimen_id in common.FINAL_COMPARISON_SPECIMENS
    }
    return {
        "input_rows": len(form_b_rows),
        "form_a_rows_examined": len(form_a_rows),
        "completed_form_a_rows_available": form_a_index["complete_count"],
        "linked_form_b_rows": linked_rows,
        "valid_post_system_review": bool(form_b_rows) and not errors,
        "valid_as_independent_evidence": False,
        "independent_evidence_boundary": (
            "Form B is supplemental post-system manufacturability review only; "
            "do not aggregate it with independent Form A expert baseline rows."
        ),
        "errors": errors,
        "warnings": warnings,
        "response_count_by_specimen": dict(counts),
        "linkage_ready_by_specimen": linkage_ready_by_specimen,
        "all_specimens_have_form_b_review": all(linkage_ready_by_specimen.values()),
        "form_a_linkage_rule": (
            "Each Form B row must match a completed Form A row by expert_id and "
            "specimen_id; when form_a_response_id is populated it must also match "
            "the Form A response_id."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("form_b_responses_csv", type=Path)
    parser.add_argument(
        "--form-a-responses",
        type=Path,
        required=True,
        help="Frozen normalized Form A response CSV used only for linkage validation.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=common.FORM_B_RESPONSE_VALIDATION,
    )
    args = parser.parse_args()

    form_b_rows = common.read_csv_rows(args.form_b_responses_csv)
    form_a_rows = common.read_csv_rows(args.form_a_responses)
    summary = validate_rows(form_b_rows, form_a_rows)
    common.write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
