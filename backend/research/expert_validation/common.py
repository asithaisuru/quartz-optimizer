"""Shared constants and helpers for external expert validation.

The expert-comparison workflow consumes frozen system-side optimizer evidence
and independently collected expert recommendations. It does not run optimizer
search, reconstruction, defect detection, facet ML, or frontend code.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SYSTEM_YIELD_RESULTS = (
    ROOT / "final_research_evidence" / "optimizer_validation"
    / "system_yield_results.csv"
)
OUTPUT_DIR = ROOT / "final_research_evidence" / "expert_validation"
EXPERT_RESPONSE_TEMPLATE = OUTPUT_DIR / "expert_response_template.csv"
FORM_B_RESPONSE_TEMPLATE = Path(__file__).resolve().parent / "form_b_response_template.csv"
FORM_B_RESPONSE_VALIDATION = OUTPUT_DIR / "form_b_response_validation.json"
COMPARISON_RESULTS = OUTPUT_DIR / "comparison_results.csv"
COMPARISON_SUMMARY = OUTPUT_DIR / "comparison_summary.json"
COMPARISON_REPORT = OUTPUT_DIR / "comparison_report.md"

FORM_A_VERSION = "objective3_external_expert_form_a_v1"
FORM_B_VERSION = "objective3_post_system_form_b_v1"
COMPARISON_LABEL = "SYSTEM vs INDEPENDENT EXPERT/TRADITIONAL CUTTER"
PROPOSAL_TARGET_WASTE_REDUCTION_PERCENT = 15.0

EXISTING_FORM_A_SPECIMENS = ("QZ-01", "QZ-05", "QZ-09", "QZ-14", "QZ-30")
FINAL_COMPARISON_SPECIMENS = ("QZ-01", "QZ-03", "QZ-05", "QZ-08", "QZ-14")
EXCLUDED_SPECIMEN_REASONS = {
    "QZ-09": "SPARSE_FALLBACK/replaced; not active TRUE_DENSE system evidence.",
    "QZ-30": "SPARSE_FALLBACK degraded low-weight evidence; preserve, do not use as primary system comparison.",
}

EXPERT_RESPONSE_FIELDS = [
    "response_id",
    "submitted_at",
    "expert_id",
    "expert_name_or_code",
    "years_experience",
    "consent_independent_blind",
    "form_version",
    "specimen_id",
    "recommended_cut_shape",
    "gem_count",
    "retained_weight_ct",
    "orientation_description",
    "orientation_vector_x",
    "orientation_vector_y",
    "orientation_vector_z",
    "defects_or_areas_to_avoid",
    "rationale",
    "confidence_1_to_5",
    "manufacturable_with_standard_saw",
    "notes",
]

FORM_B_RESPONSE_FIELDS = [
    "form_b_response_id",
    "submitted_at",
    "expert_id",
    "expert_name_or_code",
    "form_a_response_id",
    "form_a_completed_before_system_shown",
    "consent_post_system_comparison",
    "form_version",
    "specimen_id",
    "system_plan_reviewed",
    "system_plan_manufacturability",
    "manufacturing_risks_or_practical_concerns",
    "changes_expert_would_make_to_system_plan",
    "revised_retained_weight_estimate_ct",
    "revised_gem_count",
    "revised_recommended_cut_shape",
    "system_orientation_acceptable",
    "orientation_disagreement_explanation",
    "feasibility_confidence_1_to_5",
    "overall_comments",
]

COMPARISON_FIELDS = [
    "response_id",
    "expert_id",
    "specimen_id",
    "known_weight_ct",
    "system_job_id",
    "system_strategy",
    "system_strategy_name",
    "system_gem_count",
    "system_retained_weight_ct",
    "system_yield_percent",
    "system_waste_percent",
    "system_verification_scope",
    "expert_recommended_shape",
    "expert_gem_count",
    "expert_retained_weight_ct",
    "expert_yield_percent",
    "expert_waste_percent",
    "absolute_retained_weight_difference_ct",
    "system_minus_expert_yield_pp",
    "relative_waste_reduction_percent",
    "proposal_target_status",
    "shape_agreement_status",
    "orientation_angular_difference_deg",
    "confidence_1_to_5",
    "notes",
]

FORM_B_QUESTIONS = [
    {
        "id": "form_a_completed_before_system_shown",
        "prompt": (
            "Confirm that your Form A recommendation was completed and frozen "
            "before any system result was shown."
        ),
        "answer_format": "YES/NO",
    },
    {
        "id": "consent_post_system_comparison",
        "prompt": (
            "Do you consent to having this post-system review linked to your "
            "frozen Form A submission for supplemental analysis?"
        ),
        "answer_format": "YES/NO",
    },
    {
        "id": "system_plan_reviewed",
        "prompt": "Did you review the system plan for this specimen?",
        "answer_format": "YES/NO",
    },
    {
        "id": "system_plan_manufacturability",
        "prompt": (
            "Based on the reviewed system plan, is the plan manufacturable "
            "using ordinary gem-cutting/sawing practice?"
        ),
        "answer_format": "YES/NO/UNCERTAIN",
    },
    {
        "id": "manufacturing_risks_or_practical_concerns",
        "prompt": (
            "List any manufacturing risks, stability concerns, fracture/defect "
            "concerns, or missing information that would affect feasibility."
        ),
        "answer_format": "free text",
    },
    {
        "id": "changes_expert_would_make_to_system_plan",
        "prompt": (
            "Describe any changes you would make to the system plan before "
            "cutting, or state that no changes are recommended."
        ),
        "answer_format": "free text",
    },
    {
        "id": "revised_retained_weight_estimate_ct",
        "prompt": (
            "If your changes alter the retained finished weight estimate, "
            "provide the revised estimate in carats; otherwise leave blank."
        ),
        "answer_format": "number or blank",
    },
    {
        "id": "revised_gem_count",
        "prompt": (
            "If your changes alter the finished gem count, provide the revised "
            "count; otherwise leave blank."
        ),
        "answer_format": "integer or blank",
    },
    {
        "id": "revised_recommended_cut_shape",
        "prompt": (
            "If your changes alter the recommended cut shape, provide the "
            "revised shape; otherwise leave blank."
        ),
        "answer_format": "text or blank",
    },
    {
        "id": "system_orientation_acceptable",
        "prompt": "Is the system orientation acceptable for this specimen?",
        "answer_format": "YES/NO/UNCERTAIN",
    },
    {
        "id": "orientation_disagreement_explanation",
        "prompt": (
            "If you disagree with the system orientation, explain the practical "
            "or manufacturing basis for the disagreement."
        ),
        "answer_format": "free text or blank",
    },
    {
        "id": "feasibility_confidence_1_to_5",
        "prompt": "How confident are you in this feasibility assessment?",
        "answer_format": "integer 1-5",
    },
    {
        "id": "overall_comments",
        "prompt": "Provide any overall comments on the reviewed system plan.",
        "answer_format": "free text or blank",
    },
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_header(path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def is_yes(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def load_system_rows() -> list[dict[str, str]]:
    return read_csv_rows(SYSTEM_YIELD_RESULTS)


def load_primary_system_results() -> dict[str, dict[str, str]]:
    """Return the primary manufacturing-complete non-diagnostic system result.

    If multiple verified report rows exist for a specimen, use the highest
    retained weight. Diagnostic rows are never primary for the Objective-3
    proposal calculation.
    """
    primary: dict[str, dict[str, str]] = {}
    for row in load_system_rows():
        specimen_id = row["specimen_id"]
        if specimen_id not in FINAL_COMPARISON_SPECIMENS:
            continue
        if row["manufacturing_cut_sequence_complete"] != "YES":
            continue
        if row["verification_scope"] != "fully_verified_existing_report":
            continue
        current = primary.get(specimen_id)
        if current is None:
            primary[specimen_id] = row
            continue
        current_weight = to_float(current["retained_cuttable_weight_ct"]) or -1.0
        row_weight = to_float(row["retained_cuttable_weight_ct"]) or -1.0
        if row_weight > current_weight:
            primary[specimen_id] = row
    return primary


def system_specimen_reasons(primary: dict[str, dict[str, str]]) -> dict[str, str]:
    reasons = {
        "QZ-01": (
            "Include: confirmed optimizer benchmark with verified "
            "manufacturing-complete production multi-gem plan."
        ),
        "QZ-03": (
            "Include: TRUE_DENSE supplemental specimen with completed job "
            "artifact; only verified single-gem system output is available."
        ),
        "QZ-05": (
            "Include: TRUE_DENSE completed job; primary Objective-3 comparison "
            "uses verified Single Large result because Multi-Gem is diagnostic "
            "only with no exact manufacturing sequence."
        ),
        "QZ-08": (
            "Include: TRUE_DENSE completed job; only verified single-gem system "
            "output is available and poor yield is retained."
        ),
        "QZ-14": (
            "Include: TRUE_DENSE completed job and existing Form A specimen; "
            "only verified single-gem system output is available."
        ),
        "QZ-09": "Exclude: " + EXCLUDED_SPECIMEN_REASONS["QZ-09"],
        "QZ-30": "Exclude: " + EXCLUDED_SPECIMEN_REASONS["QZ-30"],
    }
    missing_primary = [
        specimen_id for specimen_id in FINAL_COMPARISON_SPECIMENS
        if specimen_id not in primary
    ]
    for specimen_id in missing_primary:
        reasons[specimen_id] = (
            "Needs correction: no manufacturing-complete non-diagnostic system "
            "row was found."
        )
    return reasons


def expert_yield_percent(retained_weight_ct: float, known_weight_ct: float) -> float:
    return retained_weight_ct / known_weight_ct * 100.0


def waste_percent(yield_percent: float) -> float:
    return 100.0 - yield_percent


def relative_waste_reduction(system_waste: float, expert_waste: float) -> float | None:
    if expert_waste <= 0:
        return None
    return (expert_waste - system_waste) / expert_waste * 100.0


def angular_difference_degrees(a: tuple[float, float, float],
                               b: tuple[float, float, float]) -> float | None:
    mag_a = math.sqrt(sum(v * v for v in a))
    mag_b = math.sqrt(sum(v * v for v in b))
    if mag_a == 0 or mag_b == 0:
        return None
    dot = sum(x * y for x, y in zip(a, b)) / (mag_a * mag_b)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
