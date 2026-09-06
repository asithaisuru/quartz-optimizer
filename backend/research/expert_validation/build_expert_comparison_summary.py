"""Build the external expert-comparison package and summary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    COMPARISON_FIELDS,
    COMPARISON_LABEL,
    COMPARISON_REPORT,
    COMPARISON_RESULTS,
    COMPARISON_SUMMARY,
    EXISTING_FORM_A_SPECIMENS,
    EXPERT_RESPONSE_FIELDS,
    EXPERT_RESPONSE_TEMPLATE,
    FINAL_COMPARISON_SPECIMENS,
    FORM_A_VERSION,
    FORM_B_QUESTIONS,
    PROPOSAL_TARGET_WASTE_REDUCTION_PERCENT,
    load_primary_system_results,
    median_or_none,
    read_csv_rows,
    rel,
    system_specimen_reasons,
    to_float,
    write_csv_header,
    write_json,
)


def build_pending_summary(comparison_rows: list[dict[str, str]]) -> dict:
    primary = load_primary_system_results()
    specimen_reasons = system_specimen_reasons(primary)
    by_specimen: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in comparison_rows:
        by_specimen[row["specimen_id"]].append(row)

    specimen_summaries = {}
    for specimen_id in FINAL_COMPARISON_SPECIMENS:
        rows = by_specimen.get(specimen_id, [])
        waste_reductions = [
            value for value in (
                to_float(row.get("relative_waste_reduction_percent"))
                for row in rows
            )
            if value is not None
        ]
        target_counts = Counter(row.get("proposal_target_status", "") for row in rows)
        specimen_summaries[specimen_id] = {
            "expert_response_count": len(rows),
            "median_relative_waste_reduction_percent": median_or_none(waste_reductions),
            "proposal_target_status_counts": dict(target_counts),
            "descriptive_ready": len(rows) >= 1,
            "consensus_ready": len({row.get("expert_id", "") for row in rows}) >= 3,
        }

    any_rows = bool(comparison_rows)
    return {
        "comparison": COMPARISON_LABEL,
        "current_form_a_status": "PARTIALLY VALID",
        "current_form_a_assessment": (
            "Existing Form A specimen list is partly reusable: QZ-01, QZ-05, "
            "and QZ-14 remain useful, while QZ-09 and QZ-30 are excluded from "
            "the primary Objective-3 comparison because they are SPARSE_FALLBACK "
            "or degraded evidence. The current final set also needs QZ-03 and QZ-08."
        ),
        "existing_form_a_specimens": list(EXISTING_FORM_A_SPECIMENS),
        "recommended_final_comparison_specimen_set": list(FINAL_COMPARISON_SPECIMENS),
        "specimen_inclusion_reasons": specimen_reasons,
        "primary_system_result_policy": (
            "Use the highest retained-weight system row per specimen only when "
            "manufacturing_cut_sequence_complete=YES and "
            "verification_scope=fully_verified_existing_report. Diagnostic "
            "rows are secondary and cannot support the >=15% proposal claim."
        ),
        "final_comparison_metrics": {
            "expert_yield_percent": "expert_retained_weight_ct / known_weight_ct * 100",
            "expert_waste_percent": "100 - expert_yield_percent",
            "system_waste_percent": "100 - system_yield_percent",
            "absolute_retained_weight_difference_ct": "abs(system_retained_weight_ct - expert_retained_weight_ct)",
            "system_vs_expert_yield_difference_pp": "system_yield_percent - expert_yield_percent",
            "relative_waste_reduction_percent": (
                "(expert_waste_percent - system_waste_percent) / "
                "expert_waste_percent * 100"
            ),
            "proposal_target": (
                f"ACHIEVED only when relative_waste_reduction_percent >= "
                f"{PROPOSAL_TARGET_WASTE_REDUCTION_PERCENT} for pre-specified "
                "eligible comparisons."
            ),
            "formula_appropriateness": (
                "Appropriate only when system and expert waste are computed "
                "against the same known specimen weight and the expert retained "
                "weight is an independent feasible recommendation."
            ),
        },
        "minimum_expert_response_recommendation": {
            "descriptive_comparison": (
                "At least 1 complete independent expert response per included "
                "specimen; report missing specimens as pending."
            ),
            "per_specimen_consensus": (
                "At least 3 independent complete expert responses per specimen; "
                "report median, range/IQR, and disagreements without inventing "
                "statistical significance."
            ),
        },
        "multiple_expert_aggregation": (
            "Primary aggregation is unweighted per-specimen median expert waste "
            "and median retained weight. Also report each expert row, range/IQR, "
            "shape mode with ties labelled no consensus, and confidence as "
            "descriptive metadata rather than a primary weight."
        ),
        "form_b_required": "YES",
        "form_b_use_boundary": (
            "Form B is supplemental manufacturability review only after Form A "
            "is frozen and system results are revealed; it is not the independent "
            "traditional baseline for the primary Objective-3 target."
        ),
        "form_b_questions": FORM_B_QUESTIONS,
        "objective2_expert_reuse": {
            "possible": "YES",
            "boundary": (
                "Only pre-system Form A orientation fields may be reused for an "
                "Objective-2 expert orientation comparison. Post-system Form B "
                "responses and Objective-3 yield fields must not be used as an "
                "independent Objective-2 baseline."
            ),
        },
        "comparison_rows": len(comparison_rows),
        "per_specimen_summary": specimen_summaries,
        "objective3_target_evaluation_readiness": (
            "READY_AFTER_REAL_FORM_A_RESPONSES"
            if any_rows else "NOT_READY_NO_REAL_EXPERT_RESPONSES"
        ),
        "outputs": {
            "expert_response_template": rel(EXPERT_RESPONSE_TEMPLATE),
            "comparison_results": rel(COMPARISON_RESULTS),
            "comparison_summary": rel(COMPARISON_SUMMARY),
            "comparison_report": rel(COMPARISON_REPORT),
        },
    }


def write_report(summary: dict) -> None:
    COMPARISON_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_REPORT.open("w", encoding="utf-8") as handle:
        handle.write("# External Gem-Cutter / Traditional-Cutting Comparison\n\n")
        handle.write("## Current Form A\n\n")
        handle.write(summary["current_form_a_status"] + ". ")
        handle.write(summary["current_form_a_assessment"] + "\n\n")
        handle.write("## Final Specimen Set\n\n")
        for specimen_id in [
            *FINAL_COMPARISON_SPECIMENS,
            "QZ-09",
            "QZ-30",
        ]:
            handle.write(f"- {specimen_id}: {summary['specimen_inclusion_reasons'][specimen_id]}\n")
        handle.write("\n")
        handle.write("## Primary System Result Policy\n\n")
        handle.write(summary["primary_system_result_policy"] + "\n\n")
        handle.write("## Metrics\n\n")
        for name, formula in summary["final_comparison_metrics"].items():
            handle.write(f"- {name}: {formula}\n")
        handle.write("\n")
        handle.write("## Expert Response Minimums\n\n")
        for name, value in summary["minimum_expert_response_recommendation"].items():
            handle.write(f"- {name}: {value}\n")
        handle.write("\n")
        handle.write("## Multiple Expert Aggregation\n\n")
        handle.write(summary["multiple_expert_aggregation"] + "\n\n")
        handle.write("## Form B\n\n")
        handle.write(summary["form_b_use_boundary"] + "\n\n")
        for question in summary["form_b_questions"]:
            handle.write(
                f"- {question['id']}: {question['prompt']} "
                f"Format: {question['answer_format']}.\n"
            )
        handle.write("\n")
        handle.write("## Objective 2 Reuse Boundary\n\n")
        handle.write(summary["objective2_expert_reuse"]["boundary"] + "\n\n")
        handle.write("## Readiness\n\n")
        handle.write(summary["objective3_target_evaluation_readiness"] + "\n")


def ensure_initial_outputs() -> None:
    write_csv_header(EXPERT_RESPONSE_TEMPLATE, EXPERT_RESPONSE_FIELDS)
    if not COMPARISON_RESULTS.exists():
        write_csv_header(COMPARISON_RESULTS, COMPARISON_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-results",
        type=Path,
        default=COMPARISON_RESULTS,
        help="Existing comparison results CSV, if real expert responses were compared.",
    )
    parser.add_argument(
        "--init-pending",
        action="store_true",
        help="Create empty template/results and a pending summary/report.",
    )
    args = parser.parse_args()

    if args.init_pending:
        ensure_initial_outputs()
    rows = []
    if args.comparison_results.exists():
        rows = read_csv_rows(args.comparison_results)
    summary = build_pending_summary(rows)
    summary["form_a_version"] = FORM_A_VERSION
    write_json(COMPARISON_SUMMARY, summary)
    write_report(summary)
    print(json.dumps({
        "current_form_a_status": summary["current_form_a_status"],
        "recommended_final_comparison_specimen_set": (
            summary["recommended_final_comparison_specimen_set"]
        ),
        "comparison_rows": summary["comparison_rows"],
        "objective3_target_evaluation_readiness": (
            summary["objective3_target_evaluation_readiness"]
        ),
        "outputs": summary["outputs"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
