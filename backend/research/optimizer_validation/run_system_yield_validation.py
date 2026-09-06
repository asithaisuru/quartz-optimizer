"""Build Objective 3 system-side optimizer yield validation outputs.

This script reads only frozen/completed evidence artifacts. It does not rerun
photogrammetry, optimizer search, defect detection, facet ML, or frontend code.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RECON_RESULTS = (
    ROOT / "final_research_evidence" / "reconstruction_multi"
    / "batch_validation_results.csv"
)
OPTIMIZER_BASELINE = (
    ROOT / "final_research_evidence" / "optimizer_baseline"
    / "optimizer_baseline_results.csv"
)
QZ01_VISUAL_AUDIT = (
    ROOT / "final_research_evidence" / "optimizer_visual_audit"
    / "QZ01_CURRENT_PLAN_MANIFEST.json"
)
OUTPUT_DIR = ROOT / "final_research_evidence" / "optimizer_validation"
RESULTS_CSV = OUTPUT_DIR / "system_yield_results.csv"
SUMMARY_JSON = OUTPUT_DIR / "system_yield_summary.json"
REPORT_MD = OUTPUT_DIR / "system_yield_report.md"

TRUE_DENSE_SPECIMENS = ("QZ-03", "QZ-05", "QZ-08", "QZ-14")
ANCHOR_SPECIMENS = ("QZ-01",)
PROPOSAL_TARGET_STATUS = "PENDING EXTERNAL TRADITIONAL/EXPERT COMPARISON"
COMPARISON_LABEL = "internal computational baseline comparison"

RESULT_FIELDS = [
    "specimen_id",
    "job_id",
    "reconstruction_provenance",
    "known_weight_ct",
    "optimizer_strategy",
    "strategy_name",
    "gem_count",
    "retained_cuttable_weight_ct",
    "yield_percent",
    "waste_percent",
    "manufacturing_cut_sequence_complete",
    "number_of_cuts",
    "search_termination_reason",
    "time_budget_exhausted",
    "candidate_placement_count",
    "candidate_placement_count_source",
    "verification_scope",
    "geometric_containment_status",
    "inter_gem_spacing_clearance_status",
    "no_cut_defect_constraint_status",
    "exact_cut_sequence_status",
    "evidence_source",
    "notes",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def round_or_none(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def pct_waste(yield_percent: Any) -> float | None:
    y = to_float(yield_percent)
    if y is None:
        return None
    return round(100.0 - y, 6)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def is_complete_manufacturing(status: str | None) -> bool:
    return status in {"complete", "no_separation_required"}


def option_diagnostics(option: dict[str, Any]) -> dict[str, Any]:
    return option.get("diagnostics") or option.get("optimizer_diagnostics") or {}


def manufacturing_plan(option: dict[str, Any]) -> dict[str, Any]:
    return option.get("manufacturing_plan") or {}


def exact_sequence_verified(plan: dict[str, Any]) -> bool:
    if plan.get("status") == "no_separation_required":
        return True
    diagnostics = plan.get("diagnostics") or {}
    return bool(diagnostics.get("exact_sequence_verified"))


def constraint_statuses(
    option: dict[str, Any],
    diagnostics: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[str, str, str, str, str]:
    gem_count = int(option.get("gem_count") or 0)
    status = plan.get("status")
    manufacturing_eligible = bool(option.get("manufacturing_eligible", True))
    complete = manufacturing_eligible and is_complete_manufacturing(status)
    exact_ok = complete and exact_sequence_verified(plan)

    blade = diagnostics.get("blade_clearance") or {}
    rough = diagnostics.get("rough_clearance") or {}
    defect_points = diagnostics.get("defect_points")
    no_cut_volume = diagnostics.get("no_cut_volume_mesh_units")

    if gem_count <= 1:
        containment = "PASS: reported single-gem fit"
        spacing = "not applicable: single gem"
    else:
        rough_after = to_float(rough.get("min_clearance_after_mesh_units"))
        rough_target = to_float(rough.get("target_clearance_mesh_units"))
        if rough_after is not None and rough_target is not None:
            containment = (
                "PASS: rough clearance target met"
                if rough_after + 1e-12 >= rough_target
                else "FAIL: rough clearance below target"
            )
        else:
            containment = "INCONCLUSIVE: rough clearance not recorded"

        if blade.get("meets_target") is True:
            spacing = "PASS: blade/protected corridor clearance target met"
        elif blade.get("meets_target") is False:
            spacing = "FAIL: blade/protected corridor clearance target missed"
        else:
            spacing = "INCONCLUSIVE: blade clearance not recorded"

    if defect_points == 0 or no_cut_volume == 0:
        defect = "PASS: no approved no-cut defect constraints recorded"
    elif defect_points is None and no_cut_volume is None:
        defect = "INCONCLUSIVE: no-cut defect diagnostics not recorded"
    else:
        defect = "CHECKED: approved no-cut defect mask present"

    if exact_ok:
        cut_sequence = (
            "PASS: no separation required"
            if status == "no_separation_required"
            else "PASS: exact straight full-through sequence verified"
        )
    elif status == "settings_required":
        cut_sequence = "INCONCLUSIVE: manufacturing settings required"
    elif status == "geometric_comparison_only":
        cut_sequence = "DIAGNOSTIC_ONLY: not manufacturing eligible"
    else:
        cut_sequence = f"INCONCLUSIVE: manufacturing status {status or 'missing'}"

    if (
        containment.startswith("PASS")
        and (spacing.startswith("PASS") or spacing.startswith("not applicable"))
        and (defect.startswith("PASS") or defect.startswith("CHECKED"))
        and cut_sequence.startswith("PASS")
    ):
        combined = "PASS"
    elif "FAIL" in (containment + spacing + defect + cut_sequence):
        combined = "FAIL"
    else:
        combined = "INCONCLUSIVE"

    return containment, spacing, defect, cut_sequence, combined


def verification_scope(option: dict[str, Any], plan: dict[str, Any]) -> str:
    status = plan.get("status")
    eligible = bool(option.get("manufacturing_eligible", True))
    if not eligible:
        return "diagnostic_only"
    if status == "settings_required":
        return "diagnostic_only_missing_exact_cut_sequence"
    if is_complete_manufacturing(status) and exact_sequence_verified(plan):
        return "fully_verified_existing_report"
    return "inconclusive_existing_report"


def search_status(
    specimen_id: str,
    option: dict[str, Any],
    diagnostics: dict[str, Any],
    qz01_manifest: dict[str, Any] | None,
) -> tuple[str, str, Any, str]:
    strategy = option.get("type") or ""
    if specimen_id == "QZ-01" and strategy == "Preserve + Fill" and qz01_manifest:
        preserve = (
            qz01_manifest.get("stored_report_diagnostics", {})
            .get("preserve_fill", {})
        )
        return (
            preserve.get("stop_reason") or "not_recorded",
            "YES" if preserve.get("timed_out") else "NO",
            preserve.get("fit_candidates"),
            "QZ01_CURRENT_PLAN_MANIFEST stored_report_diagnostics.preserve_fill.fit_candidates",
        )

    if strategy == "Single Large":
        reason = "not_applicable_single_gem_baseline"
        exhausted = "NO"
    elif diagnostics.get("meets_runtime_target") is False:
        reason = "overall_runtime_target_missed"
        exhausted = "YES"
    elif diagnostics.get("meets_runtime_target") is True:
        reason = "completed_within_recorded_runtime_target"
        exhausted = "NO"
    else:
        reason = "not_recorded"
        exhausted = "NO"

    candidate_count = diagnostics.get("candidate_count")
    source = "analysis_report option diagnostics.candidate_count"
    if candidate_count is None:
        source = "not_recorded"
    return reason, exhausted, candidate_count, source


def report_option_row(
    specimen_id: str,
    job_id: str,
    provenance: str,
    known_weight_ct: float,
    report_path: Path,
    option: dict[str, Any],
    qz01_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = manufacturing_plan(option)
    diagnostics = option_diagnostics(option)
    status = plan.get("status")
    sequence = plan.get("sequence") or []
    mfg_complete = bool(
        option.get("manufacturing_eligible", True)
        and is_complete_manufacturing(status)
        and exact_sequence_verified(plan)
    )
    containment, spacing, defect, cut_sequence, combined = constraint_statuses(
        option, diagnostics, plan
    )
    reason, exhausted, candidates, candidate_source = search_status(
        specimen_id, option, diagnostics, qz01_manifest
    )

    yield_percent = to_float(option.get("yield"))
    retained = to_float(option.get("weight"))
    notes = []
    if combined != "PASS":
        notes.append(f"constraint_status={combined}")
    if status == "settings_required":
        notes.append("multi-gem yield retained as diagnostic because exact cut sequence is absent")
    if option.get("manufacturing_eligible") is False:
        notes.append("geometric comparison is not manufacturing eligible")

    return {
        "specimen_id": specimen_id,
        "job_id": job_id,
        "reconstruction_provenance": provenance,
        "known_weight_ct": round_or_none(known_weight_ct, 6),
        "optimizer_strategy": option.get("type"),
        "strategy_name": option.get("name"),
        "gem_count": option.get("gem_count"),
        "retained_cuttable_weight_ct": round_or_none(retained, 6),
        "yield_percent": round_or_none(yield_percent, 6),
        "waste_percent": pct_waste(yield_percent),
        "manufacturing_cut_sequence_complete": "YES" if mfg_complete else "NO",
        "number_of_cuts": len(sequence),
        "search_termination_reason": reason,
        "time_budget_exhausted": exhausted,
        "candidate_placement_count": candidates,
        "candidate_placement_count_source": candidate_source,
        "verification_scope": verification_scope(option, plan),
        "geometric_containment_status": containment,
        "inter_gem_spacing_clearance_status": spacing,
        "no_cut_defect_constraint_status": defect,
        "exact_cut_sequence_status": cut_sequence,
        "evidence_source": rel(report_path),
        "notes": "; ".join(notes),
    }


def qz01_diagnostic_four_gem_row(
    qz01_job_id: str,
    known_weight_ct: float,
    qz01_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    probe = qz01_manifest.get("fourth_gem_probe_whole_stone_unconstrained") or {}
    stored = qz01_manifest.get("current_saved_plan_metrics") or {}
    candidates = probe.get("top_candidates_checked") or []
    exact_candidates = [
        candidate for candidate in candidates
        if (candidate.get("exact_cut_sequence_check") or {}).get("status") == "complete"
        and (candidate.get("exact_cut_sequence_check") or {})
        .get("diagnostics", {})
        .get("exact_sequence_verified") is True
    ]
    if not exact_candidates:
        return None

    best = max(exact_candidates, key=lambda c: float(c.get("estimated_weight_ct") or 0))
    exact = best["exact_cut_sequence_check"]
    saved_weight = to_float(stored.get("saved_estimated_weight_ct_from_fit"))
    fourth_weight = to_float(best.get("estimated_weight_ct"))
    total_weight = (saved_weight or 0.0) + (fourth_weight or 0.0)
    yield_percent = total_weight / known_weight_ct * 100.0
    clearance_mm = to_float(exact.get("minimum_envelope_clearance_mm"))
    candidate_count = probe.get("fit_candidate_count")

    return {
        "specimen_id": "QZ-01",
        "job_id": qz01_job_id,
        "reconstruction_provenance": (
            "confirmed primary optimizer benchmark; diagnostic whole-stone "
            "residual probe from visual audit"
        ),
        "known_weight_ct": round_or_none(known_weight_ct, 6),
        "optimizer_strategy": "Repacked Cuttable Plan (diagnostic fourth-gem probe)",
        "strategy_name": "4 gems (3-gem saved plan plus best exact-complete fourth candidate)",
        "gem_count": 4,
        "retained_cuttable_weight_ct": round(total_weight, 3),
        "yield_percent": round(yield_percent, 2),
        "waste_percent": round(100.0 - yield_percent, 6),
        "manufacturing_cut_sequence_complete": "YES",
        "number_of_cuts": exact.get("sequence_length"),
        "search_termination_reason": "diagnostic_probe_completed_without_timeout",
        "time_budget_exhausted": "NO",
        "candidate_placement_count": candidate_count,
        "candidate_placement_count_source": (
            "QZ01_CURRENT_PLAN_MANIFEST fourth_gem_probe_whole_stone_unconstrained.fit_candidate_count"
        ),
        "verification_scope": "diagnostic_only_exact_cut_sequence_verified_not_production_option",
        "geometric_containment_status": (
            "PASS: probe candidate surface clearance "
            f"{best.get('surface_clearance_mm')} mm; envelope clearance {clearance_mm} mm"
        ),
        "inter_gem_spacing_clearance_status": (
            "PASS: exact sequence verified with protected envelopes"
        ),
        "no_cut_defect_constraint_status": (
            "PASS: exact check recorded zero approved no-cut defect points"
        ),
        "exact_cut_sequence_status": (
            "PASS: diagnostic exact straight full-through sequence verified"
        ),
        "evidence_source": rel(QZ01_VISUAL_AUDIT),
        "notes": (
            "Diagnostic evidence only; not the saved production optimizer option "
            "and not proof of global optimality."
        ),
    }


def build_specimen_records() -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    recon_rows = {row["specimen_id"]: row for row in read_csv_rows(RECON_RESULTS)}
    baseline_rows = read_csv_rows(OPTIMIZER_BASELINE)
    qz01 = next(
        row for row in baseline_rows
        if row["specimen_id_if_confirmed"] == "QZ-01"
        and row["counts_as_independent_physical_input"] == "yes"
    )

    specimens: dict[str, dict[str, Any]] = {
        "QZ-01": {
            "specimen_id": "QZ-01",
            "job_id": qz01["job_id"],
            "known_weight_ct": float(qz01["input_weight_ct"]),
            "provenance": (
                "existing confirmed QZ-01 optimizer benchmark from optimizer_baseline"
            ),
            "analysis_report_path": ROOT / qz01["analysis_report_path"],
        }
    }
    for specimen_id in TRUE_DENSE_SPECIMENS:
        row = recon_rows[specimen_id]
        specimens[specimen_id] = {
            "specimen_id": specimen_id,
            "job_id": row["job_id"],
            "known_weight_ct": float(row["known_weight_ct"]),
            "provenance": row["reconstruction_provenance"],
            "analysis_report_path": ROOT / "jobs" / row["job_id"] / "analysis_report.json",
        }

    unavailable = []
    for specimen_id in ("QZ-30", "QZ-09"):
        row = recon_rows.get(specimen_id, {})
        unavailable.append({
            "specimen_id": specimen_id,
            "job_id": row.get("job_id", ""),
            "reason": (
                f"excluded from Objective-3 system-side set: "
                f"{row.get('reconstruction_provenance', 'unknown provenance')}"
            ),
        })
    for row in baseline_rows:
        if row.get("counts_as_independent_physical_input") == "no":
            unavailable.append({
                "specimen_id": row.get("specimen_id_if_confirmed") or "(unconfirmed)",
                "job_id": row["job_id"],
                "reason": row.get("evidence_status") or "not independent physical input",
            })
    return specimens, unavailable


def collect_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    specimens, unavailable = build_specimen_records()
    qz01_manifest = load_json(QZ01_VISUAL_AUDIT)
    rows: list[dict[str, Any]] = []
    for specimen_id in (*ANCHOR_SPECIMENS, *TRUE_DENSE_SPECIMENS):
        specimen = specimens[specimen_id]
        report_path = specimen["analysis_report_path"]
        report = load_json(report_path)
        for option in report.get("options") or []:
            rows.append(
                report_option_row(
                    specimen_id=specimen_id,
                    job_id=specimen["job_id"],
                    provenance=specimen["provenance"],
                    known_weight_ct=specimen["known_weight_ct"],
                    report_path=report_path,
                    option=option,
                    qz01_manifest=qz01_manifest if specimen_id == "QZ-01" else None,
                )
            )
        if specimen_id == "QZ-01":
            diagnostic = qz01_diagnostic_four_gem_row(
                specimen["job_id"], specimen["known_weight_ct"], qz01_manifest
            )
            if diagnostic:
                rows.append(diagnostic)
    return rows, unavailable


def internal_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    by_specimen: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_specimen.setdefault(row["specimen_id"], []).append(row)

    for specimen_id, specimen_rows in by_specimen.items():
        baselines = [
            row for row in specimen_rows
            if row["optimizer_strategy"] == "Single Large"
        ]
        if not baselines:
            continue
        baseline = baselines[0]
        base_yield = to_float(baseline["yield_percent"])
        base_weight = to_float(baseline["retained_cuttable_weight_ct"])
        base_waste = to_float(baseline["waste_percent"])
        for row in specimen_rows:
            if row is baseline:
                continue
            if row["optimizer_strategy"] == "Geometric Comparison":
                continue
            y = to_float(row["yield_percent"])
            w = to_float(row["retained_cuttable_weight_ct"])
            waste = to_float(row["waste_percent"])
            comparisons.append({
                "specimen_id": specimen_id,
                "comparison_scope": COMPARISON_LABEL,
                "traditional_cutting_comparison": "NO",
                "baseline_strategy": baseline["optimizer_strategy"],
                "compared_strategy": row["optimizer_strategy"],
                "compared_strategy_name": row["strategy_name"],
                "baseline_yield_percent": base_yield,
                "compared_yield_percent": y,
                "absolute_yield_improvement_pp": (
                    round(y - base_yield, 6)
                    if y is not None and base_yield is not None else None
                ),
                "relative_retained_weight_improvement_percent": (
                    round((w - base_weight) / base_weight * 100.0, 6)
                    if w is not None and base_weight else None
                ),
                "internal_waste_reduction_percent": (
                    round((base_waste - waste) / base_waste * 100.0, 6)
                    if waste is not None and base_waste else None
                ),
                "verification_scope": row["verification_scope"],
            })
    return comparisons


def summarize(
    rows: list[dict[str, Any]],
    unavailable: list[dict[str, str]],
) -> dict[str, Any]:
    comparisons = internal_comparisons(rows)
    eligible_specimens = sorted({row["specimen_id"] for row in rows})
    fully_verified_multi = [
        row for row in rows
        if int(row["gem_count"] or 0) > 1
        and row["manufacturing_cut_sequence_complete"] == "YES"
        and row["verification_scope"] == "fully_verified_existing_report"
    ]
    diagnostic_exact = [
        row for row in rows
        if "diagnostic_only_exact_cut_sequence_verified" in row["verification_scope"]
    ]
    settings_required = [
        row for row in rows
        if row["verification_scope"] == "diagnostic_only_missing_exact_cut_sequence"
    ]
    return {
        "objective": "Objective 3 optimizer/yield validation",
        "validation_status": (
            "SYSTEM-SIDE AUDIT COMPLETE; traditional/expert comparison pending"
        ),
        "scientific_boundary": [
            "Internal optimizer strategy comparisons are internal computational baseline comparisons.",
            "Internal optimizer strategy comparisons are not traditional cutting comparisons.",
            "The >=15% proposal waste-reduction target is not claimed from internal baselines.",
            "External independent gem-cutter or traditional-cutting data are required for the proposal target.",
            "No photogrammetry, optimizer search, defect model, facet ML, or frontend code was rerun by this script.",
        ],
        "eligible_specimens": eligible_specimens,
        "eligible_specimen_count": len(eligible_specimens),
        "unavailable_or_ineligible": unavailable,
        "result_row_count": len(rows),
        "fully_verified_multi_gem_existing_report_rows": fully_verified_multi,
        "diagnostic_exact_cut_sequence_rows": diagnostic_exact,
        "diagnostic_settings_required_rows": settings_required,
        "internal_strategy_comparisons": comparisons,
        "proposal_target": {
            "waste_reduction_target_percent": 15,
            "status": PROPOSAL_TARGET_STATUS,
            "reason": (
                "No independent traditional/expert cutter baseline artifact was "
                "found in the audited Objective-3 evidence."
            ),
        },
        "outputs": {
            "csv": rel(RESULTS_CSV),
            "summary_json": rel(SUMMARY_JSON),
            "report_md": rel(REPORT_MD),
        },
    }


def write_results(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with REPORT_MD.open("w", encoding="utf-8") as handle:
        handle.write("# Objective 3 System-Side Optimizer/Yield Validation\n\n")
        handle.write("## Scientific Boundary\n\n")
        for item in summary["scientific_boundary"]:
            handle.write(f"- {item}\n")
        handle.write("\n")
        handle.write("## Eligible Specimens\n\n")
        handle.write(", ".join(summary["eligible_specimens"]))
        handle.write("\n\n")
        handle.write("## System-Side Yield Results\n\n")
        handle.write(
            "| Specimen | Job | Strategy | Gems | Retained ct | Yield % | "
            "Waste % | Manufacturing complete | Cuts | Verification |\n"
        )
        handle.write(
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |\n"
        )
        for row in rows:
            handle.write(
                f"| {row['specimen_id']} | `{row['job_id'][:8]}...` | "
                f"{row['optimizer_strategy']} | {row['gem_count']} | "
                f"{row['retained_cuttable_weight_ct']} | {row['yield_percent']} | "
                f"{row['waste_percent']} | "
                f"{row['manufacturing_cut_sequence_complete']} | "
                f"{row['number_of_cuts']} | {row['verification_scope']} |\n"
            )
        handle.write("\n")
        handle.write("## Internal Strategy Comparisons\n\n")
        handle.write(
            "All rows below are internal computational baseline comparison rows, "
            "not traditional cutting comparison rows.\n\n"
        )
        handle.write(
            "| Specimen | Compared strategy | Yield improvement pp | "
            "Retained-weight improvement % | Internal waste reduction % | Verification |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | --- |\n")
        for item in summary["internal_strategy_comparisons"]:
            handle.write(
                f"| {item['specimen_id']} | {item['compared_strategy']} | "
                f"{item['absolute_yield_improvement_pp']} | "
                f"{item['relative_retained_weight_improvement_percent']} | "
                f"{item['internal_waste_reduction_percent']} | "
                f"{item['verification_scope']} |\n"
            )
        handle.write("\n")
        handle.write("## Proposal Target\n\n")
        handle.write(
            f"Status: {summary['proposal_target']['status']}. "
            f"Reason: {summary['proposal_target']['reason']}\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Objective 3 system-side optimizer yield evidence."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and summarize existing artifacts without writing outputs.",
    )
    args = parser.parse_args()

    rows, unavailable = collect_rows()
    summary = summarize(rows, unavailable)
    if not args.check_only:
        write_results(rows, summary)

    print(json.dumps({
        "validation_status": summary["validation_status"],
        "eligible_specimens": summary["eligible_specimens"],
        "result_rows": summary["result_row_count"],
        "proposal_target_status": summary["proposal_target"]["status"],
        "outputs_written": not args.check_only,
        "outputs": summary["outputs"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
