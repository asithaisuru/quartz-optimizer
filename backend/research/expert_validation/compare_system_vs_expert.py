"""Compare frozen system optimizer results against independent expert responses.

Run this only after Form A responses are collected and frozen. Diagnostic
system rows are not used for the primary Objective-3 proposal calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    COMPARISON_FIELDS,
    COMPARISON_RESULTS,
    PROPOSAL_TARGET_WASTE_REDUCTION_PERCENT,
    angular_difference_degrees,
    expert_yield_percent,
    load_primary_system_results,
    normalize_text,
    read_csv_rows,
    relative_waste_reduction,
    to_float,
    to_int,
    waste_percent,
)
from validate_expert_responses import validate_rows  # noqa: E402


def row_orientation(row: dict[str, str]) -> tuple[float, float, float] | None:
    values = [
        to_float(row.get("orientation_vector_x")),
        to_float(row.get("orientation_vector_y")),
        to_float(row.get("orientation_vector_z")),
    ]
    if any(value is None for value in values):
        return None
    return values[0], values[1], values[2]  # type: ignore[return-value]


def system_shape(row: dict[str, str]) -> str:
    if to_int(row["gem_count"]) == 1:
        return row["strategy_name"]
    return ""


def compare_rows(expert_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    validation = validate_rows(expert_rows)
    if not validation["valid_for_comparison"]:
        raise ValueError("Expert responses are not valid for comparison.")

    primary = load_primary_system_results()
    comparisons: list[dict[str, object]] = []
    for row in expert_rows:
        specimen_id = row["specimen_id"].strip()
        system = primary[specimen_id]
        known_weight = to_float(system["known_weight_ct"])
        system_retained = to_float(system["retained_cuttable_weight_ct"])
        system_yield = to_float(system["yield_percent"])
        system_waste = to_float(system["waste_percent"])
        expert_retained = to_float(row["retained_weight_ct"])
        expert_gems = to_int(row["gem_count"])

        if known_weight is None or system_retained is None or system_yield is None:
            raise ValueError(f"Missing system metrics for {specimen_id}.")
        if system_waste is None or expert_retained is None or expert_gems is None:
            raise ValueError(f"Missing comparison metrics for {specimen_id}.")

        expert_yield = expert_yield_percent(expert_retained, known_weight)
        expert_waste = waste_percent(expert_yield)
        waste_reduction = relative_waste_reduction(system_waste, expert_waste)
        if waste_reduction is None:
            target_status = "INCONCLUSIVE: expert waste is zero"
        elif waste_reduction >= PROPOSAL_TARGET_WASTE_REDUCTION_PERCENT:
            target_status = "ACHIEVED"
        else:
            target_status = "NOT ACHIEVED"

        s_shape = normalize_text(system_shape(system))
        e_shape = normalize_text(row["recommended_cut_shape"])
        if not s_shape:
            shape_status = "UNAVAILABLE: system primary result has multiple gems"
        elif s_shape == e_shape:
            shape_status = "MATCH"
        else:
            shape_status = "DIFFERENT"

        expert_orientation = row_orientation(row)
        orientation_diff = None
        if expert_orientation is not None:
            orientation_diff = None

        comparisons.append({
            "response_id": row["response_id"],
            "expert_id": row["expert_id"],
            "specimen_id": specimen_id,
            "known_weight_ct": round(known_weight, 6),
            "system_job_id": system["job_id"],
            "system_strategy": system["optimizer_strategy"],
            "system_strategy_name": system["strategy_name"],
            "system_gem_count": system["gem_count"],
            "system_retained_weight_ct": system_retained,
            "system_yield_percent": system_yield,
            "system_waste_percent": system_waste,
            "system_verification_scope": system["verification_scope"],
            "expert_recommended_shape": row["recommended_cut_shape"],
            "expert_gem_count": expert_gems,
            "expert_retained_weight_ct": expert_retained,
            "expert_yield_percent": round(expert_yield, 6),
            "expert_waste_percent": round(expert_waste, 6),
            "absolute_retained_weight_difference_ct": round(
                abs(system_retained - expert_retained), 6
            ),
            "system_minus_expert_yield_pp": round(system_yield - expert_yield, 6),
            "relative_waste_reduction_percent": (
                round(waste_reduction, 6)
                if waste_reduction is not None else ""
            ),
            "proposal_target_status": target_status,
            "shape_agreement_status": shape_status,
            "orientation_angular_difference_deg": (
                round(orientation_diff, 6)
                if orientation_diff is not None else "UNAVAILABLE"
            ),
            "confidence_1_to_5": row["confidence_1_to_5"],
            "notes": (
                "Primary comparison uses only manufacturing-complete "
                "non-diagnostic system result."
            ),
        })
    return comparisons


def write_comparison(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses_csv", type=Path)
    parser.add_argument("--output", type=Path, default=COMPARISON_RESULTS)
    args = parser.parse_args()

    expert_rows = read_csv_rows(args.responses_csv)
    comparisons = compare_rows(expert_rows)
    write_comparison(comparisons, args.output)
    print(json.dumps({
        "comparison_rows": len(comparisons),
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
