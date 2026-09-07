"""Build grouped uncertainty summaries from frozen Quartz evidence only.

This module does not run COLMAP, model inference, optimizer search, threshold
tuning, retraining, or label mutation. It reads existing frozen evidence and
writes a supplemental academic-validation package.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "final_research_evidence" / "final_academic_validation"
UNCERTAINTY_SUMMARY_JSON = OUTPUT_DIR / "uncertainty_summary.json"
UNCERTAINTY_RESULTS_CSV = OUTPUT_DIR / "uncertainty_results.csv"

RECON_RESULTS_CSV = (
    ROOT
    / "final_research_evidence"
    / "reconstruction_multi"
    / "batch_validation_results.csv"
)
RECON_SUMMARY_JSON = (
    ROOT
    / "final_research_evidence"
    / "reconstruction_multi"
    / "batch_validation_summary.json"
)
DEFECT_RESULTS_JSON = (
    ROOT
    / "final_research_evidence"
    / "defect_detection"
    / "final_test"
    / "independent_final_test_results"
    / "independent_final_test_results.json"
)
FACET_METRICS_JSON = (
    ROOT / "backend" / "research" / "facet_ml" / "facet_orientation_metrics.json"
)
FACET_MANIFEST_JSON = (
    ROOT / "backend" / "research" / "facet_ml" / "facet_orientation_manifest.json"
)
FACET_DATASET_CSV = (
    ROOT / "backend" / "research" / "facet_ml" / "facet_orientation_dataset.csv"
)
OPTIMIZER_SUMMARY_JSON = (
    ROOT
    / "final_research_evidence"
    / "optimizer_validation"
    / "system_yield_summary.json"
)
OPTIMIZER_RESULTS_CSV = (
    ROOT
    / "final_research_evidence"
    / "optimizer_validation"
    / "system_yield_results.csv"
)

BOOTSTRAP_SEED = 20260907
BOOTSTRAP_ITERATIONS = 10_000
TRUE_DENSE_SPECIMENS = ("QZ-03", "QZ-05", "QZ-08", "QZ-14")
DEFECT_SPECIMEN_GROUP_COUNT = 9
DEFECT_IMAGES_PER_GROUP = 5

UNCERTAINTY_FIELDS = [
    "domain",
    "metric",
    "point_estimate",
    "ci_lower",
    "ci_upper",
    "ci_method",
    "status",
    "resampling_unit",
    "group_count",
    "observation_count",
    "seed",
    "iterations",
    "notes",
]

CONFUSION_FIELDS = ("tp", "fp", "fn", "tn")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNCERTAINTY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in UNCERTAINTY_FIELDS})


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def as_float(value: Any) -> float:
    return float(str(value).strip())


def parse_ranked_dimensions(value: str) -> list[float]:
    parts = [part.strip() for part in value.split("x")]
    if len(parts) != 3:
        raise ValueError(f"Expected three ranked dimensions: {value!r}")
    return [float(part) for part in parts]


def metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty metric vector.")
    return {
        "mae": sum(values) / len(values),
        "rmse": math.sqrt(sum(value * value for value in values) / len(values)),
        "median_abs_error": float(median(values)),
        "max_abs_error": max(values),
    }


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty vector.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile_value / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    lower_weight = upper - rank
    upper_weight = rank - lower
    return ordered[lower] * lower_weight + ordered[upper] * upper_weight


def confidence_interval(values: list[float]) -> dict[str, float]:
    return {
        "lower": percentile(values, 2.5),
        "upper": percentile(values, 97.5),
    }


def bootstrap_metric_intervals(
    groups: list[dict[str, Any]],
    metric_function: Callable[[list[dict[str, Any]]], dict[str, float]],
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, dict[str, float]]:
    if not groups:
        raise ValueError("Bootstrap requires at least one group.")
    rng = random.Random(seed)
    sampled: dict[str, list[float]] = defaultdict(list)
    group_count = len(groups)
    for _ in range(iterations):
        sample = [groups[rng.randrange(group_count)] for _ in range(group_count)]
        metrics = metric_function(sample)
        for name, value in metrics.items():
            sampled[name].append(float(value))
    return {name: confidence_interval(values) for name, values in sampled.items()}


def reconstruction_groups_from_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups = []
    for row in rows:
        included = row.get("included_in_dense_aggregate", "").strip().lower() == "yes"
        provenance = row.get("reconstruction_provenance", "").strip()
        if not included or provenance != "TRUE_DENSE":
            continue
        specimen_id = row["specimen_id"].strip()
        physical = parse_ranked_dimensions(row["physical_ranked_mm"])
        principal = parse_ranked_dimensions(row["principal_oriented_ranked_mm"])
        errors = [abs(actual - predicted) for actual, predicted in zip(physical, principal)]
        groups.append(
            {
                "specimen_id": specimen_id,
                "physical_ranked_mm": physical,
                "principal_oriented_ranked_mm": principal,
                "dimension_abs_errors_mm": errors,
                "specimen_metrics": {
                    "mae_mm": as_float(row["principal_mae_mm"]),
                    "rmse_mm": as_float(row["principal_rmse_mm"]),
                    "median_error_mm": as_float(row["principal_median_error_mm"]),
                    "max_error_mm": as_float(row["principal_max_error_mm"]),
                },
            }
        )
    return sorted(groups, key=lambda item: item["specimen_id"])


def reconstruction_metrics(sampled_groups: list[dict[str, Any]]) -> dict[str, float]:
    errors = [
        error
        for group in sampled_groups
        for error in group["dimension_abs_errors_mm"]
    ]
    return metric_summary(errors)


def bootstrap_reconstruction(
    rows: list[dict[str, str]] | None = None,
    summary: dict[str, Any] | None = None,
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    rows = rows if rows is not None else read_csv_rows(RECON_RESULTS_CSV)
    summary = summary if summary is not None else read_json(RECON_SUMMARY_JSON)
    groups = reconstruction_groups_from_rows(rows)
    errors = [
        error
        for group in groups
        for error in group["dimension_abs_errors_mm"]
    ]
    point_estimates = metric_summary(errors)
    intervals = bootstrap_metric_intervals(
        groups,
        reconstruction_metrics,
        seed=seed,
        iterations=iterations,
    )
    expected_specimens = sorted(TRUE_DENSE_SPECIMENS)
    sparse_excluded = [
        row["specimen_id"]
        for row in rows
        if row.get("reconstruction_provenance") == "SPARSE_FALLBACK"
        and row.get("included_in_dense_aggregate", "").strip().lower() != "yes"
    ]
    return {
        "status": "estimated",
        "source_csv": rel(RECON_RESULTS_CSV),
        "source_summary": rel(RECON_SUMMARY_JSON),
        "resampling_unit": "physical_specimen",
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
        "specimen_count": len(groups),
        "dimension_count": len(errors),
        "specimen_ids": [group["specimen_id"] for group in groups],
        "expected_true_dense_specimens": expected_specimens,
        "sparse_fallback_excluded": sparse_excluded,
        "pixel_or_dimension_independent_bootstrap_used": False,
        "small_sample_caveat": (
            "Only four TRUE_DENSE physical specimens are available; intervals "
            "are descriptive/exploratory and do not establish population-level "
            "precision."
        ),
        "point_estimates_from_dimension_rows": point_estimates,
        "frozen_reported_principal_oriented": summary.get("principal_oriented", {}),
        "bootstrap_95_percentile_ci": intervals,
        "specimen_level_errors": groups,
        "target": {"metric": "principal-oriented absolute dimensional error", "target_mm": 0.1},
        "target_status": summary.get("proposal_target", {}).get("status", "NOT_ACHIEVED"),
        "rounding_note": (
            "Bootstrap uses ranked dimension values stored in the frozen CSV, "
            "which are rounded to three decimals; the frozen summary remains "
            "the authoritative aggregate point estimate."
        ),
    }


def aggregate_confusion_counts(
    groups: list[dict[str, Any]],
    selected_indices: list[int] | range | None = None,
) -> dict[str, int]:
    indices = list(selected_indices) if selected_indices is not None else list(range(len(groups)))
    aggregate = {field: 0 for field in CONFUSION_FIELDS}
    for index in indices:
        group = groups[index]
        for field in CONFUSION_FIELDS:
            aggregate[field] += int(group[field])
    return aggregate


def compute_defect_metrics(counts: dict[str, int]) -> dict[str, float]:
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    tn = int(counts["tn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "specificity": specificity,
    }


def defect_groups_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    per_specimen = result.get("per_specimen")
    if not isinstance(per_specimen, dict):
        raise ValueError("Defect result is missing per_specimen counts.")
    image_counts = Counter(
        row.get("specimen_id")
        for row in result.get("per_image", [])
        if isinstance(row, dict)
    )
    groups = []
    for specimen_id, values in per_specimen.items():
        group = {
            "specimen_id": specimen_id,
            "image_count": int(image_counts.get(specimen_id, 0)),
        }
        for field in CONFUSION_FIELDS:
            group[field] = int(values[field])
        group.update(compute_defect_metrics(group))
        groups.append(group)
    return sorted(groups, key=lambda item: item["specimen_id"])


def verify_defect_group_sums(
    result: dict[str, Any],
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    from_groups = aggregate_confusion_counts(groups)
    aggregate = {
        field: int(result.get("aggregate", {}).get(field, 0))
        for field in CONFUSION_FIELDS
    }
    return {
        "matches_frozen_aggregate": from_groups == aggregate,
        "from_groups": from_groups,
        "frozen_aggregate": aggregate,
    }


def defect_metrics_for_sample(sampled_groups: list[dict[str, Any]]) -> dict[str, float]:
    return compute_defect_metrics(aggregate_confusion_counts(sampled_groups))


def bootstrap_defect(
    result: dict[str, Any] | None = None,
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    result = result if result is not None else read_json(DEFECT_RESULTS_JSON)
    groups = defect_groups_from_result(result)
    intervals = bootstrap_metric_intervals(
        groups,
        defect_metrics_for_sample,
        seed=seed,
        iterations=iterations,
    )
    group_sum_check = verify_defect_group_sums(result, groups)
    image_counts = {group["specimen_id"]: group["image_count"] for group in groups}
    return {
        "status": "estimated",
        "source_json": rel(DEFECT_RESULTS_JSON),
        "resampling_unit": "independent_physical_specimen_group",
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
        "specimen_group_count": len(groups),
        "image_count": int(result.get("image_count", sum(image_counts.values()))),
        "images_per_group": image_counts,
        "all_groups_have_five_images": all(
            count == DEFECT_IMAGES_PER_GROUP for count in image_counts.values()
        ),
        "group_sum_check": group_sum_check,
        "pixel_level_bootstrap_used": False,
        "image_level_bootstrap_used": False,
        "native_ultralytics_inference_run": False,
        "thresholds_or_model_mutated": False,
        "configuration": result.get("configuration", {}),
        "point_estimates": result.get("aggregate", {}),
        "bootstrap_95_percentile_ci": {
            metric: intervals[metric]
            for metric in ("precision", "recall", "f1", "iou", "specificity")
            if metric in intervals
        },
        "native_mask_map50_ci": {
            "status": "CI_NOT_ESTIMATED",
            "reason": (
                "Native mAP50 confidence intervals are not derivable from the "
                "stored aggregate/per-image confusion artifacts without rerunning "
                "native validation or relying on unavailable per-instance AP "
                "samples."
            ),
            "frozen_mask_map50": (
                result.get("native_segmentation", {}) or {}
            ).get("mask_map_50"),
        },
        "target": {"metric": "binary_visible_defect_f1", "target_f1": 0.90},
        "target_status": (
            result.get("proposal_90_percent_target", {}) or {}
        ).get("status", "NOT ACHIEVED"),
        "specimen_level_counts": groups,
    }


def facet_uncertainty_status() -> dict[str, Any]:
    metrics = read_json(FACET_METRICS_JSON)
    manifest = read_json(FACET_MANIFEST_JSON)
    dataset_rows = read_csv_rows(FACET_DATASET_CSV)
    headers = set(dataset_rows[0].keys()) if dataset_rows else set()
    prediction_columns = sorted(
        headers
        & {
            "model_prediction",
            "model_predicted_score",
            "predicted_score",
            "model_score",
            "y_pred",
        }
    )
    if prediction_columns:
        status = "available_but_not_computed"
        reason = "Prediction columns were detected but no audited parser is defined."
    else:
        status = "unavailable"
        reason = (
            "Frozen aggregate metrics and the simulation dataset are present, "
            "but held-out per-scenario model predictions are not stored as a "
            "frozen artifact. Regenerating predictions from the model artifact "
            "was intentionally not performed."
        )
    return {
        "status": status,
        "source_metrics": rel(FACET_METRICS_JSON),
        "source_manifest": rel(FACET_MANIFEST_JSON),
        "source_dataset": rel(FACET_DATASET_CSV),
        "dataset_rows_examined": len(dataset_rows),
        "scenario_group_count": manifest.get("dataset", {}).get("scenario_group_count"),
        "test_scenario_group_count": (
            manifest.get("dataset", {}).get("split_group_counts", {}) or {}
        ).get("test"),
        "prediction_columns_found": prediction_columns,
        "bootstrap_resampling_unit_required": "scenario_group",
        "bootstrap_ci_estimated": False,
        "reason": reason,
        "frozen_test_metrics": metrics.get("splits", {}).get("test", {}),
        "boundary": (
            "No retraining, prediction regeneration, or row-level bootstrap was "
            "performed. The facet result remains simulation-bound."
        ),
    }


def optimizer_descriptive_boundary() -> dict[str, Any]:
    summary = read_json(OPTIMIZER_SUMMARY_JSON)
    rows = read_csv_rows(OPTIMIZER_RESULTS_CSV)
    return {
        "status": "descriptive_only",
        "source_summary": rel(OPTIMIZER_SUMMARY_JSON),
        "source_csv": rel(OPTIMIZER_RESULTS_CSV),
        "eligible_specimen_count": summary.get("eligible_specimen_count"),
        "eligible_specimens": summary.get("eligible_specimens"),
        "result_row_count": summary.get("result_row_count", len(rows)),
        "inferential_ci_estimated": False,
        "reason": (
            "The current optimizer evidence is a tiny system-side set with mixed "
            "verified and diagnostic rows; statistical inference is not justified."
        ),
        "external_expert_comparison_status": summary.get("proposal_target", {}).get("status"),
        "internal_strategy_comparisons": summary.get("internal_strategy_comparisons", []),
        "verified_rows": [
            row
            for row in rows
            if row.get("verification_scope") == "fully_verified_existing_report"
        ],
        "diagnostic_rows": [
            row
            for row in rows
            if row.get("verification_scope", "").startswith("diagnostic")
        ],
        "scientific_boundary": summary.get("scientific_boundary", []),
    }


def uncertainty_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reconstruction = summary["reconstruction"]
    for metric, ci in reconstruction["bootstrap_95_percentile_ci"].items():
        rows.append(
            {
                "domain": "reconstruction",
                "metric": metric,
                "point_estimate": reconstruction["point_estimates_from_dimension_rows"][metric],
                "ci_lower": ci["lower"],
                "ci_upper": ci["upper"],
                "ci_method": "specimen-level bootstrap percentile",
                "status": "estimated",
                "resampling_unit": reconstruction["resampling_unit"],
                "group_count": reconstruction["specimen_count"],
                "observation_count": reconstruction["dimension_count"],
                "seed": reconstruction["bootstrap_seed"],
                "iterations": reconstruction["bootstrap_iterations"],
                "notes": reconstruction["small_sample_caveat"],
            }
        )
    defect = summary["defect_detection"]
    for metric, ci in defect["bootstrap_95_percentile_ci"].items():
        rows.append(
            {
                "domain": "defect_detection",
                "metric": metric,
                "point_estimate": defect["point_estimates"].get(metric),
                "ci_lower": ci["lower"],
                "ci_upper": ci["upper"],
                "ci_method": "specimen-group-level bootstrap percentile",
                "status": "estimated",
                "resampling_unit": defect["resampling_unit"],
                "group_count": defect["specimen_group_count"],
                "observation_count": defect["image_count"],
                "seed": defect["bootstrap_seed"],
                "iterations": defect["bootstrap_iterations"],
                "notes": "Counts are reconstructed from stored TP/FP/FN/TN groups.",
            }
        )
    facet = summary["facet_ml"]
    rows.append(
        {
            "domain": "facet_ml",
            "metric": "ml_vs_heuristic_grouped_uncertainty",
            "status": "unavailable",
            "ci_method": "not estimated",
            "resampling_unit": facet["bootstrap_resampling_unit_required"],
            "group_count": facet.get("test_scenario_group_count"),
            "observation_count": "",
            "notes": facet["reason"],
        }
    )
    optimizer = summary["optimizer"]
    for item in optimizer["internal_strategy_comparisons"]:
        for metric in (
            "absolute_yield_improvement_pp",
            "relative_retained_weight_improvement_percent",
            "internal_waste_reduction_percent",
        ):
            rows.append(
                {
                    "domain": "optimizer",
                    "metric": metric,
                    "point_estimate": item.get(metric),
                    "ci_method": "descriptive only",
                    "status": "no_inferential_ci",
                    "resampling_unit": "specimen",
                    "group_count": optimizer["eligible_specimen_count"],
                    "observation_count": optimizer["result_row_count"],
                    "notes": (
                        f"{item.get('specimen_id')} {item.get('compared_strategy')} "
                        f"({item.get('verification_scope')}); not a traditional "
                        "cutter comparison."
                    ),
                }
            )
    return rows


def build_uncertainty_package(
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    global OUTPUT_DIR, UNCERTAINTY_SUMMARY_JSON, UNCERTAINTY_RESULTS_CSV
    OUTPUT_DIR = output_dir
    UNCERTAINTY_SUMMARY_JSON = OUTPUT_DIR / "uncertainty_summary.json"
    UNCERTAINTY_RESULTS_CSV = OUTPUT_DIR / "uncertainty_results.csv"

    summary = {
        "schema_version": "1.0",
        "report_type": "final_academic_validation_uncertainty",
        "frozen_evidence_only": True,
        "no_model_training": True,
        "no_model_inference": True,
        "no_colmap_rerun": True,
        "no_optimizer_search": True,
        "no_threshold_or_label_mutation": True,
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
        "reconstruction": bootstrap_reconstruction(seed=seed, iterations=iterations),
        "defect_detection": bootstrap_defect(seed=seed, iterations=iterations),
        "facet_ml": facet_uncertainty_status(),
        "optimizer": optimizer_descriptive_boundary(),
    }
    rows = uncertainty_rows(summary)
    outputs = {
        "uncertainty_summary_json": rel(UNCERTAINTY_SUMMARY_JSON),
        "uncertainty_results_csv": rel(UNCERTAINTY_RESULTS_CSV),
    }
    summary["outputs"] = outputs
    write_json(UNCERTAINTY_SUMMARY_JSON, summary)
    write_csv(UNCERTAINTY_RESULTS_CSV, rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    summary = build_uncertainty_package(
        seed=args.seed,
        iterations=args.iterations,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": "SAFE",
                "frozen_evidence_only": summary["frozen_evidence_only"],
                "reconstruction_status": summary["reconstruction"]["status"],
                "defect_status": summary["defect_detection"]["status"],
                "facet_status": summary["facet_ml"]["status"],
                "optimizer_status": summary["optimizer"]["status"],
                "outputs": summary["outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
