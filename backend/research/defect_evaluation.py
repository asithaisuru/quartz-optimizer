"""Defect metric utilities and final-evaluation readiness gates."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import atomic_write_json, utc_timestamp
from .model_provenance import inspect_model_provenance
from .readiness_common import (
    atomic_write_csv,
    atomic_write_text,
    load_json_config,
    resolve_path,
    write_manifest,
)
from .taxonomy_validation import validate_taxonomy_approval

EXPERT_STATUSES = {
    "approved",
    "corrected",
    "rejected",
    "needs_second_review",
    "unreviewed",
}
PRIMARY_METRICS = {
    "binary_visible_defect_f1",
    "binary_visible_defect_recall",
    "macro_class_f1",
}


def load_evaluation_config(
    path: str | Path,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    return load_json_config(path, output_override=output_override)


def _divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def binary_segmentation_metrics(
    ground_truth: Any,
    prediction: Any,
) -> dict[str, Any]:
    truth = np.asarray(ground_truth, dtype=bool)
    predicted = np.asarray(prediction, dtype=bool)
    if truth.shape != predicted.shape:
        raise ValueError("Ground-truth and prediction masks must have the same shape.")
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    tn = int(np.count_nonzero(~truth & ~predicted))
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    f1_denominator = 2 * tp + fp + fn
    union = tp + fp + fn
    if f1_denominator == 0:
        f1 = 1.0
        dice = 1.0
    else:
        f1 = float(2 * tp / f1_denominator)
        dice = f1
    iou = 1.0 if union == 0 else float(tp / union)
    defined_rates = [value for value in (recall, specificity) if value is not None]
    balanced = (
        float(sum(defined_rates) / len(defined_rates))
        if len(defined_rates) == 2
        else None
    )
    if not np.any(truth) and not np.any(predicted):
        status = "empty_ground_truth_and_prediction"
    elif np.any(truth) and not np.any(predicted):
        status = "empty_prediction_with_positive_ground_truth"
    elif not np.any(truth) and np.any(predicted):
        status = "prediction_with_empty_ground_truth"
    else:
        status = "evaluated"
    return {
        "status": status,
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "true_negative_pixels": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "undefined_metric_policy": (
            "Undefined precision/recall/specificity values are null. IoU, Dice, "
            "and F1 are 1 only when both masks contain no positive pixels."
        ),
    }


def _average(
    rows: list[dict[str, Any]],
    metric: str,
    weights: list[int] | None = None,
) -> float | None:
    values = []
    effective_weights = []
    for index, row in enumerate(rows):
        value = row.get(metric)
        if value is None:
            continue
        values.append(float(value))
        effective_weights.append(weights[index] if weights else 1)
    if not values or sum(effective_weights) == 0:
        return None
    return float(
        sum(value * weight for value, weight in zip(values, effective_weights))
        / sum(effective_weights)
    )


def per_class_segmentation_metrics(
    ground_truth: Any,
    prediction: Any,
    class_ids: Iterable[int],
) -> dict[str, Any]:
    truth = np.asarray(ground_truth)
    predicted = np.asarray(prediction)
    if truth.shape != predicted.shape:
        raise ValueError("Ground-truth and prediction arrays must have the same shape.")
    classes = list(class_ids)
    rows = []
    supports = []
    for class_id in classes:
        metrics = binary_segmentation_metrics(
            truth == class_id,
            predicted == class_id,
        )
        support = int(np.count_nonzero(truth == class_id))
        supports.append(support)
        rows.append({"class_id": class_id, "support_pixels": support, **metrics})
    aggregate_counts = {
        "true_positive_pixels": sum(row["true_positive_pixels"] for row in rows),
        "false_positive_pixels": sum(row["false_positive_pixels"] for row in rows),
        "false_negative_pixels": sum(row["false_negative_pixels"] for row in rows),
        "true_negative_pixels": sum(row["true_negative_pixels"] for row in rows),
    }
    micro_truth = np.concatenate(
        [(truth == class_id).reshape(-1) for class_id in classes]
    ) if classes else np.asarray([], dtype=bool)
    micro_predicted = np.concatenate(
        [(predicted == class_id).reshape(-1) for class_id in classes]
    ) if classes else np.asarray([], dtype=bool)
    micro = binary_segmentation_metrics(micro_truth, micro_predicted)
    metrics = ("precision", "recall", "f1", "iou", "dice")
    return {
        "per_class": rows,
        "macro_average": {
            metric: _average(rows, metric) for metric in metrics
        },
        "micro_average": {
            metric: micro[metric] for metric in metrics
        },
        "weighted_average": {
            metric: _average(rows, metric, supports) for metric in metrics
        },
        "aggregate_counts": aggregate_counts,
    }


def _mask_iou(first: Any, second: Any) -> float:
    left = np.asarray(first, dtype=bool)
    right = np.asarray(second, dtype=bool)
    if left.shape != right.shape:
        raise ValueError("Instance masks must have matching shapes.")
    union = int(np.count_nonzero(left | right))
    return 1.0 if union == 0 else float(np.count_nonzero(left & right) / union)


def instance_segmentation_metrics(
    ground_truth_instances: Iterable[Any],
    predicted_instances: Iterable[Any],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be within [0, 1].")
    ground = list(ground_truth_instances)
    predicted = list(predicted_instances)
    candidates = []
    for prediction_id, prediction in enumerate(predicted):
        for ground_id, truth in enumerate(ground):
            iou = _mask_iou(prediction, truth)
            if iou >= iou_threshold:
                candidates.append((iou, prediction_id, ground_id))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched_predictions = set()
    matched_ground = set()
    matches = []
    for iou, prediction_id, ground_id in candidates:
        if prediction_id in matched_predictions or ground_id in matched_ground:
            continue
        matched_predictions.add(prediction_id)
        matched_ground.add(ground_id)
        matches.append(
            {
                "prediction_id": prediction_id,
                "ground_truth_id": ground_id,
                "iou": iou,
            }
        )
    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(ground) - tp
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    denominator = 2 * tp + fp + fn
    f1 = 1.0 if denominator == 0 else float(2 * tp / denominator)
    return {
        "iou_threshold": float(iou_threshold),
        "true_positive_objects": tp,
        "false_positive_objects": fp,
        "false_negative_objects": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "matching": "one_to_one_greedy_descending_iou",
    }


def unavailable_yolo_native_metrics(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "mask_map_50": None,
        "mask_map_50_95": None,
        "box_metrics": None,
        "reason": reason,
        "implementation": "Ultralytics native validation required.",
    }


def summarize_expert_review(
    path: Path | None,
    test_sample_ids: set[str],
) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "status": "unavailable",
            "reviewed_image_count": 0,
            "reviewed_specimen_count": 0,
            "corrected_label_count": 0,
            "rejected_label_count": 0,
            "unresolved_disagreement_count": 0,
            "test_set_image_count": len(test_sample_ids),
            "test_set_reviewed_count": 0,
            "test_set_review_coverage": 0.0,
            "invalid_status_count": 0,
        }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {key: str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    reviewed_statuses = {"approved", "corrected", "rejected", "needs_second_review"}
    reviewed = [
        row for row in rows if row.get("review_status") in reviewed_statuses
    ]
    reviewed_test = {
        row.get("sample_id")
        for row in reviewed
        if row.get("sample_id") in test_sample_ids
    }
    invalid = sum(
        row.get("review_status") not in EXPERT_STATUSES for row in rows
    )
    return {
        "status": "available",
        "reviewed_image_count": len(reviewed),
        "reviewed_specimen_count": len(
            {
                row.get("specimen_id")
                for row in reviewed
                if row.get("specimen_id")
            }
        ),
        "corrected_label_count": sum(
            row.get("review_status") == "corrected" for row in rows
        ),
        "rejected_label_count": sum(
            row.get("review_status") == "rejected" for row in rows
        ),
        "unresolved_disagreement_count": sum(
            row.get("review_status") == "needs_second_review" for row in rows
        ),
        "test_set_image_count": len(test_sample_ids),
        "test_set_reviewed_count": len(reviewed_test),
        "test_set_review_coverage": (
            round(len(reviewed_test) / len(test_sample_ids), 6)
            if test_sample_ids else 0.0
        ),
        "invalid_status_count": invalid,
    }


def build_evaluation_gates(context: dict[str, Any]) -> list[dict[str, Any]]:
    definitions = (
        ("taxonomy_approved", "Approved taxonomy and matching approval hash"),
        ("grouped_split_approved", "Grouped split approved"),
        ("grouped_split_frozen", "Grouped split frozen"),
        ("grouping_leakage_free", "Configured grouping leakage absent"),
        ("test_set_locked", "Held-out test set lock available"),
        ("expert_review_sufficient", "Expert test-mask coverage sufficient"),
        ("test_labels_available", "Test labels available"),
        ("primary_metric_approved", "Primary 90 percent claim metric approved"),
        ("model_final_eligible", "Model training provenance compatible"),
        ("runtime_dependencies_available", "Required runtime dependencies available"),
        ("model_classes_match_taxonomy", "Model classes match taxonomy"),
        ("test_data_excluded_from_training", "Test data excluded from training"),
        ("threshold_tuned_on_validation_only", "Threshold selected on validation only"),
        ("final_output_available", "Final output directory is new or resumable"),
        ("deterministic_seed_recorded", "Deterministic seed recorded"),
    )
    gates = []
    reasons = context.get("reasons", {})
    for key, title in definitions:
        passed = bool(context.get(key, False))
        gates.append(
            {
                "gate": key,
                "title": title,
                "status": "pass" if passed else "blocked",
                "passed": passed,
                "reason": reasons.get(
                    key,
                    "Evidence is available." if passed else "Required evidence is unavailable.",
                ),
            }
        )
    return gates


def _read_split_state(directory: Path | None) -> dict[str, Any]:
    if directory is None:
        return {}
    try:
        lock = json.loads((directory / "split_lock.json").read_text(encoding="utf-8"))
        summary = json.loads(
            (directory / "split_summary.json").read_text(encoding="utf-8")
        )
        leakage = json.loads(
            (directory / "split_leakage_audit.json").read_text(encoding="utf-8")
        )
        with (directory / "grouped_split_manifest.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        return {"lock": lock, "summary": summary, "leakage": leakage, "rows": rows}
    except (OSError, ValueError):
        return {}


def preflight_defect_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["_repo_root"])
    taxonomy_path = resolve_path(config.get("taxonomy_path"), repo_root)
    approval_path = resolve_path(config.get("taxonomy_approval"), repo_root)
    dataset_yaml = resolve_path(config.get("dataset_yaml"), repo_root)
    taxonomy = (
        validate_taxonomy_approval(taxonomy_path, approval_path, dataset_yaml)
        if taxonomy_path and dataset_yaml
        else None
    )
    split_directory = resolve_path(config.get("grouped_split_directory"), repo_root)
    split = _read_split_state(split_directory)
    test_rows = [
        row for row in split.get("rows", [])
        if row.get("assigned_split") == "test"
    ]
    test_ids = {row.get("sample_id", "") for row in test_rows}
    expert_path = resolve_path(config.get("expert_ground_truth_review"), repo_root)
    expert = summarize_expert_review(expert_path, test_ids)

    model_config = config.get("model_provenance", {})
    model = inspect_model_provenance(
        model_path=resolve_path(model_config.get("model_path"), repo_root)
        or repo_root / "backend" / "best.pt",
        dataset_yaml=resolve_path(
            model_config.get("dataset_yaml", config.get("dataset_yaml")),
            repo_root,
        ),
        training_args=resolve_path(model_config.get("training_args"), repo_root),
        training_source=resolve_path(model_config.get("training_source"), repo_root),
        dataset_audit=resolve_path(model_config.get("dataset_audit"), repo_root),
        clean_provenance=model_config.get("clean_provenance"),
    )
    environment_path = resolve_path(config.get("environment_report"), repo_root)
    environment = {}
    if environment_path and environment_path.exists():
        try:
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            environment = {}

    primary_metric = config.get("primary_claim_metric")
    claim_threshold = config.get("claim_threshold")
    metric_approved = bool(
        primary_metric in PRIMARY_METRICS
        and claim_threshold == 0.90
        and config.get("approved_by")
        and config.get("approval_date")
    )
    minimum_review = float(config.get("minimum_expert_review_coverage", 1.0))
    final_output = resolve_path(config.get("final_output_directory"), repo_root)
    final_output_available = bool(
        final_output
        and (
            not final_output.exists()
            or bool(config.get("allow_resume_final_output", False))
        )
    )
    lock = split.get("lock", {})
    leakage = split.get("leakage", {})
    split_summary = split.get("summary", {})
    test_label_paths = [
        resolve_path(row.get("label_path"), repo_root) for row in test_rows
    ]
    taxonomy_approved = bool(
        taxonomy and taxonomy["summary"]["taxonomy_approved"]
    )
    context = {
        "taxonomy_approved": taxonomy_approved,
        "grouped_split_approved": bool(split_summary.get("split_approved")),
        "grouped_split_frozen": bool(lock.get("frozen")),
        "grouping_leakage_free": bool(leakage.get("passes")),
        "test_set_locked": bool(lock.get("test_set_hash")),
        "expert_review_sufficient": bool(
            expert["test_set_review_coverage"] >= minimum_review
            and expert["invalid_status_count"] == 0
            and expert["unresolved_disagreement_count"] == 0
        ),
        "test_labels_available": bool(
            test_rows
            and all(path is not None and path.is_file() for path in test_label_paths)
        ),
        "primary_metric_approved": metric_approved,
        "model_final_eligible": bool(model["final_research_eligible"]),
        "runtime_dependencies_available": bool(
            environment.get("summary", {}).get("final_model_runtime_ready")
        ),
        "model_classes_match_taxonomy": bool(
            taxonomy
            and taxonomy["class_validation"]
            and all(row["valid"] for row in taxonomy["class_validation"])
        ),
        "test_data_excluded_from_training": bool(
            model["training_split_provenance"] == "clean"
        ),
        "threshold_tuned_on_validation_only": (
            config.get("threshold_selection_split") == "valid"
        ),
        "final_output_available": final_output_available,
        "deterministic_seed_recorded": isinstance(
            config.get("deterministic_seed"), int
        ),
        "reasons": {
            "taxonomy_approved": (
                "No approved taxonomy/hash-linked approval is available."
                if not taxonomy_approved else "Taxonomy approval is valid."
            ),
            "expert_review_sufficient": (
                f"Test expert-review coverage is "
                f"{expert['test_set_review_coverage']:.3f}; required {minimum_review:.3f}."
            ),
            "primary_metric_approved": (
                "The primary operational metric and approver are not recorded."
                if not metric_approved else "Primary metric approval is recorded."
            ),
            "model_final_eligible": model["provenance_basis"],
            "runtime_dependencies_available": (
                "Environment readiness is incomplete or unavailable."
            ),
            "threshold_tuned_on_validation_only": (
                "Threshold selection must use validation data, never the test set."
            ),
        },
    }
    gates = build_evaluation_gates(context)
    blockers = [gate["reason"] for gate in gates if not gate["passed"]]
    ready = not blockers
    return {
        "schema_version": "1.0",
        "audit_type": "defect_evaluation_preflight",
        "created_utc": utc_timestamp(),
        "mode": config.get("evaluation_mode", "both"),
        "detector_policies": config.get("detector_policies", []),
        "primary_claim": {
            "proposal_wording": "visible-flaw detection accuracy target of 90 percent",
            "operational_metric": primary_metric,
            "threshold": claim_threshold,
            "precision_floor": config.get("precision_floor"),
            "approved_by": config.get("approved_by"),
            "approval_date": config.get("approval_date"),
            "status": "unavailable",
        },
        "gates": gates,
        "expert_review": expert,
        "model_provenance": {
            "training_split_provenance": model["training_split_provenance"],
            "final_research_eligible": model["final_research_eligible"],
            "exploratory_pipeline_smoke_test": model[
                "exploratory_pipeline_smoke_test"
            ],
            "model_sha256": model["model"]["sha256"],
        },
        "split": {
            "split_id": lock.get("split_id"),
            "approved": split_summary.get("split_approved", False),
            "frozen": lock.get("frozen", False),
            "test_set_hash": lock.get("test_set_hash"),
        },
        "summary": {
            "status": "ready" if ready else "blocked",
            "final_evaluation_ready": ready,
            "gate_count": len(gates),
            "passed_gate_count": sum(gate["passed"] for gate in gates),
            "blocked_gate_count": len(blockers),
            "blockers": blockers,
            "final_metrics_calculated": False,
            "claim_90_percent_available": False,
        },
        "metrics": {
            "pixel_binary": None,
            "per_class": None,
            "instance": None,
            "yolo_native": unavailable_yolo_native_metrics(
                "Preflight mode does not execute model validation."
            ),
        },
        "claim_boundary": (
            "Preflight checks readiness only. It does not run inference, tune "
            "thresholds, calculate final metrics, or support the 90 percent claim."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return "\n".join(
        [
            "# Defect Evaluation Preflight",
            "",
            f"- Status: {summary['status']}",
            f"- Final evaluation ready: {str(summary['final_evaluation_ready']).lower()}",
            f"- Passed gates: {summary['passed_gate_count']} / {summary['gate_count']}",
            f"- Final metrics calculated: {str(summary['final_metrics_calculated']).lower()}",
            f"- 90 percent claim available: {str(summary['claim_90_percent_available']).lower()}",
            "",
            *[
                f"- {gate['status']}: {gate['title']} - {gate['reason']}"
                for gate in report["gates"]
            ],
            "",
            report["claim_boundary"],
            "",
        ]
    )


def run_evaluation_preflight(
    config: dict[str, Any],
    output_override: str | Path | None = None,
    command: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    report = preflight_defect_evaluation(config)
    repo_root = Path(config["_repo_root"])
    output = resolve_path(
        output_override or config.get("output_directory"),
        repo_root,
    )
    if output is None:
        raise ValueError("An output directory is required.")
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "report": output / "defect_evaluation_preflight.json",
        "gates": output / "evaluation_gates.csv",
        "expert": output / "expert_review_summary.csv",
        "markdown": output / "audit_summary.md",
        "manifest": output / "manifest.json",
    }
    atomic_write_json(files["report"], report)
    atomic_write_csv(
        files["gates"],
        report["gates"],
        ("gate", "title", "status", "passed", "reason"),
    )
    atomic_write_csv(
        files["expert"],
        [report["expert_review"]],
        tuple(report["expert_review"].keys()),
    )
    atomic_write_text(files["markdown"], _markdown(report))
    write_manifest(
        path=files["manifest"],
        repo_root=repo_root,
        tool_name="quartz_defect_evaluation_preflight",
        command=command,
        tool_sources=(
            "backend/research/defect_evaluation.py",
            "backend/research/model_provenance.py",
            "backend/research/taxonomy_validation.py",
            "backend/research/readiness_common.py",
            "backend/research_benchmark.py",
        ),
        inputs=(
            ("configuration", config.get("_config_path")),
            ("taxonomy", config.get("taxonomy_path")),
            ("taxonomy_approval", config.get("taxonomy_approval")),
            ("environment_report", config.get("environment_report")),
            ("expert_review", config.get("expert_ground_truth_review")),
        ),
        outputs=(
            ("report", files["report"]),
            ("gates", files["gates"]),
            ("expert", files["expert"]),
            ("markdown", files["markdown"]),
        ),
        extra={
            "final_metrics_calculated": False,
            "claim_90_percent_available": False,
        },
    )
    return report, files
