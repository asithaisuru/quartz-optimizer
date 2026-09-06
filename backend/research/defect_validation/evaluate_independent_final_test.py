"""Guarded one-time evaluator for the independent defect final-test subset."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.common import (  # noqa: E402
    atomic_write_json,
    normalize_path,
    sha256_file,
    utc_timestamp,
)
from research.defect_validation.independent_final_test import (  # noqa: E402
    BINARY_CONFIDENCE_THRESHOLD,
    CLASS_NAMES,
    EXPECTED_IMAGE_COUNT,
    EXPECTED_FINAL_METRICS,
    EXPECTED_FINAL_RESULT_SHA256,
    EXPECTED_GROUP_COUNT,
    EXPECTED_LABEL_MANIFEST_SHA256,
    EXPECTED_MEMBERSHIP_SHA256,
    EXPECTED_MODEL_SHA256,
    EXPECTED_NATIVE_MASK_MAP50,
    INFERENCE_IMAGE_SIZE,
    NATIVE_VALIDATION_IOU,
    NMS_IOU_THRESHOLD,
    PROPOSAL_TARGET_F1,
    annotation_workspace,
    completed_final_result_path,
    expected_label_path,
    final_result_freeze_metadata,
    final_test_root,
    frozen_metric_configuration,
    image_id,
    label_freeze_manifest_path,
    manifest_path,
    manual_annotation_status,
    read_json,
    repo_root,
    selected_images,
    validate_frozen_manifest,
    verify_label_freeze_manifest,
    verify_selected_model,
    verify_split_leakage,
)

EVALUATION_DEVICE = "cpu"
COUNT_KEYS = ("tp", "fp", "fn", "tn")
METRIC_KEYS = ("precision", "recall", "f1", "iou", "specificity")


def _metrics_from_counts(counts: Counter | dict[str, int]) -> dict[str, float]:
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    tn = int(counts.get("tn", 0))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_value = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_value,
        "iou": iou,
        "specificity": specificity,
    }


def _count_values(record: dict[str, Any]) -> dict[str, int]:
    return {key: int(record.get(key, 0)) for key in COUNT_KEYS}


def _sum_count_values(records: list[dict[str, Any]]) -> Counter:
    totals: Counter = Counter()
    for record in records:
        totals.update(_count_values(record))
    return totals


def _float_matches(left: Any, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)


def _stored_metrics_match(record: dict[str, Any]) -> bool:
    recalculated = _metrics_from_counts(_count_values(record))
    return all(_float_matches(record[metric], recalculated[metric]) for metric in METRIC_KEYS)


def _mask_from_yolo_label(label_path: Path, width: int, height: int, class_id: int | None = None) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw

    mask = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < 7 or len(tokens[1:]) % 2:
            raise ValueError(f"Invalid YOLO polygon in {label_path}:{line_number}")
        current_class = int(float(tokens[0]))
        if current_class not in range(len(CLASS_NAMES)):
            raise ValueError(f"Invalid class id in {label_path}:{line_number}")
        if class_id is not None and current_class != class_id:
            continue
        coords = [float(value) for value in tokens[1:]]
        if any(value < 0.0 or value > 1.0 for value in coords):
            raise ValueError(f"Coordinate outside [0,1] in {label_path}:{line_number}")
        points = [
            (coords[index] * width, coords[index + 1] * height)
            for index in range(0, len(coords), 2)
        ]
        draw.polygon(points, fill=1)
    return np.array(mask, dtype=bool)


def _mask_from_yolo_result(result: Any, width: int, height: int, class_id: int | None = None) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw

    mask = Image.new("1", (width, height), 0)
    if getattr(result, "masks", None) is None or getattr(result.masks, "xy", None) is None:
        return np.array(mask, dtype=bool)
    classes = []
    if getattr(result, "boxes", None) is not None and getattr(result.boxes, "cls", None) is not None:
        classes = [int(value) for value in result.boxes.cls.detach().cpu().numpy().tolist()]
    draw = ImageDraw.Draw(mask)
    for index, polygon in enumerate(result.masks.xy):
        if class_id is not None and classes and classes[index] != class_id:
            continue
        points = [(float(x), float(y)) for x, y in polygon]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    return np.array(mask, dtype=bool)


def _confusion_counts(truth: Any, prediction: Any) -> Counter:
    import numpy as np

    return Counter(
        {
            "tp": int(np.logical_and(truth, prediction).sum()),
            "fp": int(np.logical_and(~truth, prediction).sum()),
            "fn": int(np.logical_and(truth, ~prediction).sum()),
            "tn": int(np.logical_and(~truth, ~prediction).sum()),
        }
    )


def _write_native_data_yaml(root: Path, rows: list[dict[str, Any]], temporary: Path) -> Path:
    images_dir = annotation_workspace(root) / "images"
    labels_dir = annotation_workspace(root) / "labels"
    image_list = temporary / "final_test_images.txt"
    image_list.write_text(
        "\n".join(str((root / str(row["workspace_image_path"])).resolve()) for row in rows) + "\n",
        encoding="utf-8",
    )
    data_yaml = temporary / "final_test_data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {annotation_workspace(root).as_posix()}",
                f"train: {images_dir.as_posix()}",
                f"val: {image_list.as_posix()}",
                f"test: {image_list.as_posix()}",
                f"nc: {len(CLASS_NAMES)}",
                "names: [\"fracture\", \"inclusion\"]",
                f"# labels: {labels_dir.as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def _native_map50(metrics: Any) -> dict[str, Any]:
    segment = getattr(metrics, "seg", None)
    box = getattr(metrics, "box", None)
    return {
        "status": "available",
        "mask_map_50": float(getattr(segment, "map50", 0.0)) if segment is not None else None,
        "mask_map_50_95": float(getattr(segment, "map", 0.0)) if segment is not None else None,
        "box_map_50": float(getattr(box, "map50", 0.0)) if box is not None else None,
        "box_map_50_95": float(getattr(box, "map", 0.0)) if box is not None else None,
    }


def device_selection_report() -> dict[str, Any]:
    return {
        "device": EVALUATION_DEVICE,
        "selection": "explicit_cpu",
        "cuda_autodetect_used": False,
        "thresholds_or_membership_changed": False,
    }


def readiness_report(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    frozen = validate_frozen_manifest(root)
    leakage = verify_split_leakage(root)
    model = verify_selected_model(root)
    annotations = manual_annotation_status(root)
    ready = (
        frozen["status"] == "pass"
        and leakage["status"] == "pass"
        and model["status"] == "pass"
        and annotations["status"] == "complete_and_frozen"
    )
    return {
        "schema_version": "1.0",
        "report_type": "independent_defect_final_test_readiness",
        "created_utc": utc_timestamp(),
        "final_test_manifest": frozen,
        "split_leakage_verification": leakage,
        "selected_model_verification": model,
        "manual_annotation_status": annotations,
        "frozen_inference_metric_configuration": frozen_metric_configuration(),
        "device_selection": device_selection_report(),
        "evaluation_ready": ready,
        "inference_or_evaluation_run": False,
        "blockers": [
            label for label, passed in (
                ("frozen manifest mismatch", frozen["status"] == "pass"),
                ("split leakage verification failed", leakage["status"] == "pass"),
                ("selected model hash mismatch", model["status"] == "pass"),
                (
                    "manual labels are not complete and frozen",
                    annotations["status"] == "complete_and_frozen",
                ),
            )
            if not passed
        ],
    }


def completed_final_result_integrity_report(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    result_path = completed_final_result_path(root)
    metadata = final_result_freeze_metadata()
    if not result_path.is_file():
        return {
            "schema_version": "1.0",
            "report_type": "independent_defect_final_result_integrity_audit",
            "created_utc": utc_timestamp(),
            "status": "needs_correction",
            "final_result_metadata": metadata,
            "completed_result_path": normalize_path(result_path, root),
            "inference_or_evaluation_run": False,
            "read_only_audit": True,
            "issues": ["completed final result JSON is missing"],
        }

    result = read_json(result_path)
    result_sha256 = sha256_file(result_path)
    frozen = validate_frozen_manifest(root)
    label_freeze = verify_label_freeze_manifest(root)
    freeze_payload = read_json(label_freeze_manifest_path(root))
    model = verify_selected_model(root)
    per_image = result.get("per_image") if isinstance(result.get("per_image"), list) else []
    per_specimen = result.get("per_specimen") if isinstance(result.get("per_specimen"), dict) else {}
    per_class = (
        result.get("per_class_pixel_metrics")
        if isinstance(result.get("per_class_pixel_metrics"), dict)
        else {}
    )
    aggregate = result.get("aggregate") if isinstance(result.get("aggregate"), dict) else {}
    configuration = result.get("configuration") if isinstance(result.get("configuration"), dict) else {}
    device = result.get("device_selection") if isinstance(result.get("device_selection"), dict) else {}
    proposal = (
        result.get("proposal_90_percent_target")
        if isinstance(result.get("proposal_90_percent_target"), dict)
        else {}
    )
    readiness = (
        result.get("readiness_evidence")
        if isinstance(result.get("readiness_evidence"), dict)
        else {}
    )
    readiness_annotations = (
        readiness.get("manual_annotation_status")
        if isinstance(readiness.get("manual_annotation_status"), dict)
        else {}
    )

    per_image_sum = _sum_count_values(per_image)
    per_image_sum_matches = dict(per_image_sum) == _count_values(aggregate)
    aggregate_metrics_match = bool(aggregate) and _stored_metrics_match(aggregate)
    per_image_metrics_match = all(_stored_metrics_match(row) for row in per_image)
    per_class_metrics_match = all(_stored_metrics_match(row) for row in per_class.values())

    per_specimen_sum: dict[str, Counter] = defaultdict(Counter)
    for row in per_image:
        per_specimen_sum[str(row.get("specimen_id", ""))].update(_count_values(row))
    per_specimen_counts_match = (
        set(per_specimen_sum) == set(per_specimen)
        and all(
            dict(per_specimen_sum[specimen_id]) == _count_values(values)
            for specimen_id, values in per_specimen.items()
        )
    )
    per_specimen_metrics_match = all(
        _stored_metrics_match(values) for values in per_specimen.values()
    )

    expected_configuration = frozen_metric_configuration()
    configuration_matches = all(
        configuration.get(key) == value for key, value in expected_configuration.items()
    )
    labels_frozen_before_inference = (
        label_freeze["status"] == "pass"
        and readiness_annotations.get("status") == "complete_and_frozen"
        and str(freeze_payload.get("created_utc", "")) <= str(result.get("created_utc", ""))
    )

    issues = []
    if result_sha256 != EXPECTED_FINAL_RESULT_SHA256:
        issues.append("completed final result JSON SHA-256 changed")
    if result.get("report_type") != "independent_defect_final_test_results":
        issues.append("result report_type is not independent final-test results")
    if result.get("inference_or_evaluation_run") is not True:
        issues.append("result does not record completed inference/evaluation")
    if result.get("image_count") != EXPECTED_IMAGE_COUNT:
        issues.append("result image count mismatch")
    if result.get("specimen_group_count") != EXPECTED_GROUP_COUNT:
        issues.append("result specimen group count mismatch")
    if result.get("membership_sha256") != EXPECTED_MEMBERSHIP_SHA256:
        issues.append("result membership SHA-256 mismatch")
    if result.get("model_sha256") != EXPECTED_MODEL_SHA256:
        issues.append("result model SHA-256 mismatch")
    if frozen["status"] != "pass":
        issues.append("frozen manifest verification failed")
    if label_freeze["label_manifest_sha256"] != EXPECTED_LABEL_MANIFEST_SHA256:
        issues.append("label manifest SHA-256 mismatch")
    if label_freeze["status"] != "pass":
        issues.append("label freeze verification failed")
    if model["status"] != "pass":
        issues.append("selected model verification failed")
    if not labels_frozen_before_inference:
        issues.append("labels are not proven frozen before inference")
    if not configuration_matches:
        issues.append("frozen inference configuration mismatch")
    if configuration.get("final_test_threshold_tuning_allowed") is not False:
        issues.append("result does not forbid final-test threshold tuning")
    if device.get("device") != EVALUATION_DEVICE or device.get("selection") != "explicit_cpu":
        issues.append("result does not record explicit CPU evaluation")
    if device.get("cuda_autodetect_used") is not False:
        issues.append("result used or recorded CUDA auto-detection")
    for metric, expected in EXPECTED_FINAL_METRICS.items():
        if not _float_matches(aggregate.get(metric), expected):
            issues.append(f"aggregate {metric} mismatch")
    if not per_image_sum_matches:
        issues.append("aggregate counts do not equal summed per-image counts")
    if not aggregate_metrics_match:
        issues.append("aggregate metrics do not recompute from aggregate counts")
    if not per_image_metrics_match:
        issues.append("one or more per-image metrics do not recompute from counts")
    if not per_specimen_counts_match:
        issues.append("per-specimen counts do not equal summed per-image counts")
    if not per_specimen_metrics_match:
        issues.append("one or more per-specimen metrics do not recompute from counts")
    if list(per_class) != list(CLASS_NAMES):
        issues.append("per-class metric names do not match frozen class order")
    if not per_class_metrics_match:
        issues.append("one or more per-class metrics do not recompute from counts")
    native = result.get("native_segmentation", {})
    if native.get("status") != "available":
        issues.append("native Ultralytics validation is not available in final result")
    if not _float_matches(native.get("mask_map_50"), EXPECTED_NATIVE_MASK_MAP50):
        issues.append("native mask mAP50 mismatch")
    if proposal.get("status") != "NOT ACHIEVED":
        issues.append("proposal target status mismatch")
    if not _float_matches(proposal.get("target"), PROPOSAL_TARGET_F1):
        issues.append("proposal target threshold mismatch")
    if not _float_matches(proposal.get("value"), EXPECTED_FINAL_METRICS["f1"]):
        issues.append("proposal target value does not equal final F1")

    audit_status = "PASS" if not issues else "FAIL"
    return {
        "schema_version": "1.0",
        "report_type": "independent_defect_final_result_integrity_audit",
        "created_utc": utc_timestamp(),
        "status": "pass" if not issues else "needs_correction",
        "integrity_audit": audit_status,
        "final_result_metadata": {**metadata, "post_evaluation_integrity_audit": audit_status},
        "completed_result_path": normalize_path(result_path, root),
        "completed_result_sha256": result_sha256,
        "completed_result_sha256_expected": EXPECTED_FINAL_RESULT_SHA256,
        "original_final_evaluation_output_unchanged": result_sha256 == EXPECTED_FINAL_RESULT_SHA256,
        "image_count": result.get("image_count"),
        "specimen_group_count": result.get("specimen_group_count"),
        "membership_sha256": result.get("membership_sha256"),
        "label_manifest_sha256": label_freeze.get("label_manifest_sha256"),
        "model_sha256": result.get("model_sha256"),
        "labels_frozen_before_inference": labels_frozen_before_inference,
        "no_final_test_threshold_tuning": (
            configuration.get("final_test_threshold_tuning_allowed") is False
        ),
        "device_selection": device,
        "configuration": configuration,
        "aggregate_counts": _count_values(aggregate),
        "aggregate_metrics": {key: aggregate.get(key) for key in EXPECTED_FINAL_METRICS},
        "aggregate_metrics_recomputed": _metrics_from_counts(_count_values(aggregate)),
        "aggregate_counts_equal_summed_per_image": per_image_sum_matches,
        "per_specimen_counts_match_per_image": per_specimen_counts_match,
        "per_class_metrics_match_counts": per_class_metrics_match,
        "native_validation_subset": (
            "same frozen 45 manifest rows and annotation_workspace labels"
        ),
        "native_mask_map50": native.get("mask_map_50"),
        "proposal_90_percent_target": proposal,
        "read_only_audit": True,
        "inference_rerun": False,
        "threshold_tuning_run": False,
        "issues": issues,
    }


def run_final_evaluation(output: Path | None = None, root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    readiness = readiness_report(root)
    if not readiness["evaluation_ready"]:
        return readiness

    os.environ.setdefault("YOLO_CONFIG_DIR", str(defect_validation_config_dir(root)))
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("YOLO_VERBOSE", "False")

    from PIL import Image
    from ultralytics import YOLO

    manifest = read_json(manifest_path(root))
    rows = selected_images(manifest)
    model_info = verify_selected_model(root)
    model_path = root / model_info["model_path"]
    if sha256_file(model_path) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Selected model hash changed before evaluation.")
    if manifest.get("membership_sha256") != EXPECTED_MEMBERSHIP_SHA256:
        raise RuntimeError("Frozen test membership hash changed before evaluation.")

    output = output or (final_test_root(root) / "independent_final_test_results")
    result_path = output / "independent_final_test_results.json"
    if result_path.exists():
        raise RuntimeError(f"Final-test result already exists: {result_path}")
    output.mkdir(parents=True, exist_ok=False)

    model = YOLO(str(model_path))
    device = EVALUATION_DEVICE
    totals = Counter()
    class_totals = {class_id: Counter() for class_id in range(len(CLASS_NAMES))}
    specimen_totals: dict[str, Counter] = defaultdict(Counter)
    per_image = []

    for row in rows:
        image_path = root / str(row["workspace_image_path"])
        label_path = expected_label_path(row, root)
        with Image.open(image_path) as image:
            width, height = image.size
        result = model.predict(
            source=str(image_path),
            imgsz=INFERENCE_IMAGE_SIZE,
            conf=BINARY_CONFIDENCE_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            device=device,
            verbose=False,
            save=False,
        )[0]
        truth = _mask_from_yolo_label(label_path, width, height)
        prediction = _mask_from_yolo_result(result, width, height)
        counts = _confusion_counts(truth, prediction)
        totals.update(counts)
        specimen_totals[str(row["specimen_id"])].update(counts)
        per_image.append(
            {
                "image_id": image_id(row),
                "specimen_id": row["specimen_id"],
                "image_path": normalize_path(image_path, root),
                **dict(counts),
                **_metrics_from_counts(counts),
            }
        )
        for class_id in range(len(CLASS_NAMES)):
            class_truth = _mask_from_yolo_label(label_path, width, height, class_id)
            class_prediction = _mask_from_yolo_result(result, width, height, class_id)
            class_totals[class_id].update(_confusion_counts(class_truth, class_prediction))

    native = {"status": "unavailable", "reason": "Ultralytics native validation did not run."}
    try:
        with tempfile.TemporaryDirectory(prefix="qz-independent-final-test-") as tmp:
            data_yaml = _write_native_data_yaml(root, rows, Path(tmp))
            native_metrics = model.val(
                data=str(data_yaml),
                imgsz=INFERENCE_IMAGE_SIZE,
                iou=NATIVE_VALIDATION_IOU,
                device=device,
                plots=False,
                save_json=False,
                project=str(output / "native_yolo_val"),
                name="ultralytics_native",
                exist_ok=False,
            )
        native = _native_map50(native_metrics)
    except Exception as exc:
        native = {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    aggregate = {**dict(totals), **_metrics_from_counts(totals)}
    report = {
        "schema_version": "1.0",
        "report_type": "independent_defect_final_test_results",
        "created_utc": utc_timestamp(),
        "inference_or_evaluation_run": True,
        "model_path": normalize_path(model_path, root),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "membership_sha256": EXPECTED_MEMBERSHIP_SHA256,
        "image_count": EXPECTED_IMAGE_COUNT,
        "specimen_group_count": len({row["specimen_id"] for row in rows}),
        "configuration": frozen_metric_configuration(),
        "device_selection": device_selection_report(),
        "aggregate": aggregate,
        "per_specimen": {
            specimen_id: {**dict(counts), **_metrics_from_counts(counts)}
            for specimen_id, counts in sorted(specimen_totals.items())
        },
        "per_class_pixel_metrics": {
            CLASS_NAMES[class_id]: {**dict(counts), **_metrics_from_counts(counts)}
            for class_id, counts in class_totals.items()
        },
        "per_image": per_image,
        "native_segmentation": native,
        "proposal_90_percent_target": {
            "metric": "binary_visible_defect_f1",
            "target": PROPOSAL_TARGET_F1,
            "value": aggregate["f1"],
            "status": "ACHIEVED" if aggregate["f1"] >= PROPOSAL_TARGET_F1 else "NOT ACHIEVED",
        },
        "readiness_evidence": readiness,
    }
    atomic_write_json(result_path, report)
    return report


def defect_validation_config_dir(root: Path) -> Path:
    path = final_test_root(root) / "ultralytics_config"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check or run the guarded independent defect final-test evaluation. "
            "Use --check-only until manual labels are complete and frozen."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run-once", action="store_true")
    mode.add_argument("--audit-result", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    root = repo_root()
    if args.check_only:
        report = readiness_report(root)
        print(json.dumps(report, indent=2))
        return 0
    if args.audit_result:
        report = completed_final_result_integrity_report(root)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "pass" else 2

    freeze_path = label_freeze_manifest_path(root)
    if not freeze_path.is_file():
        print(
            json.dumps(
                {
                    "execution": "blocked",
                    "reason": "manual labels are not frozen",
                    "required_freeze_manifest": normalize_path(freeze_path, root),
                },
                indent=2,
            )
        )
        return 2
    report = run_final_evaluation(
        output=Path(args.output).resolve() if args.output else None,
        root=root,
    )
    print(json.dumps(report, indent=2))
    return 0 if report.get("inference_or_evaluation_run") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
