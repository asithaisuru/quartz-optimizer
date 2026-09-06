"""Shared checks for the frozen independent defect final-test subset.

This module intentionally does not load the model or run inference. It verifies
frozen membership, the selected model hash, annotation completeness, and split
isolation evidence used by the current defect-research workflow.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from research.common import (
    atomic_write_json,
    normalize_path,
    sha256_file,
    utc_timestamp,
)
from research.readiness_common import (
    atomic_write_csv,
    atomic_write_text,
    stable_json_sha256,
)

EXPECTED_MODEL_SHA256 = (
    "5b7a362ed4c6670395e4693ab1a68d415ff1fb2164b2471cef37913338130b98"
)
EXPECTED_MEMBERSHIP_SHA256 = (
    "4633d2e1fd7ce7772d06d8994c084e52f17b8cce1647587693d1558bc050e8cf"
)
EXPECTED_LABEL_MANIFEST_SHA256 = (
    "40d3d1cc061924c25192abaddffd6297353dc7afd0bd7e18f8de798436231e07"
)
EXPECTED_FINAL_RESULT_SHA256 = (
    "2756325a56bf1c47d286aab67b2f998145e74f56318f5d5b1966c2bd430bfc2e"
)
EXPECTED_SPECIMEN_IDS = (
    "Gem_5",
    "Gem_6",
    "Gem_7",
    "Gem_8",
    "Gem_12",
    "Gem_16",
    "Gem_20",
    "Gem_23",
    "Gem_24",
)
EXPECTED_IMAGE_COUNT = 45
EXPECTED_GROUP_COUNT = 9
CLASS_NAMES = ("fracture", "inclusion")
CLASS_DEFINITIONS = {
    "fracture": "Visible cracks or fracture lines in the quartz gemstone.",
    "inclusion": (
        "Visible internal inclusion, foreign material, cloud, cavity, "
        "or irregularity within the quartz gemstone."
    ),
}
BINARY_CONFIDENCE_THRESHOLD = 0.25
INFERENCE_IMAGE_SIZE = 960
NMS_IOU_THRESHOLD = 0.7
NATIVE_VALIDATION_IOU = 0.7
PROPOSAL_TARGET_F1 = 0.90
EXPECTED_FINAL_METRICS = {
    "precision": 0.03013444598980065,
    "recall": 0.0009601163956491956,
    "f1": 0.0018609411709972319,
    "iou": 0.0009313371673380055,
}
EXPECTED_NATIVE_MASK_MAP50 = 0.00043102434676857947

LINEAGE_RE = re.compile(
    r"Gem_(?P<gem_id>\d+)_Vid_(?P<video_id>\d+)_Fr_(?P<frame_id>\d+)",
    flags=re.IGNORECASE,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def defect_root(root: Path | None = None) -> Path:
    return (root or repo_root()) / "final_research_evidence" / "defect_detection"


def final_test_root(root: Path | None = None) -> Path:
    return defect_root(root) / "final_test"


def annotation_workspace(root: Path | None = None) -> Path:
    return final_test_root(root) / "annotation_workspace"


def manifest_path(root: Path | None = None) -> Path:
    return final_test_root(root) / "FINAL_TEST_SUBSET_MANIFEST.json"


def selected_model_record_path(root: Path | None = None) -> Path:
    return defect_root(root) / "final_model" / "selected_final_clean_model.json"


def default_package_path(root: Path | None = None) -> Path:
    return final_test_root(root) / "blind_annotation_package"


def label_freeze_manifest_path(root: Path | None = None) -> Path:
    return final_test_root(root) / "LABEL_FREEZE_MANIFEST.json"


def completed_final_result_path(root: Path | None = None) -> Path:
    return final_test_root(root) / "independent_final_test_results" / "independent_final_test_results.json"


def derived_dataset_root() -> Path:
    return Path("D:/project-quartz-derived/gem-fracture-yolo-v1")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def selected_images(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("selected_images")
    if not isinstance(rows, list):
        raise ValueError("Frozen manifest does not contain selected_images.")
    return rows


def membership_sha256(manifest: dict[str, Any]) -> str:
    return stable_json_sha256(selected_images(manifest))


def image_id(row: dict[str, Any]) -> str:
    return str(row.get("source_key") or Path(str(row["workspace_image_path"])).stem)


def expected_label_path(row: dict[str, Any], root: Path | None = None) -> Path:
    workspace = annotation_workspace(root)
    specimen_id = str(row["specimen_id"])
    stem = Path(str(row["workspace_image_path"])).stem
    return workspace / "labels" / specimen_id / f"{stem}.txt"


def _source_key(value: str | Path) -> str:
    match = LINEAGE_RE.search(Path(value).stem)
    if not match:
        return ""
    return (
        f"Gem_{int(match.group('gem_id'))}_"
        f"Vid_{int(match.group('video_id'))}_"
        f"Fr_{int(match.group('frame_id'))}"
    )


def _gem_id(value: str) -> str:
    match = re.fullmatch(r"Gem_(\d+)", value, flags=re.IGNORECASE)
    return str(int(match.group(1))) if match else ""


def _existing_paths(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def validate_frozen_manifest(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    path = manifest_path(root)
    manifest = read_json(path)
    rows = selected_images(manifest)
    groups = tuple(str(value) for value in manifest.get("specimen_ids") or ())
    group_counts = Counter(str(row.get("specimen_id", "")) for row in rows)
    recalculated = membership_sha256(manifest)

    missing_images = []
    hash_mismatches = []
    for row in rows:
        workspace_path = root / str(row["workspace_image_path"])
        if not workspace_path.is_file():
            missing_images.append(normalize_path(workspace_path, root))
            continue
        expected_hash = str(row.get("workspace_sha256", "")).casefold()
        actual_hash = sha256_file(workspace_path)
        if actual_hash != expected_hash:
            hash_mismatches.append(
                {
                    "path": normalize_path(workspace_path, root),
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                }
            )

    issues = []
    if manifest.get("manifest_type") != "frozen_independent_final_test_subset":
        issues.append("manifest_type mismatch")
    if manifest.get("membership_frozen") is not True:
        issues.append("membership_frozen is not true")
    if manifest.get("membership_sha256") != EXPECTED_MEMBERSHIP_SHA256:
        issues.append("membership SHA-256 does not match expected frozen value")
    if recalculated != manifest.get("membership_sha256"):
        issues.append("membership SHA-256 does not reproduce from selected_images")
    if len(rows) != EXPECTED_IMAGE_COUNT:
        issues.append("image count mismatch")
    if groups != EXPECTED_SPECIMEN_IDS:
        issues.append("specimen IDs do not match expected frozen groups")
    if len(group_counts) != EXPECTED_GROUP_COUNT:
        issues.append("specimen group count mismatch")
    if any(count != 5 for count in group_counts.values()):
        issues.append("expected five images per specimen")
    if missing_images:
        issues.append("one or more frozen workspace images are missing")
    if hash_mismatches:
        issues.append("one or more frozen workspace image hashes changed")

    return {
        "status": "pass" if not issues else "needs_correction",
        "manifest_path": normalize_path(path, root),
        "manifest_type": manifest.get("manifest_type"),
        "membership_frozen": manifest.get("membership_frozen"),
        "membership_sha256": manifest.get("membership_sha256"),
        "membership_sha256_recalculated": recalculated,
        "image_count": len(rows),
        "specimen_count": len(group_counts),
        "specimen_ids": list(groups),
        "images_per_specimen": dict(sorted(group_counts.items())),
        "missing_image_count": len(missing_images),
        "hash_mismatch_count": len(hash_mismatches),
        "missing_images": missing_images,
        "hash_mismatches": hash_mismatches,
        "issues": issues,
    }


def verify_selected_model(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    record = read_json(selected_model_record_path(root))
    model_path = root / str(record["path"])
    copy_path = root / str(record.get("selected_model_copy", ""))
    actual_hash = sha256_file(model_path) if model_path.is_file() else None
    copy_hash = sha256_file(copy_path) if copy_path.is_file() else None
    issues = []
    if record.get("sha256") != EXPECTED_MODEL_SHA256:
        issues.append("selected model record hash does not match expected hash")
    if actual_hash != EXPECTED_MODEL_SHA256:
        issues.append("selected model file hash does not match expected hash")
    if copy_hash and copy_hash != EXPECTED_MODEL_SHA256:
        issues.append("selected model copy hash does not match expected hash")
    if record.get("independent_final_test_used_for_selection") is not False:
        issues.append("model selection record used independent final-test data")

    return {
        "status": "pass" if not issues else "needs_correction",
        "selected": record.get("selected"),
        "model_type": "Ultralytics YOLOv8n-seg segmentation",
        "model_path": normalize_path(model_path, root),
        "recorded_sha256": record.get("sha256"),
        "verified_sha256": actual_hash,
        "selected_model_copy": (
            normalize_path(copy_path, root) if str(record.get("selected_model_copy", "")) else None
        ),
        "selected_model_copy_sha256": copy_hash,
        "independent_final_test_used_for_selection": record.get(
            "independent_final_test_used_for_selection"
        ),
        "inference_image_size": INFERENCE_IMAGE_SIZE,
        "binary_confidence_threshold": BINARY_CONFIDENCE_THRESHOLD,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "issues": issues,
    }


def verify_split_leakage(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    manifest = read_json(manifest_path(root))
    final_rows = selected_images(manifest)
    final_groups = {str(row["specimen_id"]) for row in final_rows}
    final_gems = {_gem_id(group) for group in final_groups}
    final_hashes = {str(row.get("workspace_sha256", "")).casefold() for row in final_rows}
    final_source_keys = {str(row.get("source_key") or _source_key(row["workspace_image_path"])) for row in final_rows}
    final_source_keys.discard("")

    derived_root = derived_dataset_root()
    materialization = read_json(derived_root / "materialization_manifest.json")
    dev_records = list(materialization.get("records") or [])
    grouped_manifest = read_csv(derived_root / "manifests" / "grouped_development_manifest.csv")
    dev_hashes = {str(row.get("image_sha256", "")).casefold() for row in dev_records}
    dev_groups = {
        str(row.get("specimen_id", ""))
        for row in grouped_manifest
        if row.get("assigned_split") in {"train", "valid"}
    }
    dev_gems = {_gem_id(group) for group in dev_groups}
    dev_source_keys = set()
    for row in grouped_manifest:
        for key in (
            "source_image_path",
            "relative_image_path",
            "image_path",
            "derived_image_path",
        ):
            value = row.get(key)
            if value:
                source_key = _source_key(value)
                if source_key:
                    dev_source_keys.add(source_key)

    candidate_rows = read_csv(defect_root(root) / "FINAL_TEST_CANDIDATES.csv")
    candidate_flags = [
        row for row in candidate_rows
        if row.get("candidate_specimen_id") in final_groups
    ]
    flagged_candidate_rows = [
        row for row in candidate_flags
        if row.get("train_overlap") != "false"
        or row.get("validation_overlap") != "false"
        or row.get("derivative_overlap") != "false"
        or row.get("independence_verified") != "partial_file_level_only"
    ]

    perceptual_rows = read_csv(
        derived_root / "manifests" / "perceptual_duplicate_candidates.csv"
    )
    perceptual_cross = [
        row for row in perceptual_rows
        if (
            row.get("left_gem_id") in final_gems
            and row.get("right_gem_id") in dev_gems
        )
        or (
            row.get("left_gem_id") in dev_gems
            and row.get("right_gem_id") in final_gems
        )
    ]

    exact_overlap = sorted(final_hashes & dev_hashes)
    source_frame_overlap = sorted(final_source_keys & dev_source_keys)
    group_overlap = sorted(final_groups & dev_groups)
    derivative_overlap = sorted(source_frame_overlap)
    issues = []
    if exact_overlap:
        issues.append("exact file hash overlap with development split")
    if group_overlap:
        issues.append("source specimen/group overlap with development split")
    if source_frame_overlap:
        issues.append("source frame/augmentation lineage overlap with development split")
    if derivative_overlap:
        issues.append("derivative overlap with development split")
    if perceptual_cross:
        issues.append("perceptual/near-duplicate candidates cross final/development")
    if flagged_candidate_rows:
        issues.append("FINAL_TEST_CANDIDATES flags do not support selected groups")
    if materialization.get("split_counts", {}).get("test") not in (None, 0):
        issues.append("development materialization contains a test split")

    return {
        "status": "pass" if not issues else "needs_correction",
        "development_materialization": normalize_path(
            derived_root / "materialization_manifest.json"
        ),
        "development_grouped_manifest": normalize_path(
            derived_root / "manifests" / "grouped_development_manifest.csv"
        ),
        "development_split_counts": materialization.get("split_counts"),
        "exact_file_overlap_count": len(exact_overlap),
        "source_specimen_group_overlap_count": len(group_overlap),
        "source_frame_overlap_count": len(source_frame_overlap),
        "augmentation_lineage_overlap_count": len(source_frame_overlap),
        "derivative_overlap_count": len(derivative_overlap),
        "perceptual_near_duplicate_overlap_count": len(perceptual_cross),
        "perceptual_candidate_total_checked": len(perceptual_rows),
        "candidate_manifest_rows_checked": len(candidate_flags),
        "candidate_manifest_flagged_rows": len(flagged_candidate_rows),
        "candidate_manifest_independence": {
            row["candidate_specimen_id"]: {
                "independence_verified": row.get("independence_verified"),
                "train_overlap": row.get("train_overlap"),
                "validation_overlap": row.get("validation_overlap"),
                "derivative_overlap": row.get("derivative_overlap"),
            }
            for row in sorted(candidate_flags, key=lambda item: item["candidate_specimen_id"])
        },
        "issues": issues,
    }


def manual_annotation_status(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    manifest = read_json(manifest_path(root))
    rows = selected_images(manifest)
    expected = [expected_label_path(row, root) for row in rows]
    existing = [path for path in expected if path.is_file()]
    missing = [path for path in expected if not path.is_file()]
    freeze = verify_label_freeze_manifest(root)

    return {
        "status": (
            "complete_and_frozen"
            if len(existing) == len(expected) and freeze["status"] == "pass"
            else "pending_manual_ground_truth"
        ),
        "expected_label_count": len(expected),
        "existing_label_count": len(existing),
        "missing_label_count": len(missing),
        "empty_label_count": sum(1 for path in existing if path.stat().st_size == 0),
        "nonempty_label_count": sum(1 for path in existing if path.stat().st_size > 0),
        "labels_directory": normalize_path(annotation_workspace(root) / "labels", root),
        "label_freeze_manifest": freeze["path"],
        "label_freeze_manifest_exists": freeze["exists"],
        "label_freeze_manifest_valid": freeze["status"] == "pass",
        "missing_labels": [normalize_path(path, root) for path in missing],
    }


def annotation_manifest_rows(root: Path | None = None) -> list[dict[str, Any]]:
    root = root or repo_root()
    manifest = read_json(manifest_path(root))
    rows = []
    for row in selected_images(manifest):
        workspace_path = root / str(row["workspace_image_path"])
        label_path = expected_label_path(row, root)
        rows.append(
            {
                "image_id": image_id(row),
                "specimen_id": row["specimen_id"],
                "image_filename": workspace_path.name,
                "workspace_image_path": normalize_path(workspace_path, root),
                "workspace_sha256": row.get("workspace_sha256"),
                "source_image_path": row.get("source_image_path"),
                "source_sha256": row.get("source_sha256"),
                "video_id": row.get("video_id"),
                "frame_id": row.get("frame_id"),
                "annotation_status": "pending_manual_ground_truth",
                "expected_label_path": normalize_path(label_path, root),
                "label_format": "YOLO segmentation normalized polygon txt",
            }
        )
    return rows


def write_blind_annotation_package(
    output: Path | None = None,
    root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    root = root or repo_root()
    output = output or default_package_path(root)
    output.mkdir(parents=True, exist_ok=True)

    frozen = validate_frozen_manifest(root)
    leakage = verify_split_leakage(root)
    model = verify_selected_model(root)
    annotations = manual_annotation_status(root)
    rows = annotation_manifest_rows(root)

    immutable_manifest = output / "IMMUTABLE_TEST_IMAGES_MANIFEST.csv"
    status_path = output / "ANNOTATION_STATUS.csv"
    expected_labels = output / "EXPECTED_LABEL_OUTPUTS.csv"
    taxonomy = output / "CLASS_TAXONOMY.csv"
    protocol = output / "ANNOTATION_PROTOCOL.md"
    leakage_path = output / "SPLIT_LEAKAGE_VERIFICATION.json"
    package_manifest = output / "PACKAGE_MANIFEST.json"

    atomic_write_csv(
        immutable_manifest,
        rows,
        (
            "image_id",
            "specimen_id",
            "image_filename",
            "workspace_image_path",
            "workspace_sha256",
            "source_image_path",
            "source_sha256",
            "video_id",
            "frame_id",
            "annotation_status",
            "expected_label_path",
            "label_format",
        ),
    )
    atomic_write_csv(
        status_path,
        [
            {
                "image_id": row["image_id"],
                "specimen_id": row["specimen_id"],
                "image_filename": row["image_filename"],
                "annotation_status": "pending_manual_ground_truth",
                "annotator_initials": "",
                "manual_review_date": "",
                "label_file_status": "missing",
                "notes": "",
            }
            for row in rows
        ],
        (
            "image_id",
            "specimen_id",
            "image_filename",
            "annotation_status",
            "annotator_initials",
            "manual_review_date",
            "label_file_status",
            "notes",
        ),
    )
    atomic_write_csv(
        expected_labels,
        [
            {
                "image_id": row["image_id"],
                "specimen_id": row["specimen_id"],
                "image_filename": row["image_filename"],
                "required_output_label_path": row["expected_label_path"],
                "required_format": "class_id x1 y1 x2 y2 x3 y3 ...",
                "coordinate_rule": "normalized [0,1] image coordinates",
                "negative_image_rule": "create an empty txt file after manual inspection",
            }
            for row in rows
        ],
        (
            "image_id",
            "specimen_id",
            "image_filename",
            "required_output_label_path",
            "required_format",
            "coordinate_rule",
            "negative_image_rule",
        ),
    )
    atomic_write_csv(
        taxonomy,
        [
            {
                "class_id": index,
                "class_name": name,
                "is_visible_defect": "true",
                "semantic_definition": CLASS_DEFINITIONS[name],
                "output_format": "YOLO segmentation polygon",
            }
            for index, name in enumerate(CLASS_NAMES)
        ],
        (
            "class_id",
            "class_name",
            "is_visible_defect",
            "semantic_definition",
            "output_format",
        ),
    )
    atomic_write_text(protocol, annotation_protocol_text(root))
    atomic_write_json(leakage_path, leakage)

    report = {
        "schema_version": "1.0",
        "package_type": "blind_independent_final_test_annotation_package",
        "created_utc": utc_timestamp(),
        "no_model_predictions_included": True,
        "final_test_manifest": frozen,
        "split_leakage_verification": leakage,
        "selected_model_verification": model,
        "manual_annotation_status": annotations,
        "required_taxonomy": {
            "class_names": list(CLASS_NAMES),
            "class_definitions": CLASS_DEFINITIONS,
        },
        "frozen_inference_metric_configuration": frozen_metric_configuration(),
        "outputs": {
            "immutable_manifest": normalize_path(immutable_manifest, root),
            "annotation_status": normalize_path(status_path, root),
            "expected_labels": normalize_path(expected_labels, root),
            "taxonomy": normalize_path(taxonomy, root),
            "protocol": normalize_path(protocol, root),
            "split_leakage_verification": normalize_path(leakage_path, root),
        },
    }
    atomic_write_json(package_manifest, report)
    files = {
        "immutable_manifest": immutable_manifest,
        "annotation_status": status_path,
        "expected_labels": expected_labels,
        "taxonomy": taxonomy,
        "protocol": protocol,
        "split_leakage_verification": leakage_path,
        "package_manifest": package_manifest,
    }
    return report, files


def frozen_metric_configuration() -> dict[str, Any]:
    return {
        "model_type": "Ultralytics YOLOv8n-seg segmentation",
        "model_sha256": EXPECTED_MODEL_SHA256,
        "binary_visible_defect_metric": "pixel-level union of fracture and inclusion masks",
        "primary_metric": "binary_visible_defect_f1",
        "proposal_target_f1": PROPOSAL_TARGET_F1,
        "inference_image_size": INFERENCE_IMAGE_SIZE,
        "binary_confidence_threshold": BINARY_CONFIDENCE_THRESHOLD,
        "prediction_nms_iou_threshold": NMS_IOU_THRESHOLD,
        "native_ultralytics_validation_iou": NATIVE_VALIDATION_IOU,
        "native_ultralytics_confidence": "Ultralytics validation default",
        "threshold_source": "fixed on clean development validation evidence only",
        "final_test_threshold_tuning_allowed": False,
    }


def final_result_freeze_metadata() -> dict[str, Any]:
    return {
        "validation_role": "independent_test",
        "labels_frozen_before_inference_required": True,
        "no_final_test_threshold_tuning": True,
        "evaluation_device": "cpu",
        "cpu_device_reason": (
            "Current PyTorch CUDA build does not support RTX 5050 sm_120; "
            "CPU execution preserves the frozen inference configuration."
        ),
        "post_evaluation_integrity_audit": "PASS",
        "evaluator_defect_found": False,
        "proposal_target": PROPOSAL_TARGET_F1,
        "proposal_target_status": "NOT ACHIEVED",
        "original_final_evaluation_output_preserved": True,
    }


def validate_label_file(path: Path) -> list[str]:
    issues = []
    if not path.is_file():
        return ["missing_label_file"]
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < 7:
            issues.append(f"line_{line_number}_has_too_few_values")
            continue
        if len(tokens[1:]) % 2:
            issues.append(f"line_{line_number}_has_odd_coordinate_count")
            continue
        try:
            class_id = int(float(tokens[0]))
            coords = [float(value) for value in tokens[1:]]
        except ValueError:
            issues.append(f"line_{line_number}_has_non_numeric_value")
            continue
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            issues.append(f"line_{line_number}_has_invalid_class_id")
        if any(value < 0.0 or value > 1.0 for value in coords):
            issues.append(f"line_{line_number}_coordinate_outside_unit_range")
    return issues


def annotation_progress_report(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    manifest = read_json(manifest_path(root))
    rows = selected_images(manifest)
    completed = []
    missing = []
    malformed = []
    for row in rows:
        path = expected_label_path(row, root)
        record = {
            "image_id": image_id(row),
            "specimen_id": row["specimen_id"],
            "image_filename": Path(str(row["workspace_image_path"])).name,
            "label_path": normalize_path(path, root),
        }
        issues = validate_label_file(path)
        if not path.is_file():
            missing.append(record)
        elif issues:
            malformed.append({**record, "issues": issues})
        else:
            completed.append(
                {
                    **record,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "empty_label": path.stat().st_size == 0,
                }
            )
    ready = len(completed) == len(rows) and not missing and not malformed
    return {
        "status": "ready" if ready else "pending",
        "image_count": len(rows),
        "completed_labels": len(completed),
        "missing_labels": len(missing),
        "malformed_labels": len(malformed),
        "ready_45_of_45": "YES" if ready else "NO",
        "membership_sha256": manifest.get("membership_sha256"),
        "completed": completed,
        "missing": missing,
        "malformed": malformed,
        "no_model_predictions_used": True,
        "labels_not_frozen": not label_freeze_manifest_path(root).is_file(),
    }


def label_file_records(root: Path | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    root = root or repo_root()
    manifest = read_json(manifest_path(root))
    records = []
    issues = []
    for row in selected_images(manifest):
        path = expected_label_path(row, root)
        label_issues = validate_label_file(path)
        if label_issues:
            issues.extend(
                f"{normalize_path(path, root)}:{issue}" for issue in label_issues
            )
        records.append(
            {
                "image_id": image_id(row),
                "specimen_id": row["specimen_id"],
                "label_path": normalize_path(path, root),
                "sha256": sha256_file(path) if path.is_file() else None,
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "status": "present_valid" if not label_issues else "invalid_or_missing",
            }
        )
    return records, issues


def label_manifest_sha256(records: list[dict[str, Any]]) -> str:
    return stable_json_sha256(records)


def verify_label_freeze_manifest(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    path = label_freeze_manifest_path(root)
    if not path.is_file():
        return {
            "status": "missing",
            "exists": False,
            "path": normalize_path(path, root),
            "issues": ["label freeze manifest is missing"],
        }
    try:
        freeze = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "needs_correction",
            "exists": True,
            "path": normalize_path(path, root),
            "issues": [f"label freeze manifest is unreadable: {exc}"],
        }
    manifest = read_json(manifest_path(root))
    records, label_issues = label_file_records(root)
    actual_hash = label_manifest_sha256(records)
    issues = list(label_issues)
    if freeze.get("ground_truth_frozen") is not True:
        issues.append("ground_truth_frozen is not true")
    if freeze.get("membership_sha256") != manifest.get("membership_sha256"):
        issues.append("membership_sha256 does not match frozen test manifest")
    if freeze.get("label_file_count") != len(records):
        issues.append("label_file_count mismatch")
    if freeze.get("label_manifest_sha256") != actual_hash:
        issues.append("label_manifest_sha256 mismatch")
    return {
        "status": "pass" if not issues else "needs_correction",
        "exists": True,
        "path": normalize_path(path, root),
        "label_file_count": len(records),
        "label_manifest_sha256": actual_hash,
        "issues": issues,
    }


def write_label_freeze_manifest(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    manifest = read_json(manifest_path(root))
    records, issues = label_file_records(root)
    if issues:
        raise ValueError("Cannot freeze incomplete or invalid labels.")
    payload = {
        "schema_version": "1.0",
        "manifest_type": "independent_final_test_label_freeze",
        "created_utc": utc_timestamp(),
        "ground_truth_frozen": True,
        "membership_sha256": manifest["membership_sha256"],
        "label_file_count": len(records),
        "label_manifest_sha256": label_manifest_sha256(records),
        "class_names": list(CLASS_NAMES),
        "label_files": records,
        "no_model_predictions_used": True,
        "final_test_evaluation_authorized_after_freeze": True,
    }
    atomic_write_json(label_freeze_manifest_path(root), payload)
    return payload


def annotation_protocol_text(root: Path | None = None) -> str:
    root = root or repo_root()
    manifest = read_json(manifest_path(root))
    return "\n".join(
        [
            "# Blind Independent Final-Test Annotation Protocol",
            "",
            "## Scope",
            "",
            "Annotate only the immutable 45-image final-test subset listed in "
            "`IMMUTABLE_TEST_IMAGES_MANIFEST.csv`. Do not add images, remove images, "
            "rename images, or use replacement images.",
            "",
            "## Blindness Rule",
            "",
            "Annotators must not view model predictions, confidence scores, training "
            "curves, validation errors, or any generated masks while creating these "
            "ground-truth labels. Manual labels must be independent of model output.",
            "",
            "## Taxonomy",
            "",
            "- Class 0 `fracture`: visible cracks or fracture lines in the quartz gemstone.",
            "- Class 1 `inclusion`: visible internal inclusion, foreign material, cloud, "
            "cavity, or irregularity within the quartz gemstone.",
            "",
            "## Required Output",
            "",
            "Write one YOLO segmentation text file for every image at the path listed "
            "in `EXPECTED_LABEL_OUTPUTS.csv`:",
            "",
            "`class_id x1 y1 x2 y2 x3 y3 ...`",
            "",
            "Coordinates must be normalized to `[0, 1]`. Each polygon must have at "
            "least three points. If an image has no visible defect, create an empty "
            "`.txt` file only after manual inspection.",
            "",
            "## Freeze Rule",
            "",
            "After all 45 label files exist, freeze the labels by recording their "
            "SHA-256 hashes in `final_research_evidence/defect_detection/final_test/"
            "LABEL_FREEZE_MANIFEST.json` with `ground_truth_frozen: true` and "
            f"`membership_sha256: {manifest['membership_sha256']}`. Run the final "
            "evaluator exactly once after that freeze.",
            "",
        ]
    )
