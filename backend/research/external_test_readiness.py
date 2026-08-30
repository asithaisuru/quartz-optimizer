"""External-test operations readiness without training or evaluation."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import atomic_write_json, sha256_file, utc_timestamp
from .readiness_common import (
    atomic_write_csv,
    atomic_write_text,
    load_json_config,
    resolve_path,
    write_manifest,
)

APPROVED_SPECIMEN_IDS = tuple(
    f"EXT-GEM-{index:03d}" for index in range(1, 6)
)
SPECIMEN_ID_PATTERN = re.compile(r"^EXT-GEM-\d{3}$")
ROLE_VALUES = {
    "Research Team",
    "Dataset Annotator",
    "Co-Supervisor / Gem Expert",
}
PERSONAL_FIELD_MARKERS = {
    "name",
    "email",
    "email_address",
    "phone",
    "phone_number",
    "signature",
    "contact",
    "contact_details",
}
EMAIL_PATTERN = re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b")
PERSONAL_TITLE_PATTERN = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Eng)\.\s",
    re.IGNORECASE,
)
FALSE_VALUES = {"false", "0", "no"}
TRUE_VALUES = {"true", "1", "yes"}
COMPLETE_CAPTURE_STATUSES = {"complete", "completed", "accepted"}
COMPLETE_ANNOTATION_STATUSES = {"complete", "completed", "accepted", "approved"}
COMPLETE_REVIEW_STATUSES = {"approved", "corrected"}
PLACEHOLDER_MARKERS = {"placeholder", "dummy", "sample", "example", "todo"}

SPECIMEN_FIELDS = (
    "specimen_id",
    "specimen_status",
    "physically_new_confirmed",
    "development_exclusion_confirmed",
    "specimen_source",
    "received_date",
    "material",
    "notes",
)
CAPTURE_FIELDS = (
    "capture_session_id",
    "specimen_id",
    "capture_date",
    "capture_operator_role",
    "camera_id",
    "capture_mode",
    "image_target",
    "actual_image_count",
    "angular_increment_degrees",
    "lighting",
    "polarization",
    "background",
    "scale_target",
    "focus_locked",
    "exposure_locked",
    "white_balance_locked",
    "capture_status",
    "notes",
)
SAMPLE_FIELDS = (
    "sample_id",
    "specimen_id",
    "capture_session_id",
    "source_image_path",
    "original_image_sha256",
    "view_index",
    "nominal_angle_degrees",
    "image_quality_status",
    "blur_score",
    "duplicate_status",
    "annotation_path",
    "annotation_sha256",
    "test_membership_status",
    "used_for_training",
    "used_for_validation",
    "used_for_early_stopping",
    "used_for_threshold_selection",
    "used_for_architecture_selection",
    "used_for_augmentation_decisions",
    "used_for_model_selection",
    "notes",
)
ANNOTATION_FIELDS = (
    "sample_id",
    "specimen_id",
    "annotator_role",
    "annotation_date",
    "fracture_present",
    "inclusion_present",
    "negative_image",
    "annotation_status",
    "correction_path",
    "notes",
)
EXPERT_REVIEW_FIELDS = (
    "sample_id",
    "specimen_id",
    "reviewer_role",
    "review_date",
    "review_status",
    "mask_correct",
    "class_correct",
    "corrected_annotation_path",
    "review_confidence",
    "disagreement_reason",
    "notes",
)
ISOLATION_FIELDS = (
    "used_for_training",
    "used_for_validation",
    "used_for_early_stopping",
    "used_for_threshold_selection",
    "used_for_architecture_selection",
    "used_for_augmentation_decisions",
    "used_for_model_selection",
)


def load_external_test_readiness_config(
    path: str | Path,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    return load_json_config(path, output_override=output_override)


def _read_json(path: Path | None) -> tuple[dict[str, Any], list[str]]:
    if path is None or not path.is_file():
        return {}, ["File is unavailable."]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"Invalid JSON: {exc}"]
    if not isinstance(value, dict):
        return {}, ["Expected a JSON object."]
    return value, []


def _read_csv(
    path: Path | None,
    required_fields: tuple[str, ...],
) -> tuple[list[dict[str, str]], list[str]]:
    if path is None or not path.is_file():
        return [], ["Registry is unavailable."]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = [
            {key: str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    missing = sorted(set(required_fields) - set(fields))
    issues = [f"Missing registry field: {field}" for field in missing]
    return rows, issues


def _bool(value: Any) -> bool | None:
    normalized = str(value or "").strip().casefold()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def _int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _privacy_issues(value: Any, location: str = "$") -> list[str]:
    issues = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            child = f"{location}.{key}"
            if normalized in PERSONAL_FIELD_MARKERS or normalized.endswith(
                ("_email", "_phone", "_signature", "_contact")
            ):
                issues.append(f"Personal-data field is prohibited: {child}")
            issues.extend(_privacy_issues(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(_privacy_issues(item, f"{location}[{index}]"))
    elif isinstance(value, str):
        if EMAIL_PATTERN.search(value):
            issues.append(f"Email address is prohibited: {location}")
        if PERSONAL_TITLE_PATTERN.search(value):
            issues.append(f"Personal name marker is prohibited: {location}")
    return issues


def _check(
    status: str,
    reason: str,
    *,
    blockers: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "passed": status == "pass",
        "reason": reason,
        "blockers": list(dict.fromkeys(blockers or [])),
        "evidence": evidence or {},
    }


def _resolve_config_path(config: dict[str, Any], value: Any) -> Path | None:
    return resolve_path(value, config["_repo_root"])


def _validate_operations_approval(
    approval_path: Path | None,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    approval, blockers = _read_json(approval_path)
    expected = {
        "approval_reference_date": "2026-07-29",
        "approved_by": "Research Team",
        "approver_role": "Research Team",
        "external_holdout_scope": "small_sample_external_holdout",
        "specimen_count": 5,
        "capture_operator_role": "Research Team",
        "initial_annotation_creator_role": "Dataset Annotator",
        "expert_review_role": "Co-Supervisor / Gem Expert",
        "expected_capture_date": "2026-08-12",
        "expected_expert_review_completion_date": "2026-08-19",
        "source_development_exclusion_required": True,
        "operational_status": "planned",
        "data_availability_status": "unavailable",
        "final_test_frozen": False,
        "training_authorized": False,
        "final_evaluation_ready": False,
        "example_only": False,
    }
    for key, value in expected.items():
        if approval.get(key) != value:
            blockers.append(f"Operations approval {key} does not match.")
    ids = approval.get("specimen_ids")
    if not isinstance(ids, list) or tuple(ids) != APPROVED_SPECIMEN_IDS:
        blockers.append("Operations approval must reserve exactly five approved IDs.")
    roles = {
        approval.get("approved_by"),
        approval.get("approver_role"),
        approval.get("capture_operator_role"),
        approval.get("initial_annotation_creator_role"),
        approval.get("expert_review_role"),
    }
    if not roles.issubset(ROLE_VALUES):
        blockers.append("Operations approval contains a non-role identity.")
    blockers.extend(_privacy_issues(approval))
    evidence_links = approval.get("evidence_links")
    if not isinstance(evidence_links, dict) or len(evidence_links) < 9:
        blockers.append("Required hash-linked evidence is incomplete.")
    else:
        for name, link in evidence_links.items():
            if not isinstance(link, dict):
                blockers.append(f"Evidence link {name} is invalid.")
                continue
            evidence_path = resolve_path(link.get("path"), repo_root)
            if evidence_path is None or not evidence_path.is_file():
                blockers.append(f"Evidence link {name} is unavailable.")
                continue
            if link.get("sha256") != sha256_file(evidence_path):
                blockers.append(f"Evidence hash mismatch: {name}.")
    blockers = list(dict.fromkeys(blockers))
    status = "pass" if approval and not blockers else "blocked"
    return approval, _check(
        status,
        (
            "Role-only external operations plan is approved and hash-linked."
            if status == "pass"
            else "External operations approval is incomplete or inconsistent."
        ),
        blockers=blockers,
        evidence={
            "path": str(approval_path) if approval_path else None,
            "specimen_count": approval.get("specimen_count"),
            "scope": approval.get("external_holdout_scope"),
        },
    )


def _validate_specimens(
    rows: list[dict[str, str]],
    registry_issues: list[str],
    approval: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    blockers = list(registry_issues)
    ids = [row.get("specimen_id", "") for row in rows]
    duplicates = sorted(
        specimen_id
        for specimen_id, count in Counter(ids).items()
        if specimen_id and count > 1
    )
    invalid = sorted(
        specimen_id
        for specimen_id in ids
        if not SPECIMEN_ID_PATTERN.fullmatch(specimen_id)
    )
    approved_ids = tuple(approval.get("specimen_ids") or ())
    if duplicates:
        blockers.append(f"Duplicate specimen IDs: {', '.join(duplicates)}")
    if invalid:
        blockers.append(f"Invalid specimen IDs: {', '.join(invalid)}")
    if tuple(ids) != approved_ids or tuple(ids) != APPROVED_SPECIMEN_IDS:
        blockers.append(
            "Registry must contain exactly the five approved specimen IDs in order."
        )
    non_planned = [
        row.get("specimen_id", "")
        for row in rows
        if row.get("specimen_status") not in {"planned", "received", "available"}
    ]
    if non_planned:
        blockers.append("Registry contains unsupported specimen status values.")
    planned = _check(
        "pass" if not blockers else "blocked",
        (
            "Exactly five approved specimen IDs are reserved."
            if not blockers
            else "Specimen reservation does not match the approved scope."
        ),
        blockers=blockers,
        evidence={"specimen_ids": ids, "planned_count": len(rows)},
    )

    explicit_unavailable = [
        row["specimen_id"]
        for row in rows
        if _bool(row.get("physically_new_confirmed")) is False
        or row.get("specimen_status") == "rejected"
    ]
    physically_available_ids = [
        row["specimen_id"]
        for row in rows
        if row.get("specimen_status") in {"received", "available"}
        and _bool(row.get("physically_new_confirmed")) is True
    ]
    if explicit_unavailable:
        physical_status = "blocked"
        physical_reason = "One or more specimens failed physical-new confirmation."
    elif len(physically_available_ids) == len(APPROVED_SPECIMEN_IDS):
        physical_status = "pass"
        physical_reason = "All five physical specimens are available and confirmed new."
    else:
        physical_status = "unavailable"
        physical_reason = "Planned rows do not prove physical specimen availability."
    physical = _check(
        physical_status,
        physical_reason,
        blockers=[
            f"Physical-new confirmation failed: {item}"
            for item in explicit_unavailable
        ],
        evidence={
            "confirmed_available_ids": physically_available_ids,
            "planned_rows_are_receipt_evidence": False,
        },
    )

    failed_independence = [
        row["specimen_id"]
        for row in rows
        if _bool(row.get("development_exclusion_confirmed")) is False
    ]
    independent_ids = [
        row["specimen_id"]
        for row in rows
        if _bool(row.get("physically_new_confirmed")) is True
        and _bool(row.get("development_exclusion_confirmed")) is True
    ]
    if failed_independence:
        independence_status = "blocked"
        independence_reason = "Development exclusion failed for a specimen."
    elif len(independent_ids) == len(APPROVED_SPECIMEN_IDS):
        independence_status = "pass"
        independence_reason = "All specimens are confirmed independent of development."
    else:
        independence_status = "unavailable"
        independence_reason = "Development exclusion is not yet physically confirmed."
    independence = _check(
        independence_status,
        independence_reason,
        blockers=[
            f"Development exclusion failed: {item}"
            for item in failed_independence
        ],
        evidence={"independent_specimen_ids": independent_ids},
    )
    return {
        "specimen_count_planned": planned,
        "specimens_physically_available": physical,
        "specimen_independence_confirmed": independence,
    }


def _has_image_signature(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 16:
        return False
    try:
        header = path.read_bytes()[:12]
    except OSError:
        return False
    return (
        header.startswith(b"\xff\xd8\xff")
        or header.startswith(b"\x89PNG\r\n\x1a\n")
        or header.startswith((b"II*\x00", b"MM\x00*"))
    )


def _source_images(
    external_root: Path | None,
    specimen_ids: tuple[str, ...],
    extensions: set[str],
) -> tuple[dict[str, list[Path]], list[str]]:
    images: dict[str, list[Path]] = {item: [] for item in specimen_ids}
    rejected = []
    if external_root is None:
        return images, rejected
    source_root = external_root / "source"
    for specimen_id in specimen_ids:
        directory = source_root / specimen_id
        if not directory.is_dir():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            if path.suffix.casefold() not in extensions:
                continue
            lowered = path.stem.casefold()
            if any(marker in lowered for marker in PLACEHOLDER_MARKERS):
                rejected.append(str(path))
            elif _has_image_signature(path):
                images[specimen_id].append(path.resolve())
            else:
                rejected.append(str(path))
    return images, rejected


def _sample_path(value: str, external_root: Path | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute() and external_root is not None:
        candidate = external_root / candidate
    return candidate.resolve()


def _validate_capture(
    *,
    capture_rows: list[dict[str, str]],
    capture_issues: list[str],
    sample_rows: list[dict[str, str]],
    images: dict[str, list[Path]],
    rejected_images: list[str],
    target: int,
    physical_passed: bool,
    external_root: Path | None,
) -> dict[str, Any]:
    blockers = list(capture_issues)
    image_count = sum(len(value) for value in images.values())
    if rejected_images:
        blockers.append("Placeholder or invalid image files cannot be evidence.")
    rows_by_specimen: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in capture_rows:
        rows_by_specimen[row.get("specimen_id", "")].append(row)
    sample_paths: dict[Path, str] = {}
    grouping_errors = []
    for row in sample_rows:
        path = _sample_path(row.get("source_image_path", ""), external_root)
        if path is None:
            continue
        specimen_id = row.get("specimen_id", "")
        sample_paths[path] = specimen_id
        expected_parent = (
            external_root / "source" / specimen_id
            if external_root is not None and specimen_id
            else None
        )
        if expected_parent is not None:
            try:
                path.relative_to(expected_parent.resolve())
            except ValueError:
                grouping_errors.append(row.get("sample_id") or str(path))
    if grouping_errors:
        blockers.append("Sample paths cross specimen source directories.")

    count_mismatches = []
    incomplete_specimens = []
    for specimen_id in APPROVED_SPECIMEN_IDS:
        files = images.get(specimen_id, [])
        completed_rows = [
            row
            for row in rows_by_specimen.get(specimen_id, [])
            if row.get("capture_status", "").casefold()
            in COMPLETE_CAPTURE_STATUSES
        ]
        actuals = [_int(row.get("actual_image_count")) for row in completed_rows]
        if not files or len(files) < target or not completed_rows:
            incomplete_specimens.append(specimen_id)
        elif len(files) not in {value for value in actuals if value is not None}:
            count_mismatches.append(specimen_id)
    if count_mismatches:
        blockers.append("Capture registry counts do not match source files.")
    complete = (
        physical_passed
        and not incomplete_specimens
        and not count_mismatches
        and not blockers
        and len(sample_paths) == image_count
        and image_count > 0
    )
    if image_count and len(sample_paths) != image_count:
        blockers.append("Every source image requires one sample registry row.")
    return _check(
        "pass" if complete else "blocked",
        (
            "Capture evidence is complete and reconciled."
            if complete
            else "Capture is incomplete; empty or planned folders are not evidence."
        ),
        blockers=blockers,
        evidence={
            "target_images_per_specimen": target,
            "actual_image_count": image_count,
            "image_counts_by_specimen": {
                key: len(value) for key, value in images.items()
            },
            "incomplete_specimens": incomplete_specimens,
            "rejected_placeholder_or_invalid_files": rejected_images,
            "grouping_errors": grouping_errors,
        },
    )


def _validate_sample_hashes(
    rows: list[dict[str, str]],
    external_root: Path | None,
) -> tuple[list[str], dict[str, str]]:
    blockers = []
    hashes: dict[str, str] = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        path = _sample_path(row.get("source_image_path", ""), external_root)
        expected_hash = row.get("original_image_sha256", "").casefold()
        if not sample_id or path is None or not path.is_file():
            blockers.append(f"Sample image is unavailable: {sample_id or '<blank>'}")
            continue
        actual_hash = sha256_file(path)
        hashes[sample_id] = actual_hash
        if expected_hash != actual_hash:
            blockers.append(f"Source image hash mismatch: {sample_id}")
    return blockers, hashes


def _validate_annotations(
    sample_rows: list[dict[str, str]],
    annotation_rows: list[dict[str, str]],
    annotation_issues: list[str],
    external_root: Path | None,
) -> dict[str, Any]:
    blockers = list(annotation_issues)
    annotations = {
        row.get("sample_id", ""): row for row in annotation_rows
        if row.get("sample_id")
    }
    missing = []
    overwritten = []
    for sample in sample_rows:
        sample_id = sample.get("sample_id", "")
        annotation = annotations.get(sample_id)
        path = _sample_path(sample.get("annotation_path", ""), external_root)
        expected_hash = sample.get("annotation_sha256", "").casefold()
        if (
            annotation is None
            or annotation.get("annotator_role") != "Dataset Annotator"
            or annotation.get("annotation_status", "").casefold()
            not in COMPLETE_ANNOTATION_STATUSES
            or path is None
            or not path.is_file()
            or not expected_hash
            or sha256_file(path) != expected_hash
        ):
            missing.append(sample_id or "<blank>")
            continue
        correction = _sample_path(annotation.get("correction_path", ""), external_root)
        if correction is not None and correction == path:
            overwritten.append(sample_id)
    if missing:
        blockers.append("One or more samples lack a complete hashed annotation.")
    if overwritten:
        blockers.append("Corrected labels must not overwrite original labels.")
    complete = bool(sample_rows) and not blockers
    return _check(
        "pass" if complete else "blocked",
        (
            "All source images have complete original annotations."
            if complete
            else "Annotations are incomplete or not independently preserved."
        ),
        blockers=blockers,
        evidence={
            "sample_count": len(sample_rows),
            "annotation_row_count": len(annotation_rows),
            "missing_sample_ids": missing,
            "overwritten_sample_ids": overwritten,
        },
    )


def _validate_reviews(
    sample_rows: list[dict[str, str]],
    review_rows: list[dict[str, str]],
    review_issues: list[str],
    external_root: Path | None,
) -> dict[str, Any]:
    blockers = list(review_issues)
    reviews = {
        row.get("sample_id", ""): row for row in review_rows
        if row.get("sample_id")
    }
    missing = []
    unresolved = []
    overwritten = []
    for sample in sample_rows:
        sample_id = sample.get("sample_id", "")
        row = reviews.get(sample_id)
        if (
            row is None
            or row.get("reviewer_role") != "Co-Supervisor / Gem Expert"
            or row.get("review_status", "").casefold() not in COMPLETE_REVIEW_STATUSES
            or _bool(row.get("mask_correct")) is not True
            or _bool(row.get("class_correct")) is not True
            or not row.get("review_date")
        ):
            missing.append(sample_id or "<blank>")
            continue
        if row.get("disagreement_reason", "").strip():
            unresolved.append(sample_id)
        corrected = _sample_path(
            row.get("corrected_annotation_path", ""),
            external_root,
        )
        original = _sample_path(sample.get("annotation_path", ""), external_root)
        if corrected is not None and corrected == original:
            overwritten.append(sample_id)
    if missing:
        blockers.append("One or more samples lack completed expert review.")
    if unresolved:
        blockers.append("Expert-review disagreements remain unresolved.")
    if overwritten:
        blockers.append("Expert corrections must be stored separately.")
    complete = bool(sample_rows) and not blockers
    return _check(
        "pass" if complete else "blocked",
        (
            "Every sample has a resolved role-only expert review."
            if complete
            else "Expert review is incomplete or contains unresolved issues."
        ),
        blockers=blockers,
        evidence={
            "sample_count": len(sample_rows),
            "review_row_count": len(review_rows),
            "missing_sample_ids": missing,
            "unresolved_sample_ids": unresolved,
            "overwritten_sample_ids": overwritten,
        },
    )


def _development_hashes(config: dict[str, Any]) -> tuple[set[str], list[str]]:
    development = config.get("development") or {}
    path = _resolve_config_path(config, development.get("materialization_manifest"))
    value, issues = _read_json(path)
    hashes = {
        str(row.get("image_sha256", "")).casefold()
        for row in value.get("records", [])
        if isinstance(row, dict) and row.get("image_sha256")
    }
    if value and not hashes:
        issues.append("Development manifest has no image hashes.")
    return hashes, issues


def _validate_duplicate_exclusion(
    config: dict[str, Any],
    sample_hashes: dict[str, str],
) -> dict[str, Any]:
    development_hashes, blockers = _development_hashes(config)
    exact = sorted(
        sample_id
        for sample_id, value in sample_hashes.items()
        if value.casefold() in development_hashes
    )
    if exact:
        blockers.append("Exact external/development image duplicates were found.")
    review_path = _resolve_config_path(
        config,
        config.get("perceptual_duplicate_review"),
    )
    perceptual_candidates = []
    if review_path is not None and review_path.is_file():
        rows, issues = _read_csv(review_path, ("external_sample_id", "status"))
        blockers.extend(issues)
        perceptual_candidates = [
            row.get("external_sample_id", "")
            for row in rows
            if row.get("status", "").casefold()
            not in {"cleared", "different", "not_duplicate"}
        ]
        if perceptual_candidates and config.get(
            "strict_duplicate_exclusion", True
        ):
            blockers.append("Perceptual duplicate candidates remain unresolved.")
    elif sample_hashes:
        blockers.append("Perceptual duplicate review is unavailable.")
    if not sample_hashes:
        status = "unavailable"
        reason = "No external images exist for duplicate exclusion."
    elif blockers:
        status = "blocked"
        reason = "External/development duplicate exclusion is incomplete."
    else:
        status = "pass"
        reason = "Exact and perceptual duplicate exclusion passed."
    return _check(
        status,
        reason,
        blockers=blockers,
        evidence={
            "external_hash_count": len(sample_hashes),
            "development_hash_count": len(development_hashes),
            "exact_duplicate_sample_ids": exact,
            "unresolved_perceptual_sample_ids": perceptual_candidates,
        },
    )


def _validate_training_isolation(
    approval: dict[str, Any],
    sample_rows: list[dict[str, str]],
) -> dict[str, Any]:
    blockers = []
    prohibited = set(approval.get("test_data_prohibited_uses") or ())
    required = {
        "training",
        "validation",
        "early_stopping",
        "threshold_selection",
        "architecture_selection",
        "augmentation_decisions",
        "model_selection",
    }
    if prohibited != required:
        blockers.append("Operations approval does not prohibit every model use.")
    violations = []
    unknown = []
    for row in sample_rows:
        for field in ISOLATION_FIELDS:
            value = _bool(row.get(field))
            if value is True:
                violations.append(f"{row.get('sample_id', '<blank>')}:{field}")
            elif value is None:
                unknown.append(f"{row.get('sample_id', '<blank>')}:{field}")
    if violations:
        blockers.append("External samples were exposed to prohibited workflows.")
    if sample_rows and unknown:
        blockers.append("External sample isolation declarations are incomplete.")
    return _check(
        "pass" if not blockers else "blocked",
        (
            "External data is prohibited from all development workflows."
            if not blockers
            else "External test isolation is not enforceable."
        ),
        blockers=blockers,
        evidence={"violations": violations, "unknown_declarations": unknown},
    )


def _validate_development_readiness(config: dict[str, Any]) -> dict[str, Any]:
    development = config.get("development") or {}
    root = _resolve_config_path(config, development.get("dataset_root"))
    integrity_path = _resolve_config_path(
        config,
        development.get("integrity_report"),
    )
    lock_path = _resolve_config_path(config, development.get("split_lock"))
    integrity, blockers = _read_json(integrity_path)
    lock, lock_issues = _read_json(lock_path)
    blockers.extend(lock_issues)
    if root is None or not root.is_dir():
        blockers.append("Materialized development dataset is unavailable.")
    if (integrity.get("summary") or {}).get("integrity_verified") is not True:
        blockers.append("Development dataset integrity is not verified.")
    if integrity.get("split_counts") != {"train": 138, "valid": 35, "test": 0}:
        blockers.append("Development split counts differ from the frozen 138/35 plan.")
    if integrity.get("class_names") != ["fracture", "inclusion"]:
        blockers.append("Development taxonomy differs from the approved classes.")
    if lock.get("frozen") is not True or lock.get("split_approved") is not True:
        blockers.append("Development split is not approved and frozen.")
    if lock.get("training_authorized") is not False:
        blockers.append("Frozen split lock must not authorize training.")
    return _check(
        "pass" if not blockers else "blocked",
        (
            "Frozen 138/35 development dataset remains integrity-verified."
            if not blockers
            else "Development readiness evidence is incomplete or changed."
        ),
        blockers=blockers,
        evidence={
            "dataset_root": str(root) if root else None,
            "split_counts": integrity.get("split_counts"),
            "split_id": lock.get("split_id"),
            "frozen": lock.get("frozen"),
        },
    )


def _validate_membership_and_lock(
    config: dict[str, Any],
    prerequisites_pass: bool,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    membership_path = _resolve_config_path(
        config,
        config.get("external_membership_approval"),
    )
    membership, membership_issues = _read_json(membership_path)
    membership_ok = bool(
        prerequisites_pass
        and not membership_issues
        and membership.get("approved") is True
        and tuple(membership.get("specimen_ids") or ()) == APPROVED_SPECIMEN_IDS
        and membership.get("example_only") is False
    )
    membership_check = _check(
        "pass" if membership_ok else "blocked",
        (
            "Final external membership is explicitly approved."
            if membership_ok
            else "Final external membership is not approved."
        ),
        blockers=[] if membership_ok else membership_issues,
        evidence={"path": str(membership_path) if membership_path else None},
    )

    lock_path = _resolve_config_path(config, config.get("external_test_lock"))
    lock, lock_issues = _read_json(lock_path)
    frozen = bool(
        membership_ok
        and not lock_issues
        and lock.get("external_test_frozen") is True
        and lock.get("final_evaluation_authorized") is True
        and lock.get("manifest_sha256")
        and lock.get("lock_hash")
    )
    lock_check = _check(
        "pass" if frozen else "blocked",
        (
            "External test membership is sealed and frozen."
            if frozen
            else "External test lock is unavailable or freeze requirements are unmet."
        ),
        blockers=[] if frozen else lock_issues,
        evidence={
            "path": str(lock_path) if lock_path else None,
            "external_test_frozen": frozen,
        },
    )
    return membership_check, lock_check, frozen


def evaluate_external_test_readiness(
    config: dict[str, Any],
) -> dict[str, Any]:
    repo_root = Path(config["_repo_root"]).resolve()
    approval_path = _resolve_config_path(config, config.get("operations_approval"))
    approval, operations = _validate_operations_approval(approval_path, repo_root)

    registry_config = config.get("registries") or {}
    specimen_rows, specimen_issues = _read_csv(
        _resolve_config_path(config, registry_config.get("specimens")),
        SPECIMEN_FIELDS,
    )
    capture_rows, capture_issues = _read_csv(
        _resolve_config_path(config, registry_config.get("captures")),
        CAPTURE_FIELDS,
    )
    sample_rows, sample_issues = _read_csv(
        _resolve_config_path(config, registry_config.get("samples")),
        SAMPLE_FIELDS,
    )
    annotation_rows, annotation_issues = _read_csv(
        _resolve_config_path(config, registry_config.get("annotations")),
        ANNOTATION_FIELDS,
    )
    review_rows, review_issues = _read_csv(
        _resolve_config_path(config, registry_config.get("expert_reviews")),
        EXPERT_REVIEW_FIELDS,
    )
    sample_issues.extend(
        issue
        for row in sample_rows
        for issue in _privacy_issues(row, "$.samples")
    )
    specimen_checks = _validate_specimens(
        specimen_rows,
        specimen_issues,
        approval,
    )

    schedule_checks = {
        "capture_schedule_recorded": _check(
            "pass"
            if approval.get("expected_capture_date") == "2026-08-12"
            else "blocked",
            (
                "Expected capture date is recorded; it is not completion evidence."
                if approval.get("expected_capture_date") == "2026-08-12"
                else "Expected capture date is unavailable."
            ),
            evidence={
                "expected_capture_date": approval.get("expected_capture_date"),
                "treated_as_completed": False,
            },
        ),
        "review_schedule_recorded": _check(
            "pass"
            if approval.get("expected_expert_review_completion_date")
            == "2026-08-19"
            else "blocked",
            (
                "Expected review date is recorded; it is not completion evidence."
                if approval.get("expected_expert_review_completion_date")
                == "2026-08-19"
                else "Expected review date is unavailable."
            ),
            evidence={
                "expected_review_completion_date": approval.get(
                    "expected_expert_review_completion_date"
                ),
                "treated_as_completed": False,
            },
        ),
    }

    external_root = _resolve_config_path(config, config.get("external_data_root"))
    capture_config = config.get("capture") or {}
    extensions = {
        str(value).casefold()
        for value in capture_config.get(
            "allowed_image_extensions",
            [".jpg", ".jpeg", ".png", ".tif", ".tiff"],
        )
    }
    images, rejected_images = _source_images(
        external_root,
        APPROVED_SPECIMEN_IDS,
        extensions,
    )
    capture = _validate_capture(
        capture_rows=capture_rows,
        capture_issues=capture_issues + sample_issues,
        sample_rows=sample_rows,
        images=images,
        rejected_images=rejected_images,
        target=int(capture_config.get("target_images_per_specimen", 36)),
        physical_passed=specimen_checks[
            "specimens_physically_available"
        ]["passed"],
        external_root=external_root,
    )
    sample_hash_issues, sample_hashes = _validate_sample_hashes(
        sample_rows,
        external_root,
    )
    if sample_hash_issues:
        capture["blockers"] = list(
            dict.fromkeys(capture["blockers"] + sample_hash_issues)
        )
        capture["status"] = "blocked"
        capture["passed"] = False
    annotations = _validate_annotations(
        sample_rows,
        annotation_rows,
        annotation_issues,
        external_root,
    )
    reviews = _validate_reviews(
        sample_rows,
        review_rows,
        review_issues,
        external_root,
    )
    duplicates = _validate_duplicate_exclusion(config, sample_hashes)
    isolation = _validate_training_isolation(approval, sample_rows)
    development = _validate_development_readiness(config)

    prerequisites_pass = all(
        check["passed"]
        for check in (
            specimen_checks["specimens_physically_available"],
            specimen_checks["specimen_independence_confirmed"],
            capture,
            annotations,
            reviews,
            duplicates,
            isolation,
        )
    )
    membership, lock, frozen = _validate_membership_and_lock(
        config,
        prerequisites_pass,
    )

    checks = {
        "operations_plan": operations,
        **specimen_checks,
        **schedule_checks,
        "capture_complete": capture,
        "annotations_complete": annotations,
        "expert_review_complete": reviews,
        "duplicate_exclusion": duplicates,
        "test_membership_approval": membership,
        "test_lock": lock,
        "training_isolation": isolation,
        "development_dataset_readiness": development,
    }
    statuses = {
        key: value["status"] for key, value in checks.items()
    }
    statuses.update(
        {
            "test_set_frozen": frozen,
            "training_authorized": False,
            "final_evaluation_authorized": frozen,
        }
    )
    operational_pass = all(
        checks[key]["passed"]
        for key in (
            "operations_plan",
            "specimen_count_planned",
            "capture_schedule_recorded",
            "review_schedule_recorded",
        )
    )
    external_data_pass = all(
        checks[key]["passed"]
        for key in (
            "specimens_physically_available",
            "specimen_independence_confirmed",
            "capture_complete",
            "annotations_complete",
            "expert_review_complete",
            "duplicate_exclusion",
            "test_membership_approval",
            "test_lock",
        )
    )
    clean_preparation_ready = bool(
        development["passed"] and operational_pass and isolation["passed"]
    )
    gate_status = {
        "A_development_dataset_readiness": {
            "status": "pass" if development["passed"] else "blocked",
            "reason": development["reason"],
        },
        "B_external_operations_planning": {
            "status": "pass" if operational_pass else "blocked",
            "reason": (
                "The role-only external collection plan is registered."
                if operational_pass
                else "External operations planning is incomplete."
            ),
        },
        "C_external_data_readiness": {
            "status": "pass" if external_data_pass else "blocked",
            "reason": (
                "External data is complete and frozen."
                if external_data_pass
                else "Physical collection, annotation, review, or freezing remains."
            ),
        },
        "D_clean_training_preparation": {
            "status": "ready_but_disabled" if clean_preparation_ready else "blocked",
            "reason": (
                "Development inputs are prepared, but this phase cannot train."
                if clean_preparation_ready
                else "Clean training preparation prerequisites are incomplete."
            ),
        },
        "E_final_clean_training_authorization": {
            "status": "blocked",
            "reason": (
                "A separate explicit phase is required after physical specimen "
                "membership is confirmed and holdout isolation is enforceable."
            ),
        },
        "F_final_evaluation_authorization": {
            "status": "pass" if frozen else "blocked",
            "reason": (
                "The approved external lock authorizes final evaluation."
                if frozen
                else "Capture, annotation, expert review, membership, and freezing "
                "must be complete."
            ),
        },
    }
    return {
        "schema_version": "1.0",
        "report_type": "external_test_readiness",
        "created_utc": utc_timestamp(),
        "operations": {
            "external_holdout_scope": approval.get("external_holdout_scope"),
            "specimen_ids": list(approval.get("specimen_ids") or []),
            "specimen_count": approval.get("specimen_count"),
            "expected_capture_date": approval.get("expected_capture_date"),
            "expected_expert_review_completion_date": approval.get(
                "expected_expert_review_completion_date"
            ),
        },
        "checks": checks,
        "statuses": statuses,
        "gate_status": gate_status,
        "small_sample_warning": (
            "Five specimens support a limited external pilot only. "
            "Generalizability and confidence remain limited; additional "
            "independent specimens are recommended."
        ),
        "summary": {
            "status": "pass" if external_data_pass else "blocked",
            "operations_plan_ready": operational_pass,
            "external_data_ready": external_data_pass,
            "clean_training_preparation_ready": clean_preparation_ready,
            "training_authorized": False,
            "final_evaluation_authorized": frozen,
            "decision_training": (
                "NOT AUTHORIZED FOR TRAINING - EXTERNAL DATA COLLECTION PENDING"
            ),
            "decision_evaluation": (
                "NOT AUTHORIZED FOR FINAL EVALUATION - TEST SET NOT CAPTURED OR "
                "FROZEN"
                if not frozen
                else "FINAL EVALUATION AUTHORIZED BY EXTERNAL TEST LOCK"
            ),
        },
        "safety": {
            "training_occurred": False,
            "evaluation_occurred": False,
            "model_weights_downloaded": False,
            "model_weights_created": False,
            "performance_metrics_calculated": False,
            "claim_90_percent_available": False,
            "synthetic_external_evidence_created": False,
        },
    }


def _pending_pack_files(
    output: Path,
    rows: dict[str, list[dict[str, str]]],
) -> list[Path]:
    pack = output / "human_collection_pack"
    pack.mkdir(parents=True, exist_ok=True)
    registry_files = {
        "external_test_specimen_registry.pending.csv": (
            rows["specimens"],
            SPECIMEN_FIELDS,
        ),
        "external_test_capture_registry.pending.csv": (
            rows["captures"],
            CAPTURE_FIELDS,
        ),
        "external_test_sample_registry.pending.csv": (
            rows["samples"],
            SAMPLE_FIELDS,
        ),
        "external_test_annotation_registry.pending.csv": (
            rows["annotations"],
            ANNOTATION_FIELDS,
        ),
        "external_test_expert_review_registry.pending.csv": (
            rows["expert_reviews"],
            EXPERT_REVIEW_FIELDS,
        ),
    }
    created = []
    for name, (values, fields) in registry_files.items():
        path = pack / name
        atomic_write_csv(path, values, fields)
        created.append(path)
    documents = {
        "capture_day_checklist.md": """# Capture Day Checklist

- [ ] 1. Confirm the reserved specimen ID.
- [ ] 2. Confirm the stone is physically new.
- [ ] 3. Confirm it is not one of the development gemstones.
- [ ] 4. Record the traceable capture session.
- [ ] 5. Capture approximately 36 views at 10-degree intervals.
- [ ] 6. Check blur, duplicates, focus, and full rotational coverage.
- [ ] 7. Preserve source originals without resizing or augmentation.
- [ ] 8. Record the actual image count.
- [ ] 9. Hash every source file after capture.
- [ ] 10. Keep external data outside all training directories.
""",
        "annotation_checklist.md": """# Annotation Checklist

- [ ] Use class 0 only for visible fracture lines or cracks.
- [ ] Use class 1 only for visible inclusions or internal irregularities.
- [ ] Exclude reflections, dust, background, and shadows.
- [ ] Retain defect-negative images with valid empty labels.
- [ ] Flag ambiguous regions for expert review.
- [ ] Preserve original annotations and store corrections separately.
- [ ] Do not accept model predictions as final ground truth.
""",
        "expert_review_checklist.md": """# Expert Review Checklist

- [ ] Review every source image and annotation.
- [ ] Confirm mask and class correctness.
- [ ] Record role-only review status and date.
- [ ] Store corrections separately from original annotations.
- [ ] Resolve every disagreement before freezing.
- [ ] Confirm valid negative examples are retained.
""",
        "test_freeze_checklist.md": """# Test Freeze Checklist

- [ ] Confirm all five physical specimen identities.
- [ ] Confirm development exclusion for every specimen.
- [ ] Reconcile capture counts and source hashes.
- [ ] Complete annotations and expert review.
- [ ] Resolve exact and perceptual duplicate checks.
- [ ] Confirm no prohibited training or selection use.
- [ ] Approve the final membership manifest.
- [ ] Generate and verify the final lock hash.
- [ ] Mark the test set frozen only after every item passes.
""",
        "folder_naming_guide.md": """# Folder Naming Guide

Store source originals under `source/EXT-GEM-NNN/`, using only the five
reserved IDs. Keep original annotations in `annotations_original/` and expert
corrections in `annotations_expert_corrected/`. Use stable sample IDs that map
one image to one specimen and capture session. Never place external images
inside development `train` or `valid` directories.
""",
    }
    for name, text in documents.items():
        path = pack / name
        atomic_write_text(path, text)
        created.append(path)
    return created


def _markdown_summary(report: dict[str, Any]) -> str:
    rows = [
        "# External Test Readiness",
        "",
        "| Check | Status | Reason |",
        "|---|---|---|",
    ]
    for key, value in report["checks"].items():
        rows.append(f"| `{key}` | {value['status']} | {value['reason']} |")
    rows.extend(
        [
            "",
            f"Training authorized: `{str(report['summary']['training_authorized']).lower()}`",
            (
                "Final evaluation authorized: "
                f"`{str(report['summary']['final_evaluation_authorized']).lower()}`"
            ),
            "",
            report["small_sample_warning"],
            "",
        ]
    )
    return "\n".join(rows)


def run_external_test_readiness(
    config: dict[str, Any],
    *,
    output_override: str | Path | None = None,
    command: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    report = evaluate_external_test_readiness(config)
    output = resolve_path(
        output_override or config.get("output_directory"),
        config["_repo_root"],
    )
    if output is None:
        raise ValueError("An output directory is required.")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "external_test_readiness.json"
    status_path = output / "readiness_statuses.csv"
    summary_path = output / "audit_summary.md"
    manifest_path = output / "manifest.json"
    atomic_write_json(report_path, report)
    atomic_write_csv(
        status_path,
        [
            {
                "check": key,
                "status": value["status"],
                "passed": str(value["passed"]).lower(),
                "reason": value["reason"],
            }
            for key, value in report["checks"].items()
        ],
        ("check", "status", "passed", "reason"),
    )
    atomic_write_text(summary_path, _markdown_summary(report))

    registry_config = config.get("registries") or {}
    rows = {}
    for key, fields in (
        ("specimens", SPECIMEN_FIELDS),
        ("captures", CAPTURE_FIELDS),
        ("samples", SAMPLE_FIELDS),
        ("annotations", ANNOTATION_FIELDS),
        ("expert_reviews", EXPERT_REVIEW_FIELDS),
    ):
        rows[key], _ = _read_csv(
            _resolve_config_path(config, registry_config.get(key)),
            fields,
        )
    pack_files = _pending_pack_files(output, rows)
    inputs = [
        ("operations_approval", config.get("operations_approval")),
        *[
            (f"registry_{key}", value)
            for key, value in registry_config.items()
        ],
        *[
            (f"protocol_{key}", value)
            for key, value in (config.get("protocols") or {}).items()
        ],
        (
            "development_materialization_manifest",
            (config.get("development") or {}).get("materialization_manifest"),
        ),
        (
            "development_integrity_report",
            (config.get("development") or {}).get("integrity_report"),
        ),
        (
            "development_split_lock",
            (config.get("development") or {}).get("split_lock"),
        ),
        ("external_membership_approval", config.get("external_membership_approval")),
        ("external_test_lock", config.get("external_test_lock")),
    ]
    write_manifest(
        path=manifest_path,
        repo_root=config["_repo_root"],
        tool_name="quartz_external_test_readiness",
        command=command,
        tool_sources=[
            "backend/research/external_test_readiness.py",
            "backend/research/readiness_common.py",
            "backend/research_benchmark.py",
        ],
        inputs=inputs,
        outputs=[
            ("readiness_report", report_path),
            ("readiness_statuses", status_path),
            ("markdown_summary", summary_path),
            *[
                (f"human_pack_{index:02d}", path)
                for index, path in enumerate(pack_files, 1)
            ],
        ],
        extra={
            "training_authorized": False,
            "final_evaluation_authorized": report["summary"][
                "final_evaluation_authorized"
            ],
            "training_occurred": False,
            "evaluation_occurred": False,
            "model_weights_downloaded": False,
            "performance_metrics_calculated": False,
        },
    )
    return report, {
        "report": report_path,
        "statuses": status_path,
        "summary": summary_path,
        "manifest": manifest_path,
        "human_collection_pack": output / "human_collection_pack",
    }
