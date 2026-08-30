"""Phase 3B Gate 0 orchestration without training or final evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common import atomic_write_json, normalize_path, sha256_file, utc_timestamp
from .grouped_split import UnionFind
from .metric_approval import validate_metric_approval
from .readiness_common import (
    atomic_write_csv,
    atomic_write_text,
    load_json_config,
    resolve_path,
    stable_json_sha256,
    write_manifest,
)
from .taxonomy_validation import validate_taxonomy_approval

ROUTES = {"external_specimen_holdout", "recovered_legacy_lineage"}
DUPLICATE_POLICIES = {"human_reviewed", "conservative_union"}
GATE_STATUSES = {"pass", "warning", "blocked", "unavailable", "not_applicable"}
GROUPING_DECISIONS = {"same_source", "same_specimen", "augmentation_related"}
EXPERT_STATUSES = {
    "approved",
    "corrected",
    "rejected",
    "needs_second_review",
    "unreviewed",
}
TRAINING_INITIALIZATIONS = {
    "official_generic_pretrained_initialization",
    "training_from_scratch",
}

SPECIMEN_FIELDS = (
    "specimen_id",
    "specimen_source",
    "acquisition_date",
    "custodian",
    "legacy_dataset_exclusion_confirmed",
    "exclusion_confirmed_by",
    "notes",
)
CAPTURE_FIELDS = (
    "capture_session_id",
    "specimen_id",
    "capture_date",
    "camera_id",
    "capture_mode",
    "lighting",
    "polarization",
    "background",
    "image_count",
    "operator_id",
    "notes",
)
SAMPLE_FIELDS = (
    "sample_id",
    "specimen_id",
    "capture_session_id",
    "source_image_path",
    "label_path",
    "expert_review_status",
    "expert_reviewer_id",
    "original_image_sha256",
    "label_sha256",
    "test_set_status",
    "is_augmented",
    "used_for_threshold_selection",
    "used_for_architecture_selection",
    "used_for_early_stopping",
    "used_for_confidence_tuning",
    "used_for_augmentation_tuning",
    "used_for_model_selection",
    "notes",
)
EXTERNAL_EXPERT_FIELDS = (
    "sample_id",
    "specimen_id",
    "expert_reviewer_id",
    "review_date",
    "review_status",
    "mask_correct",
    "class_correct",
    "correction_file",
    "visible_defect_present",
    "review_confidence",
    "class_ids_present",
    "disagreement_reason",
    "notes",
)


def load_phase3b_gate_config(
    path: str | Path,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    return load_json_config(path, output_override=output_override)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path | None) -> tuple[list[dict[str, str]], list[str]]:
    if path is None or not path.is_file():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = [
            {key: str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    return rows, headers


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().casefold()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _approval(value: Any, *, route: str | None = None) -> bool:
    if not isinstance(value, dict) or value.get("example_only", False):
        return False
    if route is not None and value.get("dataset_route") != route:
        return False
    return bool(
        value.get("approved_by")
        and value.get("approver_role")
        and value.get("approval_date")
    )


def _gate(
    key: str,
    title: str,
    status: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in GATE_STATUSES:
        raise ValueError(f"Unsupported gate status: {status}")
    return {
        "gate": key,
        "title": title,
        "status": status,
        "passed": status == "pass",
        "reason": reason,
        "evidence": evidence or {},
    }


def duplicate_relationships(
    candidate_rows: Iterable[dict[str, Any]],
    review_rows: Iterable[dict[str, Any]],
    policy: str | None,
) -> dict[str, Any]:
    """Resolve candidate grouping without calling candidates confirmed duplicates."""
    candidates = [
        {
            "left": str(row.get("left") or row.get("image_a") or ""),
            "right": str(row.get("right") or row.get("image_b") or ""),
        }
        for row in candidate_rows
        if (row.get("left") or row.get("image_a"))
        and (row.get("right") or row.get("image_b"))
    ]
    reviews = {}
    for row in review_rows:
        left = str(row.get("image_a") or row.get("left") or "")
        right = str(row.get("image_b") or row.get("right") or "")
        reviews[tuple(sorted((left, right)))] = str(
            row.get("review_decision") or "unreviewed"
        )
    nodes = sorted(
        {item for row in candidates for item in (row["left"], row["right"])}
    )
    union = UnionFind(nodes)
    joined_pairs = []
    unresolved = 0
    reviewed = 0
    for row in candidates:
        pair = tuple(sorted((row["left"], row["right"])))
        decision = reviews.get(pair, "unreviewed")
        if decision not in {"", "unreviewed", "uncertain"}:
            reviewed += 1
        if decision in GROUPING_DECISIONS:
            should_join = True
        elif decision in {"different_specimen", "not_duplicate"}:
            should_join = False
        elif policy == "conservative_union":
            should_join = True
            unresolved += 1
        else:
            should_join = False
            unresolved += 1
        if should_join:
            union.union(*pair)
            joined_pairs.append(
                {"left": pair[0], "right": pair[1], "decision": decision}
            )
    components: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        components[union.find(node)].append(node)
    grouped = sorted(
        (sorted(values) for values in components.values() if len(values) > 1),
        key=lambda values: (-len(values), values),
    )
    return {
        "policy": policy,
        "method": (
            "conservative_near_duplicate_grouping"
            if policy == "conservative_union"
            else "human_reviewed_near_duplicate_grouping"
            if policy == "human_reviewed"
            else "unavailable"
        ),
        "candidate_count": len(candidates),
        "reviewed_count": reviewed,
        "unresolved_count": unresolved,
        "joined_pair_count": len(joined_pairs),
        "connected_component_count": len(grouped),
        "largest_component_size": max((len(value) for value in grouped), default=0),
        "components": grouped,
        "joined_pairs": joined_pairs,
        "provisional": policy == "conservative_union",
        "balance_effect": {
            "status": (
                "requires_deterministic_split_recalculation"
                if policy == "conservative_union" and grouped
                else "not_applicable"
            ),
            "images_in_multi_candidate_components": sum(
                len(value) for value in grouped
            ),
            "note": (
                "Conservative components can reduce stratification flexibility "
                "and must be reported with the resulting train/validation counts."
            ),
        },
    }


def _difference_hash(path: Path) -> int | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            pixels = list(
                image.convert("L")
                .resize((9, 8), Image.Resampling.LANCZOS)
                .tobytes()
            )
        value = 0
        for row in range(8):
            for column in range(8):
                left = pixels[row * 9 + column]
                right = pixels[row * 9 + column + 1]
                value = (value << 1) | int(left > right)
        return value
    except Exception:
        return None


def _duplicates_against_development(
    samples: list[dict[str, str]],
    development_rows: list[dict[str, str]],
    repo_root: Path,
    threshold: int,
) -> dict[str, Any]:
    development_hashes = {
        row.get("image_sha256", "")
        for row in development_rows
        if row.get("image_sha256")
    }
    exact_cross = []
    exact_internal = []
    by_hash: dict[str, list[str]] = defaultdict(list)
    external_hashes: dict[str, Path] = {}
    for row in samples:
        digest = row.get("original_image_sha256", "")
        sample_id = row.get("sample_id", "")
        if digest:
            by_hash[digest].append(sample_id)
            if digest in development_hashes:
                exact_cross.append(sample_id)
        path = resolve_path(row.get("source_image_path"), repo_root)
        if path and path.is_file():
            external_hashes[sample_id] = path
    for digest, members in by_hash.items():
        if digest and len(members) > 1:
            exact_internal.append({"sha256": digest, "sample_ids": sorted(members)})

    perceptual = []
    external_dhash = {
        sample_id: _difference_hash(path)
        for sample_id, path in external_hashes.items()
    }
    for row in development_rows:
        path = resolve_path(row.get("file_path"), repo_root)
        if path is None or not path.is_file():
            continue
        development_value = _difference_hash(path)
        if development_value is None:
            continue
        for sample_id, external_value in external_dhash.items():
            if external_value is None:
                continue
            distance = (development_value ^ external_value).bit_count()
            if distance <= threshold:
                perceptual.append(
                    {
                        "development_image": row.get("file_path", ""),
                        "external_sample_id": sample_id,
                        "hamming_distance": distance,
                        "status": "candidate_not_confirmed",
                    }
                )
    return {
        "exact_cross_set_sample_ids": sorted(set(exact_cross)),
        "exact_internal_groups": exact_internal,
        "perceptual_cross_set_candidates": perceptual,
        "perceptual_check_status": (
            "available"
            if external_hashes
            and any(value is not None for value in external_dhash.values())
            else "unavailable"
        ),
    }


def summarize_external_expert_review(
    review_rows: list[dict[str, str]],
    sample_rows: list[dict[str, str]],
) -> dict[str, Any]:
    sample_ids = {row.get("sample_id", "") for row in sample_rows}
    specimen_by_sample = {
        row.get("sample_id", ""): row.get("specimen_id", "") for row in sample_rows
    }
    relevant = [row for row in review_rows if row.get("sample_id", "") in sample_ids]
    statuses = Counter(row.get("review_status", "") for row in relevant)
    complete_statuses = {"approved", "corrected"}
    complete_ids = {
        row.get("sample_id", "")
        for row in relevant
        if row.get("review_status") in complete_statuses
    }
    reviewed_specimens = {
        specimen_by_sample.get(sample_id, "")
        for sample_id in complete_ids
        if specimen_by_sample.get(sample_id, "")
    }
    class_ids = {
        item.strip()
        for row in relevant
        for item in row.get("class_ids_present", "").replace(";", ",").split(",")
        if item.strip()
    }
    positives = sum(
        _bool(row.get("visible_defect_present")) is True for row in relevant
    )
    negatives = sum(
        _bool(row.get("visible_defect_present")) is False for row in relevant
    )
    total = len(sample_rows)
    return {
        "total_test_images": total,
        "total_test_specimens": len(
            {row.get("specimen_id", "") for row in sample_rows if row.get("specimen_id")}
        ),
        "reviewed_images": len(complete_ids),
        "reviewed_specimens": len(reviewed_specimens),
        "corrected_labels": statuses["corrected"],
        "approved_labels": statuses["approved"],
        "rejected_labels": statuses["rejected"],
        "unresolved_disagreements": sum(
            bool(row.get("disagreement_reason", ""))
            for row in relevant
            if row.get("review_status") not in complete_statuses
        ),
        "second_review_requirements": statuses["needs_second_review"],
        "unreviewed": total - len(complete_ids),
        "class_coverage": sorted(class_ids),
        "visible_defect_positive_images": positives,
        "visible_defect_negative_images": negatives,
        "review_coverage": (len(complete_ids) / total) if total else 0.0,
        "complete": bool(
            total
            and len(complete_ids) == total
            and statuses["rejected"] == 0
            and statuses["needs_second_review"] == 0
        ),
    }


def validate_external_holdout(
    external: dict[str, Any],
    repo_root: Path,
    development_rows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    development_rows = development_rows or []
    specimen_path = resolve_path(external.get("specimens_manifest"), repo_root)
    session_path = resolve_path(external.get("capture_sessions_manifest"), repo_root)
    sample_path = resolve_path(external.get("samples_manifest"), repo_root)
    review_path = resolve_path(external.get("expert_review"), repo_root)
    specimens, specimen_headers = _read_csv(specimen_path)
    sessions, session_headers = _read_csv(session_path)
    samples, sample_headers = _read_csv(sample_path)
    reviews, review_headers = _read_csv(review_path)
    blockers = []
    warnings = []

    for label, headers, required in (
        ("specimen", specimen_headers, SPECIMEN_FIELDS),
        ("capture-session", session_headers, CAPTURE_FIELDS),
        ("sample", sample_headers, SAMPLE_FIELDS),
        ("expert-review", review_headers, EXTERNAL_EXPERT_FIELDS),
    ):
        if not headers:
            blockers.append(f"External {label} manifest is unavailable.")
        else:
            missing = [field for field in required if field not in headers]
            if missing:
                blockers.append(
                    f"External {label} manifest is missing columns: {missing}"
                )
    if not specimens:
        blockers.append("No external specimen records were supplied.")
    if not sessions:
        blockers.append("No external capture-session records were supplied.")
    if not samples:
        blockers.append("No external sample records were supplied.")

    specimen_ids = [row.get("specimen_id", "") for row in specimens]
    session_ids = [row.get("capture_session_id", "") for row in sessions]
    sample_ids = [row.get("sample_id", "") for row in samples]
    for label, values in (
        ("specimen", specimen_ids),
        ("capture session", session_ids),
        ("sample", sample_ids),
    ):
        duplicates = sorted(
            value for value, count in Counter(values).items() if value and count > 1
        )
        if duplicates:
            blockers.append(f"Duplicate external {label} IDs: {duplicates}")
        if any(not value for value in values):
            blockers.append(f"External {label} IDs contain empty values.")

    specimen_set = set(specimen_ids)
    session_map = {
        row.get("capture_session_id", ""): row.get("specimen_id", "")
        for row in sessions
    }
    for row in specimens:
        if (
            _bool(row.get("legacy_dataset_exclusion_confirmed")) is not True
            or not row.get("exclusion_confirmed_by")
        ):
            blockers.append(
                f"Legacy exclusion is not confirmed for specimen "
                f"{row.get('specimen_id', '')}."
            )
    for row in sessions:
        if row.get("specimen_id", "") not in specimen_set:
            blockers.append(
                f"Capture session {row.get('capture_session_id', '')} references "
                "an unknown specimen."
            )
    test_status_by_specimen: dict[str, set[str]] = defaultdict(set)
    sample_hash_errors = 0
    forbidden_use = {
        "used_for_threshold_selection": "threshold selection",
        "used_for_architecture_selection": "architecture selection",
        "used_for_early_stopping": "early stopping",
        "used_for_confidence_tuning": "confidence tuning",
        "used_for_augmentation_tuning": "augmentation tuning",
        "used_for_model_selection": "model selection",
    }
    for row in samples:
        sample_id = row.get("sample_id", "")
        specimen_id = row.get("specimen_id", "")
        session_id = row.get("capture_session_id", "")
        if specimen_id not in specimen_set:
            blockers.append(f"Sample {sample_id} references an unknown specimen.")
        if session_id not in session_map:
            blockers.append(f"Sample {sample_id} references an unknown session.")
        elif session_map[session_id] != specimen_id:
            blockers.append(
                f"Sample {sample_id} specimen/session relationship is inconsistent."
            )
        if _bool(row.get("is_augmented")) is not False:
            blockers.append(f"External test sample {sample_id} is augmented or undecided.")
        for field, purpose in forbidden_use.items():
            if _bool(row.get(field)) is not False:
                blockers.append(
                    f"External test sample {sample_id} is used or undecided for {purpose}."
                )
        status = row.get("test_set_status", "")
        test_status_by_specimen[specimen_id].add(status)
        if status != "frozen":
            blockers.append(f"External test sample {sample_id} is not frozen.")
        for field, hash_field in (
            ("source_image_path", "original_image_sha256"),
            ("label_path", "label_sha256"),
        ):
            path = resolve_path(row.get(field), repo_root)
            if path is None or not path.is_file():
                blockers.append(f"External sample {sample_id} {field} is unavailable.")
                sample_hash_errors += 1
            elif row.get(hash_field) != sha256_file(path):
                blockers.append(f"External sample {sample_id} {hash_field} does not match.")
                sample_hash_errors += 1
    for specimen_id, statuses in test_status_by_specimen.items():
        if statuses != {"frozen"}:
            blockers.append(
                f"All images from specimen {specimen_id} must remain in the frozen test set."
            )

    duplicates = _duplicates_against_development(
        samples,
        development_rows,
        repo_root,
        int(external.get("perceptual_hamming_threshold", 5)),
    )
    if duplicates["exact_cross_set_sample_ids"]:
        blockers.append("Exact external test images occur in development data.")
    if duplicates["exact_internal_groups"]:
        blockers.append("Exact duplicate images occur within the external test set.")
    if duplicates["perceptual_cross_set_candidates"]:
        blockers.append(
            "Perceptual duplicate candidates cross development and external test data."
        )
    if duplicates["perceptual_check_status"] == "unavailable" and samples:
        blockers.append("Perceptual cross-set duplicate checking is unavailable.")

    expert = summarize_external_expert_review(reviews, samples)
    label_by_sample = {
        row.get("sample_id", ""): resolve_path(row.get("label_path"), repo_root)
        for row in samples
    }
    for row in reviews:
        if row.get("review_status") != "corrected":
            continue
        correction = resolve_path(row.get("correction_file"), repo_root)
        original = label_by_sample.get(row.get("sample_id", ""))
        if correction is None or not correction.is_file():
            blockers.append(
                f"Corrected expert label is unavailable for "
                f"{row.get('sample_id', '')}."
            )
        elif original is not None and correction == original:
            blockers.append(
                f"Expert correction for {row.get('sample_id', '')} must be "
                "stored separately from the original label."
            )
    required_coverage = float(external.get("required_expert_review_coverage", 1.0))
    if expert["review_coverage"] < required_coverage or not expert["complete"]:
        blockers.append(
            f"Expert review coverage is {expert['review_coverage']:.3f}; "
            f"required {required_coverage:.3f} with no unresolved reviews."
        )
    freeze_approval = external.get("freeze_approval")
    approved = _approval(freeze_approval)
    if not approved:
        blockers.append("External test-set freeze approval is unavailable.")
    membership = [
        {
            "sample_id": row.get("sample_id"),
            "specimen_id": row.get("specimen_id"),
            "image_sha256": row.get("original_image_sha256"),
            "label_sha256": row.get("label_sha256"),
        }
        for row in sorted(samples, key=lambda item: item.get("sample_id", ""))
    ]
    unique_blockers = list(dict.fromkeys(blockers))
    ready = bool(samples and not unique_blockers)
    return {
        "status": "pass" if ready else "blocked",
        "ready": ready,
        "specimen_count": len(specimen_set),
        "image_count": len(samples),
        "capture_session_count": len(sessions),
        "membership_hash": stable_json_sha256(membership) if samples else None,
        "membership_frozen": bool(ready and approved),
        "hash_error_count": sample_hash_errors,
        "expert_review": expert,
        "duplicates": duplicates,
        "blockers": unique_blockers,
        "warnings": warnings,
        "paths": {
            "specimens": str(specimen_path) if specimen_path else None,
            "sessions": str(session_path) if session_path else None,
            "samples": str(sample_path) if sample_path else None,
            "expert_review": str(review_path) if review_path else None,
        },
    }


def validate_initialization_policy(
    approval_path: Path | None,
    training_config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    approval = _read_json(approval_path)
    blockers = []
    policy = approval.get("initialization_policy")
    if not approval:
        blockers.append("Final model initialization approval is unavailable.")
    elif approval.get("example_only", False):
        blockers.append("Example training approval is not a real approval.")
    if policy not in TRAINING_INITIALIZATIONS:
        blockers.append("Final model initialization policy is missing or unsupported.")
    if not _approval(approval):
        blockers.append("Final model initialization policy lacks human approval.")
    base_model = str(training_config.get("base_model") or "")
    normalized = base_model.replace("\\", "/").casefold()
    if normalized.endswith("backend/best.pt") or policy == "initialize_from_backend_best_pt":
        blockers.append("backend/best.pt is prohibited as final initialization.")
    if policy == "official_generic_pretrained_initialization":
        for field in ("base_model", "base_model_source", "base_model_license"):
            if not training_config.get(field):
                blockers.append(f"Generic initialization field is missing: {field}")
    return {
        "status": "pass" if not blockers else "blocked",
        "approved": bool(not blockers),
        "policy": policy,
        "base_model": base_model or None,
        "legacy_model_excluded": not normalized.endswith("backend/best.pt"),
        "blockers": list(dict.fromkeys(blockers)),
    }


def validate_training_configuration(
    training: dict[str, Any],
    context: dict[str, bool],
) -> dict[str, Any]:
    blockers = []
    if training.get("enabled", False):
        blockers.append("Gate 0 training configuration must remain disabled.")
    if training.get("dataset_route") not in ROUTES:
        blockers.append("Training dataset route is unavailable.")
    base_model = str(training.get("base_model") or "").replace("\\", "/").casefold()
    if base_model.endswith("backend/best.pt"):
        blockers.append("Training configuration cannot use backend/best.pt.")
    if training.get("test_data_used_for_training") is not False:
        blockers.append("Training configuration must exclude test data.")
    if training.get("test_data_used_for_threshold_selection") is not False:
        blockers.append(
            "Training configuration must exclude test data from threshold selection."
        )
    for field in (
        "image_size",
        "batch_size",
        "epochs",
        "patience",
        "device",
        "workers",
        "augmentation",
        "class_handling",
        "output_directory",
    ):
        if training.get(field) in (None, ""):
            blockers.append(f"Training decision remains unset: {field}")
    required_context = {
        "taxonomy_approved": "Taxonomy is not approved.",
        "metric_approved": "Primary metric is not approved.",
        "environment_ready": "Isolated ML environment is not ready.",
        "development_split_approved": "Development split is not approved.",
        "test_set_frozen": "Independent test set is not frozen.",
        "initialization_approved": "Initialization policy is not approved.",
        "test_excluded_from_training": "Test data exclusion is not proven.",
        "test_excluded_from_threshold_selection": (
            "Test data exclusion from threshold selection is not proven."
        ),
    }
    for key, reason in required_context.items():
        if not context.get(key, False):
            blockers.append(reason)
    return {
        "status": "pass" if not blockers else "blocked",
        "ready": bool(not blockers),
        "enabled": bool(training.get("enabled", False)),
        "blockers": list(dict.fromkeys(blockers)),
    }


def _environment_status(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    location = report.get("environment_location_category")
    ready = bool(
        summary.get("final_model_runtime_ready")
        and location == "external_isolated"
        and report.get("environment_variables_recorded") is False
    )
    blockers = []
    if not report:
        blockers.append("Isolated ML environment evidence is unavailable.")
    if location != "external_isolated":
        blockers.append("ML environment is not recorded as externally isolated.")
    if not summary.get("final_model_runtime_ready"):
        blockers.append("Required ML runtime dependencies are incomplete.")
    if report.get("environment_variables_recorded") is not False:
        blockers.append("Environment evidence does not confirm secret-safe output.")
    return {
        "status": "pass" if ready else "blocked",
        "ready": ready,
        "location_category": location,
        "python_version": report.get("system", {}).get("python_version"),
        "cuda": report.get("system", {}).get("cuda", {}),
        "blockers": list(dict.fromkeys(blockers)),
    }


def _development_rows(split_directory: Path | None) -> list[dict[str, str]]:
    path = split_directory / "grouped_split_manifest.csv" if split_directory else None
    rows, _ = _read_csv(path)
    return rows


def _taxonomy_visualization_decided(path: Path | None) -> tuple[bool, str]:
    rows, headers = _read_csv(path)
    if "eligible_for_visualization" not in headers:
        return False, "Taxonomy lacks explicit visualization eligibility decisions."
    undecided = [
        row.get("class_name", "")
        for row in rows
        if _bool(row.get("eligible_for_visualization")) is None
    ]
    if undecided:
        return False, f"Visualization eligibility is undecided for: {undecided}"
    return True, "Visualization eligibility is explicitly decided."


def evaluate_phase3b_gate(config: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["_repo_root"]).resolve()
    route = config.get("dataset_route")
    route_approval = config.get("dataset_route_approval")
    route_valid = route in ROUTES
    route_approved = bool(route_valid and _approval(route_approval, route=route))

    environment_path = resolve_path(config.get("ml_environment_report"), repo_root)
    environment = _environment_status(_read_json(environment_path))
    taxonomy_path = resolve_path(config.get("taxonomy"), repo_root)
    taxonomy_approval = resolve_path(config.get("taxonomy_approval"), repo_root)
    dataset_yaml = resolve_path(config.get("dataset_yaml"), repo_root)
    if taxonomy_path and dataset_yaml:
        taxonomy = validate_taxonomy_approval(
            taxonomy_path, taxonomy_approval, dataset_yaml
        )
    else:
        taxonomy = {"summary": {"taxonomy_approved": False, "blockers": [
            "Taxonomy inputs are unavailable."
        ]}, "taxonomy": {"sha256": None}}
    visualization_decided, visualization_reason = (
        _taxonomy_visualization_decided(taxonomy_path)
        if taxonomy_path
        else (False, "Taxonomy is unavailable.")
    )
    taxonomy_approved = bool(
        taxonomy["summary"]["taxonomy_approved"] and visualization_decided
    )

    protocol_path = resolve_path(config.get("test_protocol"), repo_root)
    metric_path = resolve_path(config.get("metric_approval"), repo_root)
    if taxonomy_path and protocol_path:
        metric = validate_metric_approval(
            metric_path, taxonomy_path, protocol_path
        )
    else:
        metric = {"summary": {"metric_approved": False, "blockers": [
            "Metric approval inputs are unavailable."
        ]}, "approval": {}}
    metric_approved = bool(metric["summary"]["metric_approved"])

    split_directory = resolve_path(config.get("grouped_split_directory"), repo_root)
    split_summary = _read_json(
        split_directory / "split_summary.json" if split_directory else None
    )
    split_lock = _read_json(
        split_directory / "split_lock.json" if split_directory else None
    )
    split_leakage = _read_json(
        split_directory / "split_leakage_audit.json"
        if split_directory
        else None
    )
    development_rows = _development_rows(split_directory)

    dataset_audit = _read_json(
        resolve_path(config.get("dataset_audit"), repo_root)
    )
    candidate_rows = dataset_audit.get("near_duplicate_candidates", [])
    review_rows, _ = _read_csv(
        resolve_path(config.get("near_duplicate_review"), repo_root)
    )
    duplicate_policy = config.get("duplicate_policy")
    duplicates = duplicate_relationships(
        candidate_rows, review_rows, duplicate_policy
    )
    duplicate_decision_approved = bool(
        duplicate_policy in DUPLICATE_POLICIES
        and _approval(config.get("duplicate_policy_approval"))
    )
    duplicates_ready = bool(
        duplicate_decision_approved
        and (
            duplicate_policy == "conservative_union"
            or duplicates["unresolved_count"] == 0
        )
    )

    external = validate_external_holdout(
        config.get("external_holdout") or {},
        repo_root,
        development_rows,
    )
    metadata = split_summary.get("metadata_coverage", {})
    if (
        route == "external_specimen_holdout"
        and float(metadata.get("specimen_id_coverage", 0)) < 1.0
        and "legacy_specimen_identity_unavailable"
        not in (route_approval or {}).get("limitations_acknowledged", [])
    ):
        route_approved = False
    lineage_ready = bool(
        route == "recovered_legacy_lineage"
        and float(metadata.get("specimen_id_coverage", 0)) >= 1.0
        and float(metadata.get("recording_id_coverage", 0)) >= 1.0
        and split_summary.get("split_approved")
        and split_lock.get("frozen")
        and duplicates["unresolved_count"] == 0
    )
    external_route_ready = bool(
        route == "external_specimen_holdout"
        and route_approved
        and external["ready"]
        and duplicates_ready
    )
    final_test_ready = external_route_ready if route == "external_specimen_holdout" else lineage_ready
    expert_ready = (
        external["expert_review"]["complete"]
        if route == "external_specimen_holdout"
        else bool(
            float(metadata.get("expert_review_coverage", 0)) >= 1.0
            and lineage_ready
        )
    )

    development_split_approved = bool(
        route_approved
        and taxonomy_approved
        and duplicates_ready
        and split_leakage.get("passes")
        and (
            route == "external_specimen_holdout"
            or (
                route == "recovered_legacy_lineage"
                and split_summary.get("split_approved")
                and split_lock.get("frozen")
            )
        )
    )

    training_config_path = resolve_path(config.get("training_config"), repo_root)
    training_config = _read_json(training_config_path)
    initialization_path = resolve_path(
        config.get("training_approval"), repo_root
    )
    initialization = validate_initialization_policy(
        initialization_path, training_config, repo_root
    )
    test_excluded = bool(
        route == "external_specimen_holdout"
        and external["membership_frozen"]
        and not external["duplicates"]["exact_cross_set_sample_ids"]
        and not external["duplicates"]["perceptual_cross_set_candidates"]
    ) or bool(
        route == "recovered_legacy_lineage"
        and split_lock.get("frozen")
        and split_summary.get("split_approved")
    )
    training = validate_training_configuration(
        training_config,
        {
            "taxonomy_approved": taxonomy_approved,
            "metric_approved": metric_approved,
            "environment_ready": environment["ready"],
            "development_split_approved": development_split_approved,
            "test_set_frozen": final_test_ready,
            "initialization_approved": initialization["approved"],
            "test_excluded_from_training": test_excluded,
            "test_excluded_from_threshold_selection": test_excluded,
        },
    )

    gates = [
        _gate(
            "repository_preservation",
            "Prior-phase repository preservation",
            "pass",
            "Gate orchestration is read-only with respect to protected artifacts.",
        ),
        _gate(
            "environment_readiness",
            "Externally isolated ML environment",
            environment["status"],
            "; ".join(environment["blockers"]) or "Isolated environment is ready.",
            environment,
        ),
        _gate(
            "taxonomy_approval",
            "Expert-approved defect taxonomy",
            "pass" if taxonomy_approved else "blocked",
            (
                "Taxonomy and visualization decisions are approved."
                if taxonomy_approved
                else "; ".join(
                    taxonomy["summary"].get("blockers", []) + [visualization_reason]
                )
            ),
        ),
        _gate(
            "dataset_route",
            "Approved dataset route",
            "pass" if route_approved else "blocked",
            (
                f"Approved route: {route}."
                if route_approved
                else "Dataset route is unselected, unsupported, or unapproved."
            ),
        ),
        _gate(
            "duplicate_handling",
            "Approved duplicate policy",
            "pass" if duplicates_ready else "blocked",
            (
                f"{duplicates['method']} is approved."
                if duplicates_ready
                else "Duplicate policy is unapproved or unresolved candidates remain."
            ),
            duplicates,
        ),
        _gate(
            "external_holdout",
            "Independent external specimen holdout",
            (
                external["status"]
                if route == "external_specimen_holdout"
                else "not_applicable"
                if route == "recovered_legacy_lineage"
                else "unavailable"
            ),
            (
                "External holdout is ready."
                if external["ready"]
                else "; ".join(external["blockers"])
                if route == "external_specimen_holdout"
                else "Route not selected."
            ),
            external,
        ),
        _gate(
            "legacy_lineage",
            "Recovered legacy specimen lineage",
            (
                "pass"
                if lineage_ready
                else "blocked"
                if route == "recovered_legacy_lineage"
                else "not_applicable"
                if route == "external_specimen_holdout"
                else "unavailable"
            ),
            (
                "Legacy lineage and split are approved."
                if lineage_ready
                else "Genuine complete specimen/recording lineage is unavailable."
            ),
        ),
        _gate(
            "development_split",
            "Approved development split",
            "pass" if development_split_approved else "blocked",
            (
                "Development split is approved for the selected route."
                if development_split_approved
                else "Development split prerequisites are incomplete."
            ),
        ),
        _gate(
            "final_test_set",
            "Frozen independent test set",
            "pass" if final_test_ready else "blocked",
            (
                "Final test membership is independent and frozen."
                if final_test_ready
                else "Final test membership is not ready and frozen."
            ),
        ),
        _gate(
            "expert_ground_truth",
            "Expert-reviewed final test ground truth",
            "pass" if expert_ready else "blocked",
            (
                "Expert review requirement is satisfied."
                if expert_ready
                else "Expert-reviewed final test masks are incomplete."
            ),
        ),
        _gate(
            "primary_metric",
            "Approved operational 90 percent metric",
            "pass" if metric_approved else "blocked",
            (
                "Primary metric approval is valid."
                if metric_approved
                else "; ".join(metric["summary"].get("blockers", []))
            ),
        ),
        _gate(
            "initialization_policy",
            "Approved clean model initialization",
            initialization["status"],
            "; ".join(initialization["blockers"]) or "Initialization is approved.",
            initialization,
        ),
        _gate(
            "training_configuration",
            "Clean training configuration",
            training["status"],
            "; ".join(training["blockers"]) or "Training configuration is ready.",
            training,
        ),
    ]
    blocking = [gate for gate in gates if gate["status"] in {"blocked", "unavailable"}]
    authorized = bool(
        not blocking
        and training["ready"]
        and initialization["legacy_model_excluded"]
        and final_test_ready
        and expert_ready
    )
    development_lock = {
        "lock_type": "development_split",
        "dataset_route": route,
        "source_split_id": split_lock.get("split_id"),
        "manifest_hash": split_summary.get("manifest_hash"),
        "duplicate_method": duplicates["method"],
        "approved": development_split_approved,
        "frozen": development_split_approved,
        "approval_status": "approved" if development_split_approved else "unavailable",
    }
    development_lock["lock_hash"] = stable_json_sha256(development_lock)
    test_lock = {
        "lock_type": "final_test_split",
        "dataset_route": route,
        "membership_hash": (
            external["membership_hash"]
            if route == "external_specimen_holdout"
            else split_lock.get("test_set_hash")
        ),
        "specimen_count": (
            external["specimen_count"]
            if route == "external_specimen_holdout"
            else None
        ),
        "image_count": (
            external["image_count"]
            if route == "external_specimen_holdout"
            else None
        ),
        "approved": final_test_ready,
        "frozen": final_test_ready,
        "approval_status": "approved" if final_test_ready else "unavailable",
    }
    test_lock["lock_hash"] = stable_json_sha256(test_lock)
    return {
        "schema_version": "1.0",
        "audit_type": "phase3b_gate0",
        "created_utc": utc_timestamp(),
        "dataset_route": {
            "selected": route if route_valid else None,
            "approved": route_approved,
            "allowed_routes": sorted(ROUTES),
        },
        "environment": environment,
        "taxonomy": {
            "approved": taxonomy_approved,
            "taxonomy_sha256": taxonomy.get("taxonomy", {}).get("sha256"),
            "visualization_eligibility_decided": visualization_decided,
        },
        "metric": {
            "approved": metric_approved,
            **metric.get("approval", {}),
        },
        "duplicates": duplicates,
        "external_holdout": external,
        "legacy_lineage": {
            "ready": lineage_ready,
            "metadata_coverage": metadata,
        },
        "initialization": initialization,
        "training": training,
        "development_split_lock": development_lock,
        "test_split_lock": test_lock,
        "gates": gates,
        "summary": {
            "status": "pass" if authorized else "blocked",
            "authorized_for_clean_training": authorized,
            "gate_count": len(gates),
            "passed_gate_count": sum(gate["status"] == "pass" for gate in gates),
            "warning_gate_count": sum(gate["status"] == "warning" for gate in gates),
            "blocked_gate_count": len(blocking),
            "training_occurred": False,
            "final_metrics_calculated": False,
            "claim_90_percent_available": False,
            "blockers": [gate["reason"] for gate in blocking],
        },
        "claim_boundary": (
            "Gate 0 validates readiness only. It does not train a model, run final "
            "evaluation, or support a 90 percent performance claim."
        ),
    }


def _pending_taxonomy(source: Path, destination: Path) -> None:
    rows, headers = _read_csv(source)
    fields = list(headers)
    if "eligible_for_visualization" not in fields:
        insert_at = (
            fields.index("eligible_for_3d_mapping")
            if "eligible_for_3d_mapping" in fields
            else len(fields)
        )
        fields.insert(insert_at, "eligible_for_visualization")
    for row in rows:
        row.setdefault("eligible_for_visualization", "")
    atomic_write_csv(destination, rows, fields)


def _copy_or_header(
    source: Path | None,
    destination: Path,
    fields: Iterable[str],
) -> None:
    if source and source.is_file():
        shutil.copy2(source, destination)
    else:
        atomic_write_csv(destination, [], fields)


def write_human_input_pack(
    config: dict[str, Any],
    report: dict[str, Any],
    output: Path,
) -> list[Path]:
    repo_root = Path(config["_repo_root"]).resolve()
    pack = output / "human_input_pack"
    pack.mkdir(parents=True, exist_ok=True)
    taxonomy = resolve_path(config.get("taxonomy"), repo_root)
    protocol = resolve_path(config.get("test_protocol"), repo_root)
    taxonomy_hash = sha256_file(taxonomy) if taxonomy and taxonomy.is_file() else None
    protocol_hash = sha256_file(protocol) if protocol and protocol.is_file() else None
    files = {
        "taxonomy": pack / "defect_taxonomy.pending.csv",
        "taxonomy_approval": pack / "taxonomy_approval.pending.json",
        "metric": pack / "defect_metric_approval.pending.json",
        "route": pack / "dataset_route_decision.pending.json",
        "duplicates": pack / "near_duplicate_review.pending.csv",
        "specimens": pack / "external_holdout_specimens.pending.csv",
        "sessions": pack / "external_holdout_capture_sessions.pending.csv",
        "samples": pack / "external_holdout_samples.pending.csv",
        "expert": pack / "external_holdout_expert_review.pending.csv",
        "training": pack / "defect_training_approval.pending.json",
        "checklist": pack / "gate_closure_checklist.md",
    }
    if taxonomy and taxonomy.is_file():
        _pending_taxonomy(taxonomy, files["taxonomy"])
    else:
        atomic_write_csv(files["taxonomy"], [], ())
    atomic_write_json(
        files["taxonomy_approval"],
        {
            "taxonomy_id": None,
            "taxonomy_version": None,
            "taxonomy_file": "defect_taxonomy.pending.csv",
            "taxonomy_sha256": None,
            "approved_by": None,
            "approver_role": None,
            "approval_date": None,
            "approval_scope": None,
            "primary_visible_defect_definition": None,
            "primary_claim_metric": None,
            "claim_threshold": 0.90,
            "precision_floor": None,
            "notes": "Complete taxonomy first, then insert its SHA-256.",
            "example_only": True,
        },
    )
    atomic_write_json(
        files["metric"],
        {
            "metric_id": None,
            "metric_version": None,
            "primary_metric": "binary_visible_defect_f1",
            "claim_threshold": 0.90,
            "precision_floor": None,
            "aggregation_level": "specimen_grouped",
            "approved_by": None,
            "approver_role": None,
            "approval_date": None,
            "taxonomy_sha256": taxonomy_hash,
            "test_protocol_sha256": protocol_hash,
            "notes": "Recommendation only until formally approved.",
            "example_only": True,
        },
    )
    atomic_write_json(
        files["route"],
        {
            "dataset_route": None,
            "allowed_routes": sorted(ROUTES),
            "duplicate_policy": None,
            "approved_by": None,
            "approver_role": None,
            "approval_date": None,
            "limitations_acknowledged": [],
            "notes": "",
            "example_only": True,
        },
    )
    review = resolve_path(config.get("near_duplicate_review"), repo_root)
    _copy_or_header(
        review,
        files["duplicates"],
        (
            "candidate_group_id",
            "image_a",
            "image_b",
            "hash_a",
            "hash_b",
            "perceptual_distance",
            "review_decision",
            "reviewer_id",
            "review_date",
            "reason",
            "notes",
        ),
    )
    external = config.get("external_holdout") or {}
    _copy_or_header(
        resolve_path(external.get("specimens_manifest"), repo_root),
        files["specimens"],
        SPECIMEN_FIELDS,
    )
    _copy_or_header(
        resolve_path(external.get("capture_sessions_manifest"), repo_root),
        files["sessions"],
        CAPTURE_FIELDS,
    )
    _copy_or_header(
        resolve_path(external.get("samples_manifest"), repo_root),
        files["samples"],
        SAMPLE_FIELDS,
    )
    _copy_or_header(
        resolve_path(external.get("expert_review"), repo_root),
        files["expert"],
        EXTERNAL_EXPERT_FIELDS,
    )
    atomic_write_json(
        files["training"],
        {
            "training_approval_id": None,
            "training_approval_version": None,
            "initialization_policy": None,
            "approved_by": None,
            "approver_role": None,
            "approval_date": None,
            "notes": (
                "Allowed: official_generic_pretrained_initialization or "
                "training_from_scratch. backend/best.pt is prohibited."
            ),
            "example_only": True,
        },
    )
    checklist = """# Phase 3B Gate 0 Closure Checklist

| File | Completed by | Mandatory evidence | Blocked decision |
|---|---|---|---|
| defect_taxonomy.pending.csv | Gem expert and research team | Meaning and eligibility for every class | Taxonomy and no-cut validity |
| taxonomy_approval.pending.json | Supervisor or authorized expert | Taxonomy hash, identity, date, scope | Taxonomy approval |
| defect_metric_approval.pending.json | Supervisor or research team | Metric, threshold, protocol/taxonomy hashes | Operational 90% definition |
| dataset_route_decision.pending.json | Supervisor and research team | Selected route and duplicate policy | Dataset strategy |
| near_duplicate_review.pending.csv | Dataset reviewer | Decision, reviewer, date for every pair | Human-reviewed grouping |
| external_holdout_specimens.pending.csv | Data custodian | New specimen IDs and legacy exclusion | Test independence |
| external_holdout_capture_sessions.pending.csv | Capture operator | Camera/session provenance | Capture lineage |
| external_holdout_samples.pending.csv | Dataset manager | Paths, hashes, frozen status, no-use declarations | Test membership |
| external_holdout_expert_review.pending.csv | Gem expert | Reviewed/corrected masks and disagreements | Ground truth |
| defect_training_approval.pending.json | Supervisor | Clean initialization choice | Model initialization |

After completion, replace the example paths in `phase3b_gate.example.json` with
the reviewed files, set `example_only` to false only when actually approved,
and rerun:

`python backend/research_benchmark.py phase3b-gate --config research/configs/phase3b_gate.example.json --output research_evidence/runs/phase3b-gate-current`

Do not train until the command reports authorization.
"""
    atomic_write_text(files["checklist"], checklist)
    return list(files.values())


def _summary_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Phase 3B Gate 0",
        "",
        f"- Status: {summary['status']}",
        f"- Authorized for clean training: {str(summary['authorized_for_clean_training']).lower()}",
        f"- Gates passed: {summary['passed_gate_count']}/{summary['gate_count']}",
        f"- Blocked/unavailable gates: {summary['blocked_gate_count']}",
        "- Training occurred: false",
        "- Final metrics calculated: false",
        "- 90 percent claim available: false",
        "",
        "| Gate | Status | Reason |",
        "|---|---|---|",
    ]
    for gate in report["gates"]:
        lines.append(f"| {gate['title']} | {gate['status']} | {gate['reason']} |")
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def run_phase3b_gate(
    config: dict[str, Any],
    output_override: str | Path | None = None,
    command: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    repo_root = Path(config["_repo_root"]).resolve()
    output = resolve_path(
        output_override or config.get("output_directory"), repo_root
    )
    if output is None:
        raise ValueError("An output directory is required.")
    output.mkdir(parents=True, exist_ok=True)
    report = evaluate_phase3b_gate(config)
    files = {
        "report": output / "phase3b_gate.json",
        "summary": output / "gate_summary.csv",
        "development_lock": output / "development_split_lock.json",
        "test_lock": output / "test_split_lock.json",
        "split_summary": output / "split_summary.json",
        "leakage": output / "split_leakage_audit.json",
        "grouped_manifest": output / "grouped_split_manifest.csv",
        "yaml": output / "data_grouped.yaml",
        "markdown": output / "audit_summary.md",
        "manifest": output / "manifest.json",
    }
    atomic_write_json(files["report"], report)
    atomic_write_csv(
        files["summary"],
        report["gates"],
        ("gate", "title", "status", "passed", "reason"),
    )
    atomic_write_json(files["development_lock"], report["development_split_lock"])
    atomic_write_json(files["test_lock"], report["test_split_lock"])
    atomic_write_json(
        files["split_summary"],
        {
            "dataset_route": report["dataset_route"],
            "development_split": report["development_split_lock"],
            "test_split": report["test_split_lock"],
            "authorization": report["summary"]["authorized_for_clean_training"],
        },
    )
    atomic_write_json(
        files["leakage"],
        {
            "duplicate_policy": report["duplicates"],
            "external_duplicates": report["external_holdout"]["duplicates"],
            "passes": bool(
                report["summary"]["authorized_for_clean_training"]
            ),
            "claim_boundary": (
                "A blocked Gate 0 result is not evidence of leakage-free "
                "specimen generalization."
            ),
        },
    )
    source_rows = _development_rows(
        resolve_path(config.get("grouped_split_directory"), repo_root)
    )
    atomic_write_csv(
        files["grouped_manifest"],
        source_rows,
        tuple(source_rows[0]) if source_rows else ("sample_id", "assigned_split"),
    )
    atomic_write_text(
        files["yaml"],
        "# Gate 0 manifest only. No derived dataset was materialized.\n"
        "path: null\ntrain: null\nval: null\ntest: null\n",
    )
    atomic_write_text(files["markdown"], _summary_markdown(report))
    human_files = write_human_input_pack(config, report, output)
    outputs = [
        ("gate_report", files["report"]),
        ("gate_summary", files["summary"]),
        ("development_lock", files["development_lock"]),
        ("test_lock", files["test_lock"]),
        ("split_summary", files["split_summary"]),
        ("leakage_audit", files["leakage"]),
        ("grouped_manifest", files["grouped_manifest"]),
        ("dataset_yaml", files["yaml"]),
        ("markdown", files["markdown"]),
    ]
    outputs.extend((f"human_input_{index}", path) for index, path in enumerate(human_files))
    write_manifest(
        path=files["manifest"],
        repo_root=repo_root,
        tool_name="quartz_phase3b_gate0",
        command=command,
        tool_sources=(
            "backend/research/phase3b_gate.py",
            "backend/research/metric_approval.py",
            "backend/research/readiness_common.py",
            "backend/research_benchmark.py",
        ),
        inputs=(
            ("configuration", config.get("_config_path")),
            ("taxonomy", config.get("taxonomy")),
            ("taxonomy_approval", config.get("taxonomy_approval")),
            ("metric_approval", config.get("metric_approval")),
            ("environment", config.get("ml_environment_report")),
            ("dataset_audit", config.get("dataset_audit")),
            ("duplicate_review", config.get("near_duplicate_review")),
            ("training_config", config.get("training_config")),
            ("training_approval", config.get("training_approval")),
        ),
        outputs=outputs,
        extra={
            "training_occurred": False,
            "final_metrics_calculated": False,
            "environment_variables_recorded": False,
        },
    )
    return report, files
