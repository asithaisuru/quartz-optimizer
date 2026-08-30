"""Phase 3B Gate 2 approval and authorization validation."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .candidate_dataset import (
    audit_candidate_dataset,
    candidate_source_root,
    plan_development_split,
    validate_gem_identity_approval,
)
from .common import atomic_write_json, sha256_file, utc_timestamp
from .metric_approval import validate_metric_approval
from .readiness_common import (
    atomic_write_csv,
    atomic_write_text,
    load_json_config,
    resolve_path,
    write_manifest,
)
from .taxonomy_validation import validate_taxonomy_approval


APPROVAL_DATE = "2026-07-29"
ROLE_IDENTIFIERS = {
    "Primary Supervisor",
    "Co-Supervisor / Gem Expert",
    "Research Team",
}
ROLE_FIELDS = {
    "approved_by",
    "approver_role",
    "confirmed_by",
    "record_prepared_by",
    "required_reviewer_role",
    "capture_operator_role",
    "initial_mask_creator_role",
}
FORBIDDEN_IDENTITY_KEYS = {
    "name",
    "email",
    "email_address",
    "signature",
    "contact",
    "contact_details",
    "phone",
    "phone_number",
}
PERSONAL_MARKER = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Eng)\.\s", re.IGNORECASE)
EMAIL_MARKER = re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b")


def load_gate2_config(
    path: str | Path,
    *,
    source_override: str | Path | None = None,
    zip_override: str | Path | None = None,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    config = load_json_config(path, output_override=output_override)
    if source_override is not None:
        config["_source_override"] = str(source_override)
    if zip_override is not None:
        config["_zip_override"] = str(zip_override)
    return config


def _load_json(path: Path | None) -> tuple[dict[str, Any], list[str]]:
    if path is None or not path.is_file():
        return {}, ["Approval file is unavailable."]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"Approval file is invalid: {exc}"]
    if not isinstance(value, dict):
        return {}, ["Approval file must contain a JSON object."]
    return value, []


def _privacy_issues(value: Any, location: str = "$") -> list[str]:
    issues = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            child = f"{location}.{key}"
            if normalized in FORBIDDEN_IDENTITY_KEYS or normalized.endswith(
                ("_email", "_signature", "_contact", "_phone")
            ):
                issues.append(f"Personal-data field is prohibited: {child}")
            if key in ROLE_FIELDS and item not in (None, ""):
                if item not in ROLE_IDENTIFIERS:
                    issues.append(
                        f"Identity value must be a role-only identifier: {child}"
                    )
            issues.extend(_privacy_issues(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(_privacy_issues(item, f"{location}[{index}]"))
    elif isinstance(value, str):
        if EMAIL_MARKER.search(value):
            issues.append(f"Email address is prohibited: {location}")
        if PERSONAL_MARKER.search(value):
            issues.append(f"Personal name marker is prohibited: {location}")
    return issues


def _approval_path(config: dict[str, Any], key: str) -> Path | None:
    return resolve_path(config.get(key), config["_repo_root"])


def _validate_route(path: Path | None) -> dict[str, Any]:
    value, blockers = _load_json(path)
    if value:
        if value.get("example_only", False):
            blockers.append("Example route records are not approvals.")
        if value.get("approved") is not True:
            blockers.append("Dataset route is not approved.")
        if value.get("dataset_route") != "external_specimen_holdout":
            blockers.append("Dataset route must be external_specimen_holdout.")
        if value.get("approval_date") != APPROVAL_DATE:
            blockers.append("Dataset route approval date does not match.")
        for field in ("approved_by", "approver_role"):
            if value.get(field) != "Primary Supervisor":
                blockers.append(f"Dataset route {field} must be Primary Supervisor.")
        blockers.extend(_privacy_issues(value))
    blockers = list(dict.fromkeys(blockers))
    return {
        "path": str(path) if path else None,
        "approved": bool(value and not blockers),
        "blockers": blockers,
    }


def _validate_initialization(
    approval_path: Path | None,
    protocol_path: Path | None,
) -> dict[str, Any]:
    value, blockers = _load_json(approval_path)
    protocol_hash = (
        sha256_file(protocol_path)
        if protocol_path is not None and protocol_path.is_file()
        else None
    )
    if protocol_hash is None:
        blockers.append("Clean-training protocol is unavailable.")
    if value:
        if value.get("example_only", False):
            blockers.append("Example initialization records are not approvals.")
        if value.get("approved") is not True:
            blockers.append("Initialization policy is not approved.")
        if (
            value.get("initialization_policy")
            != "official_generic_pretrained_initialization"
        ):
            blockers.append("Initialization policy is not the approved clean policy.")
        if value.get("legacy_model_initialization_allowed") is not False:
            blockers.append("backend/best.pt must be rejected as initialization.")
        if value.get("clean_training_protocol_sha256") != protocol_hash:
            blockers.append("Initialization approval protocol hash does not match.")
        if value.get("approval_date") != APPROVAL_DATE:
            blockers.append("Initialization approval date does not match.")
        for field in ("approved_by", "approver_role"):
            if value.get(field) != "Primary Supervisor":
                blockers.append(
                    f"Initialization {field} must be Primary Supervisor."
                )
        blockers.extend(_privacy_issues(value))
    blockers = list(dict.fromkeys(blockers))
    return {
        "path": str(approval_path) if approval_path else None,
        "approved": bool(value and not blockers),
        "policy": value.get("initialization_policy"),
        "clean_training_protocol_sha256": protocol_hash,
        "legacy_model_initialization_allowed": value.get(
            "legacy_model_initialization_allowed"
        ),
        "blockers": blockers,
    }


def _validate_external_concept(path: Path | None) -> dict[str, Any]:
    value, blockers = _load_json(path)
    unavailable_fields = (
        "new_specimen_count",
        "capture_operator_role",
        "initial_mask_creator_role",
        "capture_date",
        "annotation_review_completion_date",
    )
    if value:
        if value.get("example_only", False):
            blockers.append("Example external-test records are not approvals.")
        if value.get("concept_approved") is not True:
            blockers.append("External-test concept is not approved.")
        if value.get("approval_date") != APPROVAL_DATE:
            blockers.append("External-test concept approval date does not match.")
        if value.get("approved_by") != "Primary Supervisor":
            blockers.append("External-test concept approver must be Primary Supervisor.")
        if value.get("record_prepared_by") != "Research Team":
            blockers.append("External-test record must be prepared by Research Team.")
        if (
            value.get("required_reviewer_role")
            != "Co-Supervisor / Gem Expert"
        ):
            blockers.append(
                "External-test reviewer must be Co-Supervisor / Gem Expert."
            )
        if any(value.get(field) is not None for field in unavailable_fields):
            blockers.append(
                "Unavailable external-test operational details must remain null."
            )
        if value.get("operational_ready") is not False:
            blockers.append("External-test operational readiness must remain blocked.")
        if value.get("training_authorized") is not False:
            blockers.append("Training must remain unauthorized.")
        blockers.extend(_privacy_issues(value))
    blockers = list(dict.fromkeys(blockers))
    return {
        "path": str(path) if path else None,
        "concept_approved": bool(value and not blockers),
        "operational_ready": False,
        "unavailable_fields": list(unavailable_fields),
        "blockers": blockers,
    }


def _taxonomy_role_issues(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return ["Approved taxonomy file is unavailable."]
    issues = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("confirmed_by") != "Co-Supervisor / Gem Expert":
            issues.append(
                f"Class {row.get('class_name', '')} confirmer must be role-only."
            )
        if row.get("confirmed_date") != APPROVAL_DATE:
            issues.append(
                f"Class {row.get('class_name', '')} confirmation date does not match."
            )
        issues.extend(_privacy_issues(row))
    return list(dict.fromkeys(issues))


def validate_gate2_approvals(config: dict[str, Any]) -> dict[str, Any]:
    candidate = audit_candidate_dataset(config)
    source_root = candidate_source_root(config)
    if source_root is None:
        raise ValueError("Candidate source root is unavailable.")
    dataset_yaml = source_root / str(config.get("dataset_yaml", "data.yaml"))
    taxonomy_path = _approval_path(config, "candidate_taxonomy")
    taxonomy_approval_path = _approval_path(
        config,
        "candidate_taxonomy_approval",
    )
    taxonomy = validate_taxonomy_approval(
        taxonomy_path,
        taxonomy_approval_path,
        dataset_yaml,
    )
    taxonomy_role_issues = _taxonomy_role_issues(taxonomy_path)
    taxonomy_approval_value, taxonomy_privacy = _load_json(
        taxonomy_approval_path
    )
    taxonomy_privacy.extend(_privacy_issues(taxonomy_approval_value))
    taxonomy_approved = bool(
        taxonomy["summary"]["taxonomy_approved"]
        and not taxonomy_role_issues
        and not taxonomy_privacy
    )

    identity_path = _approval_path(config, "gem_identity_approval")
    identity = validate_gem_identity_approval(
        identity_path,
        repo_root=config["_repo_root"],
        source_manifest_sha256=candidate["source"][
            "directory_manifest_sha256"
        ],
    )
    identity_value, identity_privacy = _load_json(identity_path)
    identity_privacy.extend(_privacy_issues(identity_value))
    if identity_value.get("approval_date") != APPROVAL_DATE:
        identity_privacy.append("Gem identity approval date does not match.")
    if identity_value.get("approved_by") != "Co-Supervisor / Gem Expert":
        identity_privacy.append(
            "Gem identity approver must be Co-Supervisor / Gem Expert."
        )
    identity_approved = bool(identity["approved"] and not identity_privacy)

    route = _validate_route(_approval_path(config, "dataset_route_approval"))
    evaluation_protocol = _approval_path(config, "evaluation_protocol")
    metric_path = _approval_path(config, "metric_approval")
    metric = validate_metric_approval(
        metric_path,
        taxonomy_path,
        evaluation_protocol,
    )
    metric_value, metric_privacy = _load_json(metric_path)
    metric_privacy.extend(_privacy_issues(metric_value))
    if metric_value.get("approval_date") != APPROVAL_DATE:
        metric_privacy.append("Metric approval date does not match.")
    if metric_value.get("approved_by") != "Primary Supervisor":
        metric_privacy.append("Metric approver must be Primary Supervisor.")
    metric_approved = bool(
        metric["summary"]["metric_approved"] and not metric_privacy
    )

    initialization = _validate_initialization(
        _approval_path(config, "training_approval"),
        _approval_path(config, "clean_training_protocol"),
    )
    split_plan = plan_development_split(config)
    split_approved = bool(
        split_plan["summary"]["split_approved"]
        and split_plan["summary"]["frozen"]
        and split_plan["summary"]["materialization_allowed"]
    )
    external = _validate_external_concept(
        _approval_path(config, "external_test_concept_approval")
    )

    approval_files = [
        path
        for path in (
            identity_path,
            _approval_path(config, "dataset_route_approval"),
            taxonomy_path,
            taxonomy_approval_path,
            metric_path,
            _approval_path(config, "training_approval"),
            _approval_path(config, "development_split_approval"),
            _approval_path(config, "external_test_concept_approval"),
        )
        if path is not None and path.is_file()
    ]
    approval_hashes = [
        {
            "file": path.name,
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for path in approval_files
    ]
    privacy_blockers = list(
        dict.fromkeys(
            taxonomy_role_issues
            + taxonomy_privacy
            + identity_privacy
            + metric_privacy
            + route["blockers"]
            + initialization["blockers"]
            + external["blockers"]
        )
    )
    development_ready = bool(
        candidate["summary"]["registration_ready"]
        and taxonomy_approved
        and identity_approved
        and route["approved"]
        and metric_approved
        and initialization["approved"]
        and split_approved
        and not privacy_blockers
    )
    return {
        "schema_version": "1.0",
        "audit_type": "phase3b_gate2_approval_validation",
        "created_utc": utc_timestamp(),
        "approval_date": APPROVAL_DATE,
        "allowed_role_identifiers": sorted(ROLE_IDENTIFIERS),
        "candidate_registration_ready": candidate["summary"][
            "registration_ready"
        ],
        "taxonomy": {
            "approved": taxonomy_approved,
            "validator": taxonomy,
            "role_or_privacy_blockers": list(
                dict.fromkeys(taxonomy_role_issues + taxonomy_privacy)
            ),
        },
        "gem_identity": {
            **identity,
            "approved": identity_approved,
            "role_or_privacy_blockers": identity_privacy,
        },
        "dataset_route": route,
        "metric": {
            "approved": metric_approved,
            "validator": metric,
            "role_or_privacy_blockers": metric_privacy,
        },
        "initialization": initialization,
        "development_split": {
            "approved": split_plan["summary"]["split_approved"],
            "frozen": split_plan["summary"]["frozen"],
            "materialization_allowed": split_plan["summary"][
                "materialization_allowed"
            ],
            "split_id": split_plan["development_split_lock"]["split_id"],
            "manifest_sha256": split_plan["development_split_lock"][
                "grouped_development_manifest_sha256"
            ],
            "split_counts": split_plan["summary"]["split_image_counts"],
            "gem_counts": split_plan["summary"]["split_gem_counts"],
            "leakage": split_plan["leakage_audit"],
            "blockers": split_plan["summary"]["blockers"],
        },
        "external_test": external,
        "approval_hashes": approval_hashes,
        "summary": {
            "status": "development_ready" if development_ready else "blocked",
            "development_dataset_approval_ready": development_ready,
            "external_test_concept_approved": external["concept_approved"],
            "external_test_operational_ready": False,
            "training_authorized": False,
            "authorization_status": (
                "blocked_external_test_operational_details"
            ),
            "training_occurred": False,
            "weights_downloaded": False,
            "final_evaluation_occurred": False,
            "final_metrics_calculated": False,
            "claim_90_percent_achieved": False,
            "privacy_blocker_count": len(privacy_blockers),
            "privacy_blockers": privacy_blockers,
        },
    }


def run_gate2_approval_validation(
    config: dict[str, Any],
    *,
    output_override: str | Path | None = None,
    command: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    report = validate_gate2_approvals(config)
    output_value = output_override or config.get("gate2_approval_output_directory")
    output = resolve_path(output_value, config["_repo_root"])
    if output is None:
        raise ValueError("Gate 2 approval output directory is required.")
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "report": output / "gate2_approval_validation.json",
        "hashes": output / "approval_hashes.csv",
        "markdown": output / "audit_summary.md",
        "manifest": output / "manifest.json",
    }
    atomic_write_json(files["report"], report)
    atomic_write_csv(
        files["hashes"],
        report["approval_hashes"],
        ("file", "path", "sha256"),
    )
    atomic_write_text(
        files["markdown"],
        "\n".join(
            [
                "# Phase 3B Gate 2 Approval Validation",
                "",
                f"- Status: {report['summary']['status']}",
                f"- Taxonomy approved: {str(report['taxonomy']['approved']).lower()}",
                f"- Gem identity approved: {str(report['gem_identity']['approved']).lower()}",
                f"- Dataset route approved: {str(report['dataset_route']['approved']).lower()}",
                f"- Metric approved: {str(report['metric']['approved']).lower()}",
                f"- Initialization approved: {str(report['initialization']['approved']).lower()}",
                f"- Development split frozen: {str(report['development_split']['frozen']).lower()}",
                f"- External concept approved: {str(report['external_test']['concept_approved']).lower()}",
                "- External operational readiness: false",
                "- Training authorized: false",
                "- Training occurred: false",
                "",
            ]
        ),
    )
    write_manifest(
        path=files["manifest"],
        repo_root=config["_repo_root"],
        tool_name="quartz_phase3b_gate2_approval_validation",
        command=command,
        tool_sources=(
            "backend/research/gate2.py",
            "backend/research/candidate_dataset.py",
            "backend/research/metric_approval.py",
            "backend/research/taxonomy_validation.py",
        ),
        inputs=tuple(
            (f"approval_{index}", row["path"])
            for index, row in enumerate(report["approval_hashes"])
        ),
        outputs=tuple(
            (name, path)
            for name, path in files.items()
            if name != "manifest"
        ),
        extra={
            "development_dataset_approval_ready": report["summary"][
                "development_dataset_approval_ready"
            ],
            "external_test_operational_ready": False,
            "training_authorized": False,
            "training_occurred": False,
            "weights_downloaded": False,
        },
    )
    return report, files


def approval_paths(config: dict[str, Any]) -> Iterable[Path]:
    for key in (
        "gem_identity_approval",
        "dataset_route_approval",
        "candidate_taxonomy",
        "candidate_taxonomy_approval",
        "metric_approval",
        "training_approval",
        "development_split_approval",
        "external_test_concept_approval",
    ):
        path = _approval_path(config, key)
        if path is not None:
            yield path
