"""Validate defect taxonomy approval without creating approval evidence."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from defect_policy import TAXONOMY_STATUSES

from .common import atomic_write_json, sha256_file, utc_timestamp
from .dataset_audit import load_dataset_yaml
from .readiness_common import (
    atomic_write_csv,
    atomic_write_text,
    find_repo_root,
    resolve_path,
    write_manifest,
)

APPROVED_METRICS = {
    "binary_visible_defect_f1",
    "binary_visible_defect_recall",
    "macro_class_f1",
}
CLASS_COLUMNS = (
    "class_id",
    "model_class_name",
    "taxonomy_class_name",
    "taxonomy_status",
    "semantic_definition_present",
    "mapping_decided",
    "no_cut_decided",
    "expert_confirmed",
    "minimum_confidence_valid",
    "valid",
    "blocking_reasons",
)


def _class_names(yaml_path: Path) -> list[str]:
    data = load_dataset_yaml(yaml_path)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(value) for value in names]
    return []


def _read_taxonomy_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    issues = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    required = {
        "class_id",
        "class_name",
        "semantic_definition",
        "taxonomy_status",
        "is_visible_defect",
        "eligible_for_3d_mapping",
        "eligible_for_no_cut_zone",
        "minimum_confidence",
        "expert_confirmed",
    }
    missing = sorted(required - set(reader.fieldnames or []))
    if missing:
        issues.append(f"Missing taxonomy columns: {', '.join(missing)}")
    return rows, issues


def _bool_value(value: str) -> bool | None:
    normalized = value.casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _approval(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, "Taxonomy approval record was not supplied."
    if not path.exists():
        return None, "Taxonomy approval record was not found."
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        return None, f"Taxonomy approval record is invalid: {exc}"
    if not isinstance(value, dict):
        return None, "Taxonomy approval record must be a JSON object."
    return value, None


def validate_taxonomy_approval(
    taxonomy_path: str | Path,
    approval_path: str | Path | None,
    dataset_yaml: str | Path,
) -> dict[str, Any]:
    started = utc_timestamp()
    taxonomy = Path(taxonomy_path).resolve()
    yaml_path = Path(dataset_yaml).resolve()
    approval_file = Path(approval_path).resolve() if approval_path else None
    blockers: list[str] = []
    warnings: list[str] = []

    if not taxonomy.exists():
        blockers.append("Taxonomy file was not found.")
        rows = []
        taxonomy_issues = []
    else:
        rows, taxonomy_issues = _read_taxonomy_rows(taxonomy)
        blockers.extend(taxonomy_issues)
    model_names = _class_names(yaml_path) if yaml_path.exists() else []
    if not model_names:
        blockers.append("Model class names are unavailable.")

    class_ids = [row.get("class_id", "") for row in rows]
    duplicate_ids = sorted(
        value for value, count in Counter(class_ids).items()
        if value and count > 1
    )
    if duplicate_ids:
        blockers.append(f"Duplicate taxonomy class IDs: {duplicate_ids}")

    matrix = []
    rows_by_id: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            class_id = int(row.get("class_id", ""))
        except ValueError:
            blockers.append(f"Invalid taxonomy class ID: {row.get('class_id')}")
            continue
        if class_id not in rows_by_id:
            rows_by_id[class_id] = row

    for class_id, model_name in enumerate(model_names):
        row = rows_by_id.get(class_id)
        reasons = []
        if row is None:
            reasons.append("class_missing")
            row = {}
        taxonomy_name = row.get("class_name", "")
        status = row.get("taxonomy_status", "missing")
        if taxonomy_name != model_name:
            reasons.append("class_name_mismatch")
        if status not in TAXONOMY_STATUSES:
            reasons.append("invalid_taxonomy_status")
        if status == "unknown":
            reasons.append("taxonomy_unknown")
        definition_present = bool(row.get("semantic_definition", ""))
        if status in {"confirmed_defect", "confirmed_non_defect"} and not definition_present:
            reasons.append("semantic_definition_missing")
        mapping = _bool_value(row.get("eligible_for_3d_mapping", ""))
        no_cut = _bool_value(row.get("eligible_for_no_cut_zone", ""))
        expert = _bool_value(row.get("expert_confirmed", ""))
        if mapping is None:
            reasons.append("mapping_eligibility_not_decided")
        if no_cut is None:
            reasons.append("no_cut_eligibility_not_decided")
        if no_cut and not (
            status == "confirmed_defect"
            and expert is True
            and mapping is True
        ):
            reasons.append("no_cut_without_confirmed_expert_defect")
        threshold_text = row.get("minimum_confidence", "")
        threshold_valid = True
        if threshold_text:
            try:
                threshold = float(threshold_text)
                threshold_valid = 0.0 <= threshold <= 1.0
            except ValueError:
                threshold_valid = False
        if not threshold_valid:
            reasons.append("invalid_confidence_threshold")
        matrix.append(
            {
                "class_id": class_id,
                "model_class_name": model_name,
                "taxonomy_class_name": taxonomy_name,
                "taxonomy_status": status,
                "semantic_definition_present": definition_present,
                "mapping_decided": mapping is not None,
                "no_cut_decided": no_cut is not None,
                "expert_confirmed": expert is True,
                "minimum_confidence_valid": threshold_valid,
                "valid": not reasons,
                "blocking_reasons": reasons,
            }
        )
    unknown_taxonomy_ids = sorted(set(rows_by_id) - set(range(len(model_names))))
    if unknown_taxonomy_ids:
        blockers.append(
            f"Taxonomy contains class IDs absent from model metadata: {unknown_taxonomy_ids}"
        )
    for item in matrix:
        if item["blocking_reasons"]:
            blockers.append(
                f"Class {item['class_id']} ({item['model_class_name']}): "
                + ", ".join(item["blocking_reasons"])
            )

    approval, approval_error = _approval(approval_file)
    if approval_error:
        blockers.append(approval_error)
    approval_valid = approval is not None
    taxonomy_hash = sha256_file(taxonomy) if taxonomy.exists() else None
    if approval:
        if approval.get("example_only", False):
            blockers.append("Example approval records are not valid approvals.")
            approval_valid = False
        if approval.get("taxonomy_sha256") != taxonomy_hash:
            blockers.append("Approval taxonomy hash does not match the taxonomy file.")
            approval_valid = False
        for field in (
            "taxonomy_id",
            "taxonomy_version",
            "approved_by",
            "approver_role",
            "approval_date",
            "approval_scope",
        ):
            if not approval.get(field):
                blockers.append(f"Approval field is missing: {field}")
                approval_valid = False
        metric = approval.get("primary_claim_metric")
        if metric not in APPROVED_METRICS:
            blockers.append("Primary claim metric is missing or unsupported.")
            approval_valid = False
        try:
            claim_threshold = float(approval.get("claim_threshold"))
        except (TypeError, ValueError):
            claim_threshold = None
        if claim_threshold != 0.90:
            blockers.append(
                "The proposal's 0.90 threshold was changed or omitted without approval evidence."
            )
            approval_valid = False
    else:
        metric = None
        claim_threshold = None
        approval_valid = False

    unique_blockers = list(dict.fromkeys(blockers))
    valid = bool(not unique_blockers and approval_valid and matrix)
    return {
        "schema_version": "1.0",
        "audit_type": "taxonomy_approval_validation",
        "started_utc": started,
        "ended_utc": utc_timestamp(),
        "taxonomy": {
            "path": str(taxonomy),
            "sha256": taxonomy_hash,
            "class_count": len(rows),
        },
        "model_metadata": {
            "dataset_yaml": str(yaml_path),
            "class_names": model_names,
        },
        "approval": {
            "path": str(approval_file) if approval_file else None,
            "supplied": approval is not None,
            "valid": approval_valid and not unique_blockers,
            "example_only": bool(approval and approval.get("example_only", False)),
            "primary_claim_metric": metric,
            "claim_threshold": claim_threshold,
        },
        "class_validation": matrix,
        "summary": {
            "status": "available" if valid else "unavailable",
            "taxonomy_approved": valid,
            "blocker_count": len(unique_blockers),
            "warning_count": len(warnings),
            "blockers": unique_blockers,
            "warnings": warnings,
        },
        "claim_status": "unavailable",
        "claim_boundary": (
            "Validation confirms approval evidence only. It does not evaluate "
            "detector performance or create expert approval."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    approval = report["approval"]
    return "\n".join(
        [
            "# Taxonomy Approval Validation",
            "",
            f"- Status: {summary['status']}",
            f"- Taxonomy approved: {str(summary['taxonomy_approved']).lower()}",
            f"- Approval supplied: {str(approval['supplied']).lower()}",
            f"- Example approval: {str(approval['example_only']).lower()}",
            f"- Primary metric: {approval['primary_claim_metric'] or 'unavailable'}",
            f"- Blockers: {summary['blocker_count']}",
            "",
            *[f"- {item}" for item in summary["blockers"]],
            "",
            report["claim_boundary"],
            "",
        ]
    )


def run_taxonomy_validation(
    *,
    taxonomy_path: str | Path,
    approval_path: str | Path | None,
    dataset_yaml: str | Path,
    output_directory: str | Path,
    command: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    taxonomy = Path(taxonomy_path).resolve()
    repo_root = find_repo_root(taxonomy)
    yaml_path = resolve_path(dataset_yaml, repo_root)
    approval = resolve_path(approval_path, repo_root)
    output = resolve_path(output_directory, repo_root)
    if yaml_path is None or output is None:
        raise ValueError("dataset_yaml and output_directory are required.")
    report = validate_taxonomy_approval(taxonomy, approval, yaml_path)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "report": output / "taxonomy_validation.json",
        "summary": output / "taxonomy_summary.csv",
        "classes": output / "class_validation.csv",
        "markdown": output / "audit_summary.md",
        "manifest": output / "manifest.json",
    }
    atomic_write_json(files["report"], report)
    atomic_write_csv(files["summary"], [report["summary"]], (
        "status", "taxonomy_approved", "blocker_count", "warning_count"
    ))
    atomic_write_csv(files["classes"], report["class_validation"], CLASS_COLUMNS)
    atomic_write_text(files["markdown"], _markdown(report))
    write_manifest(
        path=files["manifest"],
        repo_root=repo_root,
        tool_name="quartz_taxonomy_approval_validation",
        command=command,
        tool_sources=(
            "backend/research/taxonomy_validation.py",
            "backend/research/readiness_common.py",
            "backend/research_benchmark.py",
        ),
        inputs=(
            ("taxonomy", taxonomy),
            ("approval", approval),
            ("dataset_yaml", yaml_path),
        ),
        outputs=(
            ("report", files["report"]),
            ("summary", files["summary"]),
            ("classes", files["classes"]),
            ("markdown", files["markdown"]),
        ),
    )
    return report, files
