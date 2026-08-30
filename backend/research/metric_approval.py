"""Validate the supervisor-approved operational definition of the 90% target."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import atomic_write_json, sha256_file, utc_timestamp
from .readiness_common import (
    atomic_write_csv,
    atomic_write_text,
    find_repo_root,
    resolve_path,
    write_manifest,
)

PRIMARY_METRICS = {
    "binary_visible_defect_f1",
    "binary_visible_defect_recall",
    "macro_class_f1",
}
AGGREGATION_LEVELS = {
    "image_level",
    "specimen_grouped",
    "pixel_level",
}
REQUIRED_FIELDS = (
    "metric_id",
    "metric_version",
    "primary_metric",
    "claim_threshold",
    "aggregation_level",
    "approved_by",
    "approver_role",
    "approval_date",
    "taxonomy_sha256",
    "test_protocol_sha256",
)


def _load_record(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, "Metric approval record was not supplied."
    if not path.is_file():
        return None, "Metric approval record was not found."
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"Metric approval record is invalid: {exc}"
    if not isinstance(value, dict):
        return None, "Metric approval record must be a JSON object."
    return value, None


def validate_metric_approval(
    approval_path: str | Path | None,
    taxonomy_path: str | Path,
    test_protocol_path: str | Path,
) -> dict[str, Any]:
    """Return fail-closed approval status without choosing a metric."""
    started = utc_timestamp()
    approval_file = Path(approval_path).resolve() if approval_path else None
    taxonomy = Path(taxonomy_path).resolve()
    protocol = Path(test_protocol_path).resolve()
    taxonomy_hash = sha256_file(taxonomy) if taxonomy.is_file() else None
    protocol_hash = sha256_file(protocol) if protocol.is_file() else None
    blockers: list[str] = []
    warnings: list[str] = []
    if taxonomy_hash is None:
        blockers.append("Taxonomy file is unavailable.")
    if protocol_hash is None:
        blockers.append("Test protocol file is unavailable.")

    approval, error = _load_record(approval_file)
    if error:
        blockers.append(error)
    if approval:
        if approval.get("example_only", False):
            blockers.append("Example metric records are not valid approvals.")
        for field in REQUIRED_FIELDS:
            if approval.get(field) in (None, ""):
                blockers.append(f"Metric approval field is missing: {field}")
        metric = approval.get("primary_metric")
        if metric not in PRIMARY_METRICS:
            blockers.append("Primary metric is missing or unsupported.")
        aggregation = approval.get("aggregation_level")
        if aggregation not in AGGREGATION_LEVELS:
            blockers.append("Aggregation level is missing or unsupported.")
        try:
            threshold = float(approval.get("claim_threshold"))
        except (TypeError, ValueError):
            threshold = None
        if threshold != 0.90:
            blockers.append(
                "The proposal threshold must remain 0.90 unless separately amended."
            )
        floor = approval.get("precision_floor")
        if floor not in (None, ""):
            try:
                floor = float(floor)
            except (TypeError, ValueError):
                floor = None
                blockers.append("Precision floor must be a number between 0 and 1.")
            else:
                if not 0.0 <= floor <= 1.0:
                    blockers.append(
                        "Precision floor must be a number between 0 and 1."
                    )
        else:
            floor = None
        if metric == "binary_visible_defect_recall" and floor is None:
            blockers.append(
                "Recall as the primary metric requires an approved precision floor."
            )
        if approval.get("taxonomy_sha256") != taxonomy_hash:
            blockers.append("Metric approval taxonomy hash does not match.")
        if approval.get("test_protocol_sha256") != protocol_hash:
            blockers.append("Metric approval test-protocol hash does not match.")
    else:
        metric = None
        threshold = None
        floor = None
        aggregation = None

    unique_blockers = list(dict.fromkeys(blockers))
    approved = bool(approval and not unique_blockers)
    return {
        "schema_version": "1.0",
        "audit_type": "defect_metric_approval_validation",
        "started_utc": started,
        "ended_utc": utc_timestamp(),
        "approval": {
            "path": str(approval_file) if approval_file else None,
            "supplied": approval is not None,
            "example_only": bool(
                approval and approval.get("example_only", False)
            ),
            "approved": approved,
            "primary_metric": metric,
            "claim_threshold": threshold,
            "precision_floor": floor,
            "aggregation_level": aggregation,
        },
        "evidence": {
            "taxonomy_path": str(taxonomy),
            "taxonomy_sha256": taxonomy_hash,
            "test_protocol_path": str(protocol),
            "test_protocol_sha256": protocol_hash,
        },
        "summary": {
            "status": "pass" if approved else "blocked",
            "metric_approved": approved,
            "blocker_count": len(unique_blockers),
            "blockers": unique_blockers,
            "warnings": warnings,
        },
        "claim_status": "unavailable",
        "claim_boundary": (
            "This validates an operational metric decision. It does not calculate "
            "performance or determine whether the 90 percent target was achieved."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    approval = report["approval"]
    summary = report["summary"]
    return "\n".join(
        [
            "# Defect Metric Approval",
            "",
            f"- Status: {summary['status']}",
            f"- Approved: {str(summary['metric_approved']).lower()}",
            f"- Primary metric: {approval['primary_metric'] or 'unavailable'}",
            f"- Threshold: {approval['claim_threshold'] if approval['claim_threshold'] is not None else 'unavailable'}",
            f"- Precision floor: {approval['precision_floor'] if approval['precision_floor'] is not None else 'unavailable'}",
            f"- Blockers: {summary['blocker_count']}",
            "",
            *[f"- {item}" for item in summary["blockers"]],
            "",
            report["claim_boundary"],
            "",
        ]
    )


def run_metric_approval(
    *,
    approval_path: str | Path | None,
    taxonomy_path: str | Path,
    test_protocol_path: str | Path,
    output_directory: str | Path,
    command: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    taxonomy = Path(taxonomy_path).resolve()
    repo_root = find_repo_root(taxonomy)
    approval = resolve_path(approval_path, repo_root)
    protocol = resolve_path(test_protocol_path, repo_root)
    output = resolve_path(output_directory, repo_root)
    if protocol is None or output is None:
        raise ValueError("test_protocol_path and output_directory are required.")
    report = validate_metric_approval(approval, taxonomy, protocol)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "report": output / "metric_approval.json",
        "summary": output / "metric_approval_summary.csv",
        "markdown": output / "audit_summary.md",
        "manifest": output / "manifest.json",
    }
    atomic_write_json(files["report"], report)
    atomic_write_csv(
        files["summary"],
        [
            {
                **report["summary"],
                **report["approval"],
            }
        ],
        (
            "status",
            "metric_approved",
            "primary_metric",
            "claim_threshold",
            "precision_floor",
            "aggregation_level",
            "blocker_count",
        ),
    )
    atomic_write_text(files["markdown"], _markdown(report))
    write_manifest(
        path=files["manifest"],
        repo_root=repo_root,
        tool_name="quartz_defect_metric_approval",
        command=command,
        tool_sources=(
            "backend/research/metric_approval.py",
            "backend/research/readiness_common.py",
            "backend/research_benchmark.py",
        ),
        inputs=(
            ("approval", approval),
            ("taxonomy", taxonomy),
            ("test_protocol", protocol),
        ),
        outputs=(
            ("report", files["report"]),
            ("summary", files["summary"]),
            ("markdown", files["markdown"]),
        ),
    )
    return report, files
