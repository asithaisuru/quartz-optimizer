"""Read-only audit of detector taxonomy, policy, and output traceability."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from defect_policy import (
    POLICIES,
    load_taxonomy,
    resolve_taxonomy_path,
    validate_policy_config,
)

from . import TOOL_VERSION
from .common import (
    atomic_write_json,
    git_evidence,
    normalize_path,
    sha256_file,
    utc_timestamp,
)
from .dataset_audit import load_dataset_yaml

SUMMARY_COLUMNS = (
    "audit_status",
    "claim_ready",
    "policy",
    "strict_research_mode",
    "model_class_count",
    "taxonomy_class_count",
    "unknown_class_count",
    "unconfirmed_class_count",
    "classes_reaching_no_cut_without_confirmation",
    "opencv_no_cut_without_confirmation",
    "legacy_policy_enabled",
    "raw_accepted_outputs_separated",
    "legacy_masks_without_metadata",
    "blocker_count",
    "warning_count",
)
CLASS_COLUMNS = (
    "class_id",
    "model_class_name",
    "taxonomy_class_name",
    "taxonomy_status",
    "semantic_definition_present",
    "expert_confirmed",
    "eligible_for_visualization",
    "eligible_for_3d_mapping",
    "eligible_for_no_cut_zone",
    "minimum_confidence",
    "can_reach_no_cut",
    "decision_reason",
)
WARNING_COLUMNS = ("severity", "code", "message")


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate.resolve()
    return Path.cwd().resolve()


def _resolve(value: str | Path | None, root: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_defect_audit_config(
    path: str | Path,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Detector policy configuration must be a JSON object.")
    config["_config_path"] = str(config_path)
    config["_repo_root"] = str(_find_repo_root(config_path.parent))
    if output_override is not None:
        config["output_directory"] = str(output_override)
    return config


def _model_class_names(config: dict[str, Any], repo_root: Path) -> list[str]:
    yaml_path = _resolve(config.get("dataset_yaml"), repo_root)
    if yaml_path is None or not yaml_path.exists():
        return []
    value = load_dataset_yaml(yaml_path)
    names = value.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(item) for item in names]
    return []


def _warning(
    warnings: list[dict[str, str]],
    severity: str,
    code: str,
    message: str,
) -> None:
    warnings.append({"severity": severity, "code": code, "message": message})


def _yolo_policy_uses_classes(policy: str) -> bool:
    return policy in {"yolo_only", "union", "gated_union", "fallback", "legacy_union_all"}


def _opencv_policy_can_contribute(policy: str) -> bool:
    return policy in {"opencv_only", "union", "gated_union", "fallback", "legacy_union_all"}


def _job_traceability(job_dir: Path | None) -> dict[str, Any]:
    if job_dir is None:
        return {
            "job_directory": None,
            "raw_accepted_outputs_separated": None,
            "output_separation_status": "not_observed_without_job",
            "legacy_masks_present": False,
            "legacy_masks_without_metadata": False,
        }
    detections = job_dir / "detections"
    raw_present = (detections / "raw_predictions.json").exists()
    decisions_present = (detections / "policy_decisions.json").exists()
    channel_dirs = all(
        (detections / name).is_dir()
        for name in (
            "raw_masks",
            "visualization_masks",
            "mapping_masks",
            "no_cut_masks",
        )
    )
    legacy_present = (job_dir / "fractures").is_dir()
    metadata_present = decisions_present and (detections / "policy_summary.json").exists()
    return {
        "job_directory": str(job_dir),
        "raw_accepted_outputs_separated": bool(
            raw_present and decisions_present and channel_dirs
        ),
        "output_separation_status": (
            "observed"
            if raw_present and decisions_present and channel_dirs
            else "missing_or_incomplete"
        ),
        "legacy_masks_present": legacy_present,
        "legacy_masks_without_metadata": bool(legacy_present and not metadata_present),
    }


def audit_defect_policy(
    config: dict[str, Any],
    job_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Audit policy safety without evaluating model performance."""

    started = utc_timestamp()
    repo_root = Path(config.get("_repo_root") or Path.cwd()).resolve()
    warnings: list[dict[str, str]] = []
    policy = str(config.get("policy", ""))
    validation_error = None
    try:
        validate_policy_config(config)
    except (KeyError, TypeError, ValueError) as exc:
        validation_error = str(exc)
        _warning(warnings, "blocker", "invalid_policy_configuration", str(exc))

    taxonomy_value = config.get("taxonomy_path")
    taxonomy_path = (
        resolve_taxonomy_path(config, repo_root)
        if taxonomy_value
        else repo_root / "research" / "templates" / "missing_taxonomy.csv"
    )
    if not taxonomy_value:
        _warning(
            warnings,
            "blocker",
            "taxonomy_path_missing",
            "The policy configuration does not define taxonomy_path.",
        )
    taxonomy = {}
    if not taxonomy_path.exists():
        _warning(
            warnings,
            "blocker",
            "taxonomy_missing",
            f"Taxonomy file does not exist: {taxonomy_path}",
        )
    else:
        try:
            taxonomy = load_taxonomy(taxonomy_path)
        except (OSError, TypeError, ValueError) as exc:
            _warning(warnings, "blocker", "taxonomy_invalid", str(exc))

    model_classes = _model_class_names(config, repo_root)
    if not model_classes:
        _warning(
            warnings,
            "blocker",
            "model_class_names_unavailable",
            "Configured dataset class names could not be read.",
        )

    default_threshold = float(config.get("minimum_confidence_default", 0.25))
    per_class = config.get("minimum_confidence_by_class", {})
    strict = bool(config.get("strict_research_mode", True))
    class_rows = []
    unauthorized_no_cut = []
    for class_id, model_name in enumerate(model_classes):
        entry = taxonomy.get(class_id)
        threshold = per_class.get(str(class_id), per_class.get(model_name))
        if threshold is None and entry is not None:
            threshold = entry.minimum_confidence
        if threshold is None:
            threshold = default_threshold

        confirmed = bool(
            entry
            and entry.taxonomy_status == "confirmed_defect"
            and entry.expert_confirmed
        )
        can_reach_no_cut = bool(
            _yolo_policy_uses_classes(policy)
            and entry
            and entry.eligible_for_3d_mapping
            and entry.eligible_for_no_cut_zone
            and (
                confirmed
                or (
                    not strict
                    and entry.taxonomy_status == "provisional_defect"
                    and config.get("allow_provisional_for_no_cut", False)
                )
            )
        )
        if can_reach_no_cut and not confirmed:
            unauthorized_no_cut.append(class_id)
        if entry is None:
            reason = "class_missing_from_taxonomy"
        elif not entry.semantic_definition:
            reason = "semantic_definition_unconfirmed"
        elif entry.taxonomy_status == "confirmed_non_defect":
            reason = "confirmed_non_defect"
        elif not confirmed:
            reason = "taxonomy_not_expert_confirmed"
        elif not entry.eligible_for_no_cut_zone:
            reason = "not_eligible_for_no_cut"
        else:
            reason = "confirmed_policy_eligible"
        class_rows.append(
            {
                "class_id": class_id,
                "model_class_name": model_name,
                "taxonomy_class_name": entry.class_name if entry else "",
                "taxonomy_status": entry.taxonomy_status if entry else "missing",
                "semantic_definition_present": bool(
                    entry and entry.semantic_definition
                ),
                "expert_confirmed": bool(entry and entry.expert_confirmed),
                "eligible_for_visualization": bool(
                    entry and entry.is_visible_defect
                ),
                "eligible_for_3d_mapping": bool(
                    entry and entry.eligible_for_3d_mapping
                ),
                "eligible_for_no_cut_zone": bool(
                    entry and entry.eligible_for_no_cut_zone
                ),
                "minimum_confidence": threshold,
                "can_reach_no_cut": can_reach_no_cut,
                "decision_reason": reason,
            }
        )

    taxonomy_ids = set(taxonomy)
    unknown_ids = [
        row["class_id"]
        for row in class_rows
        if row["taxonomy_status"] in {"unknown", "missing"}
    ]
    unconfirmed_ids = [
        row["class_id"]
        for row in class_rows
        if not row["expert_confirmed"]
    ]
    if unknown_ids:
        _warning(
            warnings,
            "blocker",
            "unknown_model_classes",
            f"Model class IDs without confirmed meaning: {unknown_ids}",
        )
    if unconfirmed_ids:
        _warning(
            warnings,
            "blocker",
            "unconfirmed_model_classes",
            f"Model class IDs without expert confirmation: {unconfirmed_ids}",
        )
    extra_taxonomy_ids = sorted(taxonomy_ids - set(range(len(model_classes))))
    if extra_taxonomy_ids:
        _warning(
            warnings,
            "warning",
            "taxonomy_classes_not_in_model_config",
            f"Taxonomy class IDs absent from dataset YAML: {extra_taxonomy_ids}",
        )
    if unauthorized_no_cut:
        _warning(
            warnings,
            "blocker",
            "unconfirmed_class_can_reach_no_cut",
            f"Unconfirmed classes can reach no-cut output: {unauthorized_no_cut}",
        )

    opencv = config.get("opencv", {})
    opencv_no_cut_without_confirmation = bool(
        _opencv_policy_can_contribute(policy)
        and opencv.get("eligible_for_3d_mapping", False)
        and opencv.get("eligible_for_no_cut", False)
        and (not strict or config.get("allow_provisional_for_no_cut", False))
    )
    if opencv_no_cut_without_confirmation:
        _warning(
            warnings,
            "blocker",
            "opencv_candidate_can_reach_no_cut",
            "OpenCV candidates can reach no-cut output without class confirmation.",
        )

    legacy_enabled = bool(
        policy == "legacy_union_all"
        or config.get("legacy", {}).get("allow_legacy_union_all", False)
    )
    if legacy_enabled:
        _warning(
            warnings,
            "blocker",
            "unsafe_legacy_policy",
            "Legacy union compatibility is enabled.",
        )

    job_path = _resolve(job_dir, repo_root) if job_dir else None
    traceability = _job_traceability(job_path)
    if traceability["raw_accepted_outputs_separated"] is False:
        _warning(
            warnings,
            "blocker",
            "raw_and_accepted_outputs_not_separated",
            "The supplied job does not contain all separated output artifacts.",
        )
    if traceability["legacy_masks_without_metadata"]:
        _warning(
            warnings,
            "blocker",
            "legacy_masks_without_metadata",
            "The supplied job contains legacy masks without policy metadata.",
        )

    blockers = [item for item in warnings if item["severity"] == "blocker"]
    confirmed_defect_count = sum(
        row["taxonomy_status"] == "confirmed_defect"
        and row["expert_confirmed"]
        for row in class_rows
    )
    claim_ready = bool(
        not blockers
        and confirmed_defect_count > 0
        and validation_error is None
    )
    summary = {
        "audit_status": "pass" if not warnings else "warning",
        "claim_ready": claim_ready,
        "policy": policy if policy in POLICIES else "invalid",
        "strict_research_mode": strict,
        "model_class_count": len(model_classes),
        "taxonomy_class_count": len(taxonomy),
        "unknown_class_count": len(unknown_ids),
        "unconfirmed_class_count": len(unconfirmed_ids),
        "classes_reaching_no_cut_without_confirmation": len(
            unauthorized_no_cut
        ),
        "opencv_no_cut_without_confirmation": opencv_no_cut_without_confirmation,
        "legacy_policy_enabled": legacy_enabled,
        "raw_accepted_outputs_separated": traceability[
            "raw_accepted_outputs_separated"
        ],
        "legacy_masks_without_metadata": traceability[
            "legacy_masks_without_metadata"
        ],
        "blocker_count": len(blockers),
        "warning_count": len(warnings) - len(blockers),
    }
    return {
        "schema_version": "1.0",
        "tool_version": TOOL_VERSION,
        "audit_type": "detector_policy_and_traceability",
        "performance_metrics_calculated": False,
        "started_utc": started,
        "ended_utc": utc_timestamp(),
        "summary": summary,
        "policy": {
            "name": policy,
            "strict_research_mode": strict,
            "minimum_confidence_default": default_threshold,
            "minimum_confidence_by_class": per_class,
            "opencv": opencv,
            "fallback_conditions": {
                key: opencv.get(key, False)
                for key in (
                    "fallback_on_model_unavailable",
                    "fallback_on_inference_error",
                    "fallback_on_zero_detections",
                )
            },
            "legacy": config.get("legacy", {}),
        },
        "taxonomy": {
            "path": normalize_path(taxonomy_path, repo_root),
            "confirmed_defect_count": confirmed_defect_count,
            "unknown_class_ids": unknown_ids,
            "unconfirmed_class_ids": unconfirmed_ids,
        },
        "class_policy_matrix": class_rows,
        "traceability": traceability,
        "warnings": warnings,
        "claim_status": {
            "status": "unavailable",
            "reason": (
                "No confirmed defect taxonomy was supplied."
                if confirmed_defect_count == 0
                else "Detector performance was not evaluated in Phase 2."
            ),
        },
        "claim_boundary": (
            "This audit evaluates policy and traceability only. It does not "
            "calculate precision, recall, F1, mAP, or detection accuracy."
        ),
    }


def _write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    columns: Iterable[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns_list = list(columns)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=columns_list)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in columns_list})
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return "\n".join(
        [
            "# Defect Policy Audit",
            "",
            f"- Policy: {summary['policy']}",
            f"- Strict research mode: {str(summary['strict_research_mode']).lower()}",
            f"- Model classes: {summary['model_class_count']}",
            f"- Unknown classes: {summary['unknown_class_count']}",
            f"- Unconfirmed classes: {summary['unconfirmed_class_count']}",
            f"- Policy blockers: {summary['blocker_count']}",
            "- Model-performance metrics calculated: false",
            f"- Claim status: {report['claim_status']['status']}",
            f"- Reason: {report['claim_status']['reason']}",
            "",
            "Raw candidates are not validated physical fractures. An empty approved "
            "set does not prove a defect-free stone.",
            "",
        ]
    )


def _manifest(
    report: dict[str, Any],
    config: dict[str, Any],
    files: dict[str, Path],
    command: str,
) -> dict[str, Any]:
    repo_root = Path(config["_repo_root"])
    taxonomy_path = (
        resolve_taxonomy_path(config, repo_root)
        if config.get("taxonomy_path")
        else repo_root / "research" / "templates" / "missing_taxonomy.csv"
    )
    inputs = []
    for name, path in (
        ("config", Path(config["_config_path"])),
        ("taxonomy", taxonomy_path),
        ("dataset_yaml", _resolve(config.get("dataset_yaml"), repo_root)),
        ("model", repo_root / "backend" / "best.pt"),
    ):
        if path is not None and path.exists():
            inputs.append(
                {
                    "name": name,
                    "path": normalize_path(path, repo_root),
                    "sha256": sha256_file(path),
                }
            )
    outputs = []
    for name, path in files.items():
        if name != "manifest" and path.exists():
            outputs.append(
                {
                    "name": name,
                    "path": normalize_path(path, repo_root),
                    "sha256": sha256_file(path),
                }
            )
    tool_sources = []
    for path in (
        repo_root / "backend" / "research" / "defect_policy_audit.py",
        repo_root / "backend" / "defect_policy.py",
        repo_root / "backend" / "research_benchmark.py",
    ):
        if path.exists():
            tool_sources.append(
                {
                    "path": normalize_path(path, repo_root),
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": "1.0",
        "tool_version": TOOL_VERSION,
        "command": command,
        "created_utc": utc_timestamp(),
        "audit_type": report["audit_type"],
        "performance_metrics_calculated": False,
        "git": git_evidence(repo_root),
        "tool_sources": tool_sources,
        "inputs": inputs,
        "outputs": outputs,
    }


def write_defect_policy_outputs(
    report: dict[str, Any],
    config: dict[str, Any],
    output_directory: str | Path,
    command: str,
) -> dict[str, Path]:
    repo_root = Path(config["_repo_root"])
    output_path = _resolve(output_directory, repo_root)
    if output_path is None:
        raise ValueError("An output directory is required.")
    output_path.mkdir(parents=True, exist_ok=True)
    files = {
        "audit": output_path / "defect_policy_audit.json",
        "summary": output_path / "defect_policy_summary.csv",
        "classes": output_path / "class_policy_matrix.csv",
        "warnings": output_path / "policy_warnings.csv",
        "markdown": output_path / "audit_summary.md",
        "manifest": output_path / "manifest.json",
    }
    atomic_write_json(files["audit"], report)
    _write_csv(files["summary"], [report["summary"]], SUMMARY_COLUMNS)
    _write_csv(files["classes"], report["class_policy_matrix"], CLASS_COLUMNS)
    _write_csv(files["warnings"], report["warnings"], WARNING_COLUMNS)
    files["markdown"].write_text(
        _markdown(report),
        encoding="utf-8",
        newline="\n",
    )
    atomic_write_json(files["manifest"], _manifest(report, config, files, command))
    return files


def run_defect_policy_audit(
    config: dict[str, Any],
    job_dir: str | Path | None = None,
    output_override: str | Path | None = None,
    command: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    report = audit_defect_policy(config, job_dir=job_dir)
    output = output_override or config.get("output_directory")
    if not output:
        raise ValueError("An output directory is required.")
    files = write_defect_policy_outputs(
        report,
        config,
        output,
        command,
    )
    return report, files
