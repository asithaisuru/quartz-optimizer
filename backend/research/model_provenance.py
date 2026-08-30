"""Inspect legacy defect-model provenance without loading or changing the model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import atomic_write_json, normalize_path, sha256_file, utc_timestamp
from .dataset_audit import load_dataset_yaml
from .readiness_common import (
    atomic_write_csv,
    atomic_write_text,
    find_repo_root,
    resolve_path,
    write_manifest,
)

PROVENANCE_STATUSES = {
    "confirmed_contaminated",
    "likely_contaminated",
    "clean",
    "unknown",
}


def _class_names(yaml_path: Path | None) -> list[str]:
    if yaml_path is None or not yaml_path.exists():
        return []
    data = load_dataset_yaml(yaml_path)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(value) for value in names]
    return []


def _training_data_path(args_path: Path | None) -> str | None:
    if args_path is None or not args_path.exists():
        return None
    data = load_dataset_yaml(args_path)
    value = data.get("data")
    return str(value) if value else None


def inspect_model_provenance(
    *,
    model_path: str | Path,
    dataset_yaml: str | Path | None = None,
    training_args: str | Path | None = None,
    training_source: str | Path | None = None,
    dataset_audit: str | Path | None = None,
    clean_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = utc_timestamp()
    model = Path(model_path).resolve()
    yaml_path = Path(dataset_yaml).resolve() if dataset_yaml else None
    args_path = Path(training_args).resolve() if training_args else None
    source_path = Path(training_source).resolve() if training_source else None
    audit_path = Path(dataset_audit).resolve() if dataset_audit else None
    blockers = []
    warnings = []

    if not model.exists():
        blockers.append("Model file was not found.")
        model_hash = None
    else:
        model_hash = sha256_file(model)
    audit = {}
    if audit_path and audit_path.exists():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            warnings.append("Dataset audit could not be read.")
    leakage_groups = int(
        audit.get("totals", {}).get("augmentation_parent_leakage_groups", 0)
    )
    affected_images = int(
        audit.get("totals", {}).get("augmentation_leakage_affected_images", 0)
    )
    training_data = _training_data_path(args_path)
    same_dataset_reference = False
    if training_data and yaml_path:
        try:
            same_dataset_reference = (
                Path(training_data).resolve() == yaml_path.resolve()
            )
        except OSError:
            same_dataset_reference = False

    clean = clean_provenance or {}
    clean_hash_matches = bool(
        model_hash
        and clean.get("model_sha256") == model_hash
        and clean.get("grouped_split_frozen") is True
        and clean.get("test_data_excluded") is True
        and clean.get("threshold_tuned_on_validation_only") is True
    )
    if clean_hash_matches:
        provenance_status = "clean"
        basis = "Supplied clean-training provenance matches the model hash."
    elif (
        leakage_groups > 0
        and same_dataset_reference
        and args_path is not None
        and args_path.exists()
    ):
        provenance_status = "likely_contaminated"
        basis = (
            "Available training arguments reference the dataset whose audit found "
            "augmentation-parent leakage. The copied model cannot be proven to be "
            "the exact output of that run, so contamination is likely, not confirmed."
        )
    else:
        provenance_status = "unknown"
        basis = "Clean grouped training provenance could not be established."

    final_eligible = provenance_status == "clean" and clean_hash_matches
    if not final_eligible:
        blockers.append(
            "Model lacks clean grouped-training provenance compatible with a frozen test set."
        )
    available_metrics = []
    missing_metrics = [
        "held_out_pixel_metrics",
        "held_out_instance_metrics",
        "mask_mAP50",
        "mask_mAP50_95",
    ]
    return {
        "schema_version": "1.0",
        "audit_type": "defect_model_provenance",
        "started_utc": started,
        "ended_utc": utc_timestamp(),
        "model": {
            "path": str(model),
            "sha256": model_hash,
            "size_bytes": model.stat().st_size if model.exists() else None,
            "last_modified_utc": (
                datetime.fromtimestamp(
                    model.stat().st_mtime, timezone.utc
                ).isoformat().replace("+00:00", "Z")
                if model.exists()
                else None
            ),
            "framework": "Ultralytics YOLO",
            "architecture": (
                str(load_dataset_yaml(args_path).get("model"))
                if args_path and args_path.exists()
                else "unavailable"
            ),
            "declared_class_names": _class_names(yaml_path),
        },
        "training": {
            "args_path": str(args_path) if args_path else None,
            "args_sha256": (
                sha256_file(args_path)
                if args_path and args_path.exists()
                else None
            ),
            "source_path": str(source_path) if source_path else None,
            "source_sha256": (
                sha256_file(source_path)
                if source_path and source_path.exists()
                else None
            ),
            "dataset_yaml": str(yaml_path) if yaml_path else None,
            "dataset_yaml_sha256": (
                sha256_file(yaml_path)
                if yaml_path and yaml_path.exists()
                else None
            ),
            "training_dataset_reference": training_data,
            "augmentation_parent_separation_enforced": (
                False if leakage_groups > 0 and same_dataset_reference else None
            ),
            "specimen_level_separation_enforced": None,
            "dataset_leakage_groups": leakage_groups,
            "dataset_leakage_affected_images": affected_images,
        },
        "training_split_provenance": provenance_status,
        "provenance_basis": basis,
        "final_research_eligible": final_eligible,
        "exploratory_pipeline_smoke_test": bool(model.exists()),
        "available_metrics": available_metrics,
        "missing_metrics": missing_metrics,
        "appropriate_usage": [
            "exploratory pipeline smoke testing",
            "software integration testing",
        ],
        "prohibited_usage": [
            "final held-out research evaluation",
            "the proposal's 90 percent visible-flaw claim",
        ],
        "summary": {
            "status": "available" if model.exists() else "unavailable",
            "training_split_provenance": provenance_status,
            "final_research_eligible": final_eligible,
            "exploratory_pipeline_smoke_test": bool(model.exists()),
            "blockers": list(dict.fromkeys(blockers)),
            "warnings": warnings,
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    model = report["model"]
    return "\n".join(
        [
            "# Legacy Defect Model Provenance",
            "",
            f"- Model SHA-256: {model['sha256'] or 'unavailable'}",
            f"- Training split provenance: {summary['training_split_provenance']}",
            f"- Final research eligible: {str(summary['final_research_eligible']).lower()}",
            f"- Exploratory pipeline smoke test: {str(summary['exploratory_pipeline_smoke_test']).lower()}",
            "",
            report["provenance_basis"],
            "",
            "The model must not be used for a final held-out claim unless clean "
            "grouped training provenance is proven for this exact model hash.",
            "",
        ]
    )


def run_model_provenance(
    *,
    model_path: str | Path,
    dataset_yaml: str | Path | None,
    training_args: str | Path | None,
    training_source: str | Path | None,
    dataset_audit: str | Path | None,
    output_directory: str | Path,
    command: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    model = Path(model_path).resolve()
    repo_root = find_repo_root(model)
    yaml_path = resolve_path(dataset_yaml, repo_root)
    args_path = resolve_path(training_args, repo_root)
    source_path = resolve_path(training_source, repo_root)
    audit_path = resolve_path(dataset_audit, repo_root)
    output = resolve_path(output_directory, repo_root)
    if output is None:
        raise ValueError("An output directory is required.")
    report = inspect_model_provenance(
        model_path=model,
        dataset_yaml=yaml_path,
        training_args=args_path,
        training_source=source_path,
        dataset_audit=audit_path,
    )
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "report": output / "model_provenance.json",
        "summary": output / "model_provenance_summary.csv",
        "markdown": output / "audit_summary.md",
        "manifest": output / "manifest.json",
    }
    atomic_write_json(files["report"], report)
    atomic_write_csv(
        files["summary"],
        [report["summary"]],
        (
            "status",
            "training_split_provenance",
            "final_research_eligible",
            "exploratory_pipeline_smoke_test",
            "blockers",
            "warnings",
        ),
    )
    atomic_write_text(files["markdown"], _markdown(report))
    write_manifest(
        path=files["manifest"],
        repo_root=repo_root,
        tool_name="quartz_defect_model_provenance",
        command=command,
        tool_sources=(
            "backend/research/model_provenance.py",
            "backend/research/readiness_common.py",
            "backend/research_benchmark.py",
        ),
        inputs=(
            ("model", model),
            ("dataset_yaml", yaml_path),
            ("training_args", args_path),
            ("training_source", source_path),
            ("dataset_audit", audit_path),
        ),
        outputs=(
            ("report", files["report"]),
            ("summary", files["summary"]),
            ("markdown", files["markdown"]),
        ),
    )
    return report, files
