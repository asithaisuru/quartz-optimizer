"""Shared read-only effective saved-result resolver."""
import hashlib
import json
from pathlib import Path

def _read_object(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None

def _hash_file(path):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def resolve_effective_result(job_folder):
    """Resolve the selected report and all of its mesh artifacts as one bundle."""

    job_root = Path(job_folder)
    result_id = "result_v1"
    report_path = job_root / "analysis_report.json"
    artifact_path = job_root / "dense"

    manifest = _read_object(job_root / "extended_search" / "results.json")
    extended_report = job_root / "extended_search" / "result_v2" / "analysis_report.json"
    extended_artifacts = extended_report.parent
    if (
        manifest is not None
        and manifest.get("best_result") == "result_v2"
        and extended_report.is_file()
        and extended_artifacts.is_dir()
    ):
        result_id = "result_v2"
        report_path = extended_report
        artifact_path = extended_artifacts

    return {
        "result_id": result_id,
        "report_path": report_path,
        "artifact_path": artifact_path,
        "report_hash": _hash_file(report_path),
    }

