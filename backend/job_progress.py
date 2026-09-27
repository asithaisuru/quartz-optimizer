"""File-backed calculation progress for backward-compatible job polling."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_NAMES = (
    "capture_quality",
    "reconstruction",
    "scale_calibration",
    "gem_candidate_generation",
    "optimization",
    "manufacturing_verification",
    "report_generation",
    "completed",
)

STAGE_STATUSES = {"pending", "running", "completed", "failed"}
PROGRESS_FILENAME = "status.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stage(stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "pending",
        "message": "",
        "started_at": None,
        "completed_at": None,
    }


def _empty_progress() -> dict[str, Any]:
    return {"schema_version": "1.0", "stages": [_stage(name) for name in STAGE_NAMES]}


def _path(job_folder: str | Path) -> Path:
    return Path(job_folder) / PROGRESS_FILENAME


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read(job_folder: str | Path) -> dict[str, Any] | None:
    path = _path(job_folder)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict) or not isinstance(value.get("stages"), list):
            return None
        by_name = {
            item.get("stage"): item
            for item in value["stages"]
            if isinstance(item, dict) and item.get("stage") in STAGE_NAMES
        }
        value["stages"] = [
            {**_stage(name), **by_name.get(name, {})}
            for name in STAGE_NAMES
        ]
        return value
    except (OSError, ValueError, TypeError):
        return None


def initialize_progress(
    job_folder: str | Path,
    completed_stages: tuple[str, ...] = (),
) -> dict[str, Any]:
    value = {}
    path = _path(job_folder)
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if isinstance(existing, dict):
                value.update(existing)
        except (OSError, ValueError, TypeError):
            pass
    value.update(_empty_progress())
    now = _utc_now()
    for item in value["stages"]:
        if item["stage"] in completed_stages:
            item.update(
                status="completed",
                message="Completed before detailed progress tracking.",
                started_at=now,
                completed_at=now,
            )
    _atomic_write(path, value)
    return value


def ensure_progress(job_folder: str | Path) -> dict[str, Any]:
    return _read(job_folder) or initialize_progress(job_folder)


def update_stage(
    job_folder: str | Path,
    stage: str,
    status: str,
    message: str,
) -> dict[str, Any]:
    if stage not in STAGE_NAMES:
        raise ValueError(f"Unsupported calculation stage: {stage}")
    if status not in STAGE_STATUSES:
        raise ValueError(f"Unsupported calculation stage status: {status}")
    value = ensure_progress(job_folder)
    now = _utc_now()
    for item in value["stages"]:
        if item["stage"] != stage:
            continue
        item["status"] = status
        item["message"] = str(message)
        if status in {"running", "completed", "failed"} and not item["started_at"]:
            item["started_at"] = now
        item["completed_at"] = now if status in {"completed", "failed"} else None
        break
    _atomic_write(_path(job_folder), value)
    return value


def reset_from_stage(job_folder: str | Path, stage: str) -> dict[str, Any]:
    if stage not in STAGE_NAMES:
        raise ValueError(f"Unsupported calculation stage: {stage}")
    value = ensure_progress(job_folder)
    reset = False
    for item in value["stages"]:
        reset = reset or item["stage"] == stage
        if reset:
            item.update(_stage(item["stage"]))
    _atomic_write(_path(job_folder), value)
    return value


def fail_running_stage(job_folder: str | Path, message: str) -> dict[str, Any]:
    value = ensure_progress(job_folder)
    running = next(
        (item["stage"] for item in value["stages"] if item["status"] == "running"),
        None,
    )
    if running is None:
        return value
    return update_stage(job_folder, running, "failed", message)


def _legacy_progress(job_folder: str | Path) -> dict[str, Any]:
    """Return a non-persisted compatibility view for jobs made by older builds."""

    value = _empty_progress()
    folder = Path(job_folder)
    status = {}
    try:
        with (folder / "status.json").open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            status = loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError, TypeError):
        pass
    if status.get("status") == "Completed" and (folder / "analysis_report.json").is_file():
        timestamp = status.get("timestamp")
        completed_at = (
            datetime.fromtimestamp(float(timestamp), timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
            if isinstance(timestamp, (int, float))
            else None
        )
        for item in value["stages"]:
            item.update(
                status="completed",
                message="Completed before detailed progress tracking.",
                started_at=None,
                completed_at=completed_at,
            )
    elif status.get("status") in {"Failed", "Cancelled"}:
        value["stages"][0].update(
            status="failed",
            message=str(status.get("message") or status.get("status")),
            started_at=None,
            completed_at=None,
        )
    return value


def progress_response(job_id: str, job_folder: str | Path) -> dict[str, Any]:
    value = _read(job_folder) or _legacy_progress(job_folder)
    stages = value["stages"]
    completed = sum(item["status"] == "completed" for item in stages)
    if any(item["status"] == "failed" for item in stages):
        overall = "failed"
    elif completed == len(stages):
        overall = "completed"
    elif any(item["status"] == "running" for item in stages):
        overall = "running"
    else:
        overall = "pending"
    current = next(
        (item["stage"] for item in stages if item["status"] == "running"),
        None,
    )
    if current is None:
        current = next(
            (item["stage"] for item in stages if item["status"] == "failed"),
            None,
        )
    if current is None:
        current = next(
            (item["stage"] for item in stages if item["status"] == "pending"),
            "completed",
        )
    return {
        "job_id": job_id,
        "overall_status": overall,
        "current_stage": current,
        "progress_percent": round(100.0 * completed / len(stages), 1),
        "stages": stages,
    }
