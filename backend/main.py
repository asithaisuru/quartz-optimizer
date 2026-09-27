import sys

# Windows consoles default to a legacy codepage (e.g. cp1252) that can't
# encode the emoji used throughout this pipeline's log output, crashing
# any print() that hits one (seen as e.g. "'charmap' codec can't encode
# character ..."). Force UTF-8 stdio before anything else runs so those
# prints — including deep in the optimizer/AI pipeline — never take the
# whole request down over a log line.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import shutil
import os
import uuid
import json
import time
import logging
import math
import hashlib
import re
from pathlib import Path
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    BackgroundTasks,
    Form,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
from dotenv import load_dotenv

from capture_quality_api import router as capture_quality_router
from job_progress import (
    ensure_progress,
    fail_running_stage,
    initialize_progress,
    progress_response,
    reset_from_stage,
    update_stage,
)
from yield_calculator import calculate_gem_stats
from report_generator import create_pdf, find_existing_pdf, is_valid_pdf

load_dotenv()

app = FastAPI(title="Quartz Gemstone Optimizer")

CONFIG_PATH = os.getenv("STORAGE_PATH", os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(CONFIG_PATH) == "backend":
    CONFIG_PATH = os.path.dirname(CONFIG_PATH)
JOBS_DIR = os.path.join(CONFIG_PATH, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PDF_ERROR_FILENAME = "pdf_report_error.json"
PDF_CACHE_FILENAME = "report_pdf_cache.json"
JOB_METADATA_FILENAME = "job_metadata.json"
RECONSTRUCTION_QUALITY_FILENAME = "reconstruction_quality.json"
OPTIMIZER_MESH_VALIDATION_FILENAME = "optimizer_mesh_validation.json"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
EXTENDED_SEARCH_BUDGET_MULTIPLIER = 3.0
try:
    EXTENDED_SEARCH_RESOURCE_LIMIT_SECONDS = max(
        360.0,
        float(os.getenv("EXTENDED_SEARCH_RESOURCE_LIMIT_SECONDS", "1800")),
    )
except (TypeError, ValueError):
    EXTENDED_SEARCH_RESOURCE_LIMIT_SECONDS = 1800.0
EXTENDED_SEARCH_SCOPE_MESSAGE = (
    "Extended search explores additional candidate solutions within the "
    "implemented search strategy."
)
PDF_DOWNLOAD_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
}
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
app.mount("/files", StaticFiles(directory=JOBS_DIR), name="files")
app.include_router(capture_quality_router)


@app.on_event("startup")
async def mark_interrupted_jobs():
    """
    On every server start, any job still in Processing/Resuming state was
    killed mid-run (server restart, crash, hot-reload).  Mark it Failed so
    the frontend shows the Resume screen instead of a frozen progress bar.
    """
    if not os.path.exists(JOBS_DIR):
        return
    marked = 0
    for job_id in os.listdir(JOBS_DIR):
        extended_path = Path(JOBS_DIR) / job_id / "extended_search" / "status.json"
        extended = _read_json_object(extended_path) if extended_path.is_file() else None
        if extended and extended.get("running"):
            extended.update(
                running=False,
                finished_at_epoch=time.time(),
                outcome="failed",
                search_state="failed",
                message="Server restarted during extended optimization.",
                error="interrupted_by_server_restart",
            )
            _atomic_write_json(extended_path, extended)
        status_file = os.path.join(JOBS_DIR, job_id, "status.json")
        if not os.path.exists(status_file):
            continue
        try:
            with open(status_file) as f:
                data = json.load(f)
            if data.get("status") in ("Processing", "Resuming"):
                data["status"]  = "Failed"
                data["message"] = "Server restarted — click Resume to continue."
                with open(status_file, "w") as f:
                    json.dump(data, f)
                marked += 1
        except Exception:
            pass
    if marked:
        print(f"[startup] Marked {marked} interrupted job(s) as Failed "
              f"(Resume is available for each).", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def update_job_status(job_folder, step, progress, message, status="Processing"):
    status_file = os.path.join(job_folder, "status.json")
    existing = {}
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                existing = json.load(f)
            if existing.get("status") == "Cancelled" and status != "Failed":
                return
        except Exception:
            pass

    data = {
        **(existing if isinstance(existing, dict) else {}),
        "status": status, "step": step,
        "progress": progress, "message": message,
        "timestamp": time.time()
    }
    try:
        _atomic_write_json(status_file, data)
    except Exception:
        pass


def check_if_cancelled(job_folder):
    status_file = os.path.join(job_folder, "status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                data = json.load(f)
            if data.get("status") == "Cancelled":
                raise Exception("Cancelled by user")
        except Exception as e:
            if str(e) == "Cancelled by user":
                raise


def _log(job_id, phase, msg):
    print(f"[{job_id[:8]}] [{phase}] {msg}", flush=True)


def _atomic_write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_positive_rough_weight(value):
    if value in (None, ""):
        raise ValueError("Rough weight is required before reconstruction.")
    try:
        weight = float(value)
    except (TypeError, ValueError):
        raise ValueError("Rough weight must be a positive number in carats.")
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("Rough weight must be a positive number in carats.")
    return weight


def _required_metadata_text(value, label):
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail=f"{label} is required.")
    if len(text) > 500:
        raise HTTPException(
            status_code=422,
            detail=f"{label} must be 500 characters or fewer.",
        )
    return text


def _natural_filename_key(filename):
    value = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    )


def _safe_upload_basename(filename):
    return str(filename or "upload").replace("\\", "/").rsplit("/", 1)[-1]


def _store_upload_with_hash(upload, destination):
    digest = hashlib.sha256()
    size_bytes = 0
    with open(destination, "wb") as output:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def _require_reconstruction_quality(mesh_path, job_folder):
    from reconstruction_quality import assess_reconstruction_quality

    report = assess_reconstruction_quality(mesh_path)
    _atomic_write_json(
        Path(job_folder) / RECONSTRUCTION_QUALITY_FILENAME,
        report,
    )
    if not report.get("passed"):
        raise RuntimeError(
            report.get("message") or "Reconstruction insufficient for optimization."
        )
    return report


def _pdf_error_path(job_folder):
    return Path(job_folder) / PDF_ERROR_FILENAME


def _read_pdf_error_record(job_folder):
    error_path = _pdf_error_path(job_folder)
    if not error_path.is_file():
        return None
    try:
        with error_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return {"error": "PDF report generation failed."}


def _read_pdf_error(job_folder, cache_key=None):
    value = _read_pdf_error_record(job_folder)
    if value is None:
        return None
    if cache_key and value.get("cache_key") not in (None, cache_key):
        return None
    return str(value.get("error") or "PDF report generation failed.")


def _pdf_cache_path(job_folder):
    return Path(job_folder) / PDF_CACHE_FILENAME


def _sha256_file(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _pdf_cache_key(result_id, report_path):
    return {
        "selected_result_id": result_id,
        "report_sha256": _sha256_file(report_path),
    }


def _read_pdf_cache(job_folder):
    try:
        with _pdf_cache_path(job_folder).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _pdf_cache_matches(job_folder, cache_key):
    return bool(cache_key and _read_pdf_cache(job_folder) == cache_key)


def _generate_pdf_report(
    job_folder,
    job_id,
    stats,
    force=False,
    artifact_path=None,
    cache_key=None,
):
    if cache_key is None:
        root_report = Path(job_folder) / "analysis_report.json"
        if root_report.is_file():
            cache_key = _pdf_cache_key("result_v1", root_report)
    existing = find_existing_pdf(job_folder)
    error_path = _pdf_error_path(job_folder)
    if (
        existing
        and not force
        and not _read_pdf_error(job_folder, cache_key)
        and (cache_key is None or _pdf_cache_matches(job_folder, cache_key))
    ):
        return existing
    try:
        if artifact_path is None:
            output_path = create_pdf(job_folder, job_id, stats)
        else:
            output_path = create_pdf(
                job_folder,
                job_id,
                stats,
                artifact_path=str(artifact_path),
            )
        if not is_valid_pdf(output_path):
            raise ValueError("Generated report failed PDF signature validation.")
        if cache_key is not None:
            _atomic_write_json(_pdf_cache_path(job_folder), cache_key)
        if error_path.exists():
            error_path.unlink()
        return output_path
    except Exception as exc:
        logger.exception("PDF report generation failed for job %s", job_id)
        _atomic_write_json(
            error_path,
            {
                "error": f"PDF report generation failed: {str(exc)[:300]}",
                "timestamp": time.time(),
                "cache_key": cache_key,
            },
        )
        return None


def _pdf_download_filename(job_id):
    return f"quartz-analysis-{job_id}.pdf"


def _pdf_status_fields(job_folder, job_id):
    pdf_path = find_existing_pdf(job_folder)
    effective = _resolve_effective_result(job_folder)
    cache_key = _pdf_cache_key(
        effective["result_id"], effective["report_path"]
    )
    error = _read_pdf_error(job_folder, cache_key)
    available = bool(pdf_path and not error)
    return {
        "pdf_report_available": available,
        "pdf_report_url": (
            f"/api/jobs/{job_id}/report/pdf" if available else None
        ),
        "pdf_report_filename": _pdf_download_filename(job_id),
        "pdf_report_error": error,
    }


def _validated_job_folder(job_id):
    try:
        parsed = uuid.UUID(job_id)
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid job ID.")
    if str(parsed) != str(job_id).lower():
        raise HTTPException(status_code=400, detail="Invalid job ID.")

    jobs_root = Path(JOBS_DIR).resolve()
    job_folder = (jobs_root / str(parsed)).resolve()
    try:
        job_folder.relative_to(jobs_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID.")
    if not job_folder.is_dir():
        raise HTTPException(status_code=404, detail="Job not found.")
    return job_folder


def _read_job_state(job_folder):
    status_path = Path(job_folder) / "status.json"
    if not status_path.is_file():
        return {}
    try:
        with status_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_json_object(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _extended_search_root(job_folder):
    return Path(job_folder) / "extended_search"


def _extended_status_path(job_folder):
    return _extended_search_root(job_folder) / "status.json"


def _extended_result_path(job_folder):
    return _extended_search_root(job_folder) / "result_v2" / "analysis_report.json"


def _extended_manifest_path(job_folder):
    return _extended_search_root(job_folder) / "results.json"


def _resolve_effective_result(job_folder):
    from effective_result import resolve_effective_result
    return resolve_effective_result(job_folder)


def get_effective_result(job_id):
    """Return the effective persisted result bundle for a validated job ID."""

    return _resolve_effective_result(_validated_job_folder(job_id))


def _effective_result_status_fields(job_folder, job_id):
    effective = _resolve_effective_result(job_folder)
    job_root = Path(job_folder)
    base_url = f"{API_BASE_URL}/files/{job_id}"
    timestamp = int(time.time())
    report_path = effective["report_path"]
    artifact_path = effective["artifact_path"]
    artifact_relative = artifact_path.relative_to(job_root).as_posix()
    values = {
        "effective_result_id": effective["result_id"],
        "effective_report_hash": effective["report_hash"],
        "result_asset_base_url": f"{base_url}/{artifact_relative}",
    }
    if report_path.is_file():
        report_relative = report_path.relative_to(job_root).as_posix()
        values["report_url"] = (
            f"{base_url}/{report_relative}?t={timestamp}"
        )
    cut_path = artifact_path / "best_cut.ply"
    if cut_path.is_file():
        cut_relative = cut_path.relative_to(job_root).as_posix()
        values["cut_url"] = f"{base_url}/{cut_relative}?t={timestamp}"
    return values


def _generate_effective_pdf(job_folder, job_id, force=False):
    effective = _resolve_effective_result(job_folder)
    stats = _read_json_object(effective["report_path"])
    if stats is None:
        return None
    cache_key = _pdf_cache_key(
        effective["result_id"], effective["report_path"]
    )
    return _generate_pdf_report(
        job_folder,
        job_id,
        stats,
        force=force,
        artifact_path=effective["artifact_path"],
        cache_key=cache_key,
    )


def _manufacturing_status(report):
    plan = report.get("manufacturing_plan") if isinstance(report, dict) else None
    return plan.get("status") if isinstance(plan, dict) else None


def _best_summary(report):
    if not isinstance(report, dict):
        return {
            "yield_percent": None,
            "weight": None,
            "gem_count": None,
            "manufacturing_status": None,
        }
    details = report.get("gem_details")
    return {
        "yield_percent": report.get("yield_percent"),
        "weight": report.get("estimated_cut_carats"),
        "gem_count": len(details) if isinstance(details, list) else None,
        "manufacturing_status": _manufacturing_status(report),
    }


def _candidate_count(report):
    if not isinstance(report, dict):
        return 0
    values = []
    diagnostics = report.get("optimizer_diagnostics")
    if isinstance(diagnostics, dict):
        values.append(diagnostics.get("candidate_count"))
    for option in report.get("options") or []:
        if isinstance(option, dict):
            option_diagnostics = option.get("optimizer_diagnostics")
            if isinstance(option_diagnostics, dict):
                values.append(option_diagnostics.get("candidate_count"))
    numeric = [int(value) for value in values if isinstance(value, (int, float))]
    return max(numeric, default=0)


def _extended_search_hit_resource_limit(report):
    """Read persisted optimizer stop diagnostics without changing its search."""

    if not isinstance(report, dict):
        return False
    diagnostics = []
    top_level = report.get("optimizer_diagnostics")
    if isinstance(top_level, dict):
        diagnostics.append(top_level)
    for option in report.get("options") or []:
        if not isinstance(option, dict):
            continue
        value = option.get("optimizer_diagnostics")
        if isinstance(value, dict):
            diagnostics.append(value)

    for diagnostic in diagnostics:
        for key in ("preserve_fill", "repacked_search", "pocket_fill"):
            search = diagnostic.get(key)
            if not isinstance(search, dict):
                continue
            if search.get("timed_out"):
                return True
            if search.get("stop_reason") in {
                "time_budget_exhausted",
                "safety_ceiling_reached",
            }:
                return True
            verification = search.get("verification")
            if isinstance(verification, dict) and verification.get("timed_out"):
                return True
    return False


def _normalized_extended_state(status):
    if status.get("running"):
        return "running"
    state = status.get("search_state") or status.get("outcome")
    legacy = {
        "completed": "completed_improvement",
        "no_improvement": "completed_no_improvement",
        "timed_out": "resource_stopped",
    }
    return legacy.get(state, state or "idle")


def _remaining_space_metadata(search_state, improvements_found=0):
    if int(improvements_found or 0) > 0:
        evaluation_status = "extended_search_completed_candidate_found"
    else:
        evaluation_status = {
            "idle": "initial_bounded_search",
            "running": "extended_search_running",
            "completed_no_improvement": (
                "extended_search_completed_no_candidate"
            ),
            "completed_improvement": (
                "extended_search_completed_candidate_found"
            ),
            "cancelled": "extended_search_cancelled",
            "failed": "extended_search_failed",
            "resource_stopped": "extended_search_resource_stopped",
        }.get(search_state, "initial_bounded_search")
    diagnostic_message = {
        "initial_bounded_search": (
            "Available geometric space remains. The initial bounded search "
            "ended before exhaustive rejection, so this space is diagnostic "
            "only and not a verified gemstone candidate."
        ),
        "extended_search_running": (
            "Extended search is currently re-evaluating this remaining "
            "geometric space."
        ),
        "extended_search_completed_no_candidate": (
            "Extended search completed. No higher-yield verified gemstone "
            "placement was found in this remaining geometric space."
        ),
        "extended_search_completed_candidate_found": (
            "Extended search found an improved verified result. This "
            "remaining-space diagnostic is superseded by the updated plan."
        ),
        "extended_search_cancelled": (
            "Extended search was stopped before final evaluation of this "
            "remaining geometric space."
        ),
        "extended_search_failed": (
            "Extended search failed before final evaluation of this remaining "
            "geometric space."
        ),
        "extended_search_resource_stopped": (
            "Extended search stopped at an implemented resource boundary "
            "before proving any higher-yield verified placement for this "
            "remaining geometric space."
        ),
    }[evaluation_status]
    return {
        "remaining_space_type": "geometric_diagnostic",
        "candidate_verified": False,
        "manufacturing_verified": False,
        "search_evaluation_status": evaluation_status,
        "diagnostic_message": diagnostic_message,
    }


def _is_verified_report(report):
    return _manufacturing_status(report) in {"complete", "no_separation_required"}


def _is_better_verified(candidate, incumbent):
    if not _is_verified_report(candidate):
        return False
    try:
        candidate_yield = float(candidate.get("yield_percent"))
    except (TypeError, ValueError):
        return False
    try:
        incumbent_yield = float(incumbent.get("yield_percent"))
    except (TypeError, ValueError, AttributeError):
        incumbent_yield = float("-inf")
    return candidate_yield > incumbent_yield


def _extended_incumbent(job_folder):
    improved = _read_json_object(_extended_result_path(job_folder))
    if improved is not None:
        return improved
    return _read_json_object(Path(job_folder) / "analysis_report.json")


def _extended_available(job_folder):
    folder = Path(job_folder)
    if _read_job_state(folder).get("status") != "Completed":
        return False
    if _extended_incumbent(folder) is None:
        return False
    if _read_json_object(folder / "job_config.json") is None:
        return False
    return (folder / "dense" / "final_textured_model.ply").is_file()


def _extended_status_response(job_folder):
    job_id = Path(job_folder).name
    effective_fields = _effective_result_status_fields(job_folder, job_id)
    effective_fields.update(_pdf_status_fields(job_folder, job_id))
    status = _read_json_object(_extended_status_path(job_folder))
    if status is None:
        incumbent = _extended_incumbent(job_folder)
        return {
            "search_mode": "extended",
            "running": False,
            "elapsed_seconds": 0,
            "current_best": _best_summary(incumbent),
            "statistics": {
                "candidates_tested": 0,
                "improvements_found": 0,
                "elapsed_time": 0,
                "search_state": "idle",
            },
            "status": "idle",
            "message": "Extended optimization has not been started.",
            "remaining_space_metadata": _remaining_space_metadata("idle"),
            **effective_fields,
        }
    started_at = status.get("started_at_epoch")
    finished_at = status.get("finished_at_epoch")
    if isinstance(started_at, (int, float)):
        end = finished_at if isinstance(finished_at, (int, float)) else time.time()
        elapsed = max(0, int(end - started_at))
    else:
        elapsed = 0
    search_state = _normalized_extended_state(status)
    improvements_found = int(status.get("improvements_found", 0))
    return {
        "search_mode": "extended",
        "running": bool(status.get("running", False)),
        "elapsed_seconds": elapsed,
        "current_best": status.get("current_best") or _best_summary(
            _extended_incumbent(job_folder)
        ),
        "statistics": {
            "candidates_tested": int(status.get("candidates_tested", 0)),
            "improvements_found": improvements_found,
            "elapsed_time": elapsed,
            "search_state": search_state,
        },
        "status": search_state,
        "message": status.get("message", ""),
        "remaining_space_metadata": _remaining_space_metadata(
            search_state,
            improvements_found,
        ),
        **effective_fields,
    }


def _copy_extended_inputs(job_folder, run_folder):
    source_root = Path(job_folder)
    run_root = Path(run_folder)
    dense = run_root / "dense"
    dense.mkdir(parents=True, exist_ok=True)
    source_mesh = source_root / "dense" / "final_textured_model.ply"
    if not source_mesh.is_file():
        raise FileNotFoundError("Canonical final_textured_model.ply is missing.")
    target_mesh = dense / "final_textured_model.ply"
    shutil.copy2(source_mesh, target_mesh)
    for name in ("defects.ply", "no_cut_defects.ply", "defect_associations.json"):
        source = source_root / "dense" / name
        if source.is_file():
            shutil.copy2(source, dense / name)
    detections = run_root / "detections"
    for name in ("policy_summary.json", "policy_decisions.json"):
        source = source_root / "detections" / name
        if source.is_file():
            detections.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, detections / name)
    return target_mesh


def _store_extended_result(job_folder, run_folder, report):
    result_root = _extended_search_root(job_folder) / "result_v2"
    result_root.mkdir(parents=True, exist_ok=True)
    for source in (Path(run_folder) / "dense").glob("*.ply"):
        shutil.copy2(source, result_root / source.name)
    _atomic_write_json(result_root / "analysis_report.json", report)


class _ExtendedSearchCancelled(Exception):
    pass


class _ExtendedSearchResourceStopped(Exception):
    pass


def run_extended_search_task(job_id):
    """Run one isolated optimizer pass, retaining only a verified improvement."""

    import tempfile

    job_folder = Path(JOBS_DIR) / job_id
    status_path = _extended_status_path(job_folder)
    status = _read_json_object(status_path) or {}
    incumbent = _extended_incumbent(job_folder)
    config = _read_json_object(job_folder / "job_config.json")
    if incumbent is None or config is None:
        status.update(
            running=False,
            finished_at_epoch=time.time(),
            outcome="failed",
            search_state="failed",
            message="Extended optimization inputs are no longer available.",
            error="missing_extended_search_inputs",
        )
        _atomic_write_json(status_path, status)
        return

    extended_root = _extended_search_root(job_folder)
    extended_root.mkdir(parents=True, exist_ok=True)
    try:
        known_weight = format(
            _parse_positive_rough_weight(config.get("known_weight")),
            ".12g",
        )
        with tempfile.TemporaryDirectory(
            prefix="run-", dir=str(extended_root)
        ) as temporary:
            run_folder = Path(temporary)
            mesh_path = _copy_extended_inputs(job_folder, run_folder)
            _require_reconstruction_quality(mesh_path, job_folder)

            def reporter(_current, _total, message):
                latest = _read_json_object(status_path) or status
                if latest.get("cancel_requested"):
                    raise _ExtendedSearchCancelled("Extended optimization was cancelled.")
                started = latest.get("started_at_epoch")
                resource_limit = latest.get(
                    "resource_limit_seconds",
                    EXTENDED_SEARCH_RESOURCE_LIMIT_SECONDS,
                )
                if (
                    isinstance(started, (int, float))
                    and time.time() - started > float(resource_limit)
                ):
                    raise _ExtendedSearchResourceStopped(
                        "Extended search stopped at the configured system resource "
                        "limit; additional solutions may remain within the bounded "
                        "strategy."
                    )
                latest["message"] = f"{EXTENDED_SEARCH_SCOPE_MESSAGE} {message}"
                _atomic_write_json(status_path, latest)

            candidate = calculate_gem_stats(
                str(mesh_path),
                known_carats=known_weight,
                preferred_shape=config.get("preferred_shape"),
                cut_mode=config.get("cut_mode", "multi"),
                job_folder=str(run_folder),
                progress_callback=reporter,
                optimizer_settings=config.get("optimizer_settings"),
                optimizer_search_budget_multiplier=EXTENDED_SEARCH_BUDGET_MULTIPLIER,
            )
            latest = _read_json_object(status_path) or status
            if latest.get("cancel_requested"):
                raise _ExtendedSearchCancelled("Extended optimization was cancelled.")
            resource_limit = float(
                latest.get(
                    "resource_limit_seconds",
                    EXTENDED_SEARCH_RESOURCE_LIMIT_SECONDS,
                )
            )
            if (
                time.time() - float(latest.get("started_at_epoch", time.time()))
                > resource_limit
            ):
                raise _ExtendedSearchResourceStopped(
                    "Extended search stopped at the configured system resource "
                    "limit; additional solutions may remain within the bounded "
                    "strategy."
                )
            if not isinstance(candidate, dict) or candidate.get("error"):
                raise RuntimeError(
                    str(candidate.get("error") if isinstance(candidate, dict) else "")
                    or "Extended optimization did not return a result."
                )
            improved = _is_better_verified(candidate, incumbent)
            if improved:
                _store_extended_result(job_folder, run_folder, candidate)
                incumbent = candidate
                manifest = _read_json_object(_extended_manifest_path(job_folder)) or {}
                manifest.update(
                    result_v2={
                        "report": "extended_search/result_v2/analysis_report.json",
                        "summary": _best_summary(candidate),
                    },
                    best_result="result_v2",
                )
                _atomic_write_json(_extended_manifest_path(job_folder), manifest)
            resource_stopped = _extended_search_hit_resource_limit(candidate)
            if resource_stopped:
                search_state = "resource_stopped"
                message = (
                    "Extended search reached an implemented strategy resource "
                    "limit. A higher-yield verified result was stored, but "
                    "additional candidates may remain."
                    if improved else
                    "Extended search reached an implemented strategy resource "
                    "limit. No higher-yield verified manufacturing plan was "
                    "found before that limit."
                )
            elif improved:
                search_state = "completed_improvement"
                message = "A higher-yield verified manufacturing plan was stored."
            else:
                search_state = "completed_no_improvement"
                message = (
                    "No higher-yield verified manufacturing plan was found within "
                    "the implemented extended-search strategy."
                )
            status.update(
                running=False,
                finished_at_epoch=time.time(),
                current_best=_best_summary(incumbent),
                candidates_tested=_candidate_count(candidate),
                improvements_found=1 if improved else 0,
                outcome=search_state,
                search_state=search_state,
                message=message,
            )
    except (_ExtendedSearchCancelled, _ExtendedSearchResourceStopped) as exc:
        outcome = (
            "cancelled"
            if isinstance(exc, _ExtendedSearchCancelled)
            else "resource_stopped"
        )
        status.update(
            running=False,
            finished_at_epoch=time.time(),
            current_best=_best_summary(incumbent),
            outcome=outcome,
            search_state=outcome,
            message=str(exc),
        )
    except Exception as exc:
        logger.exception("Extended optimization failed for job %s", job_id)
        status.update(
            running=False,
            finished_at_epoch=time.time(),
            current_best=_best_summary(incumbent),
            error=str(exc)[:300],
            outcome="failed",
            search_state="failed",
            message="Extended optimization failed.",
        )
    _atomic_write_json(status_path, status)


# ---------------------------------------------------------------------------
# Checkpoint helpers — used to skip phases that already completed
# ---------------------------------------------------------------------------

def _frames_done(images_folder):
    if not os.path.exists(images_folder):
        return False
    imgs = [f for f in os.listdir(images_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return len(imgs) >= 10


def _colmap_done(job_folder):
    fused = os.path.join(job_folder, "dense", "fused.ply")
    return os.path.exists(fused) and os.path.getsize(fused) > 5000


def _mesh_done(job_folder):
    mesh = os.path.join(job_folder, "dense", "final_textured_model.ply")
    return os.path.exists(mesh) and os.path.getsize(mesh) > 1000


def _selected_viewer_mesh(job_folder):
    dense = Path(job_folder) / "dense"
    validation = _read_json_object(
        Path(job_folder) / OPTIMIZER_MESH_VALIDATION_FILENAME
    )
    if validation:
        selected = Path(str(validation.get("selected_viewer_mesh") or "")).name
        if selected in {"visual_aligned_stone.ply", "final_textured_model.ply"}:
            candidate = dense / selected
            if candidate.is_file():
                return candidate
    aligned = dense / "visual_aligned_stone.ply"
    if aligned.is_file():
        return aligned
    return dense / "final_textured_model.ply"


def _masking_done(job_folder):
    return os.path.exists(os.path.join(job_folder, "masking_done"))


def _ai_done(job_folder):
    return all(
        os.path.exists(os.path.join(job_folder, "detections", name))
        for name in ("policy_decisions.json", "policy_summary.json")
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def process_full_pipeline(
    job_id: str,
    job_folder: str,
    is_video: bool,
    scan_mode: str,
    known_weight: str = None,
    preferred_shape: str = None,
    cut_mode: str = "multi",
    optimizer_settings: dict = None
):
    ensure_progress(job_folder)
    try:
        known_weight_value = _parse_positive_rough_weight(known_weight)
        known_weight = format(known_weight_value, ".12g")
    except ValueError as exc:
        message = str(exc)
        update_job_status(job_folder, "Failed", 0, message, status="Failed")
        fail_running_stage(job_folder, message)
        return

    # Upload-only computer-vision dependencies are intentionally lazy. Existing
    # job status, report download, and recalculation must not wait for Torch,
    # rembg, or COLMAP modules to initialize.
    from colmap_runner import run_photogrammetry_pipeline
    from video_utils import extract_best_frames
    from cleanup_module import post_process_point_cloud
    from masking_utils import remove_backgrounds
    from ai_runner import run_ai_pipeline
    from fracture_mapper import map_fractures_to_3d

    images_folder = os.path.join(job_folder, "images")

    print(f"\n{'='*60}", flush=True)
    print(f"  JOB  {job_id}", flush=True)
    print(f"  scan={scan_mode}  shape={preferred_shape or 'auto'}  cut={cut_mode}", flush=True)
    print(f"{'='*60}", flush=True)

    update_stage(
        job_folder,
        "capture_quality",
        "running",
        "Preparing and validating captured frames...",
    )
    try:
        # ----------------------------------------------------------------
        # Phase 1 — Video frame extraction
        # ----------------------------------------------------------------
        check_if_cancelled(job_folder)
        if is_video:
            if _frames_done(images_folder):
                n = len([f for f in os.listdir(images_folder)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                _log(job_id, "Phase 1", f"✓ {n} frames already extracted — skipping.")
                update_job_status(job_folder, "Video Processing", 5,
                                  f"{n} frames already extracted — resuming.")
            else:
                video_files = sorted(
                    (
                        f for f in os.listdir(job_folder)
                        if f.lower().endswith(VIDEO_EXTENSIONS)
                    ),
                    key=_natural_filename_key,
                )
                _log(job_id, "Phase 1",
                     f"Extracting frames from {len(video_files)} video(s)...")
                update_job_status(job_folder, "Video Processing", 5,
                                  "Extracting frames...")
                total = 0
                idx = 0
                for vid in video_files:
                    check_if_cancelled(job_folder)
                    _log(job_id, "Phase 1", f"  Processing {vid}")
                    cnt = extract_best_frames(
                        os.path.join(job_folder, vid),
                        images_folder, target_frames=40, start_index=idx
                    )
                    _log(job_id, "Phase 1", f"  → {cnt} frames from {vid}")
                    total += cnt
                    idx += cnt
                _log(job_id, "Phase 1", f"Total frames: {total}")
                if total < 10:
                    raise Exception(
                        "Video extraction failed — too few sharp frames extracted.")

        # ----------------------------------------------------------------
        # Phase 1.5 — Background masking
        # ----------------------------------------------------------------
        check_if_cancelled(job_folder)
        if scan_mode == "turntable":
            if _masking_done(job_folder):
                _log(job_id, "Phase 1.5", "✓ Masking already done — skipping.")
                update_job_status(job_folder, "Masking", 20,
                                  "Masking already done — resuming.")
            else:
                _log(job_id, "Phase 1.5", "Removing backgrounds with AI (rembg)...")
                update_job_status(job_folder, "Masking", 20,
                                  "AI removing backgrounds...")
                remove_backgrounds(images_folder)
                # Write sentinel so resume can skip this phase
                open(os.path.join(job_folder, "masking_done"), "w").close()
                _log(job_id, "Phase 1.5", "Background removal complete.")
        else:
            _log(job_id, "Phase 1.5", "Handheld mode — skipping masking.")
            update_job_status(job_folder, "Masking", 20,
                               "Handheld mode — skipping masking...")
        update_stage(
            job_folder,
            "capture_quality",
            "completed",
            "Captured frames are ready for reconstruction.",
        )

        # ----------------------------------------------------------------
        # Phase 2 — COLMAP 3D reconstruction
        # ----------------------------------------------------------------
        check_if_cancelled(job_folder)
        update_stage(
            job_folder,
            "reconstruction",
            "running",
            "Reconstructing the rough-stone geometry...",
        )
        fused_ply_path = os.path.join(job_folder, "dense", "fused.ply")
        if _colmap_done(job_folder):
            _log(job_id, "Phase 2", f"✓ Fused point cloud exists — skipping COLMAP.")
            update_job_status(job_folder, "Reconstruction", 30,
                              "Point cloud already exists — resuming.")
            ply_path = fused_ply_path
        else:
            _log(job_id, "Phase 2", "Starting COLMAP photogrammetry pipeline...")
            update_job_status(job_folder, "Reconstruction", 30,
                              "Running photogrammetry...")
            ply_path = run_photogrammetry_pipeline(job_folder, scan_mode=scan_mode)
            _log(job_id, "Phase 2", f"Point cloud saved: {ply_path}")

        # ----------------------------------------------------------------
        # Phase 3 — Mesh cleanup
        # ----------------------------------------------------------------
        check_if_cancelled(job_folder)
        output_mesh = os.path.join(job_folder, "dense", "final_textured_model.ply")
        if _mesh_done(job_folder):
            _log(job_id, "Phase 3", "✓ Mesh exists — skipping Poisson reconstruction.")
            update_job_status(job_folder, "Meshing", 70,
                              "Mesh already exists — resuming.")
            final_model = output_mesh
        else:
            _log(job_id, "Phase 3", "Running Poisson surface reconstruction...")
            update_job_status(job_folder, "Meshing", 70,
                              "Generating watertight mesh...")
            final_model = post_process_point_cloud(ply_path, output_mesh)
            if not final_model:
                _log(job_id, "Phase 3",
                     "⚠️  Mesh cleanup failed — using raw point cloud.")
                shutil.copy(ply_path, output_mesh)
                final_model = output_mesh
            else:
                _log(job_id, "Phase 3", "Watertight mesh ready.")

        # ----------------------------------------------------------------
        # Phase 3.5 — Reconstruction quality gate
        # ----------------------------------------------------------------
        check_if_cancelled(job_folder)
        _log(job_id, "Phase 3.5", "Validating mesh topology and usable SDF volume...")
        update_job_status(
            job_folder,
            "Reconstruction Validation",
            75,
            "Validating watertightness, manifold topology, and SDF volume...",
        )
        try:
            quality_report = _require_reconstruction_quality(final_model, job_folder)
        except RuntimeError as exc:
            update_stage(job_folder, "reconstruction", "failed", str(exc))
            raise
        _log(
            job_id,
            "Phase 3.5",
            "Quality gate passed "
            f"({quality_report['metrics']['usable_sdf_voxel_count']} usable SDF voxels).",
        )

        # ----------------------------------------------------------------
        # Phase 4 — AI fracture detection
        # ----------------------------------------------------------------
        check_if_cancelled(job_folder)
        if _ai_done(job_folder):
            _log(job_id, "Phase 4",
                 "✓ Structured detector-policy results exist — skipping detection.")
            update_job_status(job_folder, "AI Analysis", 80,
                              "Detector-policy results already exist — resuming.")
        else:
            _log(job_id, "Phase 4",
                 "Running fracture/cloud detection...")
            update_job_status(job_folder, "AI Analysis", 80,
                               "Scanning for fractures & clouds...")
            run_ai_pipeline(job_folder)
        _log(job_id, "Phase 4",
             "Mapping approved 2D candidates → COLMAP sparse points...")
        map_fractures_to_3d(job_folder)
        _log(job_id, "Phase 4", "Policy-controlled defect mapping complete.")
        update_stage(
            job_folder,
            "reconstruction",
            "completed",
            "Reconstruction, mesh preparation, and 3D evidence mapping completed.",
        )

        # ----------------------------------------------------------------
        # Phase 5 — Yield & cut optimisation (always re-runs so params take effect)
        # ----------------------------------------------------------------
        check_if_cancelled(job_folder)
        _log(job_id, "Phase 5",
             f"Running gem packing optimiser (mode={cut_mode})...")
        update_job_status(job_folder, "Analysis", 90,
                          "Calculating optimal cut strategy...")
        def calculation_stage_reporter(stage, status, message):
            update_stage(job_folder, stage, status, message)

        stats = calculate_gem_stats(
            final_model,
            known_carats=known_weight,
            preferred_shape=preferred_shape,
            cut_mode=cut_mode,
            job_folder=job_folder,
            optimizer_settings=optimizer_settings,
            stage_callback=calculation_stage_reporter,
        )
        if stats.get("error"):
            raise RuntimeError(stats["error"])
        update_stage(
            job_folder,
            "report_generation",
            "running",
            "Writing analysis and PDF reports...",
        )
        with open(os.path.join(job_folder, "analysis_report.json"), "w") as f:
            json.dump(stats, f)
        pdf_path = _generate_pdf_report(job_folder, job_id, stats, force=True)
        update_stage(
            job_folder,
            "report_generation",
            "completed",
            (
                "Analysis and PDF reports generated."
                if pdf_path else "Analysis report generated; PDF is unavailable."
            ),
        )

        n_opts = len(stats.get("options", []))
        _log(job_id, "Phase 5",
             f"Optimiser done — {n_opts} cut option(s) generated.")

        update_job_status(job_folder, "Completed", 100, "Done", status="Completed")
        update_stage(job_folder, "completed", "completed", "Calculation completed.")
        print(f"\n✅ JOB COMPLETE  {job_id}\n", flush=True)

    except Exception as e:
        msg = str(e)
        if msg == "Cancelled by user":
            _log(job_id, "CANCELLED", "Job cancelled by user.")
            update_job_status(job_folder, "Cancelled", 0,
                              "Cancelled by user.", status="Cancelled")
        else:
            import traceback
            _log(job_id, "FAILED", msg)
            traceback.print_exc()
            update_job_status(job_folder, "Failed", 0, msg, status="Failed")
        fail_running_stage(job_folder, msg)


def run_recalculation_task(
    job_id: str,
    known_weight: str,
    preferred_shape: str = None,
    cut_mode: str = "multi",
    optimizer_settings: dict = None
):
    job_folder = os.path.join(JOBS_DIR, job_id)
    target_mesh = os.path.join(job_folder, "dense", "final_textured_model.ply")

    def reporter(curr, tot, msg):
        pct = int((curr / max(tot, 1)) * 100)
        update_job_status(job_folder, "Optimizing", pct,
                          f"{msg} ({curr}/{tot})")

    existing_status = _read_json_object(Path(job_folder) / "status.json") or {}
    if not isinstance(existing_status.get("stages"), list):
        initialize_progress(
            job_folder,
            completed_stages=("capture_quality", "reconstruction"),
        )
    else:
        reset_from_stage(job_folder, "scale_calibration")
    try:
        known_weight = format(
            _parse_positive_rough_weight(known_weight),
            ".12g",
        )
        print(f"\n[{job_id[:8]}] RECALCULATE  weight={known_weight}  "
              f"shape={preferred_shape or 'auto'}  cut={cut_mode}", flush=True)
        update_job_status(job_folder, "Optimizing", 0, "Starting recalculation...")
        def calculation_stage_reporter(stage, status, message):
            update_stage(job_folder, stage, status, message)

        _require_reconstruction_quality(target_mesh, job_folder)
        stats = calculate_gem_stats(
            target_mesh,
            known_carats=known_weight,
            preferred_shape=preferred_shape,
            cut_mode=cut_mode,
            job_folder=job_folder,
            progress_callback=reporter,
            optimizer_settings=optimizer_settings,
            stage_callback=calculation_stage_reporter,
        )
        if stats.get("error"):
            raise RuntimeError(stats["error"])
        update_stage(
            job_folder,
            "report_generation",
            "running",
            "Writing recalculated analysis and PDF reports...",
        )
        with open(os.path.join(job_folder, "analysis_report.json"), "w") as f:
            json.dump(stats, f)
        pdf_path = _generate_pdf_report(job_folder, job_id, stats, force=True)
        update_stage(
            job_folder,
            "report_generation",
            "completed",
            (
                "Recalculated analysis and PDF reports generated."
                if pdf_path else "Recalculated analysis generated; PDF is unavailable."
            ),
        )
        update_job_status(job_folder, "Completed", 100, "Done", status="Completed")
        update_stage(job_folder, "completed", "completed", "Calculation completed.")
        print(f"[{job_id[:8]}] RECALCULATE complete.", flush=True)
    except Exception as e:
        update_job_status(job_folder, "Failed", 0, str(e), status="Failed")
        fail_running_stage(job_folder, str(e))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _validate_cuttable_optimizer_settings(settings):
    def parse(name, label, maximum):
        value = settings.get(name)
        if value in (None, ""):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"{label} must be a number.")
        if not math.isfinite(number) or number <= 0 or number > maximum:
            raise HTTPException(
                status_code=422,
                detail=f"{label} must be greater than 0 and at most {maximum:g} mm.",
            )
        return number

    margin = parse("preform_margin_mm", "Preform margin", 5.0)
    parse("max_cut_depth_mm", "Maximum cut depth", 500.0)
    rough = settings.get("rough_clearance_mm")
    if margin is not None:
        try:
            rough_value = float(rough) if rough not in (None, "") else 0.8
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Rough clearance must be a number.")
        if not math.isfinite(rough_value) or rough_value < margin:
            raise HTTPException(
                status_code=422,
                detail="Rough clearance must be at least the preform margin.",
            )

@app.get("/shapes")
async def list_shapes():
    from gem_shapes import get_standard_shapes
    shapes = get_standard_shapes()
    return {"shapes": list(shapes.keys())}


@app.post("/upload")
async def create_job(
    bg_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    specimen_id: str = Form(...),
    source_folder: str = Form(...),
    known_weight: str = Form(...),
    is_video: bool = Form(False),
    scan_mode: str = Form("turntable"),
    preferred_shape: str = Form(None),
    cut_mode: str = Form("multi"),
    blade_kerf_mm: str = Form(None),
    rough_clearance_mm: str = Form(None),
    preform_margin_mm: str = Form(None),
    max_cut_depth_mm: str = Form(None),
    max_gems: str = Form(None),
    min_secondary_carat: str = Form(None),
    extra_gem_policy: str = Form(None)
):
    specimen_id = _required_metadata_text(specimen_id, "Specimen ID")
    source_folder = _required_metadata_text(source_folder, "Source folder")
    try:
        rough_weight_ct = _parse_positive_rough_weight(known_weight)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    known_weight = format(rough_weight_ct, ".12g")

    optimizer_settings = {
        "blade_kerf_mm": blade_kerf_mm,
        "rough_clearance_mm": rough_clearance_mm,
        "preform_margin_mm": preform_margin_mm,
        "max_cut_depth_mm": max_cut_depth_mm,
        "max_gems": max_gems,
        "min_secondary_carat": min_secondary_carat,
        "extra_gem_policy": extra_gem_policy,
    }
    _validate_cuttable_optimizer_settings(optimizer_settings)

    job_id = str(uuid.uuid4())
    job_folder = os.path.join(JOBS_DIR, job_id)
    os.makedirs(os.path.join(job_folder, "images"), exist_ok=True)

    initialize_progress(job_folder)
    update_job_status(job_folder, "Initializing", 0, "Uploading...")

    video_uploads = sorted(
        (
            upload for upload in files
            if str(upload.filename or "").lower().endswith(VIDEO_EXTENSIONS)
        ),
        key=lambda upload: _natural_filename_key(upload.filename),
    )
    video_upload_ids = {id(upload) for upload in video_uploads}
    non_video_uploads = [
        upload for upload in files if id(upload) not in video_upload_ids
    ]
    is_video = bool(video_uploads) or is_video

    video_records = []
    processing_order = []
    for position, upload in enumerate(video_uploads):
        original_filename = str(upload.filename or "")
        safe_filename = _safe_upload_basename(original_filename)
        extension = os.path.splitext(safe_filename)[1].lower()
        stored_filename = f"video_{position}{extension}"
        destination = os.path.join(job_folder, stored_filename)
        sha256, size_bytes = _store_upload_with_hash(upload, destination)
        record = {
            "processing_order": position,
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
        video_records.append(record)
        processing_order.append(
            {
                "position": position,
                "original_filename": original_filename,
                "stored_filename": stored_filename,
            }
        )

    for upload in non_video_uploads:
        safe_filename = _safe_upload_basename(upload.filename)
        destination = os.path.join(job_folder, "images", safe_filename)
        _store_upload_with_hash(upload, destination)

    metadata = {
        "schema_version": "1.0",
        "job_id": job_id,
        "specimen_id": specimen_id,
        "source_folder": source_folder,
        "rough_weight_ct": rough_weight_ct,
        "video_hashes": [
            {
                "original_filename": record["original_filename"],
                "sha256": record["sha256"],
            }
            for record in video_records
        ],
        "videos": video_records,
        "processing_order": processing_order,
        "created_at_epoch": time.time(),
    }
    _atomic_write_json(
        os.path.join(job_folder, JOB_METADATA_FILENAME),
        metadata,
    )

    # Save job parameters and immutable upload provenance for resume/audit.
    config = {
        "is_video":        is_video,
        "scan_mode":       scan_mode,
        "known_weight":    known_weight,
        "rough_weight_ct": rough_weight_ct,
        "specimen_id":     specimen_id,
        "source_folder":   source_folder,
        "video_hashes":    metadata["video_hashes"],
        "processing_order": processing_order,
        "preferred_shape": preferred_shape,
        "cut_mode":        cut_mode,
        "optimizer_settings": optimizer_settings,
    }
    _atomic_write_json(os.path.join(job_folder, "job_config.json"), config)

    bg_tasks.add_task(
        process_full_pipeline,
        job_id, job_folder, is_video, scan_mode,
        known_weight, preferred_shape, cut_mode, optimizer_settings
    )
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}/status"}


@app.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str, bg_tasks: BackgroundTasks):
    """
    Re-run the pipeline for an interrupted/failed/cancelled job.
    Phases that already produced output files are skipped automatically.
    """
    job_folder = os.path.join(JOBS_DIR, job_id)
    if not os.path.exists(job_folder):
        return JSONResponse({"error": "Job not found"}, status_code=404)

    config_path = os.path.join(job_folder, "job_config.json")
    if not os.path.exists(config_path):
        return JSONResponse(
            {"error": "No job_config.json — original parameters unknown. "
                      "Please start a new job."},
            status_code=400
        )

    with open(config_path) as f:
        config = json.load(f)
    try:
        _parse_positive_rough_weight(config.get("known_weight"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    update_job_status(job_folder, "Resuming", 0,
                      "Resuming from last checkpoint...")
    bg_tasks.add_task(
        process_full_pipeline,
        job_id, job_folder,
        config.get("is_video", True),
        config.get("scan_mode", "turntable"),
        config.get("known_weight"),
        config.get("preferred_shape"),
        config.get("cut_mode", "multi"),
        config.get("optimizer_settings")
    )
    return {"status": "Resuming", "job_id": job_id}


@app.post("/jobs/{job_id}/recalculate")
async def recalculate(
    job_id: str,
    bg_tasks: BackgroundTasks,
    known_weight: str = Form(...),
    preferred_shape: str = Form(None),
    cut_mode: str = Form("multi"),
    blade_kerf_mm: str = Form(None),
    rough_clearance_mm: str = Form(None),
    preform_margin_mm: str = Form(None),
    max_cut_depth_mm: str = Form(None),
    max_gems: str = Form(None),
    min_secondary_carat: str = Form(None),
    extra_gem_policy: str = Form(None)
):
    try:
        rough_weight_ct = _parse_positive_rough_weight(known_weight)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    known_weight = format(rough_weight_ct, ".12g")

    optimizer_settings = {
        "blade_kerf_mm": blade_kerf_mm,
        "rough_clearance_mm": rough_clearance_mm,
        "preform_margin_mm": preform_margin_mm,
        "max_cut_depth_mm": max_cut_depth_mm,
        "max_gems": max_gems,
        "min_secondary_carat": min_secondary_carat,
        "extra_gem_policy": extra_gem_policy,
    }
    _validate_cuttable_optimizer_settings(optimizer_settings)
    job_folder = _validated_job_folder(job_id)
    update_job_status(
        job_folder,
        "Optimizing",
        0,
        "Starting recalculation...",
    )
    config_path = Path(job_folder) / "job_config.json"
    config = _read_json_object(config_path) or {}
    config.update(
        known_weight=known_weight,
        rough_weight_ct=rough_weight_ct,
        preferred_shape=preferred_shape,
        cut_mode=cut_mode,
        optimizer_settings=optimizer_settings,
    )
    _atomic_write_json(config_path, config)
    metadata_path = Path(job_folder) / JOB_METADATA_FILENAME
    metadata = _read_json_object(metadata_path) or {}
    metadata["rough_weight_ct"] = rough_weight_ct
    weight_history = metadata.setdefault("rough_weight_history", [])
    if isinstance(weight_history, list):
        weight_history.append(
            {
                "rough_weight_ct": rough_weight_ct,
                "updated_at_epoch": time.time(),
                "reason": "recalculation",
            }
        )
    _atomic_write_json(metadata_path, metadata)
    bg_tasks.add_task(
        run_recalculation_task,
        job_id, known_weight, preferred_shape, cut_mode, optimizer_settings
    )
    return {"status": "Accepted", "message": "Recalculation started"}


@app.post("/jobs/{job_id}/cancel")
async def cancel(job_id: str):
    job_folder = os.path.join(JOBS_DIR, job_id)
    status_file = os.path.join(job_folder, "status.json")
    if os.path.exists(job_folder):
        extended_path = _extended_status_path(job_folder)
        extended = _read_json_object(extended_path) or {}
        if extended.get("running"):
            extended["cancel_requested"] = True
            extended["message"] = (
                "Cancellation requested; the current bounded optimizer step is "
                "stopping safely."
            )
            _atomic_write_json(extended_path, extended)
            return {"message": "Extended optimization cancellation requested"}
        existing = _read_json_object(status_file) or {}
        _atomic_write_json(
            status_file,
            {**existing, "status": "Cancelled", "message": "Cancelled by user"},
        )
        return {"message": "Cancelled"}
    return {"message": "Job not found"}


@app.get("/jobs/{job_id}/progress")
async def get_calculation_progress(job_id: str):
    job_folder = _validated_job_folder(job_id)
    return progress_response(job_id, job_folder)


@app.post("/jobs/{job_id}/extended-search")
async def start_extended_search(job_id: str, bg_tasks: BackgroundTasks):
    job_folder = _validated_job_folder(job_id)
    if not _extended_available(job_folder):
        return {"available": False}

    config = _read_json_object(Path(job_folder) / "job_config.json") or {}
    try:
        _parse_positive_rough_weight(config.get("known_weight"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    existing = _read_json_object(_extended_status_path(job_folder)) or {}
    if existing.get("running"):
        return {
            "started": False,
            "job_id": job_id,
            "message": "Extended optimization is already running",
        }

    incumbent = _extended_incumbent(job_folder)
    manifest = _read_json_object(_extended_manifest_path(job_folder))
    if manifest is None:
        _atomic_write_json(
            _extended_manifest_path(job_folder),
            {
                "schema_version": "1.0",
                "result_v1": {
                    "report": "analysis_report.json",
                    "summary": _best_summary(incumbent),
                },
                "result_v2": None,
                "best_result": "result_v1",
            },
        )
    _atomic_write_json(
        _extended_status_path(job_folder),
        {
            "schema_version": "1.0",
            "running": True,
            "started_at_epoch": time.time(),
            "finished_at_epoch": None,
            "current_best": _best_summary(incumbent),
            "candidates_tested": 0,
            "improvements_found": 0,
            "outcome": "running",
            "search_state": "running",
            "cancel_requested": False,
            "search_budget_multiplier": EXTENDED_SEARCH_BUDGET_MULTIPLIER,
            "resource_limit_seconds": EXTENDED_SEARCH_RESOURCE_LIMIT_SECONDS,
            "message": EXTENDED_SEARCH_SCOPE_MESSAGE,
        },
    )
    bg_tasks.add_task(run_extended_search_task, job_id)
    return {
        "started": True,
        "job_id": job_id,
        "message": "Extended optimization started",
    }


@app.get("/jobs/{job_id}/extended-search/status")
async def get_extended_search_status(job_id: str):
    job_folder = _validated_job_folder(job_id)
    if not _extended_available(job_folder):
        return {"available": False}
    return _extended_status_response(job_folder)


@app.get("/api/jobs/{job_id}/report/pdf")
@app.get("/jobs/{job_id}/report")
async def download_pdf_report(job_id: str):
    job_folder = _validated_job_folder(job_id)
    effective = _resolve_effective_result(job_folder)
    report_exists = effective["report_path"].is_file()
    cache_key = (
        _pdf_cache_key(effective["result_id"], effective["report_path"])
        if report_exists else None
    )
    pdf_path = find_existing_pdf(job_folder)
    pdf_error = _read_pdf_error(job_folder, cache_key)
    cache_is_current = bool(
        pdf_path
        and (
            not report_exists
            or _pdf_cache_matches(job_folder, cache_key)
        )
    )

    if not cache_is_current or pdf_error:
        job_state = _read_job_state(job_folder)
        status = str(job_state.get("status") or "").casefold()
        if status in {
            "processing",
            "resuming",
            "initializing",
            "optimizing",
        }:
            raise HTTPException(
                status_code=409,
                detail="PDF report is still being generated.",
            )

        if not report_exists:
            if status in {"failed", "cancelled"}:
                raise HTTPException(
                    status_code=409,
                    detail="PDF report is unavailable because the job did not complete.",
                )
            raise HTTPException(status_code=404, detail="PDF report not found.")

        pdf_path = _generate_effective_pdf(job_folder, job_id)
        if not pdf_path:
            raise HTTPException(
                status_code=500,
                detail="PDF report generation failed.",
            )

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=_pdf_download_filename(job_id),
        headers=PDF_DOWNLOAD_HEADERS,
    )


@app.get("/jobs/{job_id}/status")
async def get_status(job_id: str):
    job_folder = os.path.join(JOBS_DIR, job_id)
    status_file = os.path.join(job_folder, "status.json")

    if not os.path.exists(status_file):
        return {
            "status": "Not Found",
            "progress": 0,
            "job_id": job_id,
            **_pdf_status_fields(job_folder, job_id),
        }

    try:
        with open(status_file, "r") as f:
            data = json.load(f)

        if data["status"] == "Completed":
            base_url = f"{API_BASE_URL}/files"
            ts = int(time.time())

            viewer_mesh = _selected_viewer_mesh(job_folder)
            data["model_url"] = (
                f"{base_url}/{job_id}/dense/{viewer_mesh.name}?t={ts}"
            )

            data.update(_effective_result_status_fields(job_folder, job_id))

            policy_path = os.path.join(
                job_folder, "detections", "policy_summary.json"
            )
            defect_policy = None
            if os.path.exists(policy_path):
                try:
                    with open(policy_path, "r", encoding="utf-8") as handle:
                        defect_policy = json.load(handle)
                    data["defect_detection"] = defect_policy
                except (OSError, ValueError):
                    defect_policy = None
            elif (
                os.path.isdir(os.path.join(job_folder, "fractures"))
                or os.path.exists(
                    os.path.join(job_folder, "dense", "defects.ply")
                )
            ):
                data["defect_detection"] = {
                    "policy": "legacy_unverified",
                    "claim_status": "unavailable",
                    "legacy_input_used": True,
                    "reason": (
                        "Historical masks have no class-aware policy metadata."
                    ),
                }

            dp = os.path.join(job_folder, "dense", "defects.ply")
            associations_approved = False
            associations_path = os.path.join(
                job_folder, "dense", "defect_associations.json"
            )
            if os.path.exists(associations_path):
                try:
                    with open(
                        associations_path, "r", encoding="utf-8"
                    ) as handle:
                        associations = json.load(handle)
                    associations_approved = bool(
                        associations.get("status") == "complete"
                        and int(
                            associations.get("mapping_point_count", 0)
                        ) > 0
                        and defect_policy
                        and associations.get("policy")
                        == defect_policy.get("policy")
                    )
                    if defect_policy is not None:
                        data["defect_detection"]["mapped_point_count"] = int(
                            associations.get("mapping_point_count", 0)
                        )
                        data["defect_detection"]["no_cut_point_count"] = int(
                            associations.get("no_cut_point_count", 0)
                        )
                except (OSError, ValueError, TypeError):
                    associations_approved = False
            if (
                os.path.exists(dp)
                and defect_policy
                and int(defect_policy.get("mapping_count", 0)) > 0
                and not defect_policy.get("legacy_input_used", False)
                and associations_approved
            ):
                data["defects_url"] = (
                    f"{base_url}/{job_id}/dense/defects.ply?t={ts}")

        # Always expose the job_id so the frontend can offer a Resume button
        data["job_id"] = job_id
        quality_path = os.path.join(job_folder, RECONSTRUCTION_QUALITY_FILENAME)
        if os.path.exists(quality_path):
            data["reconstruction_quality_report_url"] = (
                f"{API_BASE_URL}/files/{job_id}/"
                f"{RECONSTRUCTION_QUALITY_FILENAME}?t={int(time.time())}"
            )
        data.update(_pdf_status_fields(job_folder, job_id))
        return data

    except Exception:
        return {
            "status": "Error reading status",
            "progress": 0,
            "job_id": job_id,
            **_pdf_status_fields(job_folder, job_id),
        }


# Real FastAPI instances expose router; lightweight legacy test doubles do not.
if hasattr(app, "router"):
    from preform_api import create_router as create_preform_router
    app.include_router(create_preform_router(_validated_job_folder, lambda: JOBS_DIR))
