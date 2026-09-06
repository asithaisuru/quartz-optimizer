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
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                existing = json.load(f)
            if existing.get("status") == "Cancelled" and status != "Failed":
                return
        except Exception:
            pass

    data = {
        "status": status, "step": step,
        "progress": progress, "message": message,
        "timestamp": time.time()
    }
    try:
        with open(status_file, "w") as f:
            json.dump(data, f)
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


def _pdf_error_path(job_folder):
    return Path(job_folder) / PDF_ERROR_FILENAME


def _read_pdf_error(job_folder):
    error_path = _pdf_error_path(job_folder)
    if not error_path.is_file():
        return None
    try:
        with error_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return str(value.get("error") or "PDF report generation failed.")
    except (OSError, ValueError, TypeError):
        return "PDF report generation failed."


def _generate_pdf_report(job_folder, job_id, stats, force=False):
    existing = find_existing_pdf(job_folder)
    error_path = _pdf_error_path(job_folder)
    if existing and not force and not error_path.exists():
        return existing
    try:
        output_path = create_pdf(job_folder, job_id, stats)
        if not is_valid_pdf(output_path):
            raise ValueError("Generated report failed PDF signature validation.")
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
            },
        )
        return None


def _pdf_download_filename(job_id):
    return f"quartz-analysis-{job_id}.pdf"


def _pdf_status_fields(job_folder, job_id):
    pdf_path = find_existing_pdf(job_folder)
    error = _read_pdf_error(job_folder)
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
                video_files = [f for f in os.listdir(job_folder)
                               if f.lower().endswith(('.mp4', '.mov', '.avi'))]
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

        # ----------------------------------------------------------------
        # Phase 2 — COLMAP 3D reconstruction
        # ----------------------------------------------------------------
        check_if_cancelled(job_folder)
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

        # ----------------------------------------------------------------
        # Phase 5 — Yield & cut optimisation (always re-runs so params take effect)
        # ----------------------------------------------------------------
        check_if_cancelled(job_folder)
        _log(job_id, "Phase 5",
             f"Running gem packing optimiser (mode={cut_mode})...")
        update_job_status(job_folder, "Analysis", 90,
                          "Calculating optimal cut strategy...")
        stats = calculate_gem_stats(
            final_model,
            known_carats=known_weight,
            preferred_shape=preferred_shape,
            cut_mode=cut_mode,
            job_folder=job_folder,
            optimizer_settings=optimizer_settings
        )
        with open(os.path.join(job_folder, "analysis_report.json"), "w") as f:
            json.dump(stats, f)
        _generate_pdf_report(job_folder, job_id, stats, force=True)

        n_opts = len(stats.get("options", []))
        _log(job_id, "Phase 5",
             f"Optimiser done — {n_opts} cut option(s) generated.")

        update_job_status(job_folder, "Completed", 100, "Done", status="Completed")
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


def run_recalculation_task(
    job_id: str,
    known_weight: str,
    preferred_shape: str = None,
    cut_mode: str = "multi",
    optimizer_settings: dict = None
):
    job_folder = os.path.join(JOBS_DIR, job_id)
    target_mesh = os.path.join(job_folder, "dense", "visual_aligned_stone.ply")
    if not os.path.exists(target_mesh):
        target_mesh = os.path.join(job_folder, "dense", "final_textured_model.ply")

    def reporter(curr, tot, msg):
        pct = int((curr / max(tot, 1)) * 100)
        update_job_status(job_folder, "Optimizing", pct,
                          f"{msg} ({curr}/{tot})")

    try:
        print(f"\n[{job_id[:8]}] RECALCULATE  weight={known_weight}  "
              f"shape={preferred_shape or 'auto'}  cut={cut_mode}", flush=True)
        update_job_status(job_folder, "Optimizing", 0, "Starting recalculation...")
        stats = calculate_gem_stats(
            target_mesh,
            known_carats=known_weight,
            preferred_shape=preferred_shape,
            cut_mode=cut_mode,
            job_folder=job_folder,
            progress_callback=reporter,
            optimizer_settings=optimizer_settings
        )
        with open(os.path.join(job_folder, "analysis_report.json"), "w") as f:
            json.dump(stats, f)
        _generate_pdf_report(job_folder, job_id, stats, force=True)
        update_job_status(job_folder, "Completed", 100, "Done", status="Completed")
        print(f"[{job_id[:8]}] RECALCULATE complete.", flush=True)
    except Exception as e:
        update_job_status(job_folder, "Failed", 0, str(e), status="Failed")


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
    is_video: bool = Form(False),
    scan_mode: str = Form("turntable"),
    known_weight: str = Form(None),
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
    job_id = str(uuid.uuid4())
    job_folder = os.path.join(JOBS_DIR, job_id)
    os.makedirs(os.path.join(job_folder, "images"), exist_ok=True)

    update_job_status(job_folder, "Initializing", 0, "Uploading...")

    vid_exts = ('.mp4', '.mov', '.avi')
    has_vid = any(f.filename.lower().endswith(vid_exts) for f in files)
    is_video = has_vid or is_video

    v_idx = 0
    for f in files:
        if is_video and f.filename.lower().endswith(vid_exts):
            dest = os.path.join(job_folder,
                                f"video_{v_idx}{os.path.splitext(f.filename)[1]}")
            with open(dest, "wb") as buf:
                shutil.copyfileobj(f.file, buf)
            v_idx += 1
        else:
            dest = os.path.join(job_folder, "images", f.filename)
            with open(dest, "wb") as buf:
                shutil.copyfileobj(f.file, buf)

    # Save job parameters so the pipeline can be resumed after interruption
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
    config = {
        "is_video":        is_video,
        "scan_mode":       scan_mode,
        "known_weight":    known_weight,
        "preferred_shape": preferred_shape,
        "cut_mode":        cut_mode,
        "optimizer_settings": optimizer_settings,
    }
    with open(os.path.join(job_folder, "job_config.json"), "w") as f:
        json.dump(config, f)

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
    _atomic_write_json(
        Path(job_folder) / "status.json",
        {
            "status": "Processing",
            "step": "Optimizing",
            "progress": 0,
            "message": "Starting recalculation...",
            "timestamp": time.time(),
        },
    )
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
        with open(status_file, "w") as f:
            json.dump({"status": "Cancelled", "message": "Cancelled by user"}, f)
        return {"message": "Cancelled"}
    return {"message": "Job not found"}


@app.get("/api/jobs/{job_id}/report/pdf")
@app.get("/jobs/{job_id}/report")
async def download_pdf_report(job_id: str):
    job_folder = _validated_job_folder(job_id)
    pdf_path = find_existing_pdf(job_folder)
    pdf_error = _read_pdf_error(job_folder)

    if not pdf_path or pdf_error:
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

        analysis_path = job_folder / "analysis_report.json"
        if not analysis_path.is_file():
            if status in {"failed", "cancelled"}:
                raise HTTPException(
                    status_code=409,
                    detail="PDF report is unavailable because the job did not complete.",
                )
            raise HTTPException(status_code=404, detail="PDF report not found.")

        try:
            with analysis_path.open("r", encoding="utf-8") as handle:
                stats = json.load(handle)
        except (OSError, ValueError):
            raise HTTPException(
                status_code=500,
                detail="Analysis report could not be read.",
            )

        pdf_path = _generate_pdf_report(job_folder, job_id, stats)
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

            aligned = os.path.join(job_folder, "dense", "visual_aligned_stone.ply")
            if os.path.exists(aligned):
                data["model_url"] = (
                    f"{base_url}/{job_id}/dense/visual_aligned_stone.ply?t={ts}")
            else:
                data["model_url"] = (
                    f"{base_url}/{job_id}/dense/final_textured_model.ply?t={ts}")

            rp = os.path.join(job_folder, "analysis_report.json")
            if os.path.exists(rp):
                data["report_url"] = (
                    f"{base_url}/{job_id}/analysis_report.json?t={ts}")

            cp = os.path.join(job_folder, "dense", "best_cut.ply")
            if os.path.exists(cp):
                data["cut_url"] = (
                    f"{base_url}/{job_id}/dense/best_cut.ply?t={ts}")

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
        data.update(_pdf_status_fields(job_folder, job_id))
        return data

    except Exception:
        return {
            "status": "Error reading status",
            "progress": 0,
            "job_id": job_id,
            **_pdf_status_fields(job_folder, job_id),
        }
