import shutil
import os
import uuid
import json
import time
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Optional
from dotenv import load_dotenv

from colmap_runner import run_photogrammetry_pipeline
from video_utils import extract_best_frames
from cleanup_module import post_process_point_cloud
from masking_utils import remove_backgrounds
from yield_calculator import calculate_gem_stats
from report_generator import create_pdf
from ai_runner import run_ai_pipeline
from fracture_mapper import map_fractures_to_3d

load_dotenv()

app = FastAPI(title="Quartz Gemstone Optimizer")

CONFIG_PATH = os.getenv("STORAGE_PATH", os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(CONFIG_PATH) == "backend":
    CONFIG_PATH = os.path.dirname(CONFIG_PATH)
JOBS_DIR = os.path.join(CONFIG_PATH, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
app.mount("/files", StaticFiles(directory=JOBS_DIR), name="files")


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
    cut_mode: str = "multi"
):
    images_folder = os.path.join(job_folder, "images")
    try:
        # Phase 1 — Video frame extraction
        check_if_cancelled(job_folder)
        if is_video:
            update_job_status(job_folder, "Video Processing", 5, "Extracting frames...")
            video_files = [
                f for f in os.listdir(job_folder)
                if f.lower().endswith(('.mp4', '.mov', '.avi'))
            ]
            total = 0
            idx = 0
            for vid in video_files:
                check_if_cancelled(job_folder)
                cnt = extract_best_frames(
                    os.path.join(job_folder, vid),
                    images_folder, target_frames=40, start_index=idx
                )
                total += cnt
                idx += cnt
            if total < 10:
                raise Exception("Video extraction failed — too few sharp frames.")

        # Phase 1.5 — Background masking
        check_if_cancelled(job_folder)
        if scan_mode == "turntable":
            update_job_status(job_folder, "Masking", 20, "AI removing backgrounds...")
            remove_backgrounds(images_folder)
        else:
            update_job_status(job_folder, "Masking", 20, "Handheld mode — skipping masking...")

        # Phase 2 — COLMAP 3D reconstruction
        check_if_cancelled(job_folder)
        update_job_status(job_folder, "Reconstruction", 30, "Running photogrammetry...")
        ply_path = run_photogrammetry_pipeline(job_folder, scan_mode=scan_mode)

        # Phase 3 — Mesh cleanup
        check_if_cancelled(job_folder)
        update_job_status(job_folder, "Meshing", 70, "Generating watertight mesh...")
        output_mesh = os.path.join(job_folder, "dense", "final_textured_model.ply")
        final_model = post_process_point_cloud(ply_path, output_mesh)
        if not final_model:
            shutil.copy(ply_path, output_mesh)
            final_model = output_mesh

        # Phase 4 — AI fracture detection (must run BEFORE yield so defects.ply exists)
        if os.path.exists(os.path.join(os.path.dirname(__file__), "best.pt")):
            check_if_cancelled(job_folder)
            update_job_status(job_folder, "AI Analysis", 80, "Scanning for fractures & clouds...")
            run_ai_pipeline(job_folder)
            map_fractures_to_3d(job_folder)

        # Phase 5 — Yield & cut optimization
        # defects.ply is now ready, so the optimizer can avoid fractures
        check_if_cancelled(job_folder)
        update_job_status(job_folder, "Analysis", 90, "Calculating optimal cut strategy...")
        stats = calculate_gem_stats(
            final_model,
            known_carats=known_weight,
            preferred_shape=preferred_shape,
            cut_mode=cut_mode,
            job_folder=job_folder   # <-- lets optimizer load defects.ply
        )
        with open(os.path.join(job_folder, "analysis_report.json"), "w") as f:
            json.dump(stats, f)

        update_job_status(job_folder, "Completed", 100, "Done", status="Completed")

    except Exception as e:
        msg = str(e)
        if msg == "Cancelled by user":
            print(f"[{job_id}] CANCELLED")
        else:
            update_job_status(job_folder, "Failed", 0, msg, status="Failed")


def run_recalculation_task(
    job_id: str,
    known_weight: str,
    preferred_shape: str = None,
    cut_mode: str = "multi"
):
    job_folder = os.path.join(JOBS_DIR, job_id)
    target_mesh = os.path.join(job_folder, "dense", "visual_aligned_stone.ply")
    if not os.path.exists(target_mesh):
        target_mesh = os.path.join(job_folder, "dense", "final_textured_model.ply")

    stats_path = os.path.join(job_folder, "analysis_report.json")

    def reporter(curr, tot, msg):
        pct = int((curr / max(tot, 1)) * 100)
        update_job_status(job_folder, "Optimizing", pct, f"{msg} ({curr}/{tot})")

    try:
        update_job_status(job_folder, "Optimizing", 0, "Starting recalculation...")
        stats = calculate_gem_stats(
            target_mesh,
            known_carats=known_weight,
            preferred_shape=preferred_shape,
            cut_mode=cut_mode,
            job_folder=job_folder,
            progress_callback=reporter
        )
        with open(stats_path, "w") as f:
            json.dump(stats, f)
        update_job_status(job_folder, "Completed", 100, "Done", status="Completed")
    except Exception as e:
        update_job_status(job_folder, "Failed", 0, str(e), status="Failed")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/shapes")
async def list_shapes():
    """
    Returns all available gem shape names so the frontend can
    populate the shape selector dropdown dynamically.
    """
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
    preferred_shape: str = Form(None),   # NEW
    cut_mode: str = Form("multi")        # NEW  "single" | "multi"
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
            dest = os.path.join(job_folder, f"video_{v_idx}{os.path.splitext(f.filename)[1]}")
            with open(dest, "wb") as buf:
                shutil.copyfileobj(f.file, buf)
            v_idx += 1
        else:
            dest = os.path.join(job_folder, "images", f.filename)
            with open(dest, "wb") as buf:
                shutil.copyfileobj(f.file, buf)

    bg_tasks.add_task(
        process_full_pipeline,
        job_id, job_folder, is_video, scan_mode,
        known_weight, preferred_shape, cut_mode
    )
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}/status"}


@app.post("/jobs/{job_id}/recalculate")
async def recalculate(
    job_id: str,
    bg_tasks: BackgroundTasks,
    known_weight: str = Form(...),
    preferred_shape: str = Form(None),   # NEW
    cut_mode: str = Form("multi")        # NEW
):
    bg_tasks.add_task(
        run_recalculation_task,
        job_id, known_weight, preferred_shape, cut_mode
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


@app.get("/jobs/{job_id}/report")
async def report(job_id: str):
    job_folder = os.path.join(JOBS_DIR, job_id)
    stats = json.load(open(os.path.join(job_folder, "analysis_report.json")))
    return FileResponse(create_pdf(job_folder, job_id, stats))


@app.get("/jobs/{job_id}/status")
async def get_status(job_id: str):
    job_folder = os.path.join(JOBS_DIR, job_id)
    status_file = os.path.join(job_folder, "status.json")

    if not os.path.exists(status_file):
        return {"status": "Not Found", "progress": 0}

    try:
        with open(status_file, "r") as f:
            data = json.load(f)

        if data["status"] == "Completed":
            base_url = f"{API_BASE_URL}/files"
            ts = int(time.time())

            # Rough stone model
            aligned = os.path.join(job_folder, "dense", "visual_aligned_stone.ply")
            if os.path.exists(aligned):
                data["model_url"] = f"{base_url}/{job_id}/dense/visual_aligned_stone.ply?t={ts}"
            else:
                data["model_url"] = f"{base_url}/{job_id}/dense/final_textured_model.ply?t={ts}"

            # Analysis JSON
            rp = os.path.join(job_folder, "analysis_report.json")
            if os.path.exists(rp):
                data["report_url"] = f"{base_url}/{job_id}/analysis_report.json?t={ts}"

            # Best cut PLY
            cp = os.path.join(job_folder, "dense", "best_cut.ply")
            if os.path.exists(cp):
                data["cut_url"] = f"{base_url}/{job_id}/dense/best_cut.ply?t={ts}"

            # Fracture point cloud
            dp = os.path.join(job_folder, "dense", "defects.ply")
            if os.path.exists(dp):
                data["defects_url"] = f"{base_url}/{job_id}/dense/defects.ply?t={ts}"

        return data
    except Exception:
        return {"status": "Error reading status", "progress": 0}
