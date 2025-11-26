import shutil
import os
import uuid
import json
import time
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
from dotenv import load_dotenv # <--- NEW IMPORT

# Import engines
from colmap_runner import run_photogrammetry_pipeline
from video_utils import extract_best_frames
from cleanup_module import post_process_point_cloud
from masking_utils import remove_backgrounds

# --- LOAD SETTINGS ---
load_dotenv() # Load variables from .env file

app = FastAPI(title="Quartz Gemstone Optimizer")

# Get path from .env, or default to current folder if missing
CONFIG_PATH = os.getenv("STORAGE_PATH", os.path.dirname(os.path.abspath(__file__)))
JOBS_DIR = os.path.join(CONFIG_PATH, "jobs")

# Create the folder if it doesn't exist
os.makedirs(JOBS_DIR, exist_ok=True)
print(f"✅ System initialized. Storing data in: {JOBS_DIR}")
# ---------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=JOBS_DIR), name="files")

def update_job_status(job_folder, step, progress, message, status="Processing"):
    status_file = os.path.join(job_folder, "status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                existing = json.load(f)
            if existing.get("status") == "Cancelled": return
        except: pass

    data = {
        "status": status,
        "step": step,
        "progress": progress,
        "message": message,
        "timestamp": time.time()
    }
    try:
        with open(status_file, "w") as f:
            json.dump(data, f)
    except: pass

def check_if_cancelled(job_folder):
    status_file = os.path.join(job_folder, "status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                data = json.load(f)
            if data.get("status") == "Cancelled": raise Exception("Cancelled by user")
        except Exception as e:
            if str(e) == "Cancelled by user": raise e

def process_full_pipeline(job_id: str, job_folder: str, is_video: bool, scan_mode: str):
    images_folder = os.path.join(job_folder, "images")
    
    try:
        # PHASE 1
        check_if_cancelled(job_folder)
        if is_video:
            print(f"[{job_id}] Phase 1: Video ({scan_mode})...")
            update_job_status(job_folder, "Video Processing", 5, "Extracting frames...")
            
            video_files = [f for f in os.listdir(job_folder) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
            total_extracted = 0
            current_index = 0
            for vid in video_files:
                check_if_cancelled(job_folder)
                # Extract frames (relaxed threshold)
                count = extract_best_frames(os.path.join(job_folder, vid), images_folder, target_frames=80, start_index=current_index)
                total_extracted += count
                current_index += count

            if total_extracted < 10: raise Exception("Could not extract frames.")
        
        # PHASE 1.5
        check_if_cancelled(job_folder)
        if scan_mode == "turntable":
            print(f"[{job_id}] Masking (Turntable)...")
            update_job_status(job_folder, "Masking", 20, "AI removing backgrounds...")
            remove_backgrounds(images_folder)
        else:
            print(f"[{job_id}] Handheld Mode (Skipping Masking)")
            update_job_status(job_folder, "Masking", 20, "Handheld mode enabled...")
        
        # PHASE 2
        check_if_cancelled(job_folder)
        print(f"[{job_id}] Phase 2: COLMAP ({scan_mode})...")
        
        # PASS THE SCAN MODE HERE!
        ply_path = run_photogrammetry_pipeline(job_folder, scan_mode=scan_mode)

        # PHASE 3
        check_if_cancelled(job_folder)
        print(f"[{job_id}] Phase 3: Cleanup...")
        update_job_status(job_folder, "Meshing", 85, "Generating textured mesh...")
        output_mesh = os.path.join(job_folder, "dense", "final_textured_model.ply")
        
        # Try to Mesh
        final_model = post_process_point_cloud(ply_path, output_mesh)
        
        # --- SAFETY NET (The Fix) ---
        if not final_model or not os.path.exists(final_model):
            print(f"[{job_id}] ⚠️ Meshing failed/empty. Falling back to RAW Point Cloud.")
            # Copy the raw COLMAP ply to the final output name
            shutil.copy(ply_path, output_mesh)
            # Tell Frontend it's done (even though it's just dots)
            final_model = output_mesh
        # -----------------------------

        # DONE
        check_if_cancelled(job_folder)
        print(f"[{job_id}] DONE.")
        update_job_status(job_folder, "Completed", 100, "Ready to View.", status="Completed")

    except Exception as e:
        err = str(e)
        if err == "Cancelled by user":
            print(f"[{job_id}] CANCELLED")
        else:
            print(f"[{job_id}] FAILED: {err}")
            update_job_status(job_folder, "Failed", 0, err, status="Failed")

@app.post("/upload")
async def create_job(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    is_video: bool = Form(False),
    scan_mode: str = Form("turntable")
):
    job_id = str(uuid.uuid4())
    job_folder = os.path.join(JOBS_DIR, job_id)
    images_folder = os.path.join(job_folder, "images")
    os.makedirs(images_folder, exist_ok=True)

    update_job_status(job_folder, "Initializing", 0, "Uploading files...")

    for i, file in enumerate(files):
        if is_video and file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            fname = f"video_{i}{os.path.splitext(file.filename)[1]}"
            dest = os.path.join(job_folder, fname)
            with open(dest, "wb") as f: shutil.copyfileobj(file.file, f)
        else:
            dest = os.path.join(images_folder, file.filename)
            with open(dest, "wb") as f: shutil.copyfileobj(file.file, f)

    background_tasks.add_task(process_full_pipeline, job_id, job_folder, is_video, scan_mode)

    return { "job_id": job_id, "status_url": f"/jobs/{job_id}/status" }

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job_folder = os.path.join(JOBS_DIR, job_id)
    status_file = os.path.join(job_folder, "status.json")
    if os.path.exists(job_folder):
        data = {"status": "Cancelled", "message": "Job cancelled by user."}
        with open(status_file, "w") as f: json.dump(data, f)
        return {"message": "Cancellation requested"}
    return {"message": "Job not found"}

@app.get("/jobs/{job_id}/status")
async def get_status(job_id: str):
    job_folder = os.path.join(JOBS_DIR, job_id)
    status_file = os.path.join(job_folder, "status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f: data = json.load(f)
            if data["status"] == "Completed":
                data["model_url"] = f"http://localhost:8000/files/{job_id}/dense/final_textured_model.ply"
            return data
        except: return {"status": "Error"}
    return {"status": "Not Found"}