import shutil
import os
import uuid
import json
import time
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
from dotenv import load_dotenv

# Import engines
from colmap_runner import run_photogrammetry_pipeline
from video_utils import extract_best_frames
from cleanup_module import post_process_point_cloud
from masking_utils import remove_backgrounds
from yield_calculator import calculate_gem_stats

# --- LOAD SETTINGS ---
load_dotenv()

app = FastAPI(title="Quartz Gemstone Optimizer")

# --- PATH SETUP ---
CONFIG_PATH = os.getenv("STORAGE_PATH", os.path.dirname(os.path.abspath(__file__)))
# If running from backend folder, go up one level for jobs
if os.path.basename(CONFIG_PATH) == "backend":
    CONFIG_PATH = os.path.dirname(CONFIG_PATH)

JOBS_DIR = os.path.join(CONFIG_PATH, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)
print(f"✅ System initialized. Storing data in: {JOBS_DIR}")

# --- CORS SETUP ---
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
    # Check if cancelled before overwriting
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
    except Exception as e:
        print(f"Error writing status: {e}")

def check_if_cancelled(job_folder):
    status_file = os.path.join(job_folder, "status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                data = json.load(f)
            if data.get("status") == "Cancelled": raise Exception("Cancelled by user")
        except Exception as e:
            if str(e) == "Cancelled by user": raise e

def process_full_pipeline(job_id: str, job_folder: str, is_video: bool, scan_mode: str, known_weight: str = None):
    images_folder = os.path.join(job_folder, "images")
    
    try:
        # --- PHASE 1: PREPARATION ---
        check_if_cancelled(job_folder)
        if is_video:
            print(f"[{job_id}] Phase 1: Video ({scan_mode})...")
            update_job_status(job_folder, "Video Processing", 5, "Extracting frames...")
            
            video_files = [f for f in os.listdir(job_folder) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
            total_extracted = 0
            current_index = 0
            for vid in video_files:
                check_if_cancelled(job_folder)
                count = extract_best_frames(os.path.join(job_folder, vid), images_folder, target_frames=80, start_index=current_index)
                total_extracted += count
                current_index += count

            if total_extracted < 10: raise Exception("Could not extract frames.")
        
        # --- PHASE 1.5: MASKING ---
        check_if_cancelled(job_folder)
        if scan_mode == "turntable":
            print(f"[{job_id}] Phase 1.5: Masking (Turntable)...")
            update_job_status(job_folder, "Masking", 20, "AI removing backgrounds...")
            remove_backgrounds(images_folder)
        else:
            print(f"[{job_id}] Skipping Masking (Handheld Mode)")
            update_job_status(job_folder, "Masking", 20, "Handheld mode (Skipping masking)...")
        
        # --- PHASE 2: COLMAP ---
        check_if_cancelled(job_folder)
        print(f"[{job_id}] Phase 2: COLMAP...")
        # Pass scan_mode to colmap runner
        ply_path = run_photogrammetry_pipeline(job_folder, scan_mode=scan_mode)

        # --- PHASE 3: CLEANUP & MESHING ---
        check_if_cancelled(job_folder)
        print(f"[{job_id}] Phase 3: Cleanup...")
        update_job_status(job_folder, "Meshing", 80, "Generating textured mesh...")
        output_mesh = os.path.join(job_folder, "dense", "final_textured_model.ply")
        
        final_model = post_process_point_cloud(ply_path, output_mesh)
        
        if not final_model or not os.path.exists(final_model):
            # Fallback to raw points if mesh fails
            print(f"[{job_id}] ⚠️ Meshing failed. Using Raw.")
            shutil.copy(ply_path, output_mesh)
            final_model = output_mesh

        # --- PHASE 3.5: YIELD CALCULATION ---
        check_if_cancelled(job_folder)
        print(f"[{job_id}] Calculating Yield (Ref Weight: {known_weight})...")
        update_job_status(job_folder, "Analysis", 90, "Calculating volume and yield...")
        
        stats = calculate_gem_stats(final_model, known_carats=known_weight)
        
        # Save stats
        stats_path = os.path.join(job_folder, "analysis_report.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f)

        # --- FINISH ---
        print(f"[{job_id}] Pipeline Complete.")
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
    scan_mode: str = Form("turntable"),
    known_weight: str = Form(None)
):
    job_id = str(uuid.uuid4())
    job_folder = os.path.join(JOBS_DIR, job_id)
    images_folder = os.path.join(job_folder, "images")
    os.makedirs(images_folder, exist_ok=True)

    update_job_status(job_folder, "Initializing", 0, "Uploading files...")

    # Auto-detect video
    video_exts = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    has_video_file = False
    for file in files:
        if file.filename.lower().endswith(video_exts):
            has_video_file = True
            break
    
    if has_video_file:
        is_video = True

    video_count = 0
    for file in files:
        if is_video and file.filename.lower().endswith(video_exts):
            fname = f"video_{video_count}{os.path.splitext(file.filename)[1]}"
            dest = os.path.join(job_folder, fname)
            with open(dest, "wb") as f: shutil.copyfileobj(file.file, f)
            video_count += 1
        else:
            dest = os.path.join(images_folder, file.filename)
            with open(dest, "wb") as f: shutil.copyfileobj(file.file, f)

    background_tasks.add_task(process_full_pipeline, job_id, job_folder, is_video, scan_mode, known_weight)

    return {
        "job_id": job_id, 
        "message": "Upload successful. Pipeline started.", 
        "status_url": f"/jobs/{job_id}/status"
    }

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
            with open(status_file, "r") as f:
                data = json.load(f)
            
            if data["status"] == "Completed":
                base_url = "http://localhost:8000/files"
                
                # --- FIX: ADD TIMESTAMP TO FORCE RELOAD ---
                timestamp = int(time.time())
                data["model_url"] = f"{base_url}/{job_id}/dense/final_textured_model.ply?t={timestamp}"
                # ------------------------------------------
                
                report_path = os.path.join(job_folder, "analysis_report.json")
                if os.path.exists(report_path):
                    data["report_url"] = f"{base_url}/{job_id}/analysis_report.json"
                
                cut_path = os.path.join(job_folder, "dense", "best_cut.ply")
                if os.path.exists(cut_path):
                    data["cut_url"] = f"{base_url}/{job_id}/dense/best_cut.ply?t={timestamp}"
            
            return data
        except:
            return {"status": "Error reading status", "progress": 0}
    
    return {"status": "Not Found", "progress": 0}

@app.post("/jobs/{job_id}/recalculate")
async def recalculate_yield(job_id: str, known_weight: str = Form(...)):
    print(f"[{job_id}] 🔄 Recalculating for new weight: {known_weight} cts")
    
    job_folder = os.path.join(JOBS_DIR, job_id)
    ply_path = os.path.join(job_folder, "dense", "final_textured_model.ply")
    stats_path = os.path.join(job_folder, "analysis_report.json")

    if not os.path.exists(ply_path):
        return {"status": "Error", "message": "3D Model not found. Cannot recalculate."}

    try:
        # Re-run the math (Optimizes and saves best_cut.ply)
        stats = calculate_gem_stats(ply_path, known_carats=known_weight)
        
        with open(stats_path, "w") as f:
            json.dump(stats, f)
            
        # --- NEW: Check for the cut file and return URL ---
        cut_path = os.path.join(job_folder, "dense", "best_cut.ply")
        if os.path.exists(cut_path):
            # Force a timestamp query param (?t=...) to prevent browser caching
            base_url = "http://localhost:8000/files"
            stats["cut_url"] = f"{base_url}/{job_id}/dense/best_cut.ply?t={int(time.time())}"
        # --------------------------------------------------
            
        return {"status": "Success", "data": stats}

    except Exception as e:
        return {"status": "Error", "message": str(e)}