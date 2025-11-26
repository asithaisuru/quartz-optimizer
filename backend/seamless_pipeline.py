import os
import uuid
import shutil
import time

# Import our helper modules
from video_utils import extract_best_frames
from colmap_runner import run_photogrammetry_pipeline
from cleanup_module import post_process_point_cloud

# CONFIG
BASE_DIR = r"D:\project-quartz"
JOBS_DIR = os.path.join(BASE_DIR, "jobs")

def process_video_to_model(video_path):
    # 1. Setup Job
    job_id = str(uuid.uuid4())
    job_folder = os.path.join(JOBS_DIR, job_id)
    images_folder = os.path.join(job_folder, "images")
    os.makedirs(images_folder, exist_ok=True)
    
    print(f"\n=================================================")
    print(f"🚀 STARTING SEAMLESS PIPELINE | Job: {job_id}")
    print(f"=================================================\n")
    
    # 2. Extract High-Quality Frames
    print("🎥 Phase 1: Processing Video...")
    num_frames = extract_best_frames(video_path, images_folder)
    if num_frames < 10:
        print("❌ Error: Not enough sharp frames extracted.")
        return
        
    # 3. Run COLMAP (Sparse -> Fallback -> Dense)
    print("\n⚙️ Phase 2: Running 3D Reconstruction Engine...")
    start_time = time.time()
    try:
        # This function handles the "Physics Failure" automatically now
        ply_path = run_photogrammetry_pipeline(job_folder)
    except Exception as e:
        print(f"❌ Critical Pipeline Error: {e}")
        return

    # 4. Post-Processing & Texturing
    print("\n🎨 Phase 3: Texturing & Meshing...")
    output_mesh = os.path.join(job_folder, "dense", "final_textured_model.ply")
    
    final_model = post_process_point_cloud(ply_path, output_mesh)
    
    duration = (time.time() - start_time) / 60
    
    print(f"\n=================================================")
    print(f"✅ DONE! Your 3D Model is ready.")
    print(f"⏱️ Total Time: {duration:.1f} minutes")
    print(f"📂 Location: {final_model}")
    print(f"=================================================\n")
    return final_model

if __name__ == "__main__":
    # USER: CHANGE THIS PATH TO YOUR VIDEO
    MY_VIDEO = r"D:\Photos\my_gem_video.mp4" 
    
    # Check if file exists to prevent errors
    if os.path.exists(MY_VIDEO):
        process_video_to_model(MY_VIDEO)
    else:
        print(f"Please update the 'MY_VIDEO' path in the script. File not found: {MY_VIDEO}")