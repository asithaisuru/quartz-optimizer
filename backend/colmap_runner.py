import subprocess
import os
import sys
import json
import time
import shutil

# PATH CHECK
COLMAP_BIN = shutil.which("colmap")
if not COLMAP_BIN:
    default_path = r"C:\Program Files\COLMAP\colmap.exe"
    if os.path.exists(default_path):
        COLMAP_BIN = default_path
    else:
        COLMAP_BIN = "colmap"

def update_status(job_path, step_name, percent, message):
    status_file = os.path.join(job_path, "status.json")
    data = {
        "status": "Processing",
        "step": step_name,
        "progress": percent,
        "message": message,
        "timestamp": time.time()
    }
    try:
        with open(status_file, "w") as f:
            json.dump(data, f)
    except: pass

def run_command(cmd, verbose=True):
    cmd_str = " ".join(cmd)
    if verbose: print(f"--> Running: {cmd_str}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Warning: Command failed: {cmd_str}")
        return False
    if verbose: print("Done.")
    return True

def get_largest_sparse_model(sparse_dir):
    best_folder = None
    max_size = -1
    if not os.path.exists(sparse_dir): return None
    subfolders = [f for f in os.listdir(sparse_dir) if f.isdigit()]
    if not subfolders: return None

    print(f"   Found {len(subfolders)} sub-models. Selecting the best one...")
    for folder in subfolders:
        points_file = os.path.join(sparse_dir, folder, "points3D.bin")
        if os.path.exists(points_file):
            size = os.path.getsize(points_file)
            if size > max_size:
                max_size = size
                best_folder = folder
    return best_folder

# --- UPDATED FUNCTION SIGNATURE ---
def run_photogrammetry_pipeline(job_path, scan_mode="turntable"):
    images_dir = os.path.join(job_path, "images")
    database_path = os.path.join(job_path, "database.db")
    sparse_dir = os.path.join(job_path, "sparse")
    dense_dir = os.path.join(job_path, "dense")

    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(dense_dir, exist_ok=True)

    print(f"=== Starting Photogrammetry | Mode: {scan_mode} ===")

    # --- CONFIGURE SETTINGS BASED ON MODE ---
    if scan_mode == "turntable":
        # EXTREME SENSITIVITY (For Transparent/Shiny Gems)
        feat_threshold = "0.00001" # Find invisible dust
        cam_model = "SIMPLE_RADIAL" # Better for rotation distortion
        match_overlap = "40"       # High overlap for smooth video
        stereo_window = "7"        # Detailed depth
    else:
        # STANDARD SENSITIVITY (For Handheld/Rooms)
        feat_threshold = "0.01"    # Ignore noise, focus on strong features
        cam_model = "PINHOLE"      # Standard model, faster
        match_overlap = "20"       # Standard video overlap
        stereo_window = "11"       # Smoother depth (less noise)
    # ----------------------------------------
    
    # 1. Feature Extraction
    if not os.path.exists(database_path):
        update_status(job_path, "Feature Extraction", 10, f"Extracting ({scan_mode})...")
        run_command([
            COLMAP_BIN, "feature_extractor", 
            "--database_path", database_path, 
            "--image_path", images_dir, 
            "--ImageReader.camera_model", cam_model, 
            "--ImageReader.single_camera", "1", 
            "--SiftExtraction.peak_threshold", feat_threshold
        ])

    # 2. Matching
    update_status(job_path, "Matching", 25, "Matching video frames...")
    run_command([
        COLMAP_BIN, "sequential_matcher", 
        "--database_path", database_path,
        "--SequentialMatching.overlap", match_overlap, 
        "--SequentialMatching.loop_detection", "1"
    ])

    # 3. Sparse Reconstruction
    if not os.listdir(sparse_dir):
        update_status(job_path, "Sparse Reconstruction", 40, "Building sparse model...")
        run_command([
            COLMAP_BIN, "mapper", 
            "--database_path", database_path, 
            "--image_path", images_dir, 
            "--output_path", sparse_dir,
            "--Mapper.ba_global_function_tolerance", "0.000001"
        ])

    best_model_id = get_largest_sparse_model(sparse_dir)
    if not best_model_id:
        raise Exception("Sparse reconstruction failed.")
    
    input_sparse_path = os.path.join(sparse_dir, best_model_id)

    # 4. Image Undistorter
    update_status(job_path, "Undistorting", 60, "Undistorting images...")
    run_command([
        COLMAP_BIN, "image_undistorter",
        "--image_path", images_dir,
        "--input_path", input_sparse_path,
        "--output_path", dense_dir,
        "--output_type", "COLMAP",
        "--max_image_size", "2000" 
    ])

    # 5. Patch Match Stereo
    update_status(job_path, "Depth Maps", 75, "Calculating depth maps...")
    run_command([
        COLMAP_BIN, "patch_match_stereo",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true", 
        "--PatchMatchStereo.window_radius", stereo_window,
        "--PatchMatchStereo.min_ncc", "0.05"
    ])

    # 6. Stereo Fusion
    update_status(job_path, "Point Fusion", 90, "Fusing points...")
    fused_ply_path = os.path.join(dense_dir, "fused.ply")
    
    run_command([
        COLMAP_BIN, "stereo_fusion",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP",
        "--output_path", fused_ply_path,
        "--StereoFusion.check_num_images", "1",
        "--StereoFusion.min_num_pixels", "5"
    ])

    # === FALLBACK ===
    is_empty = False
    if not os.path.exists(fused_ply_path): is_empty = True
    elif os.path.getsize(fused_ply_path) < 5000: is_empty = True

    if is_empty:
        print("⚠️ Dense reconstruction failed. Running FALLBACK to Sparse Model.")
        update_status(job_path, "Fallback", 95, "Dense failed. Using Sparse model...")
        run_command([
            COLMAP_BIN, "model_converter",
            "--input_path", input_sparse_path,
            "--output_path", fused_ply_path,
            "--output_type", "PLY"
        ])

    update_status(job_path, "Completed", 100, "Done.")
    return fused_ply_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_photogrammetry_pipeline(sys.argv[1])