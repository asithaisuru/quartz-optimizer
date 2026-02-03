import os
import numpy as np
import cv2
import trimesh
import math
from colmap_loader import read_images_binary, read_points3D_binary

def map_fractures_to_3d(job_folder):
    print(f"--- Mapping 2D Fractures to 3D Space ---")
    
    sparse_path = os.path.join(job_folder, "sparse", "0")
    if not os.path.exists(sparse_path):
        sparse_root = os.path.join(job_folder, "sparse")
        subfolders = [f for f in os.listdir(sparse_root) if f.isdigit()]
        if not subfolders:
            print("❌ No sparse model found.")
            return None
        sparse_path = os.path.join(sparse_root, subfolders[0])

    try:
        # 1. Load COLMAP Data
        images = read_images_binary(os.path.join(sparse_path, "images.bin"))
        points3D = read_points3D_binary(os.path.join(sparse_path, "points3D.bin"))
        
        # 2. Identify Defect Points
        defect_point_ids = set()
        fracture_dir = os.path.join(job_folder, "fractures")
        
        if not os.path.exists(fracture_dir):
            print("   ⚠️ No fracture masks found. Skipping.")
            return None

        print(f"   🔍 Checking {len(images)} images against AI masks...")

        for img_id, img_data in images.items():
            base_name = os.path.splitext(img_data.name)[0]
            
            # Find matching masks
            masks = [f for f in os.listdir(fracture_dir) if f.startswith(base_name) and f.endswith(".png")]
            if not masks: continue

            # Combine masks
            combined_mask = None
            for m in masks:
                mask_path = os.path.join(fracture_dir, m)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            if combined_mask is None: continue

            img_h, img_w = combined_mask.shape

            for i, p3d_id in enumerate(img_data.point3D_ids):
                if p3d_id == -1: continue 
                
                # --- BUG FIX: CHECK FOR NaN / INFINITY ---
                x = img_data.xys[i][0]
                y = img_data.xys[i][1]
                
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue # Skip invalid math values
                
                u, v = int(round(x)), int(round(y))
                # ----------------------------------------
                
                # Check bounds
                if 0 <= v < img_h and 0 <= u < img_w:
                    # Check white pixel
                    if combined_mask[v, u] > 127:
                        defect_point_ids.add(p3d_id)

        # 3. Export Defect Cloud
        if not defect_point_ids:
            print("   ⚠️ No 3D defects found overlapping with masks.")
            return None
            
        print(f"   🚨 Found {len(defect_point_ids)} confirmed 3D fracture points.")
        
        vertices = []
        colors = []
        
        # --- ALIGNMENT FIX (Match visual stone center) ---
        orig_dense_path = os.path.join(job_folder, "dense", "final_textured_model.ply")
        offset_vec = np.array([0,0,0])
        
        if os.path.exists(orig_dense_path):
             # Load ONLY vertices first to check bounds quickly
             try:
                temp_mesh = trimesh.load(orig_dense_path)
                # This must match yield_calculator logic: Geometric Center of Bounds
                offset_vec = -temp_mesh.bounds.mean(axis=0)
             except: pass
        # ----------------------------------------------
        
        for pid in defect_point_ids:
            if pid in points3D:
                pt = points3D[pid]
                vertices.append(pt.xyz + offset_vec) # Apply offset
                colors.append([255, 0, 0, 255]) 
            
        if len(vertices) == 0:
            return None

        pcd = trimesh.points.PointCloud(vertices, colors=colors)
        output_path = os.path.join(job_folder, "dense", "defects.ply")
        pcd.export(output_path)
        print(f"   💎 Saved Defects to: defects.ply")
        
        return output_path

    except Exception as e:
        print(f"❌ Fracture Mapping Failed: {e}")
        # Debug trace
        import traceback
        traceback.print_exc()
        return None