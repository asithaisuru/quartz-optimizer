import trimesh
import numpy as np
import json
import os
from optimizer import optimize_cut
# --- NEW IMPORT ---
from ray_tracer import calculate_light_performance 
# ------------------

DENSITY_QUARTZ = 2.65
CARATS_PER_GRAM = 5.0

def calculate_gem_stats(mesh_path, known_carats=None, progress_callback=None):
    print(f"--- Calculating Yield for: {mesh_path} ---")
    
    if not os.path.exists(mesh_path): return {"error": "Mesh not found"}

    try:
        # 1. Load & Center
        mesh = trimesh.load(mesh_path)
        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-center)
        
        aligned_path = mesh_path.replace("final_textured_model.ply", "visual_aligned_stone.ply")
        mesh.export(aligned_path)
        print(f"   ✅ Saved aligned visual stone (Geometric Center)")

        # 2. Scale Logic
        if not mesh.is_watertight:
            vol_mesh = mesh.convex_hull
        else:
            vol_mesh = mesh

        vol_original = vol_mesh.volume
        scale_factor = 1.0
        
        target_weight_carats = 0.0
        
        if known_carats and float(known_carats) > 0:
            target_weight_carats = float(known_carats)
            target_grams = target_weight_carats / CARATS_PER_GRAM
            target_vol_cm3 = target_grams / DENSITY_QUARTZ
            scale_factor = (target_vol_cm3 / vol_original) ** (1/3)
        else:
            scale_factor = 2.0 / np.max(vol_mesh.extents)
            target_vol_cm3 = vol_original * (scale_factor ** 3)
            target_weight_carats = target_vol_cm3 * DENSITY_QUARTZ * CARATS_PER_GRAM

        # 3. Optimize
        optimizer_mesh = mesh.copy() 
        # Optimize in arbitrary units to match viewer
        
        temp_opt_path = mesh_path.replace(".ply", "_opt_input.ply")
        optimizer_mesh.export(temp_opt_path)
        
        strategies = optimize_cut(temp_opt_path, mode="multi", progress_callback=progress_callback)
        
        # 4. Process Results
        options_data = []
        for i, strat in enumerate(strategies):
            plan_vol = strat['total_volume']
            yield_ratio = plan_vol / vol_original if vol_original > 0 else 0
            plan_weight = target_weight_carats * yield_ratio
            yield_pct = yield_ratio * 100
            
            opt_filename = f"option_{i}.ply"
            opt_full_path = os.path.join(os.path.dirname(mesh_path), opt_filename)
            
            if strat['gems']:
                combined_mesh = trimesh.util.concatenate(strat['gems'])
                combined_mesh.visual.face_colors = [255, 0, 0, 255]
                combined_mesh.export(opt_full_path)
            
            options_data.append({
                "name": strat['shape'],
                "type": strat['strategy'],
                "weight": round(plan_weight, 2),
                "yield": round(yield_pct, 1),
                "file": opt_filename
            })

        if options_data:
            best = options_data[0]
        else:
            best = {"name": "Failed", "weight": 0, "yield": 0}

        # --- NEW: RAY TRACING ANALYSIS ---
        # We run this on the 'best_cut.ply' which optimizer.py just saved
        best_cut_path = os.path.join(os.path.dirname(mesh_path), "best_cut.ply")
        light_stats = {"score": 0, "grade": "N/A"}
        
        if os.path.exists(best_cut_path):
            # Run the simulation
            light_stats = calculate_light_performance(best_cut_path)
        # ---------------------------------

        scaled_extents = vol_mesh.extents * scale_factor * 10 

        stats = {
            "volume_cm3": round(target_weight_carats / CARATS_PER_GRAM / DENSITY_QUARTZ, 2),
            "dimensions_mm": [round(x, 2) for x in scaled_extents],
            "raw_carats": round(target_weight_carats, 2),
            "estimated_cut_carats": best['weight'],
            "yield_percent": best['yield'],
            "recommended_shape": best['name'],
            "options": options_data,
            # Add Light Data to JSON
            "light_analysis": light_stats 
        }
        return stats

    except Exception as e:
        print(f"❌ Calculation failed: {e}")
        return {"error": str(e)}