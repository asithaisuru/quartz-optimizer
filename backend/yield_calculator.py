import trimesh
import numpy as np
import json
import os
from optimizer import optimize_cut

# CONSTANTS
DENSITY_QUARTZ = 2.65
CARATS_PER_GRAM = 5.0

def calculate_gem_stats(mesh_path, known_carats=None):
    print(f"--- Calculating Yield for: {mesh_path} ---")
    
    if not os.path.exists(mesh_path): return {"error": "Mesh not found"}

    try:
        # 1. Load Mesh
        mesh = trimesh.load(mesh_path)
        
        mesh.vertices -= mesh.centroid
        
        # Overwrite file so Frontend sees it centered
        mesh.export(mesh_path)
        print("   ✅ Re-centered rough stone to Center of Mass (0,0,0)")
        # --------------------------------

        if not mesh.is_watertight:
            calc_mesh = mesh.convex_hull
        else:
            calc_mesh = mesh

        # Scaling Logic
        current_volume = calc_mesh.volume
        scale_factor = 1.0
        
        if known_carats and float(known_carats) > 0:
            target_grams = float(known_carats) / CARATS_PER_GRAM
            target_vol = target_grams / DENSITY_QUARTZ
            scale_factor = (target_vol / current_volume) ** (1/3)
        else:
            scale_factor = 2.0 / np.max(calc_mesh.extents)

        scaled_extents = calc_mesh.extents * scale_factor * 10 
        final_volume = current_volume * (scale_factor ** 3)
        final_weight = final_volume * DENSITY_QUARTZ * CARATS_PER_GRAM
        
        # --- OPTIMIZATION ---
        calc_mesh.apply_scale(scale_factor)
        temp_path = mesh_path.replace(".ply", "_scaled.ply")
        calc_mesh.export(temp_path)
        
        results = optimize_cut(temp_path)
        best = results[0]
        
        cut_vol = best['max_volume_cm3']
        cut_weight = cut_vol * DENSITY_QUARTZ * CARATS_PER_GRAM
        yield_pct = (cut_weight / final_weight * 100) if final_weight > 0 else 0

        stats = {
            "volume_cm3": round(final_volume, 2),
            "dimensions_mm": [round(x, 2) for x in scaled_extents],
            "raw_carats": round(final_weight, 2),
            "estimated_cut_carats": round(cut_weight, 2),
            "yield_percent": round(yield_pct, 1),
            "recommended_shape": best['shape']
        }
        return stats

    except Exception as e:
        print(f"❌ Calculation failed: {e}")
        return {"error": str(e)}