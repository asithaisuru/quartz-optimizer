import trimesh
import numpy as np
import json
import os
from optimizer import optimize_cut

DENSITY_QUARTZ = 2.65
CARATS_PER_GRAM = 5.0

def calculate_gem_stats(mesh_path, known_carats=None, progress_callback=None):
    if not os.path.exists(mesh_path): return {"error": "Mesh not found"}

    try:
        # 1. Load & Center Rough
        mesh = trimesh.load(mesh_path)
        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-center)
        
        aligned_path = mesh_path.replace("final_textured_model.ply", "visual_aligned_stone.ply")
        mesh.export(aligned_path)

        if not mesh.is_watertight: vol_mesh = mesh.convex_hull
        else: vol_mesh = mesh

        # 2. Scale Logic (Based on Rough)
        vol_original_arbitrary = vol_mesh.volume
        scale_factor = 1.0
        target_weight_carats = 0.0
        
        if known_carats and float(known_carats) > 0:
            target_weight_carats = float(known_carats)
            target_grams = target_weight_carats / CARATS_PER_GRAM
            target_vol_cm3 = target_grams / DENSITY_QUARTZ
            
            if vol_original_arbitrary > 0:
                scale_factor = (target_vol_cm3 / vol_original_arbitrary) ** (1/3)
        else:
            scale_factor = 2.0 / np.max(vol_mesh.extents)
            target_vol_cm3 = vol_original_arbitrary * (scale_factor ** 3)
            target_weight_carats = target_vol_cm3 * DENSITY_QUARTZ * CARATS_PER_GRAM

        # 3. Calculate ROUGH Dimensions
        # These help the user check if their weight input was correct
        rough_dims_mm = vol_mesh.extents * scale_factor * 10 

        # 4. Optimize
        optimizer_mesh = mesh.copy() 
        temp_opt_path = mesh_path.replace(".ply", "_opt_input.ply")
        optimizer_mesh.export(temp_opt_path)
        
        strategies = optimize_cut(temp_opt_path, mode="multi", progress_callback=progress_callback)
        
        options_data = []
        for i, strat in enumerate(strategies):
            plan_vol = strat['total_volume']
            yield_ratio = plan_vol / vol_original_arbitrary if vol_original_arbitrary > 0 else 0
            plan_weight = target_weight_carats * yield_ratio
            yield_pct = yield_ratio * 100
            
            # --- NEW: Calculate Cut Dimensions ---
            # We take the first/largest gem in the strategy
            main_gem = strat['gems'][0]
            # Extents * Scale Factor * 10 (cm to mm)
            cut_dims = main_gem.extents * scale_factor * 10
            # -------------------------------------
            
            opt_filename = f"option_{i}.ply"
            opt_full_path = os.path.join(os.path.dirname(mesh_path), opt_filename)
            combined_mesh = trimesh.util.concatenate(strat['gems'])
            combined_mesh.visual.face_colors = [255, 0, 0, 255]
            combined_mesh.export(opt_full_path)
            
            options_data.append({
                "name": strat['shape'],
                "type": strat['strategy'],
                "weight": round(plan_weight, 2),
                "yield": round(yield_pct, 1),
                "file": opt_filename,
                # Store specific dimensions for this cut
                "cut_dims": [round(x, 2) for x in cut_dims]
            })

        best = options_data[0] if options_data else None

        stats = {
            "volume_cm3": round(target_weight_carats / CARATS_PER_GRAM / DENSITY_QUARTZ, 2),
            "raw_carats": round(target_weight_carats, 2),
            # Return Rough Dims separately
            "rough_dimensions_mm": [round(x, 2) for x in rough_dims_mm], 
            
            # Best Option Stats
            "estimated_cut_carats": best['weight'] if best else 0,
            "yield_percent": best['yield'] if best else 0,
            "recommended_shape": best['name'] if best else "None",
            "cut_dimensions_mm": best['cut_dims'] if best else [0,0,0], # The winning gem size
            "options": options_data
        }
        return stats

    except Exception as e:
        print(f"❌ Calculation failed: {e}")
        return {"error": str(e)}