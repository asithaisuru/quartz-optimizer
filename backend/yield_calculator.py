import trimesh
import numpy as np
import json
import os
from optimizer import optimize_cut
from ray_tracer import calculate_light_performance

DENSITY_QUARTZ = 2.65   # g/cm³
CARATS_PER_GRAM = 5.0


def calculate_gem_stats(
    mesh_path,
    known_carats=None,
    preferred_shape=None,
    cut_mode="multi",
    job_folder=None,
    progress_callback=None
):
    """
    Parameters
    ----------
    mesh_path       : str   - path to the rough stone PLY (final_textured_model.ply)
    known_carats    : str   - user-provided carat weight (optional)
    preferred_shape : str   - gem shape name chosen by user (None = auto)
    cut_mode        : str   - "single" or "multi"
    job_folder      : str   - job root folder (used to locate defects.ply)
    progress_callback: callable(current, total, message)
    """
    if not os.path.exists(mesh_path):
        return {"error": "Mesh not found"}

    try:
        # --- Load & center rough stone ---
        mesh = trimesh.load(mesh_path)
        mesh.apply_translation(-mesh.bounds.mean(axis=0))

        # Save the centered stone for the 3D viewer
        aligned_path = mesh_path.replace("final_textured_model.ply", "visual_aligned_stone.ply")
        mesh.export(aligned_path)

        vol_mesh = mesh if mesh.is_watertight else mesh.convex_hull
        vol_original = vol_mesh.volume

        # --- Scale calibration ---
        if known_carats and float(known_carats) > 0:
            target_weight = float(known_carats)
            target_grams = target_weight / CARATS_PER_GRAM
            target_vol = target_grams / DENSITY_QUARTZ
            scale_factor = (target_vol / vol_original) ** (1 / 3)
        else:
            scale_factor = 2.0 / np.max(vol_mesh.extents)
            target_vol = vol_original * (scale_factor ** 3)
            target_weight = target_vol * DENSITY_QUARTZ * CARATS_PER_GRAM

        # Export unscaled mesh for optimizer (optimizer works in mesh units)
        temp_opt_path = mesh_path.replace(".ply", "_opt_input.ply")
        mesh.export(temp_opt_path)

        # --- Run optimizer ---
        # Pass preferred_shape, cut_mode, and job_folder so the optimizer can:
        #   a) filter to the user's chosen shape
        #   b) run single or multi-gem packing
        #   c) load defects.ply and avoid fractures
        strategies = optimize_cut(
            temp_opt_path,
            mode=cut_mode,
            preferred_shape=preferred_shape,
            job_folder=job_folder,
            progress_callback=progress_callback
        )

        options_data = []
        for i, strat in enumerate(strategies):
            plan_vol = strat['total_volume']
            yield_ratio = plan_vol / vol_original if vol_original > 0 else 0
            plan_weight = target_weight * yield_ratio
            yield_pct = yield_ratio * 100

            # Save this option's PLY
            opt_filename = f"option_{i}.ply"
            opt_path = os.path.join(os.path.dirname(mesh_path), opt_filename)
            combined = trimesh.util.concatenate(strat['gems'])
            combined.visual.face_colors = [255, 0, 0, 255]
            combined.export(opt_path)

            # Dimensions of the cut gem (scaled to real mm)
            cut_extents_mm = [
                round(float(e) * scale_factor * 10, 2)
                for e in combined.extents
            ]

            # Light performance
            light = calculate_light_performance(opt_path)

            options_data.append({
                "name": strat['shape'],
                "type": strat['strategy'],
                "weight": round(plan_weight, 2),
                "yield": round(yield_pct, 1),
                "file": opt_filename,
                "light_score": light['score'],
                "light_grade": light['grade'],
                "cut_dims": cut_extents_mm,
                # Number of gems in this strategy
                "gem_count": len(strat['gems'])
            })

        best = options_data[0]

        # Rough stone dimensions in mm
        rough_extents_mm = [round(float(e) * scale_factor * 10, 2) for e in vol_mesh.extents]

        stats = {
            "volume_cm3": round(target_weight / CARATS_PER_GRAM / DENSITY_QUARTZ, 2),
            "rough_dimensions_mm": rough_extents_mm,
            "raw_carats": round(target_weight, 2),
            "estimated_cut_carats": best['weight'],
            "yield_percent": best['yield'],
            "recommended_shape": best['name'],
            "cut_mode": cut_mode,
            "preferred_shape": preferred_shape or "Auto",
            "options": options_data,
            "light_analysis": {
                "score": best['light_score'],
                "grade": best['light_grade']
            }
        }
        return stats

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
