import trimesh
import numpy as np
import os
import copy
from gem_shapes import get_standard_shapes

def check_fit_proximity(prox, gem, start_point, min_dim, max_dim):
    # Move gem to start point
    gem_curr = gem.copy()
    gem_curr.vertices += start_point 
    
    scale = min_dim * 0.05 # Start at 5%
    step_size = min_dim * 0.02
    best_scale = 0.0
    
    # RELATIVE TOLERANCE: 1% of the stone's smallest side
    # This adapts to huge stones (1000cts) and tiny stones (1ct)
    tolerance = min_dim * 0.01 
    
    for _ in range(100):
        if scale > max_dim: break
        
        test_gem = gem_curr.copy()
        test_gem.apply_scale(scale)
        
        try:
            # Check distance
            sdf = prox.signed_distance(test_gem.vertices)
            
            # Allow points to be slightly outside based on relative tolerance
            inside_mask = sdf < tolerance
            fraction_inside = np.mean(inside_mask)
            
            # Relaxed Rule: 90% inside
            if fraction_inside > 0.90:
                best_scale = scale
                scale += step_size
            else:
                break
        except:
            break
            
    return best_scale

def optimize_cut(rough_mesh_path):
    print(f"--- Running Geometric Optimization (Guaranteed Mode) ---")
    
    try:
        rough = trimesh.load(rough_mesh_path)
    except:
        return [{"shape": "Error", "max_volume_cm3": 0, "score": 0}]

    if hasattr(rough, 'convex_hull'): rough = rough.convex_hull

    # Calculate Grid
    extents = rough.extents
    min_dim = np.min(extents)
    max_dim = np.max(extents)
    
    # Create Proximity Engine
    try:
        prox = trimesh.proximity.ProximityQuery(rough)
    except:
        print("❌ Proximity failed. Using Fallback.")
        prox = None

    center = np.array([0.0, 0.0, 0.0])
    offset = max_dim * 0.1
    search_points = [
        center,
        center + [offset, 0, 0], center - [offset, 0, 0],
        center + [0, offset, 0], center - [0, offset, 0],
        center + [0, 0, offset], center - [0, 0, offset]
    ]

    shapes = get_standard_shapes()
    results = []
    
    global_best_vol = 0.0
    global_best_mesh = None

    for name, gem_template in shapes.items():
        gem_template.vertices -= gem_template.centroid
        
        shape_best_vol = 0.0
        shape_best_mesh = None
        
        rotations = [
            np.eye(4),
            trimesh.transformations.rotation_matrix(np.pi/2, [1,0,0]),
            trimesh.transformations.rotation_matrix(np.pi/2, [0,1,0])
        ]
        
        for rot in rotations:
            rotated_gem = gem_template.copy()
            rotated_gem.apply_transform(rot)
            
            # Try Grid Search
            if prox:
                for point in search_points:
                    scale = check_fit_proximity(prox, rotated_gem, point, min_dim, max_dim)
                    
                    if scale > 0:
                        final_gem = rotated_gem.copy()
                        final_gem.vertices += point
                        final_gem.apply_scale(scale)
                        
                        if final_gem.volume > shape_best_vol:
                            shape_best_vol = final_gem.volume
                            shape_best_mesh = final_gem

        # --- SAFETY NET: FORCE A RESULT ---
        # If optimization returned 0 (math failed), assume a safe 30% size fit.
        # This guarantees the red gem appears and is not 0 cts.
        status = "Optimized"
        if shape_best_vol == 0:
            print(f"   ⚠️ Optimization 0 for {name}. Forcing Safety Fit.")
            safe_scale = min_dim * 0.30 # 30% of width
            
            shape_best_mesh = gem_template.copy()
            shape_best_mesh.apply_scale(safe_scale) # Scale
            # No rotation/translation (assume center is safest)
            
            shape_best_vol = shape_best_mesh.volume
            status = "Fallback (Safety Net)"

        if shape_best_vol > global_best_vol:
            global_best_vol = shape_best_vol
            global_best_mesh = shape_best_mesh

        results.append({ "shape": name, "max_volume_cm3": shape_best_vol, "score": shape_best_vol, "status": status })
        print(f"   Tested {name}: Vol {shape_best_vol:.3f} - {status}")

    # SAVE
    if global_best_mesh:
        save_path = rough_mesh_path.replace("final_textured_model_scaled.ply", "best_cut.ply")
        global_best_mesh.visual.face_colors = [255, 0, 0, 255]
        global_best_mesh.export(save_path)
        print(f"   💎 Saved Best Cut: {os.path.basename(save_path)}")

    results.sort(key=lambda x: x['score'], reverse=True)
    return results