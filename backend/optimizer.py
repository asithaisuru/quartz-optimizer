import trimesh
import numpy as np
import os
import copy
import concurrent.futures
import math
from gem_shapes import get_standard_shapes

# SAFETY: 99% Max size (1% Air Gap)
SAFETY_FACTOR = 0.99
# MIN GAP: 0.05mm equivalent
SPACING = 0.001 

def get_max_scale_binary(rough_V, rough_F, gem_V, gem_F, position, min_dim, max_dim, obstacle_bounds):
    """
    Worker Function: Finds exact max scale using Binary Search.
    Does NOT require the full Trimesh object overhead in main thread.
    """
    # 1. Reconstruct Light Objects
    # We do this here to avoid passing huge objects between threads
    rough = trimesh.Trimesh(vertices=rough_V, faces=rough_F)
    base_gem = trimesh.Trimesh(vertices=gem_V, faces=gem_F)
    
    # Move gem
    base_gem.vertices += position

    low = 0.0
    high = max_dim
    best_scale = 0.0
    
    # 15 Iterations gives ~0.001 precision
    for _ in range(15):
        mid = (low + high) / 2.0
        
        # Create candidate
        # Scale relative to its own center (which is at 'position')
        # Actually base_gem is already at position. 
        # Scale: vertices = (vertices - center) * scale + center
        
        center = position
        current_verts = (base_gem.vertices - center) * mid + center
        
        # 1. Bounding Box Pre-Check (Obstacles)
        # Fast Min/Max check
        min_v = np.min(current_verts, axis=0)
        max_v = np.max(current_verts, axis=0)
        
        collides = False
        if obstacle_bounds:
            for obs_min, obs_max in obstacle_bounds:
                # If overlap
                if not (np.any(min_v > obs_max) or np.any(max_v < obs_min)):
                    collides = True # Box overlap
                    break
        
        if collides:
            high = mid
            continue

        # 2. Strict Wall Check (Ray Casting)
        # Using raw mesh contains logic
        try:
            # Check if all vertices are inside
            if np.all(rough.contains(current_verts)):
                best_scale = mid
                low = mid # Try bigger
            else:
                high = mid # Poked out
        except:
            high = mid
            
    return best_scale

# Helper to unwrap arguments for the process pool
def worker_wrapper(args):
    return get_max_scale_binary(*args)

def optimize_cut(rough_mesh_path, mode="multi", progress_callback=None):
    print(f"--- Running Geometric Optimization (Parallel Binary Search) ---")
    
    try: rough = trimesh.load(rough_mesh_path)
    except: return []

    # Force Center 0
    rough.vertices -= rough.bounds.mean(axis=0)

    # Config
    extents = rough.extents
    min_dim = np.min(extents)
    max_dim = np.max(extents)
    
    # 7-Point Grid Search
    offset = np.max(extents) * 0.15
    center = np.array([0.0, 0.0, 0.0])
    search_points = [
        center,
        center + [offset, 0, 0], center - [offset, 0, 0],
        center + [0, offset, 0], center - [0, offset, 0],
        center + [0, 0, offset], center - [0, 0, offset]
    ]

    shapes = get_standard_shapes()
    
    # 3-Axis Rotations
    rotations = [np.eye(4)]
    for axis in [[1,0,0], [0,1,0], [0,0,1]]:
        rotations.append(trimesh.transformations.rotation_matrix(np.pi/2, axis))

    strategies = []
    
    # Data Preparation for Parallel Processing
    rough_V = rough.vertices
    rough_F = rough.faces
    
    # MAIN GEM FINDER
    def find_gem(existing_obstacles):
        # Convert obstacles to AABB [min, max] for fast checking
        obstacle_bounds = []
        for ex in existing_obstacles:
            # Buffer obstacle slightly
            obstacle_bounds.append((ex.bounds[0] - SPACING, ex.bounds[1] + SPACING))

        tasks = []
        task_info = []

        # Build Job List
        for name, tmpl in shapes.items():
            # Reset Template Center
            tmpl_copy = tmpl.copy()
            tmpl_copy.vertices -= tmpl_copy.bounds.mean(axis=0)
            
            gem_V_base = tmpl_copy.vertices
            gem_F_base = tmpl_copy.faces

            for rot in rotations:
                # Pre-rotate vertices
                # Apply rotation matrix
                homo = np.hstack([gem_V_base, np.ones((len(gem_V_base), 1))])
                gem_V_rot = np.dot(homo, rot.T)[:, :3]

                for pos in search_points:
                    # Arg tuple: (rV, rF, gV, gF, pos, min, max, obs)
                    args = (rough_V, rough_F, gem_V_rot, gem_F_base, pos, min_dim, max_dim, obstacle_bounds)
                    tasks.append(args)
                    task_info.append({"name": name, "pos": pos, "rot": rot, "gem_template": tmpl_copy})

        total_tasks = len(tasks)
        print(f"   🔥 Launching {total_tasks} Threads...")
        
        # Parallel Execution
        global_best_vol = 0.0
        global_best_mesh = None
        global_best_name = ""

        max_cores = os.cpu_count() or 4
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
            # Map returns iterator in order
            results = executor.map(worker_wrapper, tasks)
            
            for i, best_scale in enumerate(results):
                if progress_callback and i % 50 == 0:
                     pct = int((i/total_tasks)*100)
                     # FIX: Ensure msg string is safe
                     progress_callback(i, total_tasks, "Solving...")
                
                if best_scale > 0:
                    info = task_info[i]
                    
                    # Estimate volume
                    # Volume = Scale^3 * BaseVol
                    # We can calc roughly without rebuilding mesh to save time
                    vol = (best_scale**3) * info['gem_template'].volume
                    
                    if vol > global_best_vol:
                        # Only rebuild mesh if it's a winner
                        global_best_vol = vol
                        global_best_name = info['name']
                        
                        win = info['gem_template'].copy()
                        win.apply_transform(info['rot'])
                        win.apply_translation(info['pos'])
                        win.apply_scale(best_scale * SAFETY_FACTOR)
                        global_best_mesh = win

        return global_best_mesh, global_best_vol, global_best_name

    # --- STEP 1 ---
    gem1, vol1, name1 = find_gem([])
    
    if not gem1:
        # Fallback
        name1 = list(shapes.keys())[0] + " (Fallback)"
        gem1 = shapes[list(shapes.keys())[0]].copy()
        gem1.apply_scale(min_dim * 0.25)
        gem1.vertices -= gem1.bounds.mean(axis=0)
        vol1 = gem1.volume

    strategies.append({
        "strategy": "Single Large",
        "shape": name1,
        "gems": [gem1],
        "total_volume": vol1
    })

    # --- STEP 2 ---
    if mode == "multi":
        print("   👥 Parallel Search for Gem 2...")
        gem2, vol2, name2 = find_gem([gem1])
        if gem2 and vol2 > (rough.volume * 0.02):
             strategies.append({
                "strategy": "Multi-Gem",
                "shape": f"{name1} + {name2}",
                "gems": [gem1, gem2],
                "total_volume": vol1 + vol2
            })

    strategies.sort(key=lambda x: x['total_volume'], reverse=True)
    winner = strategies[0]
    
    # Save
    out = os.path.dirname(rough_mesh_path)
    combined = trimesh.util.concatenate(winner['gems'])
    combined.visual.face_colors = [255, 0, 0, 255]
    combined.export(os.path.join(out, "best_cut.ply"))
    
    return strategies
    
if __name__ == '__main__':
    pass