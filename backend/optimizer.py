import trimesh
import numpy as np
import os
import copy
import concurrent.futures
from functools import partial
from gem_shapes import get_standard_shapes

# Allow gem to use 99.5% of the available space (almost touching)
SAFETY_FACTOR = 0.995
MIN_VOLUME_RATIO = 0.02 

def get_exact_max_scale(rough, gem, position, min_dim, max_dim, obstacles):
    """
    Uses Binary Search to find the EXACT limit before the gem touches a wall.
    This guarantees the largest possible gem size.
    """
    test_gem = gem.copy()
    test_gem.vertices += position
    
    # Range to search
    low = 0.0
    high = max_dim
    best_valid_scale = 0.0
    
    # 15 steps of binary search = high precision
    for _ in range(15):
        mid = (low + high) / 2.0
        
        # Scale locally at center of gem
        candidate = test_gem.copy()
        candidate.vertices -= candidate.bounds.mean(axis=0) # zero center
        candidate.apply_scale(mid)
        candidate.vertices += position + test_gem.bounds.mean(axis=0) # move back
        
        # 1. Obstacle Check
        collision = False
        if obstacles:
            c_mgr = trimesh.collision.CollisionManager()
            c_mgr.add_object('new', candidate)
            for i, obs in enumerate(obstacles):
                c_mgr.add_object(str(i), obs)
            if c_mgr.in_collision_internal():
                collision = True
        
        if collision:
            high = mid # Hit neighbor, try smaller
            continue
            
        try:
            # 2. Wall Check (Ray Casting is most accurate)
            # contains_points returns array of True/False
            # All vertices must be True (Inside)
            is_inside = rough.contains(candidate.vertices)
            
            if np.all(is_inside):
                best_valid_scale = mid
                low = mid # Fits! Try bigger.
            else:
                high = mid # Hit wall! Try smaller.
        except:
            high = mid

    return best_valid_scale

# --- MULTI-THREAD WORKER ---
def worker_check_rot(args):
    """Checks grid for ONE rotation on ONE shape"""
    (rot, search_points, rough_V, rough_F, obstacles_V, obstacles_F, gem_V, gem_F, min_dim, max_dim) = args
    
    # Rehydrate
    rough = trimesh.Trimesh(vertices=rough_V, faces=rough_F)
    obstacles = [trimesh.Trimesh(vertices=v, faces=f) for v, f in zip(obstacles_V, obstacles_F)]
    base_gem = trimesh.Trimesh(vertices=gem_V, faces=gem_F)
    base_gem.apply_transform(rot)
    
    best_scale = 0.0
    best_pos = None
    
    for pos in search_points:
        scale = get_exact_max_scale(rough, base_gem, pos, min_dim, max_dim, obstacles)
        
        if scale > best_scale:
            best_scale = scale
            best_pos = pos
            
    return best_scale, best_pos, rot

def optimize_cut(rough_mesh_path, mode="multi", progress_callback=None):
    print(f"--- Running Geometric Optimization (Binary Search Max-Size) ---")
    
    try: rough = trimesh.load(rough_mesh_path)
    except: return []

    # Center to 0
    rough.vertices -= rough.bounds.mean(axis=0)
    
    extents = rough.extents
    min_dim = np.min(extents)
    max_dim = np.max(extents)
    
    # 7-Point Search
    offset = np.max(extents) * 0.15
    center = np.array([0.0, 0.0, 0.0])
    search_points = [
        center,
        center + [offset, 0, 0], center - [offset, 0, 0],
        center + [0, offset, 0], center - [0, offset, 0],
        center + [0, 0, offset], center - [0, 0, offset]
    ]

    shapes = get_standard_shapes()
    
    # Rotations (Identity, 90s)
    rotations = [np.eye(4)]
    for axis in [[1,0,0],[0,1,0],[0,0,1]]:
        rotations.append(trimesh.transformations.rotation_matrix(np.pi/2, axis))

    strategies = []
    
    def find_gem(existing_obstacles):
        # Pack data for threads
        obs_V = [m.vertices for m in existing_obstacles]
        obs_F = [m.faces for m in existing_obstacles]
        
        tasks = []
        for name, tmpl in shapes.items():
            tmpl.vertices -= tmpl.bounds.mean(axis=0) # Zero
            for rot in rotations:
                tasks.append({
                    "name": name,
                    "args": (rot, search_points, rough.vertices, rough.faces, obs_V, obs_F, tmpl.vertices, tmpl.faces, min_dim, max_dim)
                })

        global_best = {"vol": 0, "gem": None, "name": ""}
        
        # Parallel Execution
        total = len(tasks)
        processed = 0
        if progress_callback: progress_callback(0, total, "Scanning volume...")

        # max_threads = os.cpu_count() or 4
        max_threads = os.cpu_count() or 16 # Fallback to 24 if detection fails
        print(f"   🔥 Processing {total_tasks} fit-tests on {max_threads} CORES...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
            futures = {executor.submit(worker_check_rot, t["args"]): t["name"] for t in tasks}
            
            for f in concurrent.futures.as_completed(futures):
                name = futures[f]
                processed += 1
                
                scale, pos, rot = f.result()
                if scale > 0:
                    gem_t = shapes[name].copy()
                    gem_t.vertices -= gem_t.bounds.mean(axis=0)
                    gem_t.apply_transform(rot)
                    gem_t.vertices += pos
                    gem_t.apply_scale(scale * SAFETY_FACTOR)
                    
                    if gem_t.volume > global_best["vol"]:
                        global_best["vol"] = gem_t.volume
                        global_best["gem"] = gem_t
                        global_best["name"] = name
        
        return global_best["gem"], global_best["vol"], global_best["name"]

    # STEP 1
    gem1, vol1, name1 = find_gem([])
    
    if not gem1:
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

    # STEP 2
    if mode == "multi":
        gem2, vol2, name2 = find_gem([gem1])
        if gem2 and vol2 > (rough.volume * MIN_VOLUME_RATIO):
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
    # Need this safeguard for multiprocessing on Windows
    pass