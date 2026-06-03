import trimesh
import numpy as np
import os
import concurrent.futures
import itertools
from gem_shapes import get_standard_shapes

# --- CONFIG ---
SAFETY_FACTOR = 0.92
MIN_WALL_DIST = 0.02
MIN_VOLUME_RATIO = 0.02
GEM_SPACING_BUFFER = 0.05
MAX_GEMS = 5

# How close (in mesh units) a gem vertex can be to a fracture point
FRACTURE_SAFE_DIST = 0.05

def load_fracture_points(job_folder):
    """Loads defects.ply and returns fracture positions as Nx3 numpy array."""
    defects_path = os.path.join(job_folder, "dense", "defects.ply")
    if not os.path.exists(defects_path):
        return np.empty((0, 3))
    try:
        pcd = trimesh.load(defects_path)
        pts = np.array(pcd.vertices)
        print(f"   ⚠️  Loaded {len(pts)} fracture points for avoidance.")
        return pts
    except Exception as e:
        print(f"   ⚠️  Could not load fracture points: {e}")
        return np.empty((0, 3))


def gem_hits_fracture(gem_verts, fracture_pts, safe_dist):
    if len(fracture_pts) == 0:
        return False
    gem_min = gem_verts.min(axis=0) - safe_dist
    gem_max = gem_verts.max(axis=0) + safe_dist

    in_box = np.all((fracture_pts >= gem_min) & (fracture_pts <= gem_max), axis=1)
    nearby = fracture_pts[in_box]
    if len(nearby) == 0:
        return False

    for fp in nearby:
        if np.linalg.norm(gem_verts - fp, axis=1).min() < safe_dist:
            return True
    return False

def check_overlap(new_gem, existing_gems):
    if not existing_gems:
        return False
    cutter = new_gem.copy()
    cutter.apply_scale(1.0 + GEM_SPACING_BUFFER)
    min_n, max_n = cutter.bounds

    for ex in existing_gems:
        min_e, max_e = ex.bounds
        if np.any(min_n > max_e) or np.any(max_n < min_e):
            continue
        if np.any(ex.contains(cutter.vertices)):
            return True
        if np.any(cutter.contains(ex.vertices)):
            return True
    return False


def get_max_scale_binary(rough_prox, rough_bounds, gem, position, min_dim, max_dim, obstacles, fracture_pts, normals_flipped):
    base_verts = gem.vertices.copy()
    low = 0.0
    high = max_dim
    best_scale = 0.0

    for _ in range(12):
        mid = (low + high) / 2.0
        current_verts = base_verts * mid + position

        # FAST AABB PRUNING
        g_min = current_verts.min(axis=0)
        g_max = current_verts.max(axis=0)
        if np.any(g_min < rough_bounds[0]) or np.any(g_max > rough_bounds[1]):
            high = mid
            continue

        test_gem = gem.copy()
        test_gem.vertices = current_verts

        if check_overlap(test_gem, obstacles):
            high = mid
            continue

        if gem_hits_fracture(current_verts, fracture_pts, FRACTURE_SAFE_DIST):
            high = mid
            continue

        try:
            sdf = rough_prox.signed_distance(current_verts)
            if normals_flipped:
                sdf = -sdf
                
            if np.min(sdf) > MIN_WALL_DIST:
                best_scale = mid
                low = mid
            else:
                high = mid
        except Exception:
            high = mid

    return best_scale


# --- SUBPROCESS WORKER ---
def worker_check_rot(args):
    (rot, search_points, rough_V, rough_F, rough_bounds,
     obstacles_V_list, obstacles_F_list,
     gem_V, gem_F,
     min_dim, max_dim,
     fracture_pts) = args

    rough = trimesh.Trimesh(vertices=rough_V, faces=rough_F)
    try:
        prox = trimesh.proximity.ProximityQuery(rough)
        sdf_center = prox.signed_distance([[0, 0, 0]])[0]
        normals_flipped = bool(sdf_center < 0)
    except Exception:
        return 0.0, None, rot

    obstacles =[trimesh.Trimesh(vertices=v, faces=f) for v, f in zip(obstacles_V_list, obstacles_F_list)]

    base_gem = trimesh.Trimesh(vertices=gem_V, faces=gem_F)
    base_gem.apply_transform(rot)
    base_gem.vertices -= base_gem.bounds.mean(axis=0)

    best_scale = 0.0
    best_pos = None

    for pos in search_points:
        scale = get_max_scale_binary(
            prox, rough_bounds, base_gem, pos, min_dim, max_dim, obstacles, fracture_pts, normals_flipped
        )
        if scale > best_scale:
            best_scale = scale
            best_pos = pos

    return best_scale, best_pos, rot

def find_best_gem_in_space(
    rough, shapes, obstacles, search_points,
    min_dim, max_dim, fracture_pts,
    preferred_shape=None, progress_callback=None
):
    rough_V = rough.vertices
    rough_F = rough.faces
    rough_bounds = rough.bounds
    obs_V_list = [m.vertices for m in obstacles]
    obs_F_list = [m.faces for m in obstacles]

    if preferred_shape and preferred_shape in shapes:
        active_shapes = {preferred_shape: shapes[preferred_shape]}
        print(f"   🔷 Preferred shape: {preferred_shape}")
    else:
        active_shapes = shapes

    # SMART ROTATION SEARCH (16 fast angles instead of 64 slow ones)
    rotations =[]
    # 90-degree flips on X and Y, 45-degree spins on Z
    for rx in[0, np.pi/2]:
        for ry in [0, np.pi/2]:
            for rz in[0, np.pi/4, np.pi/2, 3*np.pi/4]:
                rotations.append(trimesh.transformations.euler_matrix(rx, ry, rz))

    tasks = []
    task_info =[]

    for name, tmpl in active_shapes.items():
        tmpl_copy = tmpl.copy()
        tmpl_copy.vertices -= tmpl_copy.bounds.mean(axis=0)

        for rot in rotations:
            tasks.append((
                rot, search_points,
                rough_V, rough_F, rough_bounds,
                obs_V_list, obs_F_list,
                tmpl_copy.vertices, tmpl_copy.faces,
                min_dim, max_dim,
                fracture_pts
            ))
            task_info.append({"name": name, "gem_template": tmpl_copy})

    global_best = {"vol": 0, "gem": None, "name": ""}
    total_tasks = len(tasks)

    max_cores = os.cpu_count() or 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
        for i, result in enumerate(executor.map(worker_check_rot, tasks)):
            # Update UI less frequently to save UI rendering lag
            if progress_callback and not obstacles and i % 5 == 0:
                progress_callback(i, total_tasks, "Scanning angles...")

            scale, pos, rot = result
            if scale > 0:
                info = task_info[i]
                final_scale = scale * SAFETY_FACTOR
                vol = (final_scale ** 3) * info['gem_template'].volume

                if vol > global_best["vol"]:
                    global_best["vol"] = vol
                    global_best["name"] = info['name']

                    win = info['gem_template'].copy()
                    win.apply_transform(rot)
                    win.vertices -= win.bounds.mean(axis=0)
                    win.apply_scale(final_scale)
                    win.vertices += pos
                    global_best["gem"] = win

    return global_best["gem"], global_best["vol"], global_best["name"]


def optimize_cut(rough_mesh_path, mode="multi", preferred_shape=None, job_folder=None, progress_callback=None):
    print(f"--- Optimizer | mode={mode} | shape={preferred_shape or 'auto'} ---")
    try:
        rough = trimesh.load(rough_mesh_path)
    except Exception:
        return[]

    rough.vertices -= rough.bounds.mean(axis=0)

    extents = rough.extents
    min_dim = np.min(extents)
    max_dim = np.max(extents)
    center = np.zeros(3)

    # FAST TRANSLATION SEARCH (7 core positions instead of 27)
    offset_x = extents[0] * 0.15
    offset_y = extents[1] * 0.15
    offset_z = extents[2] * 0.15
    
    search_points = [
        center,
        center +[offset_x, 0, 0], center - [offset_x, 0, 0],
        center +[0, offset_y, 0], center - [0, offset_y, 0],
        center +[0, 0, offset_z], center - [0, 0, offset_z],
    ]

    shapes = get_standard_shapes()

    fracture_pts = np.empty((0, 3))
    if job_folder:
        fracture_pts = load_fracture_points(job_folder)

    print("   🔍 Finding primary gem...")
    gem1, vol1, name1 = find_best_gem_in_space(
        rough, shapes,[], search_points,
        min_dim, max_dim, fracture_pts,
        preferred_shape=preferred_shape,
        progress_callback=progress_callback
    )

    if not gem1:
        print("   ⚠️ Optimization failed. Using safety fallback.")
        fallback_key = list(shapes.keys())[0]
        gem1 = shapes[fallback_key].copy()
        gem1.apply_scale(min_dim * 0.2)
        gem1.vertices -= gem1.bounds.mean(axis=0)
        vol1 = gem1.volume
        name1 = f"{fallback_key} (Fallback)"

    strategies =[{
        "strategy": "Single Large",
        "shape": name1,
        "gems":[gem1],
        "total_volume": vol1
    }]

    if mode == "multi":
        if progress_callback:
            progress_callback(100, 100, "Fitting secondary gems...")

        current_gems = [gem1]
        current_vol = vol1
        current_names = [name1]

        for i in range(MAX_GEMS - 1):
            g_next, v_next, n_next = find_best_gem_in_space(
                rough, shapes, current_gems, search_points,
                min_dim, max_dim, fracture_pts,
                preferred_shape=preferred_shape
            )

            if g_next and v_next > rough.volume * MIN_VOLUME_RATIO:
                print(f"      + Gem {i + 2}: {n_next} (vol={v_next:.3f})")
                current_gems.append(g_next)
                current_names.append(n_next)
                current_vol += v_next
            else:
                break

        if len(current_gems) > 1:
            strategies.append({
                "strategy": "Multi-Gem",
                "shape": f"{len(current_gems)} × {name1}",
                "gems": current_gems,
                "total_volume": current_vol
            })

    strategies.sort(key=lambda x: x['total_volume'], reverse=True)

    winner = strategies[0]
    output_dir = os.path.dirname(rough_mesh_path)
    combined = trimesh.util.concatenate(winner['gems'])
    combined.visual.face_colors =[255, 0, 0, 255]
    combined.export(os.path.join(output_dir, "best_cut.ply"))

    return strategies