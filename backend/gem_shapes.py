import trimesh
import numpy as np
import os

# Path to your models folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_and_normalize_gem(filepath):
    """
    Loads a mesh, fixes orientation, centers it, and normalizes size.
    """
    try:
        gem = trimesh.load(filepath)
        
        # Handle Scenes (Multiple objects in one file)
        if isinstance(gem, trimesh.Scene):
            if len(gem.geometry) > 0:
                gem = trimesh.util.concatenate(tuple(gem.geometry.values()))
            else:
                return None

        # --- STANDARD FIXES FOR INTERNET MODELS ---
        
        # 1. Rotation Fix (Most internet models are Y-up, we need Z-up)
        # We rotate 90 degrees on X to stand them up
        rot_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        gem.apply_transform(rot_matrix)

        # 2. Hard Centering (Bounding Box Center -> 0,0,0)
        gem.vertices -= gem.bounds.mean(axis=0)

        # 3. Normalize Size (Scale to Unit Radius = 1.0)
        # This ensures a 100mm model and a 1mm model are treated equally
        current_radius = np.max(gem.extents) / 2.0
        if current_radius == 0: current_radius = 1.0
        
        target_radius = 1.0
        scale_factor = target_radius / current_radius
        gem.apply_scale(scale_factor)
        
        gem.fix_normals()

        # Cap vertex count so the optimizer's per-vertex containment checks
        # stay fast regardless of how detailed the downloaded model is.
        MAX_FACES = 800
        if len(gem.faces) > MAX_FACES:
            gem = gem.simplify_quadric_decimation(MAX_FACES)
            gem.fix_normals()

        return gem
        
    except Exception as e:
        print(f"   Failed to load {os.path.basename(filepath)}: {e}")
        return None


def _renormalize_shape(mesh):
    mesh.vertices -= mesh.bounds.mean(0)
    radius = np.max(mesh.extents) / 2.0
    if radius <= 0:
        radius = 1.0
    mesh.apply_scale(1.0 / radius)
    mesh.fix_normals()
    return mesh


def _rect_octagon_loop(x, y, z, chamfer=0.18):
    cx = x * chamfer
    cy = y * chamfer
    return np.array([
        [-x + cx, -y, z],
        [x - cx, -y, z],
        [x, -y + cy, z],
        [x, y - cy, z],
        [x - cx, y, z],
        [-x + cx, y, z],
        [-x, y - cy, z],
        [-x, -y + cy, z],
    ], dtype=float)


def _radial_loop(radius_x, radius_y, z, count=16, phase=0.0):
    angles = np.linspace(0.0, np.pi * 2.0, count, endpoint=False) + phase
    return np.column_stack([
        np.cos(angles) * radius_x,
        np.sin(angles) * radius_y,
        np.full(count, z),
    ])


def _mesh_from_loops(loops):
    vertices = np.vstack(loops)
    faces = []
    loop_len = len(loops[0])
    for li in range(len(loops) - 1):
        a0 = li * loop_len
        b0 = (li + 1) * loop_len
        for i in range(loop_len):
            j = (i + 1) % loop_len
            faces.append([a0 + i, a0 + j, b0 + j])
            faces.append([a0 + i, b0 + j, b0 + i])

    top_center = len(vertices)
    bottom_center = top_center + 1
    vertices = np.vstack([
        vertices,
        loops[0].mean(axis=0),
        loops[-1].mean(axis=0),
    ])
    for i in range(loop_len):
        j = (i + 1) % loop_len
        faces.append([top_center, i, j])
        start = (len(loops) - 1) * loop_len
        faces.append([bottom_center, start + j, start + i])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=True)
    return _renormalize_shape(mesh)


def _procedural_round():
    return _mesh_from_loops([
        _radial_loop(0.45, 0.45, 0.42, 16, phase=np.pi / 16),
        _radial_loop(0.78, 0.78, 0.18, 16),
        _radial_loop(1.0, 1.0, 0.0, 16, phase=np.pi / 16),
        _radial_loop(0.52, 0.52, -0.36, 16),
        _radial_loop(0.08, 0.08, -0.72, 16, phase=np.pi / 16),
    ])


def _procedural_emerald(x=1.0, y=0.66):
    return _mesh_from_loops([
        _rect_octagon_loop(x * 0.52, y * 0.52, 0.40, 0.12),
        _rect_octagon_loop(x * 0.82, y * 0.82, 0.18, 0.16),
        _rect_octagon_loop(x, y, 0.0, 0.20),
        _rect_octagon_loop(x * 0.55, y * 0.55, -0.36, 0.16),
        _rect_octagon_loop(x * 0.16, y * 0.16, -0.70, 0.10),
    ])

def get_standard_shapes():
    """
    Scans the /models/ folder and returns a dictionary of all valid gems.
    """
    shapes = {}
    
    # 1. Check if folder exists
    if not os.path.exists(MODELS_DIR):
        print(f"Models folder missing: {MODELS_DIR}")
        return shapes

    # 2. Loop through all files
    files = sorted(os.listdir(MODELS_DIR))
    print(f"   Scanning for Gem Models in: {MODELS_DIR}")
    
    for f in files:
        if f.lower().endswith(('.stl', '.obj', '.ply')):
            # Create a pretty name (e.g. "Emerald_Cut.obj" -> "Emerald Cut")
            name = os.path.splitext(f)[0].replace("_", " ").replace("-", " ").title()
            
            filepath = os.path.join(MODELS_DIR, f)
            mesh = load_and_normalize_gem(filepath)
            
            if mesh:
                shapes[name] = mesh
                print(f"      Loaded: {name}")
    
    if not shapes:
        print("   No models found. Please add .obj/.stl files to backend/models/")

    if "Round Brilliant Cut" not in shapes:
        shapes["Round Brilliant Cut"] = _procedural_round()
        print("      Generated: Round Brilliant Cut")

    if "Emerald Cut" not in shapes:
        shapes["Emerald Cut"] = _procedural_emerald(1.0, 0.64)
        print("      Generated: Emerald Cut")

    # --- Derived shapes (no additional model files needed) ------------------
    # Generated by axis-scaling existing loaded templates, then re-normalizing.
    if "Round Brilliant Cut" in shapes:
        base = shapes["Round Brilliant Cut"]

        oval = base.copy()
        oval.vertices[:, 0] *= 1.25         # moderate oval outline
        oval.vertices -= oval.bounds.mean(0)
        oval.apply_scale(1.0 / (np.max(oval.extents) / 2.0))
        oval.fix_normals()
        shapes["Oval Brilliant"] = oval
        print("      Generated: Oval Brilliant")

        mar = base.copy()
        mar.vertices[:, 0] *= 1.55          # avoid overly skinny auto cuts
        mar.vertices -= mar.bounds.mean(0)
        mar.apply_scale(1.0 / (np.max(mar.extents) / 2.0))
        mar.fix_normals()
        shapes["Marquise Cut"] = mar
        print("      Generated: Marquise Cut")

        pear = base.copy()
        y = pear.vertices[:, 1]
        y_min, y_max = float(y.min()), float(y.max())
        y_norm = (y - y_min) / max(y_max - y_min, 1e-6)
        width_profile = 0.64 + 0.48 * y_norm
        pear.vertices[:, 0] *= width_profile
        pear.vertices[:, 1] *= 1.1
        pear.vertices[:, 0] += (0.5 - y_norm) * 0.05
        pear = _renormalize_shape(pear)
        shapes["Pear Cut"] = pear
        print("      Generated: Pear Cut")

    if "Emerald Cut" in shapes:
        base = shapes["Emerald Cut"]
        princess = base.copy()
        ex, ey = princess.extents[0], princess.extents[1]
        princess.vertices[:, 1] *= ex / ey  # equalize X/Y → square outline
        princess.vertices -= princess.bounds.mean(0)
        princess.apply_scale(1.0 / (np.max(princess.extents) / 2.0))
        princess.fix_normals()
        shapes["Princess Cut"] = princess
        print("      Generated: Princess Cut")

        cushion = princess.copy()
        xy = cushion.vertices[:, :2]
        denom = np.maximum(np.max(np.abs(xy), axis=0), 1e-6)
        nx = np.abs(xy[:, 0]) / denom[0]
        ny = np.abs(xy[:, 1]) / denom[1]
        corner_soften = 1.0 - 0.18 * (nx * ny) ** 0.7
        cushion.vertices[:, 0] *= corner_soften
        cushion.vertices[:, 1] *= corner_soften
        cushion = _renormalize_shape(cushion)
        shapes["Cushion Cut"] = cushion
        print("      Generated: Cushion Cut")

    return shapes
