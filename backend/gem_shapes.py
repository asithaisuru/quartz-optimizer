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
        return gem
        
    except Exception as e:
        print(f"   ❌ Failed to load {os.path.basename(filepath)}: {e}")
        return None

def get_standard_shapes():
    """
    Scans the /models/ folder and returns a dictionary of all valid gems.
    """
    shapes = {}
    
    # 1. Check if folder exists
    if not os.path.exists(MODELS_DIR):
        print(f"⚠️ Models folder missing: {MODELS_DIR}")
        return shapes

    # 2. Loop through all files
    files = sorted(os.listdir(MODELS_DIR))
    print(f"   📂 Scanning for Gem Models in: {MODELS_DIR}")
    
    for f in files:
        if f.lower().endswith(('.stl', '.obj', '.ply')):
            # Create a pretty name (e.g. "Emerald_Cut.obj" -> "Emerald Cut")
            name = os.path.splitext(f)[0].replace("_", " ").replace("-", " ").title()
            
            filepath = os.path.join(MODELS_DIR, f)
            mesh = load_and_normalize_gem(filepath)
            
            if mesh:
                shapes[name] = mesh
                print(f"      🔹 Loaded: {name}")
    
    if not shapes:
        print("   ⚠️ No models found! Please add .obj/.stl files to backend/models/")
        
    return shapes