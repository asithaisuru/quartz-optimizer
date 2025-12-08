import trimesh
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_custom_gem(filename, target_radius=1.0):
    filepath = os.path.join(MODELS_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: Model {filename} not found.")
        return create_simple_fallback(target_radius)

    try:
        gem = trimesh.load(filepath)
        if isinstance(gem, trimesh.Scene):
            gem = trimesh.util.concatenate(tuple(gem.geometry.values()))

        # --- FIX 1: HARD CENTERING ---
        # Instead of 'apply_translation', we move the vertices directly.
        # This deletes any weird origin history the file might have.
        gem.vertices -= gem.bounds.mean(axis=0)

        # --- FIX 2: ROTATION (Standardize Y-Up/Z-Up) ---
        # Apply 90 degree X rotation (Common for internet models)
        rot_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        gem.apply_transform(rot_matrix)
        
        # Center again after rotation just to be safe
        gem.vertices -= gem.bounds.mean(axis=0)

        # --- FIX 3: NORMALIZE SIZE ---
        current_radius = np.max(gem.extents) / 2.0
        if current_radius == 0: current_radius = 1.0
        
        scale_factor = target_radius / current_radius
        gem.apply_scale(scale_factor)
        
        gem.fix_normals()
        return gem
        
    except Exception as e:
        print(f"❌ Error loading gem {filename}: {e}")
        return create_simple_fallback(target_radius)

def create_simple_fallback(radius=1.0):
    pavilion = trimesh.creation.cone(radius=radius, height=radius*0.8)
    crown = trimesh.creation.cone(radius=radius, height=radius*0.4)
    crown.apply_translation([0, 0, radius*0.6])
    return trimesh.util.concatenate([pavilion, crown])

def get_standard_shapes():
    return {
        "Round Brilliant": load_custom_gem("Round_Brilliant_Cut.stl", target_radius=1.0),
        # NOW LOADING YOUR OBJ FILE:
        "Emerald Cut": load_custom_gem("Emerald_Cut.obj", target_radius=1.0)
    }