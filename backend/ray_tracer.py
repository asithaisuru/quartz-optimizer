import trimesh
import numpy as np

def calculate_light_performance(mesh_path, num_rays=1000):
    print(f"--- Running Light Performance Simulation (Auto-Align) ---")
    try:
        mesh = trimesh.load(mesh_path)
    except:
        return {"score": 0, "grade": "Error"}

    # 1. FIND THE TABLE (The largest flat facet)
    # This determines the "Up" direction of the cut gem
    facets, facets_area = mesh.facets, mesh.area_faces
    
    # Group faces by normal to find large flat planes
    # facets returns list of lists of face indices
    # We want the group with the largest total area (usually the Table)
    largest_facet_idx = np.argmax([mesh.area_faces[f].sum() for f in facets])
    table_faces = facets[largest_facet_idx]
    
    # Calculate the normal of this table (The "Up" Vector)
    table_normal = mesh.face_normals[table_faces[0]]
    
    print(f"   🧭 Detected Table Normal: {table_normal.round(2)}")

    # 2. Identify Pavilion (Opposite to Table)
    # Faces pointing mostly OPPOSITE to the table normal
    dot_products = np.dot(mesh.face_normals, table_normal)
    # Look for faces pointing away (dot product < -0.2)
    pavilion_faces_indices = np.where(dot_products < -0.2)[0]

    # 3. Analyze Angles relative to Table
    ideal_angle_count = 0
    total_pavilion_area = 0
    
    for idx in pavilion_faces_indices:
        normal = mesh.face_normals[idx]
        area = mesh.area_faces[idx]
        total_pavilion_area += area
        
        # Angle between Pavilion Face Normal and Table Down Vector
        # We want the angle relative to the "Girdle Plane" (Horizontal)
        # Angle from Vertical = arccos(dot(normal, -table_normal))
        # Angle from Horizontal = 90 - Angle from Vertical
        
        angle_from_vertical_rad = np.arccos(np.dot(normal, -table_normal))
        angle_from_horizontal_deg = 90 - np.degrees(angle_from_vertical_rad)
        
        # Quartz Sweet Spot: 40° to 45° relative to girdle
        if 40.0 <= angle_from_horizontal_deg <= 45.0:
            ideal_angle_count += area * 1.0
        elif 35.0 <= angle_from_horizontal_deg < 40.0 or 45.0 < angle_from_horizontal_deg <= 50.0:
            ideal_angle_count += area * 0.5
            
    if total_pavilion_area == 0:
        score = 0
    else:
        score = (ideal_angle_count / total_pavilion_area) * 100

    if score >= 90: grade = "Excellent"
    elif score >= 75: grade = "Very Good"
    elif score >= 60: grade = "Good"
    elif score >= 40: grade = "Fair"
    else: grade = "Poor"

    print(f"   ✨ Brilliance Score: {score:.1f}/100 ({grade})")
    
    return {
        "score": round(score, 1),
        "grade": grade,
        "details": f"Analyzed {len(pavilion_faces_indices)} facets relative to Table."
    }