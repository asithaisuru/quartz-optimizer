import open3d as o3d
import os
import sys
import numpy as np
import copy

def post_process_point_cloud(ply_path, output_mesh_path):
    print(f"--- Starting Post-Processing & Texturing for: {ply_path} ---")
    
    if not os.path.exists(ply_path):
        print("Error: PLY file not found.")
        return None
        
    # 1. Load Point Cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    
    if len(pcd.points) < 10:
        print("Error: Point cloud has too few points.")
        return None

    # 2. Cleanup Noise
    print("Step 1: Removing Noise...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    pcd_clean = pcd.select_by_index(ind)
    
    # 3. FORCE OUTWARD NORMALS (The "Balloon" Fix)
    print("Step 2: Forcing Normals Outward...")
    
    # Calculate the center of the object
    center = pcd_clean.get_center()
    
    # Access points and normals
    pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Manually force normals to point away from the center
    # This prevents the "Sheet" collapse
    points = np.asarray(pcd_clean.points)
    normals = np.asarray(pcd_clean.normals)
    
    for i in range(len(points)):
        # Vector from center to point
        direction = points[i] - center
        # Normalize it
        direction = direction / np.linalg.norm(direction)
        # Set normal to this direction
        normals[i] = direction
        
    pcd_clean.normals = o3d.utility.Vector3dVector(normals)

    # 4. Create Mesh (Poisson Reconstruction)
    print("Step 3: Creating Watertight Mesh...")
    # Lower depth slightly to make it smoother/more blobby if sparse
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_clean, depth=8, width=0, scale=1.1, linear_fit=True)
            
    # 5. Very Gentle Trim (Keep the back closed)
    print("Step 4: Cleaning Mesh...")
    vertices_to_remove = densities < np.quantile(densities, 0.01) 
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # 6. TEXTURING
    print("Step 5: Applying Texture...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_clean)
    mesh_vertices = np.asarray(mesh.vertices)
    vertex_colors = []
    
    for i in range(len(mesh_vertices)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(mesh_vertices[i], 1)
        # If far from real data, make it dark grey
        dist = np.linalg.norm(mesh_vertices[i] - pcd_clean.points[idx[0]])
        if dist > 0.5: 
            vertex_colors.append([0.1, 0.1, 0.1]) 
        else:
            color = pcd_clean.colors[idx[0]]
            vertex_colors.append(color)
        
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vertex_colors))
    
    # 7. Save
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print(f"✅ Texturing Complete. Saved to: {output_mesh_path}")
    
    return output_mesh_path