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
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.2)
    pcd_clean = pcd.select_by_index(ind)
    
    # 3. Estimate Normals
    print("Step 2: Estimating Normals...")
    pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_clean.orient_normals_consistent_tangent_plane(k=15)

    # 4. Create Mesh (Poisson)
    print("Step 3: Creating Watertight Mesh...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        # linear_fit=True helps with sharp edges
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_clean, depth=9, width=0, scale=1.1, linear_fit=True)
            
    # --- CRITICAL FIX: DO NOT TRIM MESH ---
    # Previous code removed low-density vertices, causing holes.
    # We SKIP that now to ensure it remains a closed solid.
    # --------------------------------------
    
    # 5. TEXTURING
    print("Step 4: Applying Texture...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_clean)
    mesh_vertices = np.asarray(mesh.vertices)
    vertex_colors = []
    
    for i in range(len(mesh_vertices)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(mesh_vertices[i], 1)
        dist = np.linalg.norm(mesh_vertices[i] - pcd_clean.points[idx[0]])
        
        # Paint "fake" geometry (filled holes) dark grey
        if dist > 0.5: 
            vertex_colors.append([0.2, 0.2, 0.2]) 
        else:
            color = pcd_clean.colors[idx[0]]
            vertex_colors.append(color)
        
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vertex_colors))
    
    # 6. Save
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print(f"✅ Texturing Complete. Saved to: {output_mesh_path}")
    
    return output_mesh_path