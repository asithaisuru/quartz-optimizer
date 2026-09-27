"""Canonical-supported connectivity; never synthesize positive-volume material."""
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def consolidate_topology(mesh, stock, scale, blocked_test):
    indices = stock.indices
    shape = tuple(indices.max(axis=0) + 2)
    occupied = np.zeros(shape, dtype=bool)
    occupied[tuple(indices.T)] = True
    raw_labels, raw_count = ndimage.label(occupied)
    raw_ids = raw_labels[tuple(indices.T)]
    raw_components = [np.flatnonzero(raw_ids == i) for i in range(1, raw_count + 1)]
    edges = mesh.edges_unique
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(len(mesh.vertices), len(mesh.vertices)))
    source_count, vertex_source = connected_components(graph, directed=False)
    # Assign existing boundary cells to canonical sources; never add or reweight
    # cells. Nearest surface-vertex attribution is a disclosed raster approximation.
    nearest = cKDTree(mesh.vertices).query(stock.centres)[1]
    stock.source_labels = vertex_source[nearest]
    stock._components = None
    before = stock.connected_components()
    cell_component = np.empty(len(indices), dtype=int)
    for i, part in enumerate(before):
        cell_component[part] = i
    lookup = {tuple(cell): i for i, cell in enumerate(indices)}
    vertex_cells = np.rint((mesh.vertices - stock.origin) / stock.pitch).astype(int)
    mapping = np.array([lookup.get(tuple(cell), -1) for cell in vertex_cells])
    a, b = mapping[edges[:, 0]], mapping[edges[:, 1]]
    valid = (a >= 0) & (b >= 0)
    candidates = edges[valid]
    a, b = a[valid], b[valid]
    valid = ((cell_component[a] != cell_component[b])
             & (stock.source_labels[a] == stock.source_labels[b])
             & (stock.source_labels[a] == vertex_source[candidates[:, 0]]))
    candidates = candidates[valid]
    lengths = np.linalg.norm(mesh.vertices[candidates[:, 1]] - mesh.vertices[candidates[:, 0]], axis=1)
    candidates = candidates[np.argsort(lengths)]
    triangles = mesh.triangles
    signed = np.einsum("ij,ij->i", triangles[:, 0],
                      np.cross(triangles[:, 1], triangles[:, 2])) / 6
    source_volume = np.bincount(vertex_source[mesh.faces[:, 0]], weights=signed,
                                minlength=source_count)
    normals = mesh.vertex_normals
    links, repairs, attempted, tried = [], [], set(), 0
    for edge in candidates:
        if tried >= 256:
            break
        first, last = mesh.vertices[edge]
        if np.linalg.norm(last - first) > 2 * stock.pitch:
            continue
        pair = tuple(sorted((int(cell_component[mapping[edge[0]]]),
                             int(cell_component[mapping[edge[1]]]))))
        if pair in attempted:
            continue
        tried += 1
        # Local inward supersampling along an actual canonical edge. Distance
        # alone never authorizes a repair. Multiple inset scales cover thin webs.
        t = np.linspace(0, 1, 17)[:, None]
        normal = normals[edge[0]] * (1-t) + normals[edge[1]] * t
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
        direction = 1 if source_volume[vertex_source[edge[0]]] >= 0 else -1
        for fraction in (.01, .001, .0001):
            path = first * (1-t) + last * t - direction * normal * stock.pitch * fraction
            ends = np.rint((path[[0, -1]] - stock.origin) / stock.pitch).astype(int)
            left, right = (lookup.get(tuple(x), -1) for x in ends)
            if left < 0 or right < 0 or left == right:
                continue
            if stock.source_labels[left] != stock.source_labels[right]:
                continue
            if blocked_test(path).any() or not mesh.contains(path).all():
                continue
            links.append((left, right, path))
            repairs.append({"component_ids": list(pair),
                "reason": "mesh_supported_subvoxel_connection",
                "gap_mm": float(np.linalg.norm(stock.centres[left]-stock.centres[right]) * scale),
                "added_voxels": 0, "added_mass_ct": 0.0,
                "support": "canonical_edge_with_17_interior_samples",
                "interior_path_mesh_units": path.tolist()})
            attempted.add(pair)
            break
    stock.support_links = tuple(links)
    stock._components = None
    consolidated = stock.connected_components()
    classification = []
    for index, active in enumerate(raw_components, 1):
        sources = np.unique(stock.source_labels[active])
        classification.append({"raw_component_id":index,
            "canonical_source_ids":sources.tolist(),
            "classification":("multiple_canonical_sources_in_coarse_component" if len(sources)>1
                              else "single_canonical_source"),
            "representation_repair_ids":[i for i,(a,b,_) in enumerate(links)
                                          if a in active or b in active]})
    return {
        "component_classification":classification,
        "raw_component_count": raw_count,
        "raw_26_component_count": ndimage.label(occupied, np.ones((3,3,3)))[1],
        "canonical_component_count": int(source_count),
        "source_attributed_component_count": len(before),
        "consolidated_component_count": len(consolidated),
        "resolution_repairs": repairs, "local_edges_tested": tried,
        "representation_mass_change_ct": 0.0,
        "source_attribution": "nearest_canonical_surface_vertex_for_existing_voxel_cells",
        "genuine_source_groups": len(np.unique(stock.source_labels)),
        "note": "Watertightness does not imply one connected solid. No positive-volume bridge is added.",
    }, raw_components
