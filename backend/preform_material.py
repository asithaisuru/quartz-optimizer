"""Physical stock accounting, independent of placement/clearance masks.

The filled rough voxel solid is calibrated once to the measured mass. Capped
half-space intersections partition that solid using the *verified* cut planes.
No safety erosion participates in this ledger.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError

MODEL_VERSION = "v2_mass_conserving"


def mass_balance(original, *, retained=0.0, kerf=0.0, defects=0.0,
                 discarded=0.0, unresolved=0.0, numerical=0.0):
    """Expose the residual; never clamp recovery or relabel error as waste."""
    values = (original, retained, kerf, defects, discarded, unresolved, numerical)
    if not all(math.isfinite(float(value)) for value in values) or original <= 0:
        raise ValueError("Physical masses must be finite and the original mass positive.")
    error = original - math.fsum((retained, kerf, defects, discarded, unresolved, numerical))
    tolerance = max(1e-8, original * 1e-8)
    balanced = abs(error) <= tolerance and min(values[1:]) >= -tolerance
    return {
        "mass_balance_error_ct": error,
        "mass_balance_error_percent": error / original * 100.0,
        "mass_balance_tolerance_ct": tolerance,
        "mass_balance_valid": balanced,
        "warnings": [] if balanced else ["Physical mass balance exceeds numerical tolerance."],
    }


class VoxelStock:
    """Union of disjoint voxel interiors, retaining exact clipped boundary cells.

    Edge/corner contacts in a voxel surface can be non-manifold. Clipping closed
    convex cells avoids repairing those contacts or deleting physical material.
    Shared internal faces have zero volume; exported cells keep separate topology.
    """

    def __init__(self, indices, origin, pitch, *, active=None, fragments=None,
                 source_labels=None, support_links=()):
        self.indices = np.asarray(indices, dtype=int)
        self.origin = np.asarray(origin, dtype=float)
        self.pitch = float(pitch)
        self.centres = self.origin + self.indices * self.pitch
        self.active = np.arange(len(self.indices)) if active is None else active
        self.fragments = {} if fragments is None else fragments
        self.source_labels = source_labels
        self.support_links = support_links
        self._mesh = None
        self._cube = trimesh.creation.box(extents=[self.pitch] * 3)
        self.volume = ((len(self.active) - len(self.fragments)) * self.pitch ** 3
                       + math.fsum(v[1] for v in self.fragments.values()))

    def copy(self):
        # Stock geometry is immutable; only the surrounding piece records change.
        return self

    @staticmethod
    def _polyhedron(vertices):
        vertices = np.unique(vertices, axis=0)
        if len(vertices) < 4:
            return None
        try:
            hull = ConvexHull(vertices)
        except QhullError:
            if np.linalg.matrix_rank(vertices - vertices[0]) < 3:
                return None  # Plane/line contact has zero volume.
            raise
        return vertices[hull.vertices], float(hull.volume)

    def clip(self, normal, offset):
        radius = self.pitch * np.abs(normal).sum() / 2
        signed = self.centres[self.active] @ normal - offset
        full = self.active[signed - radius >= 0]
        partial = self.active[(signed - radius < 0) & (signed + radius > 0)]
        kept = list(full)
        fragments = {int(i): self.fragments[int(i)] for i in full if int(i) in self.fragments}
        cube_edges = self._cube.edges_unique
        for index in partial:
            index = int(index)
            previous = self.fragments.get(index)
            vertices = previous[0] if previous else self._cube.vertices + self.centres[index]
            distances = vertices @ normal - offset
            if np.all(distances >= 0):
                kept.append(index)
                if previous:
                    fragments[index] = previous
                continue
            if np.all(distances <= 0):
                continue
            if previous:
                hull = ConvexHull(vertices)
                edges = np.unique(np.sort(np.vstack(
                    [hull.simplices[:, [0, 1]], hull.simplices[:, [1, 2]],
                     hull.simplices[:, [2, 0]]]), axis=1), axis=0)
            else:
                edges = cube_edges
            a, b = edges.T
            cross = (distances[a] < 0) != (distances[b] < 0)
            a, b = a[cross], b[cross]
            t = distances[a] / (distances[a] - distances[b])
            intersections = vertices[a] + t[:, None] * (vertices[b] - vertices[a])
            clipped = self._polyhedron(np.vstack((vertices[distances >= 0], intersections)))
            if clipped is not None:
                kept.append(index)
                fragments[index] = clipped
        kept_set = set(kept)
        links = tuple((a,b,path) for a,b,path in self.support_links
                      if a in kept_set and b in kept_set
                      and np.all(path @ normal - offset >= 0))
        return VoxelStock(self.indices, self.origin, self.pitch,
                          active=np.asarray(kept, dtype=int), fragments=fragments,
                          source_labels=self.source_labels, support_links=links)


    def connected_components(self):
        """Positive-area face adjacency; edge/corner contacts are not handles.

        Clipped cells on either side of an original shared face have identical
        constraints. Checking that face's remaining area prevents a cut plane
        touching a lattice vertex/edge from fabricating a physical connection.
        """
        cached = getattr(self, "_components", None)
        if cached is not None:
            return cached
        lookup = {tuple(self.indices[i]): int(i) for i in self.active}
        remaining = set(int(i) for i in self.active)
        supported = {}
        for a,b,path in self.support_links:
            if a in remaining and b in remaining:
                supported.setdefault(a,[]).append(b)
                supported.setdefault(b,[]).append(a)
        result = []

        def face_exists(index, axis, sign):
            previous = self.fragments.get(index)
            if previous is None:
                return True
            points = previous[0]
            boundary = self.centres[index, axis] + sign * self.pitch / 2
            on_face = points[np.abs(points[:, axis] - boundary) <= self.pitch * 1e-8]
            if len(on_face) < 3:
                return False
            planar = np.delete(on_face, axis, axis=1)
            try:
                return ConvexHull(planar).volume > self.pitch ** 2 * 1e-12
            except QhullError:
                return False

        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            stack, component = [seed], []
            while stack:
                index = stack.pop()
                component.append(index)
                for other in supported.get(index, []):
                    if other in remaining:
                        remaining.remove(other)
                        stack.append(other)
                cell = self.indices[index]
                for axis in range(3):
                    for sign in (-1, 1):
                        neighbour = cell.copy()
                        neighbour[axis] += sign
                        other = lookup.get(tuple(neighbour))
                        if (other not in remaining
                                or (self.source_labels is not None
                                    and self.source_labels[index] != self.source_labels[other])
                                or not face_exists(index, axis, sign)
                                or not face_exists(other, axis, -sign)):
                            continue
                        remaining.remove(other)
                        stack.append(other)
            result.append(np.asarray(sorted(component), dtype=int))
        self._components = result
        return result

    def subset(self, active):
        """Diagnostic sub-solid with the same global calibration and cell IDs."""
        active = np.asarray(active, dtype=int)
        return VoxelStock(self.indices, self.origin, self.pitch, active=active,
                          fragments={int(i): self.fragments[int(i)] for i in active
                                     if int(i) in self.fragments},
                          source_labels=self.source_labels,
                          support_links=tuple((a,b,path) for a,b,path in self.support_links
                                              if a in set(active) and b in set(active)))

    def as_mesh(self):
        if self._mesh is not None:
            return self._mesh
        vertices, faces = [], []
        full = np.array([i for i in self.active if int(i) not in self.fragments], dtype=int)
        if len(full):
            vertices.append((self.centres[full, None, :] + self._cube.vertices[None, :, :]).reshape(-1, 3))
            faces.append((np.arange(len(full))[:, None, None] * 8 + self._cube.faces[None, :, :]).reshape(-1, 3))
        count = len(full) * 8
        for points, _ in self.fragments.values():
            hull = ConvexHull(points)
            triangles = hull.simplices.copy()
            cross = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                             points[triangles[:, 2]] - points[triangles[:, 0]])
            reverse = np.einsum("ij,ij->i", cross, hull.equations[:, :3]) < 0
            triangles[reverse] = triangles[reverse, ::-1]
            vertices.append(points)
            faces.append(triangles + count)
            count += len(points)
        self._mesh = trimesh.Trimesh(
            vertices=np.vstack(vertices) if vertices else np.empty((0, 3)),
            faces=np.vstack(faces) if faces else np.empty((0, 3), dtype=int),
            process=False)
        return self._mesh

    @property
    def vertices(self):
        return self.as_mesh().vertices

    def export(self, path):
        return self.as_mesh().export(path)


def _clip(mesh, normal, offset):
    """Closed portion normal.dot(x) >= offset, using existing capped slicing."""
    if isinstance(mesh, VoxelStock):
        return mesh.clip(normal, offset)
    signed = np.asarray(mesh.vertices) @ normal - offset
    if np.all(signed >= 0):
        return mesh.copy()
    if np.all(signed <= 0):
        return trimesh.Trimesh(vertices=[], faces=[], process=False)
    clipped = trimesh.intersections.slice_mesh_plane(
        mesh, normal, normal * offset, cap=True, engine="earcut")
    if len(clipped.faces) and (not clipped.is_watertight or not np.isfinite(clipped.volume)):
        raise ValueError("Physical cut did not produce a closed finite solid.")
    return clipped


def partition_stock(stock, plan, weight, defect_points, cell_weight, kerf_mesh,
                    defect_cell_radius=0.0):
    """Return physical leaf solids and independently measured blade slabs.

    Defect cells are conservative full cells. Verified blades must not intersect
    them; this additional check catches serialization/rounding discrepancies.
    Their mass therefore belongs to exactly one terminal physical piece.
    """
    reference = abs(float(stock.volume))
    if reference <= 0:
        raise ValueError("Physical stock has no positive volume.")
    points = np.asarray(defect_points, dtype=float).reshape((-1, 3))
    pieces = {"rough_piece_1": {"mesh": stock, "defect_points": points,
                               "parent_piece_id": None, "created_by_cut_step": None}}
    cuts = []
    kerf = 0.0
    for cut in plan["sequence"]:
        parent = pieces.pop(cut["parent_piece_id"])
        normal = np.asarray(cut["plane"]["normal"], dtype=float)
        length = float(np.linalg.norm(normal))
        if not math.isfinite(length) or length <= 0:
            raise ValueError("Invalid physical cut plane.")
        normal = normal / length
        offset = float(cut["plane"]["offset"]) / length
        half = kerf_mesh / 2.0
        defects = parent["defect_points"]
        signed = defects @ normal - offset
        if len(defects) and np.any(np.abs(signed) <= half + defect_cell_radius):
            raise ValueError("Physical blade intersects a confirmed defect cell.")
        mesh = parent["mesh"]
        left = _clip(mesh, -normal, -offset + half)
        right = _clip(mesh, normal, offset + half)
        slab = (_clip(_clip(mesh, normal, offset - half), -normal, -offset - half)
                if kerf_mesh > 0 else None)
        parent_ct = weight * (abs(float(mesh.volume)) / reference)
        left_ct = weight * (abs(float(left.volume)) / reference)
        right_ct = weight * (abs(float(right.volume)) / reference)
        kerf_ct = weight * (abs(float(slab.volume)) / reference) if slab is not None else 0.0
        check = mass_balance(parent_ct, retained=left_ct + right_ct, kerf=kerf_ct)
        if not check["mass_balance_valid"]:
            raise ValueError("Physical partition mass balance exceeds tolerance.")
        if left_ct <= 0 or right_ct <= 0:
            raise ValueError("Verified cut produces an empty physical child.")
        ids = cut["result_piece_ids"]
        pieces[ids[0]] = {"mesh": left, "defect_points": defects[signed < -half],
                          "parent_piece_id": cut["parent_piece_id"], "created_by_cut_step": cut["step"]}
        pieces[ids[1]] = {"mesh": right, "defect_points": defects[signed > half],
                          "parent_piece_id": cut["parent_piece_id"], "created_by_cut_step": cut["step"]}
        kerf += kerf_ct
        cuts.append({
            "step": cut["step"], "parent_piece_id": cut["parent_piece_id"],
            "result_piece_ids": ids, "parent_weight_ct": parent_ct,
            "child_weights_ct": [left_ct, right_ct], "actual_kerf_removed_ct": kerf_ct,
            **check,
        })
    for piece_id, piece in pieces.items():
        piece["piece_id"] = piece_id
        piece["weight_ct"] = weight * (abs(float(piece["mesh"].volume)) / reference)
        piece["confirmed_defect_loss_ct"] = len(piece["defect_points"]) * cell_weight
    return pieces, kerf, cuts


def classify_pieces(pieces, minimum_secondary):
    """Keep the largest clean piece regardless of secondary minimum.

    A dirty leaf remains a physical waste piece; subtracting defect cells never
    creates a supposedly extractable cavity. Every excluded gram has a reason.
    """
    clean = [piece for piece in pieces.values() if piece["confirmed_defect_loss_ct"] == 0]
    primary = max(clean, key=lambda p: p["weight_ct"]) if clean else None
    for index, piece in enumerate(pieces.values(), 1):
        if piece["confirmed_defect_loss_ct"] > 0:
            reason = "confirmed_defect_containing_piece"
        elif piece is not primary and piece["weight_ct"] < minimum_secondary:
            reason = "below_minimum_secondary_mass"
        else:
            reason = None
        piece["discard_reason"] = reason
        piece["retained"] = reason is None
        piece["region_id"] = ("R" if reason is None else "W") + str(index)


def evaluate_plan(stock, plan, weight, defect_points, cell_weight, kerf_mesh,
                  minimum_secondary, defect_cell_radius=0.0):
    pieces, kerf, partitions = partition_stock(
        stock, plan, weight, defect_points, cell_weight, kerf_mesh, defect_cell_radius)
    # One job is one physical stock. Reconstruction labels are geometry evidence;
    # only partition_stock's verified cut tree creates physical children.
    natural = []  # Compatibility field: reconstruction never creates partitions.
    parents = {identifier: [identifier] for identifier in pieces}
    if len(pieces) != len(partitions) + 1:
        raise ValueError("Physical leaves must equal selected binary cuts plus one.")
    classify_pieces(pieces, minimum_secondary)
    retained = math.fsum(p["weight_ct"] for p in pieces.values() if p["retained"])
    defects = math.fsum(p["confirmed_defect_loss_ct"] for p in pieces.values())
    discarded = math.fsum(p["weight_ct"] - p["confirmed_defect_loss_ct"]
                          for p in pieces.values() if not p["retained"])
    check = mass_balance(weight, retained=retained, kerf=kerf,
                         defects=defects, discarded=discarded)
    if not check["mass_balance_valid"]:
        raise ValueError("Physical plan mass balance exceeds tolerance.")
    # Rebind the verifier's advisory gem identifiers to physical leaf identifiers.
    # The verified tree and its planes are unchanged.
    result_plan = copy.deepcopy(plan)
    mapping = {}
    def collect(node):
        if node["type"] == "leaf":
            regions = [pieces[k]["region_id"] for k in parents[node["piece_id"]]]
            node["physical_component_ids"] = parents[node["piece_id"]]
            node["physical_piece_id"] = node["piece_id"]
            mapping.update({old: regions for old in node["gem_ids"]})
        else:
            collect(node["left"])
            collect(node["right"])
    collect(result_plan["cut_tree"])
    def remap(value):
        if isinstance(value, dict):
            return {key: remap(child) for key, child in value.items()}
        if isinstance(value, list):
            out = []
            for child in value:
                if isinstance(child,str) and child in mapping:
                    out.extend(mapping[child])
                else:
                    out.append(remap(child))
            return out
        return mapping.get(value, value) if isinstance(value, str) else value
    result_plan = remap(result_plan)
    result_plan["retained_preform_region_ids"] = [p["region_id"] for p in pieces.values() if p["retained"]]
    result_plan["discarded_region_ids"] = [p["region_id"] for p in pieces.values() if not p["retained"]]
    return {
        "pieces": pieces, "plan": result_plan, "retained": retained,
        "kerf": kerf, "defects": defects, "discarded": discarded,
        "partitions": partitions, "natural_components": natural, "balance": check,
    }
