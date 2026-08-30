"""Geometry-verified, straight-through rough separation planning.

The planner intentionally produces operator guidance rather than machine code.
Every accepted cut is a complete plane through the current parent piece, keeps
the kerf slab outside protected gem envelopes and approved defect zones, and
leaves at least one assigned gem on each side.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError


PLAN_VERSION = "straight_full_through_v1"
DEFAULT_TIME_LIMIT_SECONDS = 18.0
MAX_PARENT_POINTS = 30000
MAX_BRANCHES_PER_NODE = 16
MAX_SECTION_CANDIDATES_PER_NODE = 96


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-9 else None


def _canonical(vector):
    vector = _unit(vector)
    if vector is None:
        return None
    for component in vector:
        if abs(component) > 1e-8:
            if component < 0:
                vector = -vector
            break
    return vector


def _dedupe_directions(vectors, tolerance=0.025):
    result = []
    for vector in vectors:
        vector = _canonical(vector)
        if vector is None:
            continue
        if any(abs(float(np.dot(vector, known))) > 1.0 - tolerance
               for known in result):
            continue
        result.append(vector)
    return result


def _fibonacci_directions(count=30):
    vectors = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        angle = golden * index
        vectors.append([math.cos(angle) * radius, y,
                        math.sin(angle) * radius])
    return vectors


def _mesh_vertices(mesh):
    return np.asarray(mesh.vertices, dtype=float)


def _mesh_centres(meshes):
    return np.asarray([mesh.bounds.mean(axis=0) for mesh in meshes], dtype=float)


def _mesh_separating_axes(mesh, face_limit=40, edge_limit=14):
    face_axes = []
    if len(getattr(mesh, "face_normals", [])):
        areas = np.asarray(mesh.area_faces)
        ranked = np.argsort(areas)[::-1]
        face_axes = _dedupe_directions(
            np.asarray(mesh.face_normals)[ranked],
            tolerance=0.012,
        )[:face_limit]

    edge_axes = []
    edges = np.asarray(getattr(mesh, "edges_unique", []), dtype=int)
    if len(edges):
        vectors = _mesh_vertices(mesh)[edges[:, 1]] - _mesh_vertices(mesh)[edges[:, 0]]
        lengths = np.linalg.norm(vectors, axis=1)
        ranked = np.argsort(lengths)[::-1]
        edge_axes = _dedupe_directions(
            vectors[ranked],
            tolerance=0.018,
        )[:edge_limit]
    return face_axes, edge_axes


def _candidate_normals(rough, gem_meshes):
    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    vertices = _mesh_vertices(rough)
    if len(vertices) >= 3:
        covariance = np.cov(vertices.T)
        _, axes = np.linalg.eigh(covariance)
        vectors.extend(axes.T)

    centres = _mesh_centres(gem_meshes)
    for left in range(len(centres)):
        for right in range(left + 1, len(centres)):
            vectors.append(centres[right] - centres[left])

    edge_axes = []
    for mesh in gem_meshes:
        faces, edges = _mesh_separating_axes(mesh)
        vectors.extend(faces)
        edge_axes.append(edges)

    for left in range(len(edge_axes)):
        for right in range(left + 1, len(edge_axes)):
            for left_axis in edge_axes[left]:
                for right_axis in edge_axes[right]:
                    vectors.append(np.cross(left_axis, right_axis))

    vectors.extend(_fibonacci_directions())
    return _dedupe_directions(vectors)


def _base_plan(status, blade_kerf_mm, preform_margin_mm,
               max_cut_depth_mm, mm_per_mesh_unit):
    return {
        "version": PLAN_VERSION,
        "status": status,
        "machine_ready": False,
        "operator_guidance_only": True,
        "machine_ready_reason": (
            "This plan provides conservative operator guidance only; it is "
            "not G-code or CNC output and requires inspection, stable holding, "
            "and operator approval before every cut."
        ),
        "settings": {
            "blade_kerf_mm": blade_kerf_mm,
            "preform_margin_mm": preform_margin_mm,
            "protected_corridor_mm": (
                round(float(blade_kerf_mm + 2.0 * preform_margin_mm), 4)
                if blade_kerf_mm is not None and preform_margin_mm is not None
                else None
            ),
            "max_cut_depth_mm": max_cut_depth_mm,
            "mm_per_mesh_unit": mm_per_mesh_unit,
        },
        "coordinate_frame": {
            "name": "centered_rough_mesh",
            "units": "mesh_units",
            "real_world_scale": "mm_per_mesh_unit",
            "axes": {"x": "right", "y": "up", "z": "forward"},
            "viewer_transform": "shared_backend_to_threejs_group",
        },
        "cut_tree": None,
        "sequence": [],
        "retained_gems": [],
        "resulting_piece_ids": [],
        "warnings": [],
        "rejection_reasons": {},
        "diagnostics": {},
    }


def settings_required_plan(blade_kerf_mm=None, mm_per_mesh_unit=None):
    plan = _base_plan(
        "settings_required",
        blade_kerf_mm,
        None,
        None,
        mm_per_mesh_unit,
    )
    plan["warnings"] = [
        "Enter a positive preform allowance and maximum usable saw depth to "
        "generate a verified multi-gem sequence."
    ]
    return plan


def no_separation_plan(blade_kerf_mm, preform_margin_mm,
                       max_cut_depth_mm, mm_per_mesh_unit, gem_id="gem_1"):
    plan = _base_plan(
        "no_separation_required",
        blade_kerf_mm,
        preform_margin_mm,
        max_cut_depth_mm,
        mm_per_mesh_unit,
    )
    plan["retained_gems"] = [gem_id]
    plan["resulting_piece_ids"] = ["rough_piece_1"]
    plan["cut_tree"] = {
        "piece_id": "rough_piece_1",
        "type": "leaf",
        "gem_ids": [gem_id],
    }
    return plan


def _rough_points(rough, pitch=None):
    if pitch is None or pitch <= 0:
        pitch = float(np.max(rough.extents)) / 90.0
    try:
        points = np.asarray(rough.voxelized(float(pitch)).fill().points)
    except Exception:
        points = _mesh_vertices(rough)
    if len(points) > MAX_PARENT_POINTS:
        stride = max(1, len(points) // MAX_PARENT_POINTS)
        points = points[::stride][:MAX_PARENT_POINTS]
    return points, float(pitch)


def _plane_basis(normal):
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = _unit(np.cross(normal, reference))
    second = _unit(np.cross(normal, first))
    return first, second


def _section_geometry(parent_points, normal, offset, pitch, kerf_mesh):
    signed = parent_points @ normal - offset
    band = max(float(pitch) * 1.25, float(kerf_mesh) * 0.55)
    section = parent_points[np.abs(signed) <= band]
    if len(section) < 3:
        return None

    first, second = _plane_basis(normal)
    plane_origin = normal * offset
    projected = np.column_stack([
        (section - plane_origin) @ first,
        (section - plane_origin) @ second,
    ])
    try:
        hull = ConvexHull(projected)
        contour_2d = projected[hull.vertices]
        area = float(hull.volume)
    except QhullError:
        return None

    best_width = float("inf")
    best_direction = np.array([1.0, 0.0])
    for index in range(len(contour_2d)):
        edge = contour_2d[(index + 1) % len(contour_2d)] - contour_2d[index]
        edge = _unit(edge)
        if edge is None:
            continue
        direction = np.array([-edge[1], edge[0]])
        values = contour_2d @ direction
        width = float(values.max() - values.min())
        if width < best_width:
            best_width = width
            best_direction = direction

    contour_3d = (
        plane_origin
        + contour_2d[:, 0, None] * first
        + contour_2d[:, 1, None] * second
    )
    if len(contour_3d) > 80:
        stride = max(1, len(contour_3d) // 80)
        contour_3d = contour_3d[::stride][:80]
    feed = _unit(best_direction[0] * first + best_direction[1] * second)
    return {
        "contour": contour_3d,
        "area": area,
        "required_depth": best_width,
        "feed_direction": feed,
    }


def _orientation_angles(vector):
    vector = _unit(vector)
    if vector is None:
        return {"azimuth_deg": 0.0, "elevation_deg": 0.0}
    return {
        "azimuth_deg": round(math.degrees(math.atan2(vector[1], vector[0])), 2),
        "elevation_deg": round(
            math.degrees(math.atan2(vector[2], math.hypot(vector[0], vector[1]))),
            2,
        ),
    }


def _candidate_partitions(indices, gem_vertices, normals,
                          preform_mesh, kerf_mesh,
                          preferred_normals=None):
    preferred_normals = [
        normal for normal in (
            _canonical(vector) for vector in (preferred_normals or [])
        ) if normal is not None
    ]
    candidates = []
    for normal_index, normal in enumerate(normals):
        intervals = {}
        for gem_index in indices:
            values = gem_vertices[gem_index] @ normal
            intervals[gem_index] = (
                float(values.min() - preform_mesh),
                float(values.max() + preform_mesh),
            )

        ordered = sorted(indices, key=lambda index: intervals[index][0])
        for split_index in range(1, len(ordered)):
            left = tuple(ordered[:split_index])
            right = tuple(ordered[split_index:])
            left_max = max(intervals[index][1] for index in left)
            right_min = min(intervals[index][0] for index in right)
            clearance = right_min - left_max
            if clearance + 1e-9 < kerf_mesh:
                continue
            offset = (left_max + right_min) * 0.5
            candidates.append({
                "normal": normal,
                "normal_index": normal_index,
                "preferred": any(
                    abs(float(np.dot(normal, preferred))) > 0.999
                    for preferred in preferred_normals
                ),
                "offset": offset,
                "left": left,
                "right": right,
                "envelope_clearance": clearance - kerf_mesh,
                "balance": min(len(left), len(right)) / max(len(indices), 1),
            })
    candidates.sort(
        key=lambda item: (
            item["preferred"],
            item["balance"],
            item["envelope_clearance"],
        ),
        reverse=True,
    )
    seen = set()
    unique = []
    for candidate in candidates:
        partition = frozenset((frozenset(candidate["left"]),
                               frozenset(candidate["right"])))
        key = (partition, candidate["normal_index"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def is_recursively_separable(gem_meshes, blade_kerf_mesh,
                             preform_margin_mesh=0.0,
                             extra_normals=None):
    """Fast envelope-only check used while ranking packing states."""
    if len(gem_meshes) <= 1:
        return True
    centres = _mesh_centres(gem_meshes)
    facet_normals = []
    edge_axes = []
    for mesh in gem_meshes:
        faces, edges = _mesh_separating_axes(mesh, face_limit=32, edge_limit=10)
        facet_normals.extend(faces)
        edge_axes.append(edges)
    edge_cross_axes = []
    for left in range(len(edge_axes)):
        for right in range(left + 1, len(edge_axes)):
            for left_axis in edge_axes[left]:
                for right_axis in edge_axes[right]:
                    edge_cross_axes.append(np.cross(left_axis, right_axis))
    normals = _dedupe_directions(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        + [centres[right] - centres[left]
           for left in range(len(centres))
           for right in range(left + 1, len(centres))]
        + facet_normals
        + edge_cross_axes
        + list(extra_normals or [])
        + _fibonacci_directions(18)
    )
    vertices = [_mesh_vertices(mesh) for mesh in gem_meshes]
    memo = {}

    def recurse(indices):
        key = tuple(sorted(indices))
        if len(key) <= 1:
            return True
        if key in memo:
            return memo[key]
        for candidate in _candidate_partitions(
                key, vertices, normals, preform_margin_mesh,
                blade_kerf_mesh)[:12]:
            if recurse(candidate["left"]) and recurse(candidate["right"]):
                memo[key] = True
                return True
        memo[key] = False
        return False

    return recurse(tuple(range(len(gem_meshes))))


@dataclass
class _SearchContext:
    gem_vertices: list
    gem_values: list
    gem_ids: list
    normals: list
    preferred_normals: list
    defect_points: np.ndarray
    defect_radius: float
    kerf_mesh: float
    preform_mesh: float
    max_depth_mesh: float
    pitch: float
    deadline: float
    rejected: dict
    nodes: int = 0


def _reject(context, reason):
    context.rejected[reason] = context.rejected.get(reason, 0) + 1


def _search_tree(indices, parent_points, context):
    if time.monotonic() >= context.deadline:
        raise TimeoutError
    context.nodes += 1
    if len(indices) == 1:
        return {
            "type": "leaf",
            "indices": tuple(indices),
            "gem_ids": [context.gem_ids[indices[0]]],
            "points": parent_points,
        }

    candidates = _candidate_partitions(
        indices,
        context.gem_vertices,
        context.normals,
        context.preform_mesh,
        context.kerf_mesh,
        context.preferred_normals,
    )
    if not candidates:
        _reject(context, "no_envelope_corridor")
        return None

    ranked = []
    for candidate in candidates[:MAX_SECTION_CANDIDATES_PER_NODE]:
        if time.monotonic() >= context.deadline:
            raise TimeoutError
        normal = candidate["normal"]
        offset = candidate["offset"]
        signed = parent_points @ normal - offset
        half_kerf = context.kerf_mesh * 0.5
        left_points = parent_points[signed < -half_kerf]
        right_points = parent_points[signed > half_kerf]
        if len(left_points) < 4 or len(right_points) < 4:
            _reject(context, "invalid_parent_split")
            continue

        if len(context.defect_points):
            defect_distance = np.abs(context.defect_points @ normal - offset)
            if np.any(defect_distance <= half_kerf + context.defect_radius):
                _reject(context, "approved_defect_safety_zone")
                continue

        section = _section_geometry(
            parent_points,
            normal,
            offset,
            context.pitch,
            context.kerf_mesh,
        )
        if section is None:
            _reject(context, "section_not_resolved")
            continue
        if section["required_depth"] > context.max_depth_mesh + 1e-9:
            _reject(context, "maximum_cut_depth_exceeded")
            continue

        score = (
            candidate["preferred"],
            candidate["balance"],
            candidate["envelope_clearance"],
            -section["required_depth"],
            -section["area"],
        )
        ranked.append((score, candidate, section, left_points, right_points))

    ranked.sort(key=lambda item: item[0], reverse=True)
    for _, candidate, section, left_points, right_points in ranked[:MAX_BRANCHES_PER_NODE]:
        left_value = sum(context.gem_values[index]
                         for index in candidate["left"])
        right_value = sum(context.gem_values[index]
                          for index in candidate["right"])
        branches = [
            (candidate["left"], left_points, "left"),
            (candidate["right"], right_points, "right"),
        ]
        if right_value > left_value:
            branches.reverse()

        built = {}
        failed = False
        for child_indices, child_points, side in branches:
            child = _search_tree(child_indices, child_points, context)
            if child is None:
                failed = True
                break
            built[side] = child
        if failed:
            continue

        return {
            "type": "cut",
            "indices": tuple(indices),
            "gem_ids": [context.gem_ids[index] for index in indices],
            "normal": candidate["normal"],
            "offset": candidate["offset"],
            "section": section,
            "envelope_clearance": candidate["envelope_clearance"],
            "left": built["left"],
            "right": built["right"],
            "points": parent_points,
        }

    _reject(context, "recursive_child_not_cuttable")
    return None


def _round_vector(vector, digits=6):
    return [round(float(value), digits) for value in vector]


def _serialize_tree_and_sequence(root, context, mm_per_mesh_unit):
    sequence = []
    leaves = []
    piece_counter = [1]

    def next_piece_id():
        piece_counter[0] += 1
        return f"rough_piece_{piece_counter[0]}"

    def visit(node, piece_id):
        if node["type"] == "leaf":
            leaves.append(piece_id)
            return {
                "piece_id": piece_id,
                "type": "leaf",
                "gem_ids": node["gem_ids"],
            }

        left_id = next_piece_id()
        right_id = next_piece_id()
        normal = node["normal"]
        origin = normal * node["offset"]
        section = node["section"]
        step_number = len(sequence) + 1
        left_values = sum(context.gem_values[index]
                          for index in node["left"]["indices"])
        right_values = sum(context.gem_values[index]
                           for index in node["right"]["indices"])
        process_first = left_id if left_values >= right_values else right_id
        contour = [_round_vector(point) for point in section["contour"]]
        step = {
            "step": step_number,
            "operation": "straight_full_through_saw_cut",
            "parent_piece_id": piece_id,
            "result_piece_ids": [left_id, right_id],
            "process_next_piece_id": process_first,
            "retained_gems": node["gem_ids"],
            "negative_side_gems": node["left"]["gem_ids"],
            "positive_side_gems": node["right"]["gem_ids"],
            "plane": {
                "origin": _round_vector(origin),
                "normal": _round_vector(normal),
                "offset": round(float(node["offset"]), 6),
                "orientation": _orientation_angles(normal),
            },
            "feed_direction": _round_vector(section["feed_direction"]),
            "feed_orientation": _orientation_angles(section["feed_direction"]),
            "kerf_slab": {
                "thickness_mesh_units": round(context.kerf_mesh, 6),
                "thickness_mm": round(context.kerf_mesh * mm_per_mesh_unit, 4),
            },
            "section_contour": contour,
            "required_depth_mesh_units": round(section["required_depth"], 6),
            "required_depth_mm": round(
                section["required_depth"] * mm_per_mesh_unit, 3),
            "estimated_saw_area_mesh_units2": round(section["area"], 6),
            "estimated_saw_area_mm2": round(
                section["area"] * mm_per_mesh_unit ** 2, 3),
            "estimated_kerf_loss_mesh_units3": round(
                section["area"] * context.kerf_mesh, 6),
            "estimated_kerf_loss_mm3": round(
                section["area"] * context.kerf_mesh
                * mm_per_mesh_unit ** 3,
                3,
            ),
            "minimum_envelope_clearance_mesh_units": round(
                node["envelope_clearance"], 6),
            "minimum_envelope_clearance_mm": round(
                node["envelope_clearance"] * mm_per_mesh_unit, 3),
            "operator_checks": [
                "Confirm the displayed parent piece and orientation.",
                "Confirm stable, flat support and secure clamping.",
                "Inspect the proposed kerf corridor for visible defects.",
                "Stop if the stone shifts, chips unexpectedly, or contacts the blade obliquely.",
            ],
        }
        sequence.append(step)

        first_key = "left" if process_first == left_id else "right"
        second_key = "right" if first_key == "left" else "left"
        ids = {"left": left_id, "right": right_id}
        serialized = {}
        for key in (first_key, second_key):
            serialized[key] = visit(node[key], ids[key])
        return {
            "piece_id": piece_id,
            "type": "cut",
            "step": step_number,
            "gem_ids": node["gem_ids"],
            "left": serialized["left"],
            "right": serialized["right"],
        }

    tree = visit(root, "rough_piece_1")
    return tree, sequence, leaves


def plan_cut_sequence(rough, gem_meshes, *, blade_kerf_mm,
                      preform_margin_mm, max_cut_depth_mm,
                      mm_per_mesh_unit, gem_values=None, gem_ids=None,
                      defect_points=None, defect_radius_mesh=0.0,
                      pitch=None, time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
                      preferred_normals=None):
    started = time.monotonic()
    plan = _base_plan(
        "planning",
        blade_kerf_mm,
        preform_margin_mm,
        max_cut_depth_mm,
        mm_per_mesh_unit,
    )
    if (preform_margin_mm is None or max_cut_depth_mm is None
            or mm_per_mesh_unit is None or mm_per_mesh_unit <= 0):
        return settings_required_plan(blade_kerf_mm, mm_per_mesh_unit)
    if len(gem_meshes) <= 1:
        gem_id = (gem_ids or ["gem_1"])[0]
        return no_separation_plan(
            blade_kerf_mm,
            preform_margin_mm,
            max_cut_depth_mm,
            mm_per_mesh_unit,
            gem_id,
        )

    gem_ids = list(gem_ids or [f"gem_{index + 1}"
                               for index in range(len(gem_meshes))])
    gem_values = list(gem_values or [float(mesh.volume) for mesh in gem_meshes])
    rough_points, pitch = _rough_points(rough, pitch)
    kerf_mesh = float(blade_kerf_mm) / float(mm_per_mesh_unit)
    preform_mesh = float(preform_margin_mm) / float(mm_per_mesh_unit)
    max_depth_mesh = float(max_cut_depth_mm) / float(mm_per_mesh_unit)
    rejected = {}
    context = _SearchContext(
        gem_vertices=[_mesh_vertices(mesh) for mesh in gem_meshes],
        gem_values=gem_values,
        gem_ids=gem_ids,
        normals=_candidate_normals(rough, gem_meshes),
        preferred_normals=list(preferred_normals or []),
        defect_points=np.asarray(defect_points if defect_points is not None else [],
                                 dtype=float).reshape((-1, 3)),
        defect_radius=float(max(0.0, defect_radius_mesh)),
        kerf_mesh=kerf_mesh,
        preform_mesh=preform_mesh,
        max_depth_mesh=max_depth_mesh,
        pitch=float(pitch),
        deadline=started + max(0.05, float(time_limit_seconds)),
        rejected=rejected,
    )

    timed_out = False
    try:
        root = _search_tree(tuple(range(len(gem_meshes))), rough_points, context)
    except TimeoutError:
        root = None
        timed_out = True

    elapsed = time.monotonic() - started
    plan["rejection_reasons"] = rejected
    plan["diagnostics"] = {
        "runtime_seconds": round(elapsed, 3),
        "time_limit_seconds": float(time_limit_seconds),
        "search_nodes": int(context.nodes),
        "candidate_normal_count": len(context.normals),
        "rough_section_sample_count": len(rough_points),
        "approved_defect_point_count": len(context.defect_points),
        "exact_sequence_verified": root is not None,
    }
    if root is None:
        plan["status"] = "timeout" if timed_out else "not_cuttable"
        plan["warnings"] = [
            "No complete straight, full-through separation sequence was verified "
            "for this gem arrangement and the entered machine limits."
        ]
        return plan

    tree, sequence, leaves = _serialize_tree_and_sequence(
        root,
        context,
        float(mm_per_mesh_unit),
    )
    plan["status"] = "complete"
    plan["cut_tree"] = tree
    plan["sequence"] = sequence
    plan["retained_gems"] = gem_ids
    plan["resulting_piece_ids"] = leaves
    plan["gem_count"] = len(gem_ids)
    plan["minimum_envelope_clearance_mm"] = round(min(
        step["minimum_envelope_clearance_mm"] for step in sequence
    ), 3)
    plan["maximum_required_depth_mm"] = round(max(
        step["required_depth_mm"] for step in sequence
    ), 3)
    plan["estimated_total_saw_area_mm2"] = round(sum(
        step["estimated_saw_area_mm2"] for step in sequence
    ), 3)
    plan["estimated_total_kerf_loss_mm3"] = round(sum(
        step["estimated_kerf_loss_mm3"] for step in sequence
    ), 3)
    plan["reorientation_count"] = max(0, len(sequence) - 1)
    plan["warnings"] = [
        "Verify fixture stability and the visible defect condition before each cut.",
        "The sequence has not been validated by a physical workshop trial.",
    ]
    return plan
