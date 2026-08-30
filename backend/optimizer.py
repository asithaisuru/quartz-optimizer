"""
Voxel occupancy gem-packing optimiser.

The public entrypoint remains optimize_cut(...). Internally this builds a
rough-stone SDF/voxel grid, generates feasible gem placements, combines them
with beam search, then runs a small local refinement pass so selected gems can
grow into remaining free space.
"""

from dataclasses import dataclass
import math
import os
import time

import numpy as np
import trimesh
from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay, QhullError, cKDTree

from gem_shapes import get_standard_shapes
from defect_policy import approved_no_cut_cloud_path
from cut_sequence import (
    is_recursively_separable,
    no_separation_plan,
    plan_cut_sequence,
    settings_required_plan,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAFETY_FACTOR = 0.985
DEFAULT_BLADE_KERF_MM = 0.5
DEFAULT_ROUGH_CLEARANCE_MM = 0.8
DEFAULT_EXTRA_GEM_POLICY = "saleable"
DEFAULT_MIN_SECONDARY_CARAT = 0.5
MAX_GEMS = 12
MIN_GEM_VOL_RATIO = 0.01

SDF_RESOLUTION = 128
SDF_BIAS_VOXELS = 0.45
MIN_WALL_RATIO = 0.003
BLADE_KERF_RATIO = 0.018
BLADE_KERF_VOXELS = 2.0
FRACTURE_SAFE_RATIO = 0.035

MAX_SURFACE_SAMPLES = 900
MAX_OCCUPANCY_SAMPLES = 7000
OCCUPANCY_STEP = 0.07
TOP_PER_VARIANT = 5
MAX_CANDIDATES = 330
BEAM_WIDTH = 9
REFINE_PASSES = 2
STRICT_SHRINK_FACTORS = (1.0, 0.97, 0.94, 0.90, 0.86)
CANDIDATE_SCALE_FACTORS = (1.0, 0.84, 0.68, 0.54, 0.42)
CANDIDATE_SCALE_SOURCE_LIMIT = 100
# Free-space filling below keeps searching until a pass/round genuinely
# finds nothing left to place (or max_gems / a time budget is hit) rather
# than stopping after an arbitrary fixed pass count — a fixed count used to
# leave placeable, saleable pockets unfilled even when plenty of search
# budget and max_gems headroom remained.
GAP_FILL_TIME_LIMIT_SECONDS = 18.0
GAP_FILL_SAFETY_ITERATION_CEILING = 500
GAP_FILL_MIN_VOL_RATIO = 0.0010
GAP_FILL_VARIANT_LIMIT = 18
GAP_FILL_POINT_LIMIT = 26
BLADE_FINISH_SHRINK = 0.97
POCKET_FILL_COMPONENT_LIMIT = 10
POCKET_FILL_POINTS_PER_COMPONENT = 3
POCKET_FILL_VARIANT_LIMIT = 20
POCKET_FILL_TIME_LIMIT_SECONDS = 18.0
POCKET_FILL_SAFETY_ITERATION_CEILING = 500
CUTTABLE_FRONTIER_LIMIT = 8
CUTTABLE_POCKET_COMPONENT_LIMIT = 6
CUTTABLE_POCKET_POINTS_PER_COMPONENT = 8
CUTTABLE_POCKET_CANDIDATE_LIMIT = 24
CUTTABLE_POCKET_SAFETY_ROUND_CEILING = 500
CUTTABLE_CANDIDATE_EXPANSION_RESERVE_SECONDS = 2.0
CUTTABLE_DEADLINE_RESERVE_FRACTION = 0.25
CUTTABLE_GLOBAL_DEADLINE_SECONDS = 60.0
CUTTABLE_CAVITY_DEADLINE_SECONDS = 98.0
# Leave two seconds for serialization, diagnostics, and caller overhead.
CUTTABLE_TOTAL_DEADLINE_SECONDS = 118.0
POCKET_FILL_SHAPES = (
    "Round Brilliant Cut",
    "Princess Cut",
    "Cushion Cut",
    "Pear Cut",
    "Oval Brilliant",
)


@dataclass
class FitContext:
    grid: np.ndarray
    origin: np.ndarray
    pitch: float
    rough_mask: np.ndarray
    no_cut_mask: np.ndarray
    fit: dict


@dataclass
class TemplateVariant:
    key: str
    name: str
    tmpl: trimesh.Trimesh
    rot: np.ndarray
    verts: np.ndarray
    surface: np.ndarray
    occupancy: np.ndarray
    volume: float


@dataclass
class Placement:
    variant: TemplateVariant
    pos: np.ndarray
    scale: float
    volume: float
    occ_flat: np.ndarray
    occ_set: set
    collision_set: set
    surface_clearance: float

    @property
    def name(self):
        return self.variant.name


@dataclass
class BeamState:
    placements: list
    occupied: set
    volume: float


# ---------------------------------------------------------------------------
# Mesh/grid helpers
# ---------------------------------------------------------------------------

def _as_mesh(obj):
    if isinstance(obj, trimesh.Scene):
        if not obj.geometry:
            return None
        return trimesh.util.concatenate(tuple(obj.geometry.values()))
    return obj


def _transform_points(points, matrix):
    pts = np.c_[points, np.ones(len(points))]
    return (pts @ matrix.T)[:, :3]


def _normalize(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _build_sdf_grid(rough, res=SDF_RESOLUTION, bias_vox=SDF_BIAS_VOXELS, pad=3):
    pitch = float(np.max(rough.extents)) / res
    vox = rough.voxelized(pitch).fill()
    mat = np.pad(vox.matrix.astype(bool), pad)
    grid = (((ndimage.distance_transform_edt(mat)
              - ndimage.distance_transform_edt(~mat)) - bias_vox)
            * pitch).astype(np.float32)
    origin = vox.transform[:3, 3] - pad * pitch
    return grid, origin, pitch


def _fit_settings(extents, sdf_pitch, blade_kerf_mm=DEFAULT_BLADE_KERF_MM,
                  rough_clearance_mm=DEFAULT_ROUGH_CLEARANCE_MM,
                  mm_per_mesh_unit=None, min_secondary_volume=None,
                  min_secondary_carat=None, carats_per_mesh_volume=None,
                  extra_gem_policy=DEFAULT_EXTRA_GEM_POLICY,
                  max_gems=MAX_GEMS, preform_margin_mm=None,
                  max_cut_depth_mm=None):
    min_dim = float(np.min(extents))
    pitch = float(sdf_pitch)
    min_wall = max(min_dim * MIN_WALL_RATIO, pitch * 0.25)
    blade_kerf_mesh = None
    rough_clearance_mesh = None
    if mm_per_mesh_unit and mm_per_mesh_unit > 0:
        blade_kerf_mesh = float(blade_kerf_mm) / float(mm_per_mesh_unit)
        rough_clearance_mesh = (
            float(rough_clearance_mm) / float(mm_per_mesh_unit)
        )
    preform_margin_mesh = None
    if (preform_margin_mm is not None and mm_per_mesh_unit
            and mm_per_mesh_unit > 0):
        preform_margin_mesh = float(preform_margin_mm) / float(mm_per_mesh_unit)
    spacing = blade_kerf_mesh or max(min_dim * BLADE_KERF_RATIO,
                                     pitch * BLADE_KERF_VOXELS)
    if preform_margin_mesh is not None:
        spacing += 2.0 * preform_margin_mesh
    spacing = max(float(spacing), pitch * 0.75)
    collision_radius = spacing * 0.5
    collision_radius_voxels = max(
        1,
        int(math.ceil(collision_radius / max(pitch, 1e-9))),
    )
    voxelized_corridor = 2.0 * collision_radius_voxels * pitch
    protected_corridor_mm = (
        float(spacing * mm_per_mesh_unit)
        if mm_per_mesh_unit and mm_per_mesh_unit > 0 else None
    )
    return {
        "safety_factor": SAFETY_FACTOR,
        "min_wall_dist": min_wall,
        "rough_surface_clearance": max(
            min_wall,
            float(rough_clearance_mesh or 0.0),
            float(preform_margin_mesh or 0.0),
        ),
        "gem_spacing_buffer": spacing,
        "protected_corridor_mesh_units": spacing,
        "protected_corridor_mm": protected_corridor_mm,
        "collision_radius_mesh_units": collision_radius,
        "collision_radius_voxels": collision_radius_voxels,
        "voxelized_protected_corridor_mesh_units": voxelized_corridor,
        "voxelized_protected_corridor_mm": (
            float(voxelized_corridor * mm_per_mesh_unit)
            if mm_per_mesh_unit and mm_per_mesh_unit > 0 else None
        ),
        "fracture_safe_dist": max(min_dim * FRACTURE_SAFE_RATIO, pitch * 2.0),
        "surface_tolerance": pitch * 0.35,
        "voxel_tolerance": pitch * 0.20,
        "blade_kerf_mm": float(blade_kerf_mm),
        "rough_clearance_mm": float(rough_clearance_mm),
        "preform_margin_mm": (
            float(preform_margin_mm) if preform_margin_mm is not None else None
        ),
        "preform_margin_mesh_units": (
            float(preform_margin_mesh) if preform_margin_mesh is not None else None
        ),
        "max_cut_depth_mm": (
            float(max_cut_depth_mm) if max_cut_depth_mm is not None else None
        ),
        "mm_per_mesh_unit": float(mm_per_mesh_unit or 0.0),
        "min_secondary_volume": float(min_secondary_volume or 0.0),
        "min_secondary_carat": (
            float(min_secondary_carat) if min_secondary_carat is not None else None
        ),
        "carats_per_mesh_volume": float(carats_per_mesh_volume or 0.0),
        "extra_gem_policy": extra_gem_policy,
        "max_gems": int(max_gems),
    }


def _diagnostic_value(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    return value


def _make_no_cut_mask(grid_shape, origin, pitch, fpts, radius):
    mask = np.zeros(grid_shape, dtype=bool)
    if fpts is None or len(fpts) == 0:
        return mask

    idx = np.rint((fpts - origin) / pitch).astype(int)
    valid = np.all((idx >= 0) & (idx < np.array(grid_shape)), axis=1)
    idx = idx[valid]
    if len(idx) == 0:
        return mask

    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    iterations = max(1, int(math.ceil(radius / max(pitch, 1e-9))))
    return ndimage.binary_dilation(mask, iterations=iterations)


def _load_fracture_points(job_folder):
    if not job_folder:
        return np.empty((0, 3))
    path, reason = approved_no_cut_cloud_path(job_folder)
    if path is None:
        print(f"   No optimizer-approved no-cut defect cloud ({reason}).")
        return np.empty((0, 3))
    try:
        obj = trimesh.load(str(path))
        pts = np.asarray(obj.vertices)
        print(f"   Loaded {len(pts)} policy-approved no-cut point(s).")
        return pts
    except Exception as e:
        print(f"   Could not load defect points: {e}")
        return np.empty((0, 3))


def _sdf_values(ctx, points):
    idx = ((points - ctx.origin) / ctx.pitch).T
    return map_coordinates(ctx.grid, idx, order=1, mode="constant", cval=-1e6)


def _points_to_flat(points, ctx):
    idx = np.rint((points - ctx.origin) / ctx.pitch).astype(np.int32)
    shape = np.array(ctx.grid.shape, dtype=np.int32)
    if np.any(idx < 0) or np.any(idx >= shape):
        return None
    flat = np.ravel_multi_index(idx.T, ctx.grid.shape)
    return np.unique(flat)


def _flat_mask_values(mask, flat):
    return mask.ravel()[flat]


def _mesh_for_candidate(variant, pos, scale):
    mesh = variant.tmpl.copy()
    mesh.apply_transform(variant.rot)
    mesh.vertices -= mesh.bounds.mean(0)
    mesh.apply_scale(scale)
    mesh.vertices += pos
    return mesh


def _inflate_flat(flat, shape, iterations):
    if iterations <= 0:
        return set(flat.tolist())
    mask = np.zeros(shape, dtype=bool)
    mask.ravel()[flat] = True
    struct = ndimage.generate_binary_structure(3, 1)
    inflated = ndimage.binary_dilation(mask, structure=struct,
                                       iterations=iterations)
    return set(np.flatnonzero(inflated.ravel()).tolist())


def _spacing_voxels(ctx):
    configured = ctx.fit.get("collision_radius_voxels")
    if configured is not None:
        return max(1, int(configured))
    spacing = float(ctx.fit.get("gem_spacing_buffer", ctx.pitch)) * 0.5
    return max(1, int(math.ceil(spacing / max(ctx.pitch, 1e-9))))


def _mask_from_flat(shape, flat_values):
    mask = np.zeros(shape, dtype=bool)
    if flat_values:
        mask.ravel()[list(flat_values)] = True
    return mask


def _free_mask(ctx, occupied):
    allowed = ctx.rough_mask & ~ctx.no_cut_mask
    if not occupied:
        return allowed
    return allowed & ~_mask_from_flat(ctx.grid.shape, occupied)


def _component_summary(ctx, occupied, limit=8):
    free = _free_mask(ctx, occupied)
    labels, count = ndimage.label(free)
    if count == 0:
        return {
            "count": 0,
            "largest_volume_mesh_units": 0.0,
            "top_components": [],
        }, labels

    objects = ndimage.find_objects(labels)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    clearance = ndimage.distance_transform_edt(free) * ctx.pitch
    ranked = [int(i) for i in np.argsort(sizes[1:])[::-1] + 1
              if sizes[i] > 0]
    top = []
    for label in ranked[:limit]:
        slc = objects[label - 1]
        if slc is None:
            continue
        ext = [float((s.stop - s.start) * ctx.pitch) for s in slc]
        component_flat = np.flatnonzero(labels.ravel() == label)
        component_clearance = clearance.ravel()[component_flat]
        peak_flat = int(component_flat[int(np.argmax(component_clearance))])
        peak_idx = np.asarray(np.unravel_index(peak_flat, ctx.grid.shape))
        peak = ctx.origin + peak_idx * ctx.pitch
        bbox_min = ctx.origin + np.asarray([s.start for s in slc]) * ctx.pitch
        bbox_max = ctx.origin + np.asarray([s.stop for s in slc]) * ctx.pitch
        radius = float(np.max(component_clearance))
        carats_per_volume = float(ctx.fit.get("carats_per_mesh_volume", 0.0) or 0.0)
        sphere_carat = (
            (4.0 / 3.0) * math.pi * (radius ** 3) * carats_per_volume
            if carats_per_volume > 0 else None
        )
        top.append({
            "volume_mesh_units": round(float(sizes[label] * ctx.pitch ** 3), 6),
            "voxel_count": int(sizes[label]),
            "bbox_extent_mesh_units": [round(v, 5) for v in ext],
            "bbox_min_mesh_units": [round(float(v), 5) for v in bbox_min],
            "bbox_max_mesh_units": [round(float(v), 5) for v in bbox_max],
            "centre_mesh_units": [
                round(float(v), 5) for v in ((bbox_min + bbox_max) * 0.5)
            ],
            "clearance_peak_mesh_units": [round(float(v), 5) for v in peak],
            "max_inscribed_radius_mesh_units": round(radius, 6),
            "max_inscribed_diameter_mesh_units": round(radius * 2.0, 6),
            "estimated_inscribed_sphere_carat": (
                round(float(sphere_carat), 3) if sphere_carat is not None else None
            ),
        })

    largest = top[0]["volume_mesh_units"] if top else 0.0
    return {
        "count": int(count),
        "largest_volume_mesh_units": largest,
        "top_components": top,
    }, labels


def _placement_collision_set(placement):
    return getattr(placement, "collision_set", None) or placement.occ_set


def _surface_samples(mesh):
    verts = np.asarray(mesh.vertices)
    tris = verts[np.asarray(mesh.faces)]
    centers = tris.mean(axis=1)
    edge_mid = np.concatenate([
        (tris[:, 0] + tris[:, 1]) * 0.5,
        (tris[:, 1] + tris[:, 2]) * 0.5,
        (tris[:, 2] + tris[:, 0]) * 0.5,
    ], axis=0)
    pts = np.vstack([verts, centers, edge_mid])
    if len(pts) > MAX_SURFACE_SAMPLES:
        stride = max(1, len(pts) // MAX_SURFACE_SAMPLES)
        pts = pts[::stride][:MAX_SURFACE_SAMPLES]
    return pts


def _gap_surface_points(mesh, limit=2400):
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(verts) == 0 or len(faces) == 0:
        return verts
    tris = verts[faces]
    centers = tris.mean(axis=1)
    edge_mid = np.concatenate([
        (tris[:, 0] + tris[:, 1]) * 0.5,
        (tris[:, 1] + tris[:, 2]) * 0.5,
        (tris[:, 2] + tris[:, 0]) * 0.5,
    ], axis=0)
    pts = np.vstack([verts, centers, edge_mid])
    if len(pts) > limit:
        stride = max(1, len(pts) // limit)
        pts = pts[::stride][:limit]
    return pts


def _occupancy_samples(mesh):
    verts = np.asarray(mesh.vertices)
    try:
        hull = Delaunay(verts)
    except QhullError:
        return verts

    mn, mx = verts.min(0), verts.max(0)
    step = max(float(np.max(mx - mn)) * OCCUPANCY_STEP, 0.035)
    axes = [np.arange(mn[i], mx[i] + step * 0.5, step) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = hull.find_simplex(grid) >= 0
    pts = grid[inside]
    pts = np.vstack([pts, verts, [[0.0, 0.0, 0.0]]])
    if len(pts) > MAX_OCCUPANCY_SAMPLES:
        stride = max(1, len(pts) // MAX_OCCUPANCY_SAMPLES)
        pts = pts[::stride][:MAX_OCCUPANCY_SAMPLES]
    return pts


def _make_rotations(n_rz=4, n_tilt=2):
    tilt_vals = np.linspace(0, np.pi / 2, n_tilt)
    rz_vals = np.linspace(0, np.pi, n_rz, endpoint=False)
    rots = []
    for rx in tilt_vals:
        for ry in tilt_vals:
            for rz in rz_vals:
                rots.append(trimesh.transformations.euler_matrix(rx, ry, rz))
    return rots


def _make_variants(shapes):
    variants = []
    rotations = _make_rotations()
    for name, tmpl in shapes.items():
        base = tmpl.copy()
        base.vertices -= base.bounds.mean(0)
        base_surface = _surface_samples(base)
        base_occ = _occupancy_samples(base)
        base_verts = np.asarray(base.vertices)
        for ridx, rot in enumerate(rotations):
            verts = _transform_points(base_verts, rot)
            center = (verts.min(0) + verts.max(0)) * 0.5
            verts -= center
            surface = _transform_points(base_surface, rot) - center
            occ = _transform_points(base_occ, rot) - center
            variants.append(TemplateVariant(
                key=f"{name}:{ridx}",
                name=name,
                tmpl=base,
                rot=rot,
                verts=verts,
                surface=surface,
                occupancy=occ,
                volume=float(base.volume),
            ))
    return variants


def _make_search_points(center, extents, ctx):
    half = extents / 2
    seen, pts = set(), []

    def add(pos):
        key = tuple(np.round(pos, 4))
        if key not in seen:
            seen.add(key)
            pts.append(pos.copy())

    for fx in np.linspace(-0.70, 0.70, 4):
        for fy in np.linspace(-0.70, 0.70, 4):
            for fz in np.linspace(-0.70, 0.70, 4):
                add(center + np.array([fx, fy, fz]) * half)

    for axis in range(3):
        for frac in np.linspace(-0.90, 0.90, 7):
            pos = center.copy()
            pos[axis] += frac * half[axis]
            add(pos)

    allowed = ctx.rough_mask & ~ctx.no_cut_mask
    inside_idx = np.argwhere(allowed)
    if len(inside_idx):
        stride = max(1, len(inside_idx) // 35)
        for idx in inside_idx[::stride][:35]:
            add(ctx.origin + idx * ctx.pitch)

    rough_min, rough_max = center - half, center + half
    safe = ctx.fit["min_wall_dist"] + ctx.fit["gem_spacing_buffer"]
    bounded = []
    for p in pts:
        if np.all(p >= rough_min + safe) and np.all(p <= rough_max - safe):
            bounded.append(p)
    return bounded


def _candidate_occ(variant, pos, scale, ctx):
    pts = variant.occupancy * scale + pos
    return _points_to_flat(pts, ctx)


def _candidate_dense_occ(variant, pos, scale, ctx):
    surface = variant.surface * scale + pos
    verts = variant.verts * scale + pos
    interior = variant.occupancy * scale + pos
    pts = np.vstack([interior, surface, verts])

    return _points_to_flat(pts, ctx)


def _stricten_placement(placement, ctx, occupied=None):
    flat = _candidate_dense_occ(
        placement.variant,
        placement.pos,
        placement.scale,
        ctx,
    )
    if flat is None or len(flat) == 0:
        return None, "dense_bounds"
    if np.any(~_flat_mask_values(ctx.rough_mask, flat)):
        return None, "dense_rough_voxel"
    if np.any(_flat_mask_values(ctx.no_cut_mask, flat)):
        return None, "dense_defect_zone"

    spacing_voxels = _spacing_voxels(ctx)
    collision_set = _inflate_flat(flat, ctx.grid.shape, spacing_voxels)
    if occupied and not collision_set.isdisjoint(occupied):
        return None, "overlap"

    return Placement(
        variant=placement.variant,
        pos=placement.pos.copy(),
        scale=float(placement.scale),
        volume=float(placement.volume),
        occ_flat=flat,
        occ_set=set(flat.tolist()),
        collision_set=collision_set,
        surface_clearance=float(placement.surface_clearance),
    ), "ok"


def _candidate_fit(variant, pos, scale, ctx, occupied=None):
    surface = variant.surface * scale + pos
    sdf_min = float(np.min(_sdf_values(ctx, surface)))
    if sdf_min < -ctx.fit["surface_tolerance"]:
        return None, "surface_outside", sdf_min
    if sdf_min < ctx.fit.get("rough_surface_clearance", 0.0):
        return None, "surface_margin", sdf_min

    flat = _candidate_occ(variant, pos, scale, ctx)
    if flat is None or len(flat) == 0:
        return None, "bounds", sdf_min
    if np.any(~_flat_mask_values(ctx.rough_mask, flat)):
        return None, "rough_voxel", sdf_min
    if np.any(_flat_mask_values(ctx.no_cut_mask, flat)):
        return None, "defect_zone", sdf_min
    if occupied and not occupied.isdisjoint(set(flat.tolist())):
        return None, "overlap", sdf_min
    return flat, "ok", sdf_min


def _max_scale_for(variant, pos, rough_bounds, max_dim, ctx, occupied=None,
                   start_scale=0.0, limit_scale=None, strict=False):
    hi = limit_scale or max_dim
    lo = max(0.0, start_scale)
    best_flat = None
    best_sdf = -1e6
    best = 0.0

    # If start_scale is already valid, keep it as the lower bound.
    if lo > 0:
        flat, reason, sdf_min = _candidate_fit(variant, pos, lo, ctx, occupied)
        if flat is not None:
            best, best_flat, best_sdf = lo, flat, sdf_min
        else:
            lo = 0.0

    # Bound the search by the rough AABB to avoid wasting iterations.
    vext = np.maximum(np.ptp(variant.verts, axis=0), 1e-6)
    local_hi = np.min((rough_bounds[1] - rough_bounds[0]) / vext) * 1.15
    hi = min(hi, float(local_hi))

    for _ in range(12):
        mid = (lo + hi) * 0.5
        verts = variant.verts * mid + pos
        if np.any(verts.min(0) < rough_bounds[0]) or np.any(verts.max(0) > rough_bounds[1]):
            hi = mid
            continue
        flat, reason, sdf_min = _candidate_fit(variant, pos, mid, ctx, occupied)
        if flat is not None:
            best, best_flat, best_sdf = mid, flat, sdf_min
            lo = mid
        else:
            hi = mid

    if best <= 0 or best_flat is None:
        return None, "no_fit"

    last_reason = "no_fit"
    for shrink in (STRICT_SHRINK_FACTORS if strict else (1.0,)):
        final_scale = best * ctx.fit["safety_factor"] * shrink
        if final_scale <= 0:
            continue
        flat, reason, sdf_min = _candidate_fit(
            variant, pos, final_scale, ctx, occupied
        )
        if flat is None:
            last_reason = reason
            continue
        volume = float(variant.volume * (final_scale ** 3))
        placement = Placement(
            variant=variant,
            pos=pos.copy(),
            scale=final_scale,
            volume=volume,
            occ_flat=flat,
            occ_set=set(flat.tolist()),
            collision_set=set(flat.tolist()),
            surface_clearance=sdf_min,
        )
        if not strict:
            return placement, "ok"

        strict_placement, strict_reason = _stricten_placement(
            placement, ctx, occupied=occupied
        )
        if strict_placement is not None:
            return strict_placement, "ok"
        last_reason = strict_reason

    return None, last_reason


def _scaled_strict_placement(placement, factor, ctx, occupied=None):
    scale = float(placement.scale) * float(factor)
    if scale <= 0:
        return None, "no_fit"
    flat, reason, sdf_min = _candidate_fit(
        placement.variant,
        placement.pos,
        scale,
        ctx,
        occupied=occupied,
    )
    if flat is None:
        return None, reason

    candidate = Placement(
        variant=placement.variant,
        pos=placement.pos.copy(),
        scale=scale,
        volume=float(placement.variant.volume * (scale ** 3)),
        occ_flat=flat,
        occ_set=set(flat.tolist()),
        collision_set=set(flat.tolist()),
        surface_clearance=sdf_min,
    )
    return _stricten_placement(candidate, ctx, occupied=occupied)


def _generate_candidates(rough, shapes, search_points, ctx, preferred_shape=None,
                         progress_callback=None):
    if preferred_shape and preferred_shape in shapes:
        active = {preferred_shape: shapes[preferred_shape]}
    else:
        active = shapes

    variants = _make_variants(active)
    rough_bounds = rough.bounds
    max_dim = float(np.max(rough.extents))
    rejected = {}
    candidates = []

    total = max(len(variants), 1)
    for i, variant in enumerate(variants):
        if progress_callback and i % max(1, total // 20) == 0:
            progress_callback(i, total, "Voxel nesting placements")

        local = []
        for pos in search_points:
            placement, reason = _max_scale_for(variant, pos, rough_bounds, max_dim, ctx)
            if placement is None:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            local.append(placement)

        local.sort(key=lambda p: p.volume, reverse=True)
        candidates.extend(_spatially_diverse_candidates(local)[:TOP_PER_VARIANT])

    candidates.sort(key=lambda p: p.volume, reverse=True)
    candidates = _spatially_diverse_candidates(candidates[:MAX_CANDIDATES * 2])

    strict_candidates = []
    for placement in candidates[:CANDIDATE_SCALE_SOURCE_LIMIT]:
        for factor in CANDIDATE_SCALE_FACTORS:
            strict_placement, reason = _scaled_strict_placement(
                placement,
                factor,
                ctx,
            )
            if strict_placement is None:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            strict_candidates.append(strict_placement)

    strict_candidates.sort(key=lambda p: p.volume, reverse=True)
    strict_candidates = _unique_candidate_variants(strict_candidates)
    return strict_candidates[:MAX_CANDIDATES], rejected


def _spatially_diverse_candidates(candidates):
    selected = []
    seen = set()
    for p in candidates:
        key = (
            p.name,
            round(float(p.pos[0]), 2),
            round(float(p.pos[1]), 2),
            round(float(p.pos[2]), 2),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(p)
        if len(selected) >= MAX_CANDIDATES:
            break
    return selected


def _unique_candidate_variants(candidates):
    selected = []
    seen = set()
    for p in candidates:
        key = (
            p.name,
            round(float(p.pos[0]), 2),
            round(float(p.pos[1]), 2),
            round(float(p.pos[2]), 2),
            round(float(p.scale), 3),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(p)
    return selected


def _beam_pack(candidates, max_gems=MAX_GEMS, state_filter=None):
    if not candidates:
        return BeamState([], set(), 0.0)

    indexed = list(enumerate(candidates))
    beam = [BeamState([], set(), 0.0)]
    best = beam[0]

    for _depth in range(max_gems):
        expanded = list(beam)
        for state in beam:
            largest = max([p.volume for p in state.placements], default=None)
            for idx, cand in indexed:
                if any(existing is cand for existing in state.placements):
                    continue
                if largest is not None and cand.volume < largest * MIN_GEM_VOL_RATIO:
                    continue
                cand_collision = _placement_collision_set(cand)
                if not state.occupied.isdisjoint(cand_collision):
                    continue
                occupied = set(state.occupied)
                occupied.update(cand_collision)
                placements = state.placements + [cand]
                expanded.append(BeamState(
                    placements=placements,
                    occupied=occupied,
                    volume=state.volume + cand.volume,
                ))

        expanded.sort(key=lambda s: s.volume, reverse=True)
        ranked = _unique_states(expanded)
        if state_filter is not None:
            filtered = []
            # Exact cut-tree verification is reserved for final states. During
            # beam search, evaluate a bounded high-value pool with the same
            # protected-envelope separability rule.
            pool_multiplier = 12
            for state in ranked[:BEAM_WIDTH * pool_multiplier]:
                if len(state.placements) <= 1 or state_filter(state):
                    filtered.append(state)
                if len(filtered) >= BEAM_WIDTH:
                    break
            beam = filtered
        else:
            beam = ranked[:BEAM_WIDTH]
        if not beam:
            break
        if beam and beam[0].volume > best.volume:
            best = beam[0]

    return best


def _beam_pack_frontier(candidates, max_gems=MAX_GEMS, state_filter=None,
                        limit=CUTTABLE_FRONTIER_LIMIT, deadline=None):
    if not candidates:
        return [BeamState([], set(), 0.0)]

    indexed = list(enumerate(candidates))
    width = max(BEAM_WIDTH, int(limit))
    beam = [BeamState([], set(), 0.0)]
    collected = list(beam)

    for _depth in range(max_gems):
        if deadline is not None and time.time() >= deadline:
            break
        expanded = list(beam)
        for state in beam:
            largest = max([p.volume for p in state.placements], default=None)
            for _idx, cand in indexed:
                if deadline is not None and time.time() >= deadline:
                    break
                if any(existing is cand for existing in state.placements):
                    continue
                if largest is not None and cand.volume < largest * MIN_GEM_VOL_RATIO:
                    continue
                collision = _placement_collision_set(cand)
                if not state.occupied.isdisjoint(collision):
                    continue
                trial = BeamState(
                    placements=state.placements + [cand],
                    occupied=state.occupied | collision,
                    volume=state.volume + cand.volume,
                )
                expanded.append(trial)

        ranked = _unique_states(sorted(
            expanded,
            key=lambda state: (state.volume, len(state.placements)),
            reverse=True,
        ))
        if state_filter is not None:
            filtered = []
            for state in ranked[:width * 16]:
                if deadline is not None and time.time() >= deadline:
                    break
                if len(state.placements) <= 1 or state_filter(state):
                    filtered.append(state)
                if len(filtered) >= width:
                    break
            beam = filtered
        else:
            beam = ranked[:width]
        if not beam:
            break
        collected.extend(beam)

    ranked = _unique_states(sorted(
        collected,
        key=lambda state: (state.volume, len(state.placements)),
        reverse=True,
    ))
    non_empty = [state for state in ranked if state.placements]
    return (non_empty or ranked)[:max(1, int(limit))]


def _cuttable_candidate_fill(state, candidates, ctx, max_gems=MAX_GEMS):
    placements = list(state.placements)
    occupied = set(state.occupied)
    accepts = _quick_cuttable_filter(ctx)
    ranked = sorted(candidates, key=lambda placement: placement.volume,
                    reverse=True)
    minimum = _minimum_pocket_volume(ctx)
    diagnostics = {
        "added": 0,
        "checked": 0,
        "rejected_overlap": 0,
        "rejected_cut_tree": 0,
        "minimum_volume_mesh_units": round(float(minimum), 6),
    }

    while len(placements) < max_gems:
        best = None
        for candidate in ranked:
            if any(existing is candidate for existing in placements):
                continue
            if candidate.volume < minimum:
                break
            diagnostics["checked"] += 1
            collision = _placement_collision_set(candidate)
            if not occupied.isdisjoint(collision):
                diagnostics["rejected_overlap"] += 1
                continue
            trial = _state_from_placements(placements + [candidate])
            if not accepts(trial):
                diagnostics["rejected_cut_tree"] += 1
                continue
            best = candidate
            break
        if best is None:
            break
        placements.append(best)
        occupied.update(_placement_collision_set(best))
        diagnostics["added"] += 1

    return _state_from_placements(placements), diagnostics


def _merge_rejection_counts(target, source):
    for reason, count in source.items():
        target[reason] = int(target.get(reason, 0)) + int(count)


def _plan_leaf_constraints(plan, state):
    """Return the verified cut half-spaces that lead to each existing gem leaf."""
    if not plan or plan.get("status") != "complete":
        return []

    sequence = plan.get("sequence") or []
    leaves = []
    for index, placement in enumerate(state.placements):
        gem_id = f"gem_{index + 1}"
        constraints = []
        for step in sequence:
            negative = step.get("negative_side_gems") or []
            positive = step.get("positive_side_gems") or []
            if gem_id not in negative and gem_id not in positive:
                continue
            plane = step.get("plane") or {}
            normal = np.asarray(plane.get("normal") or [], dtype=float)
            if normal.shape != (3,):
                continue
            length = float(np.linalg.norm(normal))
            if length <= 1e-9:
                continue
            normal /= length
            constraints.append({
                "normal": normal,
                "offset": float(plane.get("offset", 0.0)),
                "side": "negative" if gem_id in negative else "positive",
                "half_kerf": 0.5 * float(
                    (step.get("kerf_slab") or {}).get(
                        "thickness_mesh_units",
                        0.0,
                    )
                ),
            })
        if constraints:
            leaves.append({
                "gem_id": gem_id,
                "placement": placement,
                "constraints": constraints,
            })
    return leaves


def _constraint_scale_limit(variant, pos, constraints, ctx, max_dim):
    """Limit a candidate so its preform envelope preserves verified cuts."""
    if not constraints:
        return float(max_dim)

    preform = float(ctx.fit.get("preform_margin_mesh_units") or 0.0)
    limit = float(max_dim)
    for constraint in constraints:
        normal = constraint["normal"]
        centre_projection = float(np.dot(pos, normal))
        relative = np.asarray(variant.verts, dtype=float) @ normal
        offset = float(constraint["offset"])
        half_kerf = float(constraint["half_kerf"])
        if constraint["side"] == "negative":
            available = offset - half_kerf - preform - centre_projection
            outward = float(np.max(relative))
        else:
            available = centre_projection - offset - half_kerf - preform
            outward = float(-np.min(relative))
        if available <= 0.0:
            return 0.0
        if outward > 1e-9:
            limit = min(limit, available / outward)
    return max(0.0, float(limit))


def _deadline_before_reserve(deadline, reserve_seconds):
    remaining = float(deadline - time.time())
    if remaining <= 0.0:
        return float(deadline)
    reserve = min(
        float(reserve_seconds),
        remaining * CUTTABLE_DEADLINE_RESERVE_FRACTION,
    )
    return float(deadline) - max(0.0, reserve)


def _leaf_aware_points(points, plan, state, ctx):
    leaves = _plan_leaf_constraints(plan, state)
    if not leaves:
        return [(point, []) for point in points], 0

    preform = float(ctx.fit.get("preform_margin_mesh_units") or 0.0)
    result = []
    seen = set()
    for leaf in leaves:
        for point in points:
            valid = True
            for constraint in leaf["constraints"]:
                projection = float(np.dot(point, constraint["normal"]))
                protected_offset = constraint["half_kerf"] + preform
                if constraint["side"] == "negative":
                    valid = projection < constraint["offset"] - protected_offset
                else:
                    valid = projection > constraint["offset"] + protected_offset
                if not valid:
                    break
            if not valid:
                continue
            key = (leaf["gem_id"], tuple(np.round(point, 5)))
            if key in seen:
                continue
            seen.add(key)
            result.append((point, leaf["constraints"]))
    return result, len(leaves)


def _local_pocket_candidates(state, shapes, fallback_candidates, rough, ctx,
                             deadline, structural_plan=None):
    points, component_summary = _free_component_points(
        ctx,
        state.occupied,
        component_limit=CUTTABLE_POCKET_COMPONENT_LIMIT,
        points_per_component=(
            CUTTABLE_POCKET_POINTS_PER_COMPONENT * 2
            if structural_plan else CUTTABLE_POCKET_POINTS_PER_COMPONENT
        ),
    )
    search_points, leaf_count = _leaf_aware_points(
        points,
        structural_plan,
        state,
        ctx,
    )
    variants = _pocket_variant_pool(shapes, fallback_candidates)
    minimum = _minimum_pocket_volume(ctx)
    diagnostics = {
        "components_inspected": min(
            CUTTABLE_POCKET_COMPONENT_LIMIT,
            len(component_summary.get("top_components", [])),
        ),
        "points_generated": len(points),
        "leaf_aware_points": len(search_points),
        "verified_leaf_regions": leaf_count,
        "checked_fits": 0,
        "fit_candidates": 0,
        "below_saleable_size": 0,
        "rejected_reasons": {},
        "timed_out": False,
        "free_space_components": component_summary,
    }
    if not search_points or not variants:
        return [], diagnostics

    rough_bounds = rough.bounds
    max_dim = float(np.max(rough.extents))
    candidates = []
    seen = set()
    for variant in variants:
        for pos, constraints in search_points:
            if time.time() >= deadline:
                diagnostics["timed_out"] = True
                break
            diagnostics["checked_fits"] += 1
            scale_limit = _constraint_scale_limit(
                variant,
                pos,
                constraints,
                ctx,
                max_dim,
            )
            if scale_limit <= 0.0:
                diagnostics["rejected_reasons"]["cut_leaf_boundary"] = (
                    diagnostics["rejected_reasons"].get(
                        "cut_leaf_boundary", 0
                    ) + 1
                )
                continue
            placement, reason = _max_scale_for(
                variant,
                pos,
                rough_bounds,
                max_dim,
                ctx,
                occupied=state.occupied,
                limit_scale=scale_limit,
                strict=True,
            )
            if placement is None:
                diagnostics["rejected_reasons"][reason] = (
                    diagnostics["rejected_reasons"].get(reason, 0) + 1
                )
                continue
            if placement.volume < minimum:
                diagnostics["below_saleable_size"] += 1
                continue
            key = (
                placement.variant.key,
                tuple(np.round(placement.pos, 3)),
                round(float(placement.scale), 4),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(placement)
        if diagnostics["timed_out"]:
            break

    candidates.sort(key=lambda placement: placement.volume, reverse=True)
    candidates = candidates[:CUTTABLE_POCKET_CANDIDATE_LIMIT]
    diagnostics["fit_candidates"] = len(candidates)
    return candidates, diagnostics


def _cuttable_cavity_frontier(seed_states, shapes, fallback_candidates, rough,
                              ctx, max_gems, deadline, route,
                              structural_plan=None):
    structural_normals = [
        (step.get("plane") or {}).get("normal")
        for step in ((structural_plan or {}).get("sequence") or [])
        if (step.get("plane") or {}).get("normal") is not None
    ]
    accepts = _quick_cuttable_filter(ctx, extra_normals=structural_normals)
    frontier = _unique_states(sorted(
        [state for state in seed_states if state.placements],
        key=lambda state: (state.volume, len(state.placements)),
        reverse=True,
    ))[:CUTTABLE_FRONTIER_LIMIT]
    collected = list(frontier)
    seed_count = max([len(state.placements) for state in frontier], default=0)
    diagnostics = {
        "route": route,
        "seed_states": len(frontier),
        "frontier_limit": CUTTABLE_FRONTIER_LIMIT,
        "component_limit": CUTTABLE_POCKET_COMPONENT_LIMIT,
        "points_per_component": CUTTABLE_POCKET_POINTS_PER_COMPONENT,
        "candidate_limit_per_state": CUTTABLE_POCKET_CANDIDATE_LIMIT,
        "rounds_completed": 0,
        "checked_fits": 0,
        "fit_candidates": 0,
        "below_saleable_size": 0,
        "rejected_reasons": {},
        "rejected_cut_tree": 0,
        "accepted_additions": 0,
        "timed_out": False,
        "stop_reason": "",
        "free_space_components": {},
        "verified_leaf_regions": 0,
        "leaf_aware_points": 0,
    }

    round_index = 0
    while True:
        if round_index >= CUTTABLE_POCKET_SAFETY_ROUND_CEILING:
            diagnostics["stop_reason"] = "safety_ceiling_reached"
            break
        if not frontier:
            diagnostics["stop_reason"] = "no_frontier_states"
            break
        if time.time() >= deadline:
            diagnostics["timed_out"] = True
            diagnostics["stop_reason"] = "time_budget_exhausted"
            break
        round_frontier = frontier
        expanded = list(frontier)
        additions = 0
        for state in frontier:
            if time.time() >= deadline:
                diagnostics["timed_out"] = True
                break
            if len(state.placements) >= max_gems:
                continue
            local_deadline = _deadline_before_reserve(
                deadline,
                CUTTABLE_CANDIDATE_EXPANSION_RESERVE_SECONDS,
            )
            if time.time() >= local_deadline:
                diagnostics["timed_out"] = True
                break
            local, local_diag = _local_pocket_candidates(
                state,
                shapes,
                fallback_candidates,
                rough,
                ctx,
                local_deadline,
                structural_plan=structural_plan,
            )
            diagnostics["checked_fits"] += local_diag["checked_fits"]
            diagnostics["fit_candidates"] += local_diag["fit_candidates"]
            diagnostics["below_saleable_size"] += local_diag[
                "below_saleable_size"
            ]
            diagnostics["free_space_components"] = local_diag.get(
                "free_space_components", {}
            )
            diagnostics["verified_leaf_regions"] = max(
                diagnostics["verified_leaf_regions"],
                int(local_diag.get("verified_leaf_regions", 0)),
            )
            diagnostics["leaf_aware_points"] += int(
                local_diag.get("leaf_aware_points", 0)
            )
            _merge_rejection_counts(
                diagnostics["rejected_reasons"],
                local_diag["rejected_reasons"],
            )
            if local_diag["timed_out"]:
                diagnostics["timed_out"] = True
            for candidate in local:
                if time.time() >= deadline:
                    diagnostics["timed_out"] = True
                    break
                trial = _state_from_placements(state.placements + [candidate])
                if not accepts(trial):
                    diagnostics["rejected_cut_tree"] += 1
                    continue
                expanded.append(trial)
                additions += 1
                diagnostics["accepted_additions"] += 1

        ranked = _unique_states(sorted(
            expanded,
            key=lambda state: (state.volume, len(state.placements)),
            reverse=True,
        ))
        frontier = ranked[:CUTTABLE_FRONTIER_LIMIT]
        collected.extend(frontier)
        round_index += 1
        diagnostics["rounds_completed"] = round_index
        if additions == 0:
            if diagnostics["timed_out"]:
                diagnostics["stop_reason"] = "time_budget_exhausted"
            elif all(len(state.placements) >= max_gems for state in round_frontier):
                diagnostics["stop_reason"] = "max_gems_reached"
            else:
                diagnostics["stop_reason"] = "no_additional_placements_found"
            break

    ranked = _unique_states(sorted(
        collected,
        key=lambda state: (state.volume, len(state.placements)),
        reverse=True,
    ))
    diagnostics["best_gem_count"] = max(
        [len(state.placements) for state in ranked], default=seed_count
    )
    diagnostics["added_to_best"] = max(
        diagnostics["best_gem_count"] - seed_count,
        0,
    )
    diagnostics["best_volume_mesh_units"] = round(
        float(ranked[0].volume) if ranked else 0.0,
        6,
    )
    return ranked[:CUTTABLE_FRONTIER_LIMIT], diagnostics


def _free_pocket_points(ctx, occupied, count=70):
    allowed = ctx.rough_mask & ~ctx.no_cut_mask
    if occupied:
        occupied_mask = np.zeros(ctx.grid.shape, dtype=bool)
        occupied_mask.ravel()[list(occupied)] = True
        allowed = allowed & ~occupied_mask
    if not np.any(allowed):
        return []

    clearance = ndimage.distance_transform_edt(allowed)
    flat = clearance.ravel()
    usable = np.flatnonzero(flat > 1.0)
    if len(usable) == 0:
        usable = np.flatnonzero(allowed.ravel())
    if len(usable) == 0:
        return []

    usable = usable[np.argsort(flat[usable])[::-1]]
    pts, seen = [], set()
    for flat_idx in usable:
        idx = np.array(np.unravel_index(int(flat_idx), ctx.grid.shape))
        key = tuple((idx // 3).tolist())
        if key in seen:
            continue
        seen.add(key)
        pts.append(ctx.origin + idx * ctx.pitch)
        if len(pts) >= count:
            break
    return pts


def _candidate_variant_pool(candidates, limit=GAP_FILL_VARIANT_LIMIT):
    best_by_key = {}
    for candidate in candidates:
        key = candidate.variant.key
        if key not in best_by_key or candidate.volume > best_by_key[key].volume:
            best_by_key[key] = candidate
    ranked = sorted(best_by_key.values(), key=lambda p: p.volume, reverse=True)
    return [p.variant for p in ranked[:limit]]


def _pocket_variant_pool(shapes, fallback_candidates=None):
    active = {name: shapes[name] for name in POCKET_FILL_SHAPES if name in shapes}
    if not active:
        active = shapes
    grouped = {name: [] for name in active}
    for variant in _make_variants(active):
        grouped.setdefault(variant.name, []).append(variant)

    variants = []
    while len(variants) < POCKET_FILL_VARIANT_LIMIT:
        added = False
        for name in active:
            if grouped.get(name):
                variants.append(grouped[name].pop(0))
                added = True
                if len(variants) >= POCKET_FILL_VARIANT_LIMIT:
                    break
        if not added:
            break

    if fallback_candidates and len(variants) < POCKET_FILL_VARIANT_LIMIT:
        seen = {v.key for v in variants}
        for variant in _candidate_variant_pool(
            fallback_candidates,
            POCKET_FILL_VARIANT_LIMIT,
        ):
            if variant.key in seen:
                continue
            variants.append(variant)
            seen.add(variant.key)
            if len(variants) >= POCKET_FILL_VARIANT_LIMIT:
                break
    return variants


def _free_component_points(ctx, occupied, component_limit=POCKET_FILL_COMPONENT_LIMIT,
                           points_per_component=POCKET_FILL_POINTS_PER_COMPONENT):
    summary, labels = _component_summary(ctx, occupied, limit=component_limit)
    if summary["count"] == 0:
        return [], summary

    free = labels > 0
    clearance = ndimage.distance_transform_edt(free)
    sizes = np.bincount(labels.ravel(), minlength=summary["count"] + 1)
    ranked = [int(i) for i in np.argsort(sizes[1:])[::-1] + 1
              if sizes[i] > 1]

    points = []
    seen = set()
    directions = np.asarray([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0], [-1.0, 1.0, -1.0],
    ], dtype=float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for label in ranked[:component_limit]:
        flat = np.flatnonzero(labels.ravel() == label)
        if len(flat) == 0:
            continue
        scores = clearance.ravel()[flat]
        indices = np.column_stack(np.unravel_index(flat, ctx.grid.shape))
        positions = ctx.origin + indices * ctx.pitch
        score_span = max(float(np.ptp(scores)), 1e-9)
        clearance_score = (scores - float(np.min(scores))) / score_span
        candidate_order = list(np.argsort(scores)[::-1][:points_per_component * 3])
        for direction in directions:
            projection = positions @ direction
            projection_span = max(float(np.ptp(projection)), 1e-9)
            projection_score = (
                projection - float(np.min(projection))
            ) / projection_span
            combined = clearance_score * 0.58 + projection_score * 0.42
            candidate_order.append(int(np.argmax(combined)))
        ranked_flat = [int(flat[index]) for index in candidate_order]
        added = 0
        for flat_idx in ranked_flat:
            idx = np.array(np.unravel_index(int(flat_idx), ctx.grid.shape))
            key = tuple((idx // 3).tolist())
            if key in seen:
                continue
            seen.add(key)
            points.append(ctx.origin + idx * ctx.pitch)
            added += 1
            if added >= points_per_component:
                break
    return points, summary


def _minimum_pocket_volume(ctx):
    configured = float(ctx.fit.get("min_secondary_volume", 0.0) or 0.0)
    fallback = float(np.count_nonzero(ctx.rough_mask) * (ctx.pitch ** 3)
                     * GAP_FILL_MIN_VOL_RATIO)
    if ctx.fit.get("extra_gem_policy") == "maximum_count":
        return min(configured or fallback, fallback)
    return max(configured, fallback)


def _state_from_placements(placements):
    occupied = set()
    total = 0.0
    for p in placements:
        occupied.update(_placement_collision_set(p))
        total += p.volume
    return BeamState(placements, occupied, total)


def _dominant_rejection(rejected):
    if not rejected:
        return "no candidate pockets tested"
    reason, _ = max(rejected.items(), key=lambda item: item[1])
    messages = {
        "too_small": "remaining pockets are below saleable size",
        "overlap": "remaining pockets are blocked by blade clearance",
        "dense_rough_voxel": "remaining pockets are too thin near the rough boundary",
        "rough_voxel": "remaining pockets are too thin near the rough boundary",
        "surface_outside": "remaining pockets would exit the rough stone",
        "surface_margin": "remaining pockets would sit too close to the rough surface",
        "defect_zone": "remaining pockets intersect mapped defect zones",
        "dense_defect_zone": "remaining pockets intersect mapped defect zones",
        "no_fit": "no pocket shape fits the remaining free space",
        "bounds": "remaining pockets are outside searchable bounds",
        "dense_bounds": "remaining pockets are outside searchable bounds",
    }
    return messages.get(reason, f"limited by {reason}")


def _pocket_fill_state(state, shapes, candidates, rough, ctx, max_gems=MAX_GEMS,
                       started_at=None):
    placements = list(state.placements)
    occupied = set(state.occupied)
    variants = _pocket_variant_pool(shapes, candidates)
    rough_bounds = rough.bounds
    max_dim = float(np.max(rough.extents))
    min_volume = _minimum_pocket_volume(ctx)
    started_at = started_at or time.time()
    diag = {
        "passes_completed": 0,
        "added": 0,
        "checked_fits": 0,
        "min_saleable_volume": round(float(min_volume), 6),
        "max_gems": int(max_gems),
        "rejected_reasons": {},
        "timed_out": False,
        "unused_space_reason": "",
        "free_space_components": {},
    }

    while True:
        if diag["passes_completed"] >= POCKET_FILL_SAFETY_ITERATION_CEILING:
            diag["unused_space_reason"] = "internal search safety limit reached"
            break
        if len(placements) >= max_gems:
            diag["unused_space_reason"] = "maximum gem count reached"
            break
        if time.time() - started_at > POCKET_FILL_TIME_LIMIT_SECONDS:
            diag["timed_out"] = True
            diag["unused_space_reason"] = "pocket fill stopped at runtime limit"
            break

        points, free_summary = _free_component_points(ctx, occupied)
        diag["free_space_components"] = free_summary
        if not points:
            diag["unused_space_reason"] = "no connected free pocket remains"
            break

        best = None
        for variant in variants:
            for pos in points:
                if time.time() - started_at > POCKET_FILL_TIME_LIMIT_SECONDS:
                    diag["timed_out"] = True
                    break
                diag["checked_fits"] += 1
                placement, reason = _max_scale_for(
                    variant,
                    pos,
                    rough_bounds,
                    max_dim,
                    ctx,
                    occupied=occupied,
                    strict=True,
                )
                if placement is None:
                    diag["rejected_reasons"][reason] = (
                        diag["rejected_reasons"].get(reason, 0) + 1
                    )
                    continue
                if placement.volume < min_volume:
                    diag["rejected_reasons"]["too_small"] = (
                        diag["rejected_reasons"].get("too_small", 0) + 1
                    )
                    continue
                if best is None or placement.volume > best.volume:
                    best = placement
            if diag["timed_out"]:
                break

        if best is None:
            if not diag["unused_space_reason"]:
                diag["unused_space_reason"] = _dominant_rejection(
                    diag["rejected_reasons"]
                )
            break

        placements.append(best)
        occupied.update(_placement_collision_set(best))
        diag["added"] += 1
        diag["passes_completed"] += 1

    final_summary, _ = _component_summary(ctx, occupied)
    diag["free_space_components"] = final_summary
    if not diag["unused_space_reason"]:
        diag["unused_space_reason"] = _dominant_rejection(
            diag["rejected_reasons"]
        )

    return _state_from_placements(placements), diag


def _merge_pocket_diagnostics(first, second):
    if not first:
        return second or {}
    if not second:
        return first

    merged = dict(first)
    merged["added"] = int(first.get("added", 0)) + int(second.get("added", 0))
    merged["checked_fits"] = (
        int(first.get("checked_fits", 0)) + int(second.get("checked_fits", 0))
    )
    merged["timed_out"] = bool(first.get("timed_out")) or bool(second.get("timed_out"))
    merged["unused_space_reason"] = second.get(
        "unused_space_reason",
        first.get("unused_space_reason", ""),
    )
    merged["free_space_components"] = second.get(
        "free_space_components",
        first.get("free_space_components", {}),
    )
    rejected = dict(first.get("rejected_reasons", {}))
    for reason, count in second.get("rejected_reasons", {}).items():
        rejected[reason] = rejected.get(reason, 0) + count
    merged["rejected_reasons"] = rejected
    merged["stages"] = {
        "post_blade": first,
        "post_rough_clearance": second,
    }
    return merged


def _gap_fill_state(state, candidates, search_points, rough, ctx, max_gems=MAX_GEMS,
                    started_at=None):
    placements = list(state.placements)
    occupied = set(state.occupied)
    total = float(state.volume)
    variants = _candidate_variant_pool(candidates)
    rough_bounds = rough.bounds
    max_dim = float(np.max(rough.extents))
    min_volume = _minimum_pocket_volume(ctx)
    started_at = started_at or time.time()
    diag = {
        "candidate_pool_passes_completed": 0,
        "pocket_search_passes_completed": 0,
        "added": 0,
        "checked_candidates": 0,
        "checked_pocket_fits": 0,
        "min_added_volume": round(min_volume, 6),
        "rejected": {},
        "timed_out": False,
    }

    ranked = sorted(candidates, key=lambda p: p.volume, reverse=True)
    used = set(id(p) for p in placements)

    while True:
        if diag["candidate_pool_passes_completed"] >= GAP_FILL_SAFETY_ITERATION_CEILING:
            break
        if len(placements) >= max_gems:
            break
        if time.time() - started_at > GAP_FILL_TIME_LIMIT_SECONDS:
            diag["timed_out"] = True
            break

        best = None
        for candidate in ranked:
            diag["checked_candidates"] += 1
            if id(candidate) in used:
                continue
            if candidate.volume < min_volume:
                diag["rejected"]["too_small"] = diag["rejected"].get("too_small", 0) + 1
                break
            collision = _placement_collision_set(candidate)
            if not occupied.isdisjoint(collision):
                diag["rejected"]["overlap"] = diag["rejected"].get("overlap", 0) + 1
                continue
            best = candidate
            break

        if best is None:
            break

        placements.append(best)
        used.add(id(best))
        occupied.update(_placement_collision_set(best))
        total += best.volume
        diag["added"] += 1
        diag["candidate_pool_passes_completed"] += 1

    while True:
        if diag["pocket_search_passes_completed"] >= GAP_FILL_SAFETY_ITERATION_CEILING:
            break
        if len(placements) >= max_gems:
            break
        if time.time() - started_at > GAP_FILL_TIME_LIMIT_SECONDS:
            diag["timed_out"] = True
            break

        pocket_points = _free_pocket_points(
            ctx, occupied, count=GAP_FILL_POINT_LIMIT
        )
        best = None
        for variant in variants:
            for pos in pocket_points:
                diag["checked_pocket_fits"] += 1
                placement, reason = _max_scale_for(
                    variant,
                    pos,
                    rough_bounds,
                    max_dim,
                    ctx,
                    occupied=occupied,
                    strict=True,
                )
                if placement is None:
                    diag["rejected"][reason] = diag["rejected"].get(reason, 0) + 1
                    continue
                if placement.volume < min_volume:
                    diag["rejected"]["too_small"] = diag["rejected"].get("too_small", 0) + 1
                    continue
                if best is None or placement.volume > best.volume:
                    best = placement

        if best is None:
            break

        placements.append(best)
        occupied.update(_placement_collision_set(best))
        total += best.volume
        diag["added"] += 1
        diag["pocket_search_passes_completed"] += 1

    return BeamState(placements, occupied, total), diag


def _unique_states(states):
    out, seen = [], set()
    for state in states:
        sig = tuple(sorted((p.name, tuple(np.round(p.pos, 3)), round(p.scale, 4))
                           for p in state.placements))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(state)
    return out


def _refine_state(state, rough, ctx):
    if len(state.placements) <= 0:
        return state

    placements = list(state.placements)
    rough_bounds = rough.bounds
    max_dim = float(np.max(rough.extents))
    steps = [
        np.zeros(3),
        np.array([1, 0, 0]), np.array([-1, 0, 0]),
        np.array([0, 1, 0]), np.array([0, -1, 0]),
        np.array([0, 0, 1]), np.array([0, 0, -1]),
    ]

    for _ in range(REFINE_PASSES):
        for i, current in enumerate(placements):
            occupied = set()
            for j, other in enumerate(placements):
                if i != j:
                    occupied.update(_placement_collision_set(other))

            best = current
            for direction in steps:
                for amount in (0.0, ctx.pitch, ctx.pitch * 2):
                    pos = current.pos + direction * amount
                    placement, _ = _max_scale_for(
                        current.variant,
                        pos,
                        rough_bounds,
                        max_dim,
                        ctx,
                        occupied=occupied,
                        start_scale=current.scale,
                        limit_scale=current.scale * 1.18,
                        strict=True,
                    )
                    if placement and placement.volume > best.volume:
                        best = placement
            placements[i] = best

    occupied = set()
    total = 0.0
    for p in placements:
        occupied.update(_placement_collision_set(p))
        total += p.volume
    return BeamState(placements, occupied, total)


def _mesh_from_placement(p):
    return _mesh_for_candidate(p.variant, p.pos, p.scale)


def _cuttable_settings_present(ctx):
    return (
        ctx.fit.get("preform_margin_mm") is not None
        and ctx.fit.get("max_cut_depth_mm") is not None
        and float(ctx.fit.get("mm_per_mesh_unit", 0.0) or 0.0) > 0
    )


def _quick_cuttable_filter(ctx, extra_normals=None):
    kerf = float(ctx.fit["blade_kerf_mm"]) / float(ctx.fit["mm_per_mesh_unit"])
    margin = float(ctx.fit["preform_margin_mm"]) / float(
        ctx.fit["mm_per_mesh_unit"]
    )
    cache = {}

    def accepts(state):
        signature = tuple(sorted(
            (p.variant.key, tuple(np.round(p.pos, 3)), round(p.scale, 4))
            for p in state.placements
        ))
        if signature not in cache:
            cache[signature] = is_recursively_separable(
                [_mesh_from_placement(p) for p in state.placements],
                kerf,
                margin,
                extra_normals=extra_normals,
            )
        return cache[signature]

    return accepts


def _geometric_comparison_plan(ctx, gem_count):
    plan = settings_required_plan(
        ctx.fit.get("blade_kerf_mm"),
        ctx.fit.get("mm_per_mesh_unit"),
    )
    plan["status"] = "geometric_comparison_only"
    plan["gem_count"] = int(gem_count)
    plan["warnings"] = [
        "This unconstrained maximum-yield arrangement is shown only for "
        "comparison and is not a recommended manufacturing plan."
    ]
    return plan


def _plan_or_reduce_state(state, rough, ctx, defect_points, deadline,
                          preferred_normals=None):
    """Verify a state, dropping the least valuable secondary when necessary."""
    ordered = sorted(state.placements, key=lambda placement: placement.volume,
                     reverse=True)
    original_volume = float(sum(placement.volume for placement in ordered))
    dropped = []
    latest_plan = None

    while len(ordered) > 1:
        remaining = deadline - time.time()
        if remaining <= 0.15:
            break
        planned_state = _state_from_placements(ordered)
        latest_plan = plan_cut_sequence(
            rough,
            [_mesh_from_placement(placement) for placement in ordered],
            blade_kerf_mm=ctx.fit["blade_kerf_mm"],
            preform_margin_mm=ctx.fit["preform_margin_mm"],
            max_cut_depth_mm=ctx.fit["max_cut_depth_mm"],
            mm_per_mesh_unit=ctx.fit["mm_per_mesh_unit"],
            gem_values=[placement.volume for placement in ordered],
            gem_ids=[f"gem_{index + 1}" for index in range(len(ordered))],
            defect_points=defect_points,
            defect_radius_mesh=ctx.fit["fracture_safe_dist"],
            pitch=ctx.pitch,
            time_limit_seconds=min(6.0, max(0.1, remaining)),
            preferred_normals=preferred_normals,
        )
        if latest_plan.get("status") == "complete":
            latest_plan["removed_for_cutability"] = dropped
            latest_plan["unconstrained_volume_mesh_units"] = round(
                original_volume, 6)
            latest_plan["cuttable_volume_mesh_units"] = round(
                planned_state.volume, 6)
            latest_plan["volume_difference_mesh_units"] = round(
                planned_state.volume - original_volume, 6)
            latest_plan["cuttable_yield_percent"] = round(
                100.0 * planned_state.volume / max(float(rough.volume), 1e-9),
                2,
            )
            return planned_state, latest_plan

        removed = ordered.pop()
        dropped.append({
            "shape": removed.name,
            "volume_mesh_units": round(float(removed.volume), 6),
            "reason": "no_complete_straight_cut_tree",
        })

    if len(ordered) == 1:
        single = _state_from_placements(ordered)
        plan = no_separation_plan(
            ctx.fit["blade_kerf_mm"],
            ctx.fit["preform_margin_mm"],
            ctx.fit["max_cut_depth_mm"],
            ctx.fit["mm_per_mesh_unit"],
        )
        plan["removed_for_cutability"] = dropped
        plan["unconstrained_volume_mesh_units"] = round(original_volume, 6)
        plan["cuttable_volume_mesh_units"] = round(single.volume, 6)
        plan["volume_difference_mesh_units"] = round(
            single.volume - original_volume, 6)
        return single, plan

    if latest_plan is None:
        latest_plan = _geometric_comparison_plan(ctx, len(state.placements))
        latest_plan["status"] = "timeout"
    latest_plan["removed_for_cutability"] = dropped
    return _state_from_placements(ordered), latest_plan


# ---------------------------------------------------------------------------
# Leaf-piece repacking: once a strategy has a verified straight full-through
# cut sequence, each leaf of that cut tree is a real, separately-holdable
# piece of rough with (so far) exactly one gem assigned to it. The packer
# that produced the gem layout never knew about that final cut geometry, so
# a leaf piece can still contain real, cuttable leftover space beyond its
# one gem — e.g. slicing a gem off one side first can reveal a usable void
# in what remains that the original whole-stone packing pass never
# considered. This scopes a fresh, cheap search to each leaf's *actual*
# post-cut sub-volume and, if it finds another gem that fits there, hands
# the augmented gem list back through the same verified drop-on-failure
# path (_plan_or_reduce_state) so any acceptance is re-proven safe, never
# assumed.
# ---------------------------------------------------------------------------

LEAF_REPACK_SDF_RESOLUTION = 48
LEAF_REPACK_TIME_LIMIT_SECONDS = 20.0
LEAF_REPACK_VERIFY_TIME_LIMIT_SECONDS = 8.0


def _leaf_pieces(plan):
    """List (gem_id, half-space constraints) for every single-gem leaf of a
    complete manufacturing plan's cut tree, constraints ordered root-first.
    Each constraint is (unit_normal, offset) meaning "keep dot(normal, x)
    >= offset" in the rough's own coordinate frame."""
    tree = plan.get("cut_tree")
    sequence = plan.get("sequence") or []
    steps_by_number = {step["step"]: step for step in sequence}
    if not isinstance(tree, dict):
        return []

    results = []

    def walk(node, constraints):
        if not isinstance(node, dict):
            return
        if node.get("type") == "leaf":
            gem_ids = node.get("gem_ids") or []
            if len(gem_ids) == 1:
                results.append((gem_ids[0], constraints))
            return
        step = steps_by_number.get(node.get("step"))
        if step is None:
            return
        plane = step.get("plane") or {}
        normal = plane.get("normal")
        offset = plane.get("offset")
        if normal is None or offset is None:
            return
        normal = np.asarray(normal, dtype=float)
        offset = float(offset)
        walk(node.get("left"), constraints + [(-normal, -offset)])
        walk(node.get("right"), constraints + [(normal, offset)])

    walk(tree, [])
    return results


def _leaf_submesh(rough, constraints):
    mesh = rough
    for normal, offset in constraints:
        plane_origin = normal * offset
        try:
            mesh = trimesh.intersections.slice_mesh_plane(
                mesh, normal, plane_origin, cap=True,
            )
        except Exception:
            return None
        if mesh is None or len(getattr(mesh, "faces", [])) < 4:
            return None
    return mesh


def _best_extra_placement_in_leaf(submesh, existing_placement, shapes,
                                  fit, defect_points, preferred_shape,
                                  deadline, diag):
    if submesh.volume <= 0 or float(np.min(submesh.extents)) <= 0:
        diag["leaves_skipped_geometry"] += 1
        return None
    try:
        sub_grid, sub_origin, sub_pitch = _build_sdf_grid(
            submesh, res=LEAF_REPACK_SDF_RESOLUTION,
        )
    except Exception:
        diag["leaves_skipped_geometry"] += 1
        return None

    sub_fit = _fit_settings(
        submesh.extents,
        sub_pitch,
        blade_kerf_mm=fit["blade_kerf_mm"],
        rough_clearance_mm=fit["rough_clearance_mm"],
        mm_per_mesh_unit=fit.get("mm_per_mesh_unit"),
        min_secondary_volume=fit.get("min_secondary_volume"),
        min_secondary_carat=fit.get("min_secondary_carat"),
        carats_per_mesh_volume=fit.get("carats_per_mesh_volume"),
        extra_gem_policy=fit.get("extra_gem_policy"),
        max_gems=fit.get("max_gems"),
        preform_margin_mm=fit.get("preform_margin_mm"),
        max_cut_depth_mm=fit.get("max_cut_depth_mm"),
    )
    sub_rough_mask = sub_grid > -sub_fit["voxel_tolerance"]
    sub_no_cut_mask = _make_no_cut_mask(
        sub_grid.shape, sub_origin, sub_pitch,
        defect_points, sub_fit["fracture_safe_dist"],
    )
    sub_ctx = FitContext(
        sub_grid, sub_origin, sub_pitch, sub_rough_mask, sub_no_cut_mask, sub_fit,
    )

    existing_dense = _candidate_dense_occ(
        existing_placement.variant, existing_placement.pos,
        existing_placement.scale, sub_ctx,
    )
    if existing_dense is None:
        return None
    occupied = _inflate_flat(
        existing_dense, sub_ctx.grid.shape, _spacing_voxels(sub_ctx),
    )

    min_volume = _minimum_pocket_volume(sub_ctx)
    center = submesh.bounds.mean(axis=0)
    search_points = _make_search_points(center, submesh.extents, sub_ctx)
    candidates, _ = _generate_candidates(
        submesh, shapes, search_points, sub_ctx,
        preferred_shape=preferred_shape,
    )

    best = None
    for placement in candidates:
        if time.time() >= deadline:
            break
        if placement.volume < min_volume:
            diag["rejected_reasons"]["too_small"] = (
                diag["rejected_reasons"].get("too_small", 0) + 1
            )
            continue
        flat, reason, _ = _candidate_fit(
            placement.variant, placement.pos, placement.scale, sub_ctx,
            occupied=occupied,
        )
        if flat is None:
            diag["rejected_reasons"][reason] = (
                diag["rejected_reasons"].get(reason, 0) + 1
            )
            continue
        if best is None or placement.volume > best.volume:
            best = placement

    if best is None:
        return None

    diag["candidates_found"] += 1
    axis = best.pos - existing_placement.pos
    length = float(np.linalg.norm(axis))
    separating_normal = (axis / length).tolist() if length > 1e-9 else None
    return best, separating_normal


def _repack_leaf_pieces(state, plan, rough, ctx, shapes, defect_points,
                        deadline, preferred_shape=None):
    diag = {
        "leaves_checked": 0,
        "leaves_skipped_geometry": 0,
        "candidates_found": 0,
        "gems_added": 0,
        "rejected_reasons": {},
    }
    if plan.get("status") != "complete":
        return state, plan, diag

    leaves = _leaf_pieces(plan)
    if not leaves:
        return state, plan, diag

    gem_by_id = {
        f"gem_{index + 1}": placement
        for index, placement in enumerate(state.placements)
    }
    preferred_normals = [
        (step.get("plane") or {}).get("normal")
        for step in (plan.get("sequence") or [])
        if (step.get("plane") or {}).get("normal") is not None
    ]

    new_placements = []
    leaf_deadline = min(
        _deadline_before_reserve(
            deadline,
            LEAF_REPACK_VERIFY_TIME_LIMIT_SECONDS,
        ),
        time.time() + LEAF_REPACK_TIME_LIMIT_SECONDS,
    )
    for gem_id, constraints in leaves:
        diag["leaves_checked"] += 1
        if time.time() >= leaf_deadline:
            break
        existing = gem_by_id.get(gem_id)
        if existing is None:
            continue

        submesh = _leaf_submesh(rough, constraints)
        if submesh is None:
            diag["leaves_skipped_geometry"] += 1
            continue

        # Searching every shape x rotation per leaf is far too slow for
        # this pass's tight time budget (~10x slower per leaf). Default to
        # the leaf's own existing gem shape, which is both cheap and a
        # reasonable match for what the stone was already being cut as.
        leaf_shape = preferred_shape or existing.variant.name
        found = _best_extra_placement_in_leaf(
            submesh, existing, shapes, ctx.fit, defect_points,
            leaf_shape, leaf_deadline, diag,
        )
        if found is None:
            continue
        placement, separating_normal = found
        new_placements.append(placement)
        if separating_normal is not None:
            preferred_normals.append(separating_normal)

    if not new_placements:
        return state, plan, diag

    augmented = list(state.placements) + new_placements
    verify_deadline = min(
        deadline, time.time() + LEAF_REPACK_VERIFY_TIME_LIMIT_SECONDS,
    )
    reduced_state, reduced_plan = _plan_or_reduce_state(
        _state_from_placements(augmented),
        rough,
        ctx,
        defect_points,
        verify_deadline,
        preferred_normals=preferred_normals,
    )

    original_ids = {id(p) for p in state.placements}
    kept_ids = {id(p) for p in reduced_state.placements}
    if (reduced_plan.get("status") == "complete"
            and len(reduced_state.placements) > len(state.placements)
            and original_ids.issubset(kept_ids)):
        diag["gems_added"] = len(reduced_state.placements) - len(state.placements)
        return reduced_state, reduced_plan, diag

    # Never regress: if the augmented layout couldn't be re-verified (or
    # would have dropped an already-proven gem to make room), keep the
    # original, already-safe state and plan untouched.
    return state, plan, diag


def _finalize_cuttable_frontier(states, rough, ctx, defect_points, deadline,
                                verification_limit=6,
                                structural_plan=None,
                                refine_states=True):
    best = None
    diagnostics = {
        "states_considered": 0,
        "states_verified": 0,
        "rejected_gap": 0,
        "rejected_plan": 0,
        "states_reduced_for_cutability": 0,
        "gems_removed_for_cutability": 0,
        "reduction_reasons": {},
        "timed_out": False,
    }
    target_gap = float(ctx.fit["gem_spacing_buffer"])
    preferred_normals = [
        (step.get("plane") or {}).get("normal")
        for step in ((structural_plan or {}).get("sequence") or [])
        if (step.get("plane") or {}).get("normal") is not None
    ]
    for state in states[:verification_limit]:
        if time.time() >= deadline:
            diagnostics["timed_out"] = True
            break
        diagnostics["states_considered"] += 1
        # Cavity-generated states already encode the candidate placement being
        # verified. Moving or growing it here can invalidate an otherwise
        # complete straight-cut tree, so callers may preserve the generated
        # geometry and let the exact planner decide.
        refined = state if (
            structural_plan is not None or not refine_states
        ) else _refine_state(
            state,
            rough,
            ctx,
        )
        refined, blade_initial = _enforce_blade_clearance(refined, ctx)
        refined, rough_diag = _enforce_rough_clearance(refined, ctx)
        refined, blade_final = _enforce_blade_clearance(refined, ctx)
        actual_gap = _state_min_surface_gap(refined.placements)
        tolerance = max(ctx.pitch * 0.08, 1e-6)
        if (actual_gap is not None
                and actual_gap + tolerance < target_gap):
            diagnostics["rejected_gap"] += 1
            continue

        planned, plan = _plan_or_reduce_state(
            refined,
            rough,
            ctx,
            defect_points,
            deadline,
            preferred_normals=preferred_normals,
        )
        removed = plan.get("removed_for_cutability") or []
        if removed:
            diagnostics["states_reduced_for_cutability"] += 1
            diagnostics["gems_removed_for_cutability"] += len(removed)
            for entry in removed:
                reason = entry.get("reason", "unknown")
                diagnostics["reduction_reasons"][reason] = (
                    diagnostics["reduction_reasons"].get(reason, 0) + 1
                )
        if len(planned.placements) <= 1 or plan.get("status") != "complete":
            diagnostics["rejected_plan"] += 1
            continue
        diagnostics["states_verified"] += 1
        candidate = {
            "state": planned,
            "plan": plan,
            "blade_initial": blade_initial,
            "blade_final": blade_final,
            "rough_clearance": rough_diag,
        }
        rank = (
            planned.volume,
            len(planned.placements),
            float(plan.get("minimum_envelope_clearance_mm") or 0.0),
            -float(plan.get("maximum_required_depth_mm") or 0.0),
        )
        if best is None or rank > best[0]:
            best = (rank, candidate)

    return (best[1] if best is not None else None), diagnostics


def _baseline_comparison(state, baseline, rough):
    rough_volume = max(float(rough.volume), 1e-9)
    return {
        "baseline_gem_count": len(baseline.placements),
        "baseline_volume_mesh_units": round(float(baseline.volume), 6),
        "baseline_yield_percent": round(100.0 * baseline.volume / rough_volume, 2),
        "result_gem_count": len(state.placements),
        "result_volume_mesh_units": round(float(state.volume), 6),
        "result_yield_percent": round(100.0 * state.volume / rough_volume, 2),
        "added_gems": len(state.placements) - len(baseline.placements),
        "volume_delta_mesh_units": round(float(state.volume - baseline.volume), 6),
        "yield_delta_percentage_points": round(
            100.0 * (state.volume - baseline.volume) / rough_volume,
            2,
        ),
    }


def _remaining_space_diagnostics(state, ctx, search_diagnostics=None):
    summary, _ = _component_summary(
        ctx,
        state.occupied,
        limit=CUTTABLE_POCKET_COMPONENT_LIMIT,
    )
    search_diagnostics = search_diagnostics or {}
    minimum_volume = _minimum_pocket_volume(ctx)
    minimum_carat = ctx.fit.get("min_secondary_carat")
    dominant = _dominant_rejection(
        search_diagnostics.get("rejected_reasons", {})
    )
    components = []
    for component in summary.get("top_components", []):
        item = dict(component)
        sphere_carat = item.get("estimated_inscribed_sphere_carat")
        if float(item.get("volume_mesh_units", 0.0)) < minimum_volume:
            reason = "below minimum saleable volume"
        elif (minimum_carat is not None and sphere_carat is not None
              and float(sphere_carat) < float(minimum_carat)):
            reason = "too thin for the minimum saleable carat"
        elif search_diagnostics.get("timed_out"):
            reason = "search time limit reached before exhaustive rejection"
        elif search_diagnostics.get("stop_reason") == "max_gems_reached":
            reason = "maximum gem count setting reached (raise Max gems to search further)"
        elif search_diagnostics.get("rejected_cut_tree", 0) > 0:
            reason = "candidate placements failed recursive straight-cut separation"
        elif search_diagnostics.get("fit_candidates", 0) == 0:
            reason = dominant
        else:
            reason = "no higher-yield verified placement remained in the search frontier"
        item["rejection_reason"] = reason
        components.append(item)
    return {
        "component_count": int(summary.get("count", 0)),
        "largest_volume_mesh_units": float(
            summary.get("largest_volume_mesh_units", 0.0)
        ),
        "minimum_saleable_volume_mesh_units": round(float(minimum_volume), 6),
        "minimum_saleable_carat": (
            round(float(minimum_carat), 3) if minimum_carat is not None else None
        ),
        "components": components,
    }


def _placement_boundary_points(p):
    return _gap_surface_points(_mesh_from_placement(p))


def _pair_surface_gap(a, b):
    a_pts = _placement_boundary_points(a)
    b_pts = _placement_boundary_points(b)
    if len(a_pts) == 0 or len(b_pts) == 0:
        return float("inf")
    d_ab = cKDTree(b_pts).query(a_pts, k=1)[0].min()
    d_ba = cKDTree(a_pts).query(b_pts, k=1)[0].min()
    return float(min(d_ab, d_ba))


def _state_min_surface_gap(placements):
    if len(placements) < 2:
        return None
    best = float("inf")
    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            best = min(best, _pair_surface_gap(placements[i], placements[j]))
    return best


def _placement_rough_clearance(placement, ctx):
    pts = _gap_surface_points(_mesh_from_placement(placement), limit=3200)
    if len(pts) == 0:
        return float("inf")
    return float(np.min(_sdf_values(ctx, pts)))


def _enforce_rough_clearance(state, ctx):
    placements = list(state.placements)
    target = float(ctx.fit.get("rough_surface_clearance", 0.0))
    enforcement_target = target + ctx.pitch * 0.15
    diag = {
        "target_clearance_mesh_units": round(target, 6),
        "target_clearance_mm": round(float(ctx.fit.get("rough_clearance_mm", 0.0)), 3),
        "min_clearance_before_mesh_units": None,
        "min_clearance_after_mesh_units": None,
        "adjustments": 0,
    }
    if target <= 0 or not placements:
        return state, diag

    before = [_placement_rough_clearance(p, ctx) for p in placements]
    diag["min_clearance_before_mesh_units"] = round(float(min(before)), 6)

    for _ in range(5):
        changed = False
        for i, p in enumerate(placements):
            clearance = _placement_rough_clearance(p, ctx)
            if clearance >= enforcement_target:
                continue

            occupied = set()
            for j, other in enumerate(placements):
                if i != j:
                    occupied.update(_placement_collision_set(other))

            radius = max(
                float(np.max(np.ptp(p.variant.verts, axis=0)) * p.scale * 0.5),
                1e-9,
            )
            deficit = enforcement_target - clearance
            factor = max(0.86, 1.0 - (deficit * 0.95) / radius)
            replacement, _ = _scaled_strict_placement(
                p,
                factor,
                ctx,
                occupied=occupied,
            )
            if replacement is not None and replacement.volume < p.volume:
                placements[i] = replacement
                diag["adjustments"] += 1
                changed = True

        if not changed:
            break

    final = [_placement_rough_clearance(p, ctx) for p in placements]
    diag["min_clearance_after_mesh_units"] = round(float(min(final)), 6)
    return _state_from_placements(placements), diag


def _shrink_state_for_blade_finish(state, ctx, factor=BLADE_FINISH_SHRINK):
    placements = []
    diag = {
        "scale_factor": factor,
        "adjusted_gems": 0,
    }
    for p in state.placements:
        replacement, _ = _scaled_strict_placement(p, factor, ctx)
        if replacement is not None:
            placements.append(replacement)
            diag["adjusted_gems"] += 1
        else:
            placements.append(p)

    occupied = set()
    total = 0.0
    for p in placements:
        occupied.update(_placement_collision_set(p))
        total += p.volume
    return BeamState(placements, occupied, total), diag


def _enforce_blade_clearance(state, ctx):
    placements = list(state.placements)
    target = float(ctx.fit["gem_spacing_buffer"])
    before = _state_min_surface_gap(placements)
    diag = {
        "target_gap_mesh_units": round(target, 6),
        "min_gap_before_mesh_units": round(float(before or 0.0), 6),
        "min_gap_after_mesh_units": round(float(before or 0.0), 6),
        "adjustments": 0,
    }

    if len(placements) < 2:
        return state, diag

    for _ in range(6):
        changed = False
        for i in range(len(placements)):
            for j in range(i + 1, len(placements)):
                gap = _pair_surface_gap(placements[i], placements[j])
                if gap >= target:
                    continue

                deficit = target - gap
                for idx in (i, j):
                    p = placements[idx]
                    radius = max(float(np.max(np.ptp(p.variant.verts, axis=0)) * p.scale * 0.5), 1e-9)
                    factor = max(0.78, 1.0 - (deficit * 0.75) / radius)
                    replacement, _ = _scaled_strict_placement(p, factor, ctx)
                    if replacement is not None and replacement.volume < p.volume:
                        placements[idx] = replacement
                        diag["adjustments"] += 1
                        changed = True

        if not changed:
            break

    occupied = set()
    total = 0.0
    for p in placements:
        occupied.update(_placement_collision_set(p))
        total += p.volume

    after = _state_min_surface_gap(placements)
    diag["min_gap_after_mesh_units"] = round(float(after or 0.0), 6)
    return BeamState(placements, occupied, total), diag


def _strategy_from_state(label, shape_label, state, rough, ctx, base_diag,
                         manufacturing_plan=None,
                         manufacturing_eligible=True):
    gems = [_mesh_from_placement(p) for p in state.placements]
    diag = _strategy_diagnostics(state, rough, ctx)
    diag.update(base_diag)
    return {
        "strategy": label,
        "shape": shape_label,
        "gems": gems,
        "total_volume": float(sum(p.volume for p in state.placements)),
        "diagnostics": diag,
        "manufacturing_plan": manufacturing_plan,
        "manufacturing_eligible": bool(manufacturing_eligible),
        "placements": [
            {
                "shape": p.name,
                "center": [round(float(v), 5) for v in p.pos],
                "scale": round(float(p.scale), 5),
                "volume": round(float(p.volume), 6),
                "surface_clearance": round(float(p.surface_clearance), 6),
            }
            for p in state.placements
        ],
    }


def _strategy_diagnostics(state, rough, ctx):
    if state.placements:
        all_bounds = []
        for p in state.placements:
            verts = p.variant.verts * p.scale + p.pos
            all_bounds.append([verts.min(0), verts.max(0)])
        all_bounds = np.asarray(all_bounds)
        cmin = all_bounds[:, 0, :].min(0)
        cmax = all_bounds[:, 1, :].max(0)
    else:
        cmin = cmax = np.zeros(3)

    rmin, rmax = rough.bounds
    rough_ext = np.maximum(rmax - rmin, 1e-9)
    cut_ext = cmax - cmin
    usable_voxels = int(np.count_nonzero(ctx.rough_mask & ~ctx.no_cut_mask))
    exact_voxels = len(set().union(*(p.occ_set for p in state.placements))) \
        if state.placements else 0
    collision_voxels = len(state.occupied)
    usable_volume = usable_voxels * (ctx.pitch ** 3)
    occupied_volume = float(state.volume)
    free_summary, _ = _component_summary(ctx, state.occupied)
    min_gap = _state_min_surface_gap(state.placements)
    mm_per_mesh = float(ctx.fit.get("mm_per_mesh_unit", 0.0) or 0.0)
    protected_corridor_mesh = float(ctx.fit.get(
        "protected_corridor_mesh_units",
        ctx.fit["gem_spacing_buffer"],
    ))
    protected_corridor_mm = ctx.fit.get("protected_corridor_mm")
    voxelized_corridor_mesh = float(ctx.fit.get(
        "voxelized_protected_corridor_mesh_units",
        protected_corridor_mesh,
    ))
    voxelized_corridor_mm = ctx.fit.get("voxelized_protected_corridor_mm")
    clearance_model = {
        "blade_kerf_mm": round(float(ctx.fit.get("blade_kerf_mm", 0.0)), 3),
        "preform_margin_mm": (
            round(float(ctx.fit["preform_margin_mm"]), 3)
            if ctx.fit.get("preform_margin_mm") is not None else None
        ),
        "protected_corridor_mesh_units": round(protected_corridor_mesh, 6),
        "protected_corridor_mm": (
            round(float(protected_corridor_mm), 3)
            if protected_corridor_mm is not None else None
        ),
        "per_gem_collision_radius_mesh_units": round(
            float(ctx.fit.get("collision_radius_mesh_units", protected_corridor_mesh * 0.5)),
            6,
        ),
        "per_gem_collision_radius_voxels": int(_spacing_voxels(ctx)),
        "voxelized_protected_corridor_mesh_units": round(voxelized_corridor_mesh, 6),
        "voxelized_protected_corridor_mm": (
            round(float(voxelized_corridor_mm), 3)
            if voxelized_corridor_mm is not None else None
        ),
        "actual_exported_gap_mm": (
            round(float(min_gap * mm_per_mesh), 3)
            if min_gap is not None and mm_per_mesh > 0 else None
        ),
    }

    return {
        "space_utilization": {
            "occupied_percent": round(100 * occupied_volume / max(usable_volume, 1e-9), 1),
            "axis_percent": [round(float(v), 1) for v in (cut_ext / rough_ext * 100)],
            "occupied_volume_mesh_units": round(float(occupied_volume), 6),
            "free_volume_mesh_units": round(float(max(usable_volume - occupied_volume, 0)), 6),
            "usable_volume_mesh_units": round(float(usable_volume), 6),
            "occupied_voxel_count": exact_voxels,
            "collision_voxel_count": collision_voxels,
        },
        "free_space_components": free_summary,
        "clearance_model": clearance_model,
        "blade_gap": {
            "target_gap_mesh_units": round(protected_corridor_mesh, 6),
            "target_gap_mm": clearance_model["protected_corridor_mm"],
            "blade_kerf_mm": clearance_model["blade_kerf_mm"],
            "preform_margin_mm": clearance_model["preform_margin_mm"],
            "voxelized_target_gap_mm": clearance_model[
                "voxelized_protected_corridor_mm"
            ],
            "estimated_min_gap_mesh_units": (
                round(float(min_gap), 6) if min_gap is not None else None
            ),
            "estimated_min_gap_mm": (
                round(float(min_gap * mm_per_mesh), 3)
                if min_gap is not None and mm_per_mesh > 0 else None
            ),
        },
        "axis_margins": {
            "min": [round(float(v), 5) for v in (cmin - rmin)],
            "max": [round(float(v), 5) for v in (rmax - cmax)],
        },
        "gem_count": len(state.placements),
    }


def _fallback_strategy(rough, shapes, preferred_shape, ctx, base_diag):
    key = preferred_shape if preferred_shape in shapes else next(iter(shapes))
    tmpl = shapes[key].copy()
    tmpl.vertices -= tmpl.bounds.mean(0)
    scale = float(np.min(rough.extents)) * 0.18
    tmpl.apply_scale(scale)
    p = Placement(
        variant=TemplateVariant(
            key=f"{key}:fallback",
            name=f"{key} (Fallback)",
            tmpl=shapes[key],
            rot=np.eye(4),
            verts=np.asarray(shapes[key].vertices),
            surface=_surface_samples(shapes[key]),
            occupancy=_occupancy_samples(shapes[key]),
            volume=float(shapes[key].volume),
        ),
        pos=np.zeros(3),
        scale=scale,
        volume=float(tmpl.volume),
        occ_flat=np.array([], dtype=np.int64),
        occ_set=set(),
        collision_set=set(),
        surface_clearance=0.0,
    )
    return _strategy_from_state("Single Large", f"{key} (Fallback)",
                                BeamState([p], set(), p.volume), rough, ctx, base_diag)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def optimize_cut(rough_mesh_path, mode="multi", preferred_shape=None,
                 job_folder=None, progress_callback=None,
                 blade_kerf_mm=DEFAULT_BLADE_KERF_MM,
                 rough_clearance_mm=DEFAULT_ROUGH_CLEARANCE_MM,
                 mm_per_mesh_unit=None,
                 min_secondary_volume=None,
                 min_secondary_carat=None,
                 carats_per_mesh_volume=None,
                 extra_gem_policy=DEFAULT_EXTRA_GEM_POLICY,
                 max_gems=MAX_GEMS, preform_margin_mm=None,
                 max_cut_depth_mm=None):
    print(f"--- Optimizer | mode={mode} | shape={preferred_shape or 'auto'} ---")
    t0 = time.time()

    try:
        rough = _as_mesh(trimesh.load(rough_mesh_path))
    except Exception as e:
        print(f"   Could not load rough mesh: {e}")
        return []

    if rough is None or len(getattr(rough, "vertices", [])) == 0:
        return []

    if not isinstance(rough, trimesh.Trimesh) or len(rough.faces) == 0:
        rough = trimesh.PointCloud(np.asarray(rough.vertices)).convex_hull
    elif not rough.is_watertight:
        rough.merge_vertices()
        trimesh.repair.fill_holes(rough)
        rough.fix_normals()
        if not rough.is_watertight:
            print("   Mesh not watertight after repair; voxel fill will close gaps.")

    rough.vertices -= rough.bounds.mean(0)
    extents = rough.extents
    center = np.zeros(3)

    shapes = get_standard_shapes()
    if not shapes:
        return []

    fpts = _load_fracture_points(job_folder)
    grid, origin, pitch = _build_sdf_grid(rough)
    max_gems = int(max(1, max_gems or MAX_GEMS))
    fit = _fit_settings(
        extents,
        pitch,
        blade_kerf_mm=blade_kerf_mm,
        rough_clearance_mm=rough_clearance_mm,
        mm_per_mesh_unit=mm_per_mesh_unit,
        min_secondary_volume=min_secondary_volume,
        min_secondary_carat=min_secondary_carat,
        carats_per_mesh_volume=carats_per_mesh_volume,
        extra_gem_policy=extra_gem_policy,
        max_gems=max_gems,
        preform_margin_mm=preform_margin_mm,
        max_cut_depth_mm=max_cut_depth_mm,
    )
    rough_mask = grid > -fit["voxel_tolerance"]
    no_cut_mask = _make_no_cut_mask(grid.shape, origin, pitch, fpts,
                                    fit["fracture_safe_dist"])
    ctx = FitContext(grid, origin, pitch, rough_mask, no_cut_mask, fit)

    usable_volume = float(np.count_nonzero(rough_mask & ~no_cut_mask) * pitch ** 3)
    no_cut_volume = float(np.count_nonzero(no_cut_mask & rough_mask) * pitch ** 3)
    base_diag = {
        "optimizer_version": "voxel_beam_v11_cuttable_space_recovery",
        "fit": {k: _diagnostic_value(v) for k, v in fit.items()},
        "sdf_pitch": round(float(pitch), 6),
        "sdf_shape": list(grid.shape),
        "usable_volume_mesh_units": round(usable_volume, 6),
        "no_cut_volume_mesh_units": round(no_cut_volume, 6),
        "defect_points": int(len(fpts)),
    }

    print(f"   Voxel grid: {grid.shape}, pitch={pitch:.5f}")
    print(f"   Fit margins: wall={fit['min_wall_dist']:.5f}, "
          f"surface={fit['rough_surface_clearance']:.5f}, "
          f"spacing={fit['gem_spacing_buffer']:.5f}, "
          f"defect={fit['fracture_safe_dist']:.5f}")

    search_points = _make_search_points(center, extents, ctx)
    print(f"   Search points: {len(search_points)}")

    candidates, rejected = _generate_candidates(
        rough, shapes, search_points, ctx,
        preferred_shape=preferred_shape,
        progress_callback=progress_callback,
    )
    base_diag["candidate_count"] = len(candidates)
    base_diag["rejected_candidates"] = rejected
    print(f"   Candidate placements: {len(candidates)}")

    settings_present = _cuttable_settings_present(ctx)
    if not candidates:
        strategies = [_fallback_strategy(rough, shapes, preferred_shape, ctx, base_diag)]
    else:
        best_single = max(candidates, key=lambda p: p.volume)
        single_state = _refine_state(BeamState(
            [best_single],
            set(_placement_collision_set(best_single)),
            best_single.volume
        ), rough, ctx)
        single_plan = no_separation_plan(
            fit["blade_kerf_mm"],
            fit.get("preform_margin_mm"),
            fit.get("max_cut_depth_mm"),
            fit.get("mm_per_mesh_unit"),
        )
        strategies = [_strategy_from_state(
            "Single Large",
            single_state.placements[0].name,
            single_state,
            rough,
            ctx,
            base_diag,
            manufacturing_plan=single_plan,
        )]

        if mode == "multi":
            if progress_callback:
                progress_callback(85, 100, "Beam-search geometric comparison")
            geometric_state = _beam_pack(candidates, max_gems)
            if progress_callback:
                progress_callback(92, 100, "Refining packed gems")
            geometric_state = _refine_state(geometric_state, rough, ctx)
            if settings_present:
                # Preserve time for a second, cutability-constrained beam and
                # exact cut-tree verification. The comparison intentionally
                # omits cavity filling and is never manufacturing-eligible.
                gap_diag = {"added": 0, "reason": "reserved_for_cut_tree_search"}
                pocket_diag = {
                    "added": 0,
                    "unused_space_reason": "geometric comparison only",
                }
                final_pocket_diag = {}
                geometric_state, finish_diag = _shrink_state_for_blade_finish(
                    geometric_state, ctx)
                geometric_state, blade_diag = _enforce_blade_clearance(
                    geometric_state, ctx)
                geometric_state, rough_diag = _enforce_rough_clearance(
                    geometric_state, ctx)
                final_blade_diag = blade_diag
                final_rough_diag = rough_diag
            else:
                if progress_callback:
                    progress_callback(95, 100, "Filling initial pockets")
                geometric_state, gap_diag = _gap_fill_state(
                    geometric_state,
                    candidates,
                    search_points,
                    rough,
                    ctx,
                    max_gems,
                )
                if progress_callback:
                    progress_callback(97, 100, "Applying blade clearance")
                geometric_state = _refine_state(geometric_state, rough, ctx)
                geometric_state, finish_diag = _shrink_state_for_blade_finish(
                    geometric_state, ctx)
                geometric_state, blade_diag = _enforce_blade_clearance(
                    geometric_state, ctx)
                if progress_callback:
                    progress_callback(98, 100, "Filling blade-safe cavities")
                geometric_state, pocket_diag = _pocket_fill_state(
                    geometric_state,
                    shapes,
                    candidates,
                    rough,
                    ctx,
                    max_gems,
                )
                geometric_state = _refine_state(geometric_state, rough, ctx)
                geometric_state, rough_diag = _enforce_rough_clearance(
                    geometric_state, ctx)
                geometric_state, final_blade_diag = _enforce_blade_clearance(
                    geometric_state, ctx)
                geometric_state, final_pocket_diag = _pocket_fill_state(
                    geometric_state,
                    shapes,
                    candidates,
                    rough,
                    ctx,
                    max_gems,
                )
                geometric_state = _refine_state(geometric_state, rough, ctx)
                geometric_state, final_rough_diag = _enforce_rough_clearance(
                    geometric_state, ctx)
                geometric_state, final_blade_diag = _enforce_blade_clearance(
                    geometric_state, ctx)

            multi_diag = dict(base_diag)
            multi_diag["gap_fill"] = gap_diag
            multi_diag["blade_finish"] = finish_diag
            multi_diag["blade_clearance"] = final_blade_diag
            multi_diag["blade_clearance_initial"] = blade_diag
            multi_diag["rough_clearance"] = final_rough_diag
            multi_diag["rough_clearance_initial"] = rough_diag
            multi_diag["pocket_fill"] = _merge_pocket_diagnostics(
                pocket_diag,
                final_pocket_diag,
            )

            if settings_present:
                if progress_callback:
                    progress_callback(98, 100, "Searching cuttable packing states")
                global_frontier = _beam_pack_frontier(
                    candidates,
                    max_gems,
                    state_filter=_quick_cuttable_filter(ctx),
                    limit=CUTTABLE_FRONTIER_LIMIT,
                    deadline=t0 + CUTTABLE_GLOBAL_DEADLINE_SECONDS,
                )
                global_frontier = [
                    state for state in global_frontier if state.placements
                ]
                baseline_state = (
                    global_frontier[0]
                    if global_frontier else _beam_pack(
                        candidates,
                        max_gems,
                        state_filter=_quick_cuttable_filter(ctx),
                    )
                )
                baseline_result, baseline_verify_diag = (
                    _finalize_cuttable_frontier(
                        global_frontier or [baseline_state],
                        rough,
                        ctx,
                        fpts,
                        t0 + 68.0,
                        verification_limit=CUTTABLE_FRONTIER_LIMIT,
                    )
                )
                if baseline_result is not None:
                    baseline_state = baseline_result["state"]

                if progress_callback:
                    progress_callback(98, 100, "Filling preserved cuttable layout")
                preserve_states, preserve_diag = _cuttable_cavity_frontier(
                    [baseline_state],
                    shapes,
                    candidates,
                    rough,
                    ctx,
                    max_gems,
                    t0 + 80.0,
                    "preserve_fill",
                    structural_plan=(
                        baseline_result["plan"]
                        if baseline_result is not None else None
                    ),
                )

                if progress_callback:
                    progress_callback(99, 100, "Repacking cuttable free-space frontier")
                repacked_seed_states = _unique_states([
                    *(preserve_states or []),
                    *(global_frontier or []),
                    baseline_state,
                ])
                repacked_seed_states = [
                    state for state in repacked_seed_states if state.placements
                ]
                repacked_states, repacked_diag = _cuttable_cavity_frontier(
                    repacked_seed_states or [baseline_state],
                    shapes,
                    candidates,
                    rough,
                    ctx,
                    max_gems,
                    t0 + CUTTABLE_CAVITY_DEADLINE_SECONDS,
                    "repacked_cuttable",
                )

                if progress_callback:
                    progress_callback(99, 100, "Verifying full-through cut tree")
                preserve_result, preserve_verify_diag = _finalize_cuttable_frontier(
                    preserve_states or [baseline_state],
                    rough,
                    ctx,
                    fpts,
                    t0 + CUTTABLE_TOTAL_DEADLINE_SECONDS,
                    verification_limit=3,
                    structural_plan=(
                        baseline_result["plan"]
                        if baseline_result is not None else None
                    ),
                    refine_states=False,
                )
                repacked_result, repacked_verify_diag = _finalize_cuttable_frontier(
                    repacked_states or global_frontier or [baseline_state],
                    rough,
                    ctx,
                    fpts,
                    t0 + CUTTABLE_TOTAL_DEADLINE_SECONDS,
                    verification_limit=6,
                    refine_states=False,
                )
                if preserve_result is None and baseline_result is not None:
                    preserve_result = baseline_result
                    preserve_verify_diag["baseline_fallback_used"] = True

                verified_states = []
                if preserve_result is not None:
                    preserve_state = preserve_result["state"]
                    preserve_plan = preserve_result["plan"]
                    if progress_callback:
                        progress_callback(
                            99, 100, "Repacking leftover space in cut pieces")
                    preserve_state, preserve_plan, leaf_repack_diag = (
                        _repack_leaf_pieces(
                            preserve_state, preserve_plan, rough, ctx,
                            shapes, fpts, t0 + CUTTABLE_TOTAL_DEADLINE_SECONDS,
                            preferred_shape=preferred_shape,
                        )
                    )
                    preserve_diag["verification"] = preserve_verify_diag
                    preserve_diag["baseline_verification"] = baseline_verify_diag
                    preserve_diag["leaf_repack"] = leaf_repack_diag
                    preserve_strategy_diag = dict(base_diag)
                    preserve_strategy_diag["blade_clearance"] = preserve_result[
                        "blade_final"
                    ]
                    preserve_strategy_diag["rough_clearance"] = preserve_result[
                        "rough_clearance"
                    ]
                    preserve_strategy_diag["preserve_fill"] = preserve_diag
                    preserve_strategy_diag["baseline_comparison"] = (
                        _baseline_comparison(preserve_state, baseline_state, rough)
                    )
                    preserve_strategy_diag["remaining_free_space"] = (
                        _remaining_space_diagnostics(
                            preserve_state,
                            ctx,
                            preserve_diag,
                        )
                    )
                    preserve_strategy_diag["manufacturing_plan"] = (
                        preserve_plan.get("diagnostics", {})
                    )
                    strategies.append(_strategy_from_state(
                        "Preserve + Fill",
                        f"{len(preserve_state.placements)} gems (preserved fill)",
                        preserve_state,
                        rough,
                        ctx,
                        preserve_strategy_diag,
                        manufacturing_plan=preserve_plan,
                        manufacturing_eligible=True,
                    ))
                    verified_states.append(preserve_state)

                if repacked_result is not None:
                    repacked_state = repacked_result["state"]
                    repacked_plan = repacked_result["plan"]
                    if progress_callback:
                        progress_callback(
                            99, 100, "Repacking leftover space in cut pieces")
                    repacked_state, repacked_plan, repacked_leaf_diag = (
                        _repack_leaf_pieces(
                            repacked_state, repacked_plan, rough, ctx,
                            shapes, fpts, t0 + CUTTABLE_TOTAL_DEADLINE_SECONDS,
                            preferred_shape=preferred_shape,
                        )
                    )
                    repacked_diag["verification"] = repacked_verify_diag
                    repacked_diag["leaf_repack"] = repacked_leaf_diag
                    repacked_strategy_diag = dict(base_diag)
                    repacked_strategy_diag["blade_clearance"] = repacked_result[
                        "blade_final"
                    ]
                    repacked_strategy_diag["rough_clearance"] = repacked_result[
                        "rough_clearance"
                    ]
                    repacked_strategy_diag["repacked_search"] = repacked_diag
                    repacked_strategy_diag["baseline_comparison"] = (
                        _baseline_comparison(repacked_state, baseline_state, rough)
                    )
                    repacked_strategy_diag["remaining_free_space"] = (
                        _remaining_space_diagnostics(
                            repacked_state,
                            ctx,
                            repacked_diag,
                        )
                    )
                    repacked_strategy_diag["manufacturing_plan"] = (
                        repacked_plan.get("diagnostics", {})
                    )
                    strategies.append(_strategy_from_state(
                        "Repacked Cuttable Plan",
                        f"{len(repacked_state.placements)} gems (repacked)",
                        repacked_state,
                        rough,
                        ctx,
                        repacked_strategy_diag,
                        manufacturing_plan=repacked_plan,
                        manufacturing_eligible=True,
                    ))
                    verified_states.append(repacked_state)

                comparison_state = max(
                    [geometric_state, baseline_state, *verified_states],
                    key=lambda state: state.volume,
                )
                if len(comparison_state.placements) > 1:
                    strategies.append(_strategy_from_state(
                        "Geometric Comparison",
                        f"{len(comparison_state.placements)} gems (unconstrained)",
                        comparison_state,
                        rough,
                        ctx,
                        multi_diag,
                        manufacturing_plan=_geometric_comparison_plan(
                            ctx, len(comparison_state.placements)),
                        manufacturing_eligible=False,
                    ))
            elif len(geometric_state.placements) > 1:
                strategies.append(_strategy_from_state(
                    "Multi-Gem",
                    f"{len(geometric_state.placements)} gems (mixed)",
                    geometric_state,
                    rough,
                    ctx,
                    multi_diag,
                    manufacturing_plan=settings_required_plan(
                        fit["blade_kerf_mm"], fit.get("mm_per_mesh_unit")),
                ))

    for strategy in strategies:
        if strategy.get("manufacturing_plan") is None:
            if len(strategy.get("gems", [])) <= 1:
                strategy["manufacturing_plan"] = no_separation_plan(
                    fit["blade_kerf_mm"],
                    fit.get("preform_margin_mm"),
                    fit.get("max_cut_depth_mm"),
                    fit.get("mm_per_mesh_unit"),
                )
            else:
                strategy["manufacturing_plan"] = settings_required_plan(
                    fit["blade_kerf_mm"], fit.get("mm_per_mesh_unit"))

    # Ranking/selection of "the" winning strategy — and exporting the file
    # that the 3D viewer/report treat as the result — is owned entirely by
    # yield_calculator.calculate_gem_stats() (it already re-sorts every
    # strategy returned here into `options_data`). Keeping a second,
    # independently tie-broken ranking here previously let this function
    # export a different "best_cut.ply" than whichever strategy
    # yield_calculator picked as options[0], so the viewer and the report
    # could silently describe two different cut plans. Order here is not
    # relied upon downstream.
    elapsed = time.time() - t0
    for strategy in strategies:
        diag = strategy.setdefault("diagnostics", {})
        diag["runtime_seconds"] = round(float(elapsed), 2)
        diag["runtime_target_seconds"] = 120
        diag["meets_runtime_target"] = elapsed <= 120.0

    print(f"   Generated {len(strategies)} candidate strateg"
          f"{'y' if len(strategies) == 1 else 'ies'}.")
    print(f"   Total optimiser time: {elapsed:.1f}s")
    return strategies
