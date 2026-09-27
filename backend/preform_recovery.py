"""Bounded preform-region search, separate from finished faceted-template yield."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import QhullError

from optimizer import _build_sdf_grid
from cut_sequence import _candidate_normals, plan_cut_sequence
from defect_review import confirmed_annotations, finite

SHAPES = {
    "pointed": ["pear", "marquise", "kite_diamond_preform"],
    "elongated": ["marquise", "pear", "emerald"],
    "blocky": ["emerald", "cushion", "princess"],
    "rounded": ["round", "oval", "cushion"],
    "irregular": ["cushion", "oval", "pear"],
}
DEFAULTS = dict(target_recovery_percent=85.0, defect_policy="confirmed_only",
                blade_kerf_mm=0.5, preform_mm=0.5, max_cut_depth_mm=60.0,
                rough_inset_mm=0.8, max_regions=12, min_secondary_carat=0.5)


def settings(value):
    extra = set(value) - set(DEFAULTS) - {"optimization_mode"}
    if extra:
        raise ValueError("Unsupported settings: " + ", ".join(sorted(extra)))
    if value.get("optimization_mode", "preform_recovery") != "preform_recovery":
        raise ValueError("This endpoint accepts preform_recovery only; use existing APIs for legacy_faceted_pack.")
    result = {**DEFAULTS, **{k: v for k, v in value.items() if k in DEFAULTS}}
    if result["defect_policy"] != "confirmed_only":
        raise ValueError("defect_policy must be confirmed_only.")
    finite(result["target_recovery_percent"], "target_recovery_percent", 0, 100)
    for key in ("blade_kerf_mm", "preform_mm", "rough_inset_mm", "min_secondary_carat"):
        finite(result[key], key, 0, 1000)
    if finite(result["max_cut_depth_mm"], "max_cut_depth_mm", 0, 10000) <= 0:
        raise ValueError("max_cut_depth_mm must be positive.")
    if isinstance(result["max_regions"], bool) or not isinstance(result["max_regions"], int) or not 1 <= result["max_regions"] <= 12:
        raise ValueError("max_regions must be an integer from 1 to 12.")
    return result


def recovery_metrics(rough_weight, retained_weight, target=85.0, constrained=False):
    finite(rough_weight, "rough_weight_ct", 0)
    if rough_weight <= 0:
        raise ValueError("rough_weight_ct must be positive.")
    finite(retained_weight, "retained_preform_weight_ct", 0, rough_weight)
    finite(target, "target_recovery_percent", 0, 100)
    percent = retained_weight / rough_weight * 100.0
    return {
        "rough_weight_ct": rough_weight, "retained_preform_weight_ct": retained_weight,
        "preform_recovery_percent": percent, "target_recovery_percent": target,
        "target_recovery_source": "expert_defined",
        "target_label": "Expert-defined recovery target",
        "target_weight_ct": rough_weight * target / 100.0,
        "target_applicable": not constrained,
        "target_context": "defect_constrained" if constrained else "defect_free",
        "target_met": None if constrained else bool(percent >= target),
    }


def morphology(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 8:
        return "irregular", {"reason": "insufficient_samples"}
    centred = points - points.mean(axis=0)
    values, axes = np.linalg.eigh(np.cov(centred.T))
    local = centred @ axes[:, ::-1]
    extents = np.ptp(local, axis=0)
    aspect = float(extents[0] / max(extents[1], extents[2], 1e-12))
    # Equal-length slabs measure taper without depending on surface triangulation.
    hist, _ = np.histogram(local[:, 0], bins=10)
    taper = float(min(np.mean(hist[:2]), np.mean(hist[-2:])) / max(1, np.mean(hist[4:6])))
    occupancy = len(points)
    try:
        hull = trimesh.convex.convex_hull(points)
        # Primitive extents are in the oriented box's own axes.
        box_volume = float(np.prod(hull.bounding_box_oriented.primitive.extents))
        compactness = float(abs(hull.volume) / max(box_volume, 1e-12))
    except (ValueError, QhullError, ZeroDivisionError):
        compactness = 0.0
    if taper < 0.35 and aspect > 1.15:
        kind = "pointed"
    elif aspect >= 1.8:
        kind = "elongated"
    elif compactness >= 0.82:
        kind = "blocky"
    elif compactness >= 0.42 and aspect < 1.8 and taper >= 0.25:
        kind = "rounded"
    else:
        kind = "irregular"
    return kind, {"aspect_ratio": aspect, "end_to_middle_section_ratio": taper,
                  "convex_hull_to_oriented_box_ratio": compactness,
                  "pca_eigenvalues": values[::-1].tolist(), "sample_count": occupancy}


def safety_mask(points_mm, annotations, padding_mm):
    mask = np.zeros(len(points_mm), dtype=bool)
    for item in annotations:
        geometry = item["geometry"]
        if item["geometry_type"] == "ellipsoid":
            centre = np.asarray(geometry["center_mm"])
            radii = np.asarray(geometry["radii_mm"])
            # Conservative Minkowski expansion, including the cell half diagonal.
            factor = 1.0 + padding_mm / float(np.min(radii))
            mask |= np.sum(((points_mm - centre) / (radii * factor)) ** 2, axis=1) <= 1.0
        else:
            points = np.asarray(geometry["points_mm"])
            radius = geometry["radius_mm"] + padding_mm
            for start, end in zip(points[:-1], points[1:]):
                direction = end - start
                t = np.clip((points_mm - start) @ direction / max(float(direction @ direction), 1e-20), 0, 1)
                distance = points_mm - (start + t[:, None] * direction)
                mask |= np.einsum("ij,ij->i", distance, distance) <= radius * radius
    return mask


def voxel_surface(indices, origin, pitch):
    """Export exposed voxel faces; never fill a concavity with a convex hull."""
    lower = indices.min(axis=0)
    local = indices - lower
    matrix = np.zeros(tuple(local.max(axis=0) + 3), dtype=bool)
    local = local + 1
    matrix[tuple(local.T)] = True
    vertices, faces = [], []
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        for sign in (-1, 1):
            neighbour = local.copy()
            neighbour[:, axis] += sign
            surface = indices[~matrix[tuple(neighbour.T)]]
            centres = origin + surface * pitch
            corners = np.zeros((4, 3))
            corners[:, axis] = sign * 0.5
            corners[:, other] = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
            block = (centres[:, None, :] + corners[None, :, :] * pitch).reshape(-1, 3)
            base = sum(len(v) for v in vertices)
            quad = np.arange(len(surface))[:, None] * 4 + base
            triangles = np.concatenate([quad + [0, 1, 2], quad + [0, 2, 3]], axis=0)
            if sign != (1 if axis != 1 else -1):
                triangles = triangles[:, ::-1]
            faces.append(triangles)
            vertices.append(block)
    mesh = trimesh.Trimesh(vertices=np.concatenate(vertices), faces=np.concatenate(faces), process=True)
    return mesh


def _envelope(points, pitch):
    try:
        vertices = trimesh.convex.convex_hull(points).vertices
    except (QhullError, ValueError, ZeroDivisionError):
        vertices = points
    # Include cell corners so the existing cut verifier protects entire voxels.
    corners = np.array(np.meshgrid(*[[-.5, .5]] * 3)).T.reshape(-1, 3) * pitch
    return trimesh.convex.convex_hull((vertices[:, None, :] + corners).reshape(-1, 3))


def optimize_preforms(rough, rough_weight_ct, request, snapshot, output_dir,
                      *, resolution=56, beam_width=3, candidate_limit=60, time_limit=40.0):
    cfg = settings(request)
    recovery_metrics(rough_weight_ct, 0)
    if not isinstance(rough, trimesh.Trimesh) or not rough.is_watertight or abs(rough.volume) <= 0:
        raise ValueError("Preform recovery requires a watertight canonical rough mesh with positive volume.")
    if not np.isfinite(rough.vertices).all():
        raise ValueError("Rough mesh contains non-finite coordinates.")
    # Same quartz density and carat calibration used by the legacy calculator.
    scale = (rough_weight_ct / (5.0 * 2.65) / abs(float(rough.volume))) ** (1.0 / 3.0) * 10.0
    annotations = confirmed_annotations(snapshot)
    started = time.monotonic()
    grid, origin, pitch = _build_sdf_grid(rough, res=resolution, bias_vox=0.5)
    raw = grid > 0
    original_count = int(raw.sum())
    if original_count == 0:
        raise ValueError("Rough volume could not be resolved.")
    per_cell = rough_weight_ct / original_count
    half_diagonal = np.sqrt(3) * pitch / 2
    usable = grid > cfg["rough_inset_mm"] / scale + half_diagonal
    indices = np.argwhere(usable)
    points = origin + indices * pitch
    raw_points = origin + np.argwhere(raw) * pitch
    blocked_raw = safety_mask(raw_points * scale, annotations, cfg["preform_mm"] + half_diagonal * scale)
    blocked = safety_mask(points * scale, annotations, cfg["preform_mm"] + half_diagonal * scale)
    defect_points = raw_points[blocked_raw]
    diagnostics = {
        "resolution": resolution, "pitch_mesh_units": pitch, "pitch_mm": pitch * scale,
        "mass_basis": "uniform-density voxel fractions calibrated to measured rough weight",
        "original_voxel_count": original_count, "beam_width": beam_width,
        "candidate_limit": candidate_limit, "time_limit_seconds": time_limit,
        "candidates_tested": 0, "manufacturing_rejections": {},
        "rough_inset_loss_ct": (original_count - len(indices)) * per_cell,
        "morphology_limitations": "Geometric advisory classification; no optical or standardized facet design validation.",
    }
    min_count = max(4, int(np.ceil(cfg["min_secondary_carat"] / per_cell)))
    def retained_indices(chunks):
        clean = [i for i, chunk in enumerate(chunks)
                 if len(chunk) >= 4 and not blocked[chunk].any()]
        primary = max(clean, key=lambda i: len(chunks[i])) if clean else None
        return {i for i in clean if i == primary or len(chunks[i]) >= min_count}

    def rank(chunks):
        keep = retained_indices(chunks)
        return (sum(len(chunks[i]) for i in keep), -len(chunks))
    # Dirty leaves remain in the physical cut tree as discarded material.
    # A safety-zone subtraction alone never makes an internal cavity extractable.
    initial = [np.arange(len(points))]
    best = None
    comparisons = []
    seen = set()
    def assess(chunks):
        nonlocal best
        if not chunks or any(len(c) == 0 for c in chunks):
            return
        # An inset or plane can disconnect a leaf. Such pieces must be distinct
        # physical leaves and pass the multi-region verifier.
        connected = []
        for chunk in chunks:
            local = indices[chunk] - indices[chunk].min(axis=0)
            occupied = np.zeros(tuple(local.max(axis=0) + 1), dtype=bool)
            occupied[tuple(local.T)] = True
            labels, count = ndimage.label(occupied)
            membership = labels[tuple(local.T)]
            connected.extend(chunk[membership == i] for i in range(1, count + 1))
        chunks = connected
        if len(chunks) > cfg["max_regions"]:
            rejected = diagnostics["manufacturing_rejections"]
            rejected["region_limit_exceeded"] = rejected.get("region_limit_exceeded", 0) + 1
            return
        score = rank(chunks)
        if score[0] == 0:
            return
        try:
            envelopes = [_envelope(points[c], pitch) for c in chunks]
        except (QhullError, ValueError, ZeroDivisionError):
            return
        remaining = time_limit - (time.monotonic() - started)
        if remaining <= 0:
            return
        keep = retained_indices(chunks)
        plan = plan_cut_sequence(
            rough, envelopes, blade_kerf_mm=cfg["blade_kerf_mm"],
            preform_margin_mm=cfg["preform_mm"], max_cut_depth_mm=cfg["max_cut_depth_mm"],
            mm_per_mesh_unit=scale, pitch=pitch, time_limit_seconds=min(3.0, remaining),
            gem_values=[len(c) * per_cell for c in chunks],
            gem_ids=[f"R{i + 1}" if i in keep else f"W{i + 1}" for i, c in enumerate(chunks)],
            defect_points=defect_points, defect_radius_mesh=half_diagonal,
        )
        if plan["status"] not in {"complete", "no_separation_required"}:
            for reason, count in plan.get("rejection_reasons", {}).items():
                diagnostics["manufacturing_rejections"][reason] = diagnostics["manufacturing_rejections"].get(reason, 0) + count
            comparisons.append({"manufacturing_status": "geometric_comparison_only",
                                "retained_preform_weight_ct": score[0] * per_cell,
                                "reason": plan["status"]})
            return
        diagnostics["verified_plan_count"] = diagnostics.get("verified_plan_count", 0) + 1
        # Mass, fewer cuts, clearance, then compactness are ordered tie-breakers.
        compactness = sum(len(chunks[i]) * pitch ** 3 / max(abs(envelopes[i].volume), 1e-12)
                          for i in keep) / max(1, len(keep))
        score = (*score, plan.get("minimum_envelope_clearance_mm", 0), compactness)
        if best is None or score > best[0]:
            best = (score, chunks, plan)

    assess(initial)
    labels, count = ndimage.label(usable)
    components = [np.flatnonzero(labels[tuple(indices.T)] == i) for i in range(1, count + 1)]
    components = [c for c in components if len(c) >= 4]
    if 1 < len(components) <= cfg["max_regions"] and sum(map(len, components)) == len(points):
        assess(components)
    frontier = [initial] if len(points) >= 4 else []
    for depth in range(cfg["max_regions"] - 1):
        successors = []
        for chunks in frontier:
            # Split the largest dirty/irregular leaves first, preserving alternatives.
            order = sorted(range(len(chunks)), key=lambda i: (blocked[chunks[i]].any(), len(chunks[i])), reverse=True)
            for index in order[:3]:
                chunk = chunks[index]
                if len(chunk) < 8:
                    continue
                cloud = points[chunk]
                _, axes = np.linalg.eigh(np.cov(cloud.T))
                normals = list(axes[:, ::-1].T)
                normals.extend(_candidate_normals(rough, [])[:3])
                for normal in normals:
                    projection = cloud @ normal
                    histogram, edges = np.histogram(projection, bins=12)
                    neck_bins = [i for i in range(2, 10) if histogram[i] <= min(histogram[i - 1], histogram[i + 1])]
                    offsets = list(np.quantile(projection, [.25, .5, .75]))
                    offsets += [float((edges[i] + edges[i + 1]) / 2) for i in neck_bins[:2]]
                    if blocked[chunk].any():
                        defect_projection = cloud[blocked[chunk]] @ normal
                        margin = (cfg["blade_kerf_mm"] / 2 + cfg["preform_mm"]) / scale + pitch * 2
                        offsets += [float(defect_projection.min() - margin), float(defect_projection.max() + margin)]
                    for offset in offsets:
                        if diagnostics["candidates_tested"] >= candidate_limit or time.monotonic() - started >= time_limit:
                            break
                        corridor = (cfg["blade_kerf_mm"] / 2 + cfg["preform_mm"]) / scale + half_diagonal + pitch * .05
                        left = chunk[projection < offset - corridor]
                        right = chunk[projection > offset + corridor]
                        if min(len(left), len(right)) < 4:
                            continue
                        candidate = chunks[:index] + [left, right] + chunks[index + 1:]
                        key = tuple(sorted((len(c), int(c[0]), int(c[-1])) for c in candidate))
                        if key in seen:
                            continue
                        seen.add(key)
                        diagnostics["candidates_tested"] += 1
                        assess(candidate)
                        successors.append(candidate)
                    if diagnostics["candidates_tested"] >= candidate_limit or time.monotonic() - started >= time_limit:
                        break
                if diagnostics["candidates_tested"] >= candidate_limit or time.monotonic() - started >= time_limit:
                    break
            if diagnostics["candidates_tested"] >= candidate_limit or time.monotonic() - started >= time_limit:
                break
        successors.sort(key=rank, reverse=True)
        frontier = successors[:beam_width]
        if not frontier or diagnostics["candidates_tested"] >= candidate_limit or time.monotonic() - started >= time_limit:
            break

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    regions, cuts = [], []
    retained = 0.0
    kerf_loss = 0.0
    plan = None
    if best:
        _, chunks, plan = best
        cuts = plan["sequence"]
        plan["retained_preform_region_ids"] = [f"R{i + 1}" for i in sorted(retained_indices(chunks))]
        plan["discarded_region_ids"] = [f"W{i + 1}" for i in range(len(chunks))
                                       if i not in retained_indices(chunks)]
        for index, chunk in enumerate(chunks):
            if index not in retained_indices(chunks):
                continue
            region_id = f"R{index + 1}"
            mesh = voxel_surface(indices[chunk], origin, pitch)
            mesh.export(output / (region_id + ".ply"))
            kind, metrics = morphology(points[chunk])
            weight = len(chunk) * per_cell
            retained += weight
            regions.append({
                "region_id": region_id, "retained_weight_ct": weight,
                "volume_mesh_units": len(chunk) * pitch ** 3,
                "morphology": kind, "suggested_finish_shapes": SHAPES[kind],
                "shape_compatibility_score": None, "morphology_metrics": metrics,
                "mesh_file": region_id + ".ply", "confirmed_defects_intersecting": [],
            })
        # Account for actual verifier planes over their own parent pieces, without double counting.
        parents = {"rough_piece_1": raw_points}
        removed = 0.0
        for cut in cuts:
            parent = parents.pop(cut["parent_piece_id"])
            signed = parent @ np.asarray(cut["plane"]["normal"]) - cut["plane"]["offset"]
            half = cfg["blade_kerf_mm"] / scale / 2
            # A thin kerf can pass between voxel centres. Estimate the fraction
            # of each projected cell intersecting the blade slab instead.
            cell_radius = pitch * np.abs(cut["plane"]["normal"]).sum() / 2
            overlap = np.maximum(0, np.minimum(signed + cell_radius, half)
                                 - np.maximum(signed - cell_radius, -half))
            removed += float(np.sum(overlap / (2 * cell_radius)))
            left_id, right_id = cut["result_piece_ids"]
            parents[left_id] = parent[signed < -half]
            parents[right_id] = parent[signed > half]
        kerf_loss = removed * per_cell
    metrics = recovery_metrics(rough_weight_ct, retained, cfg["target_recovery_percent"], bool(blocked_raw.any()))
    limited = diagnostics["candidates_tested"] >= candidate_limit or time.monotonic() - started >= time_limit
    diagnostics.update({
        "runtime_seconds": time.monotonic() - started,
        "resource_limit_reached": limited,
        "kerf_estimation": "fractional projected voxel overlap with verified blade slabs",
        "confirmed_safety_zone_weight_ct": int(blocked_raw.sum()) * per_cell,
        "undersized_or_defect_containing_leaves_ct": sum(len(c) * per_cell for i, c in enumerate(best[1]) if i not in retained_indices(best[1])) if best else 0,
        "remaining_protection_and_discard_loss_ct": max(0.0, rough_weight_ct - retained - kerf_loss),
        "single_region_can_rank_highest": True,
    })
    message = "Highest-ranked recovery plan found within implemented search limits."
    if metrics["target_met"] is False:
        message = "Expert-defined recovery target not reached under current bounded search."
    if not best:
        message += " No manufacturing-valid useful preform plan was found."
    return {
        "mode": "preform_recovery", "optimization_mode": "preform_recovery",
        "recovery_basis": "retained_preform_mass", **metrics,
        "estimated_kerf_loss_ct": kerf_loss,
        "confirmed_defect_excluded_ct": int(blocked_raw.sum()) * per_cell,
        "regions": regions, "cuts": cuts,
        "manufacturing_status": plan["status"] if best else "no_verified_plan",
        "manufacturing_plan": plan,
        "search_state": "resource_limit_reached" if limited else "bounded_search_complete",
        "message": message, "diagnostics": diagnostics,
        "geometric_comparisons": comparisons[:20],
        "confirmed_defect_constraints": annotations,
        "coordinate_frame": {"name": "centered_rough_mesh", "mesh_units": "mesh_units",
                             "annotation_units": "mm", "mm_per_mesh_unit": scale},
        "limitations": [
            "Retained preform mass is not polished final gemstone weight.",
            "Expert-defined recovery target is configurable and is not a universal industry constant.",
            "Defect geometries are expert/manual safety-zone approximations.",
            "Voxel discretization and uniform density approximate retained mass and kerf.",
            "Dirty leaves are discarded conservatively; enclosed defects are not assumed extractable.",
            "Shape suggestions are advisory; kite_diamond_preform is not a standardized facet design.",
            "The cut sequence is operator guidance and requires physical workshop validation.",
        ],
    }
