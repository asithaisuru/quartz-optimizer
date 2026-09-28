"""Bounded preform-region search, separate from finished faceted-template yield."""
from __future__ import annotations

import time
import hashlib
import math
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import QhullError

from optimizer import _build_sdf_grid
from cut_sequence import plan_cut_sequence
from defect_review import confirmed_annotations, finite
from preform_material import MODEL_VERSION as PHYSICAL_MODEL_VERSION, VoxelStock, evaluate_plan, mass_balance
from preform_usability import MODEL_VERSION, evaluate_usability, plan_score, validate_region
from preform_topology import consolidate_topology
from preform_features import neck_candidates, plane_key, component_candidates
from preform_defect_candidates import defect_candidates

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
    finite(retained_weight, "retained_preform_weight_ct", 0,
           rough_weight + max(1e-8, rough_weight * 1e-8))
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


def defect_constrained_plan_score(physical, usability):
    """Rank verified plans without treating an unresolved dirty leaf as loss."""
    dirty_weight = math.fsum(
        piece["weight_ct"] for piece in physical["pieces"].values()
        if piece["retained"] and piece["confirmed_defect_loss_ct"] > 0)
    # Adding kerf back measures healthy material preserved from explicit
    # discard. Kerf is compared separately after dirty-leaf isolation.
    healthy_preserved = physical["retained"] + physical["kerf"]
    return (healthy_preserved, -dirty_weight, 1, -physical["kerf"],
            -len(physical["plan"]["sequence"]),
            usability["usable_preform_weight_ct"])


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
                      *, resolution=56, beam_width=3, candidate_limit=60, time_limit=40.0,
                      preservation=False, geometry_base=None, closeout_seconds=None):
    cfg = settings(request)
    recovery_metrics(rough_weight_ct, 0)
    if not isinstance(rough, trimesh.Trimesh) or not rough.is_watertight or abs(rough.volume) <= 0:
        raise ValueError("Preform recovery requires a watertight canonical rough mesh with positive volume.")
    if not np.isfinite(rough.vertices).all():
        raise ValueError("Rough mesh contains non-finite coordinates.")
    scale = (rough_weight_ct / (5.0 * 2.65) / abs(float(rough.volume))) ** (1.0 / 3.0) * 10.0
    annotations = confirmed_annotations(snapshot)
    target_constrained = any(a["type"] in {"fracture", "inclusion"} for a in annotations)
    started = time.monotonic()
    grid, origin, pitch = (geometry_base if geometry_base is not None else
                          _build_sdf_grid(rough, res=resolution, bias_vox=0.5))
    classify = evaluate_usability
    if preservation:
        from stone_preservation import classify_preservation, preservation_score
        classify = classify_preservation
    raw = grid > 0
    original_count = int(raw.sum())
    if original_count == 0:
        raise ValueError("Rough volume could not be resolved.")
    per_cell = rough_weight_ct / original_count
    half_diagonal = np.sqrt(3) * pitch / 2
    raw_indices = np.argwhere(raw)
    raw_points = origin + raw_indices * pitch
    # Physical stock is the complete rough occupancy. This calibration conserves
    # the measured weight regardless of coarse boundary rasterization.
    stock = VoxelStock(raw_indices, origin, pitch)
    reference_volume = abs(float(stock.volume))
    # Virtual placement cores guide candidate generation and the SAME verifier.
    # Their excluded shell remains physical material until an actual operation.
    usable = grid > cfg["rough_inset_mm"] / scale + half_diagonal
    indices = np.argwhere(usable)
    points = origin + indices * pitch
    blocked_raw = safety_mask(raw_points * scale, annotations, half_diagonal * scale)
    validation_blocked_raw = safety_mask(
        raw_points * scale, annotations, cfg["preform_mm"] + half_diagonal * scale)
    blocked = safety_mask(points * scale, annotations, cfg["preform_mm"] + half_diagonal * scale)
    physical_defects = raw_points[blocked_raw]
    defect_points = raw_points[validation_blocked_raw]
    defect_weight = int(blocked_raw.sum()) * per_cell
    virtual_exclusion = (original_count - len(indices)) * per_cell
    topology, raw_components = consolidate_topology(
        rough, stock, scale,
        lambda path: safety_mask(path*scale, annotations, half_diagonal*scale))
    # Physical feature search includes tips with no eroded validation core.
    safe_raw = usable[tuple(raw_indices.T)]
    indices, points, blocked = raw_indices, raw_points, validation_blocked_raw
    trace = {"candidate_planes_generated":0, "duplicate_planes_removed":0,
             "candidates_geometrically_valid":0, "candidates_manufacturing_validated":0,
             "states_explored":0, "states_pruned":0, "maximum_depth_reached":0,
             "termination_reason":None, "best_usable_recovery_progression":[],
             "candidate_kinds":{}, "defect_candidates_generated":0,
             "defect_candidates_verified":0, "generic_candidates_generated":0,
             "evaluation_order":[], "stage_runtime_seconds":{}}
    diagnostics = {
        "resolution": resolution, "pitch_mesh_units": pitch, "pitch_mm": pitch * scale,
        "mass_basis": "complete rough voxel solid calibrated once to measured rough weight",
        "original_voxel_count": original_count, "beam_width": beam_width,
        "candidate_limit": candidate_limit, "time_limit_seconds": time_limit,
        "candidates_tested": 0, "manufacturing_rejections": {},
        "rough_inset_loss_ct": 0.0, "virtual_safety_excluded_ct": virtual_exclusion,
        "physical_reference_volume_mesh_units": reference_volume,
        "canonical_reference_volume_mesh_units": abs(float(rough.volume)),
        "voxel_to_canonical_volume_ratio": reference_volume / abs(float(rough.volume)),
        "accounting_warnings": [], "usability_rejections": {}, "usable_plan_count": 0,
        "topology": topology, "search_trace": trace,
        "morphology_limitations": "Geometric advisory classification; physical utility is not workshop validated.",
    }
    if preservation:
        from optimizer_parallel import configuration
        diagnostics["performance"] = {**configuration(), "optimizer_worker_count": 1,
            "manufacturing_seconds": 0.0}
    initial = [np.arange(len(points))]
    best = None
    uncut_physical = None
    assessment_unusable = set()
    whole_validation = {
        "usable": False, "usability_status": "manufacturing_invalid",
        "usability_reasons": [],
        "rejection_reasons": ["whole_rough_manufacturing_check_not_completed"],
    }
    comparisons = []
    search_deadline = time_limit
    search_stage = "generic"
    seen = set()

    def reject(reason):
        counts = diagnostics["manufacturing_rejections"]
        counts[reason] = counts.get(reason, 0) + 1

    def assess(chunks, uncut=False, preferred_normal=None, preverified=None):
        nonlocal best, uncut_physical, whole_validation, assessment_unusable
        assessment_unusable = set()
        trace["states_explored"] += 1
        if uncut:
            # There is no separation or interior placement to validate. A
            # disconnected *safety* mask must not manufacture a physical cut.
            # The unchanged verifier handles the existing single-stock case.
            envelopes = [rough]
        else:
            if any(not len(chunk) for chunk in chunks):
                return (0.0, -len(chunks))
            # One envelope protects each proposed physical leaf, including
            # disconnected safety cores. Their convex envelope is conservative.
            # Usability checks physical connectivity AFTER verified partitioning;
            # safety fragmentation itself must not demand extra separation cuts.
            if len(chunks) > cfg["max_regions"]:
                reject("region_limit_exceeded")
                return (0.0, -len(chunks))
            try:
                envelopes = [_envelope(points[c[safe_raw[c]]] if safe_raw[c].sum() >= 4
                                        else points[c], pitch) for c in chunks]
            except (QhullError, ValueError, ZeroDivisionError):
                reject("invalid_validation_envelope")
                return (0.0, -len(chunks))
        remaining = min(time_limit, search_deadline) - (time.monotonic() - started)
        if remaining <= 0:
            return (0.0, -len(chunks))
        verification_started = time.monotonic()
        plan = preverified if preverified is not None else plan_cut_sequence(
            rough, envelopes, blade_kerf_mm=cfg["blade_kerf_mm"],
            preform_margin_mm=cfg["preform_mm"], max_cut_depth_mm=cfg["max_cut_depth_mm"],
            mm_per_mesh_unit=scale, pitch=pitch, time_limit_seconds=min(1.5 if search_stage == "defect" else 3.0, remaining),
            gem_values=([rough_weight_ct] if uncut else [len(c) * per_cell for c in chunks]),
            gem_ids=[f"G{i + 1}" for i in range(len(envelopes))],
            defect_points=defect_points, defect_radius_mesh=half_diagonal,
            preferred_normals=[] if preferred_normal is None else [preferred_normal])
        if preservation and preverified is None:
            diagnostics["performance"]["manufacturing_seconds"] += time.monotonic()-verification_started
        if plan["status"] not in {"complete", "no_separation_required"}:
            for reason, count in plan.get("rejection_reasons", {}).items():
                counts = diagnostics["manufacturing_rejections"]
                counts[reason] = counts.get(reason, 0) + count
            comparisons.append({"manufacturing_status": "geometric_comparison_only",
                                "retained_preform_weight_ct": None, "reason": plan["status"]})
            return (0.0, -len(chunks))
        if not uncut:
            trace["candidates_manufacturing_validated"] += 1
        try:
            physical = evaluate_plan(
                stock, plan, rough_weight_ct, physical_defects, per_cell,
                cfg["blade_kerf_mm"] / scale, cfg["min_secondary_carat"], half_diagonal)
            physical["usability"] = classify(physical, cfg, scale, morphology)
            if uncut:
                whole_validation = validate_region(
                    {"mesh":stock,"weight_ct":rough_weight_ct,"confirmed_defect_loss_ct":defect_weight,
                     "retained":True,"discard_reason":None}, cfg, scale, morphology,
                    manufacturing_valid=True)
                whole_validation["assessment"] = "entire_unmodified_physical_stock_after_canonical_consolidation"
                whole_validation["physical_piece_count"] = len(physical["pieces"])
                if preservation:
                    whole_validation = {**whole_validation, **next(iter(physical["pieces"].values()))["usability"]}
                # No-cut credit belongs only to the entire original physical piece.
                if len(physical["pieces"]) != 1:
                    raise ValueError("Uncut stock must remain one physical piece.")
                if physical["retained"] > 0 or annotations:
                    uncut_physical = physical
        except (ValueError, QhullError, ZeroDivisionError, IndexError):
            reject("physical_partition_failed")
            diagnostics["accounting_warnings"].append(
                "A candidate physical partition failed closed-solid or mass-balance validation.")
            comparisons.append({"manufacturing_status": "geometric_comparison_only",
                                "retained_preform_weight_ct": None,
                                "reason": "physical_partition_failed"})
            return (0.0, -len(chunks))
        if not uncut and search_stage == "defect":
            trace["defect_candidates_verified"] += 1
        diagnostics["verified_plan_count"] = diagnostics.get("verified_plan_count", 0) + 1
        diagnostics["usable_plan_count"] += int(physical["usability"]["usable_preform_weight_ct"] > 0)
        for piece in physical["pieces"].values():
            if not piece["usability"]["usable"]:
                if piece["retained"]:
                    assessment_unusable.update(int(i) for i in piece["mesh"].active)
                for reason in piece["usability"]["rejection_reasons"]:
                    rejected = diagnostics["usability_rejections"]
                    rejected[reason] = rejected.get(reason, 0) + 1
        # Usable mass dominates mere physical retention. A no-cut candidate
        # counts only the explicitly validated entire original physical piece.
        if not plan["sequence"] and not whole_validation["usable"]:
            if physical["usability"]["usable_preform_weight_ct"] != 0:
                raise ValueError("Unusable uncut stock cannot credit candidate lobes.")
        usable_weight = physical["usability"]["usable_preform_weight_ct"]
        if preservation:
            score = preservation_score(physical, physical["usability"])
        elif target_constrained:
            score = defect_constrained_plan_score(physical, physical["usability"])
        else:
            score = plan_score(physical, physical["usability"])
        tolerance = physical["balance"]["mass_balance_tolerance_ct"]
        better = (best is None or score[0] > best[0][0] + tolerance
                  or (abs(score[0] - best[0][0]) <= tolerance
                      and score[1:] > best[0][1:]))
        if usable_weight > 0 and better:
            best = (score, physical)
            trace["best_usable_recovery_progression"].append({
                "state":trace["states_explored"],"usable_weight_ct":usable_weight,
                "usable_recovery_percent":usable_weight/rough_weight_ct*100,
                "cuts":len(plan["sequence"])})
        return score

    assess(initial, uncut=True)
    frontier = [(initial, assessment_unusable)] if len(points) >= 4 else []
    _, rough_axes = np.linalg.eigh(np.cov(points.T))
    phases = ([("defect", cfg["max_regions"] - 1 if preservation else min(2, cfg["max_regions"] - 1),
                time_limit if preservation else time_limit * .65)]
              if annotations else [])
    if not preservation:
        phases.append(("generic", cfg["max_regions"] - 1, time_limit))
    for search_stage, depth_limit, search_deadline in phases:
        phase_started = time.monotonic()
        if search_stage == "generic" and annotations:
            frontier = (frontier + [(initial, set())])[:beam_width]
        for depth in range(depth_limit):
            if not frontier:
                break
            successors = []
            trace["maximum_depth_reached"] = max(
                trace["maximum_depth_reached"], depth + 1)
            for chunks, needs in frontier:
                proposals = []
                for index, chunk in enumerate(chunks):
                    if len(chunk) < 8:
                        continue
                    if search_stage == "defect" and not np.any(blocked[chunk]):
                        continue
                    solid = stock.subset(chunk)
                    priority = sum(int(i) in needs for i in chunk) / len(chunk) if needs else 0
                    if search_stage == "defect":
                        features = defect_candidates(solid, snapshot, scale, cfg, rough_axes=rough_axes.T)
                    else:
                        features = (neck_candidates(solid, cfg["min_secondary_carat"], per_cell)
                                    + component_candidates(solid, cfg["min_secondary_carat"], per_cell))
                    for feature in features:
                        variants = [(np.asarray(feature["normal"]), feature["offset"], feature["kind"])]
                        if feature["strong"]:
                            normal = np.asarray(feature["normal"])
                            tangent = np.cross(normal, np.eye(3)[np.argmin(abs(normal))])
                            tangent /= np.linalg.norm(tangent)
                            for angle in (-np.pi/36, np.pi/36):
                                rotated = normal*np.cos(angle)+tangent*np.sin(angle)
                                variants.append((rotated, float(rotated @ (normal*feature["offset"])), "neck_angular_neighbour"))
                        for normal,offset,kind in variants:
                            trace["candidate_planes_generated"] += 1
                            trace["defect_candidates_generated" if search_stage == "defect"
                                  else "generic_candidates_generated"] += 1
                            signature = hashlib.sha1(chunk.tobytes()).hexdigest()
                            key = (signature, plane_key(normal,offset,pitch))
                            if key in seen:
                                trace["duplicate_planes_removed"] += 1
                                continue
                            seen.add(key)
                            projection = points[chunk] @ normal
                            corridor = (cfg["blade_kerf_mm"]/2+cfg["preform_mm"])/scale + half_diagonal + pitch*.05
                            left, right = chunk[projection < offset-corridor], chunk[projection > offset+corridor]
                            if min(len(left),len(right)) < 4:
                                trace["states_pruned"] += 1
                                continue
                            if len(physical_defects) and np.any(
                                np.abs(physical_defects @ normal-offset) <= cfg["blade_kerf_mm"]/scale/2+half_diagonal):
                                trace["states_pruned"] += 1
                                continue
                            trace["candidates_geometrically_valid"] += 1
                            estimate_kerf = np.count_nonzero(abs(projection-offset) < cfg["blade_kerf_mm"]/scale/2)*per_cell
                            rank = (priority, feature["rank"], -estimate_kerf, min(len(left),len(right)))
                            proposals.append((rank,index,left,right,normal,kind))
                proposals.sort(key=lambda p:p[0], reverse=True)
                # Reserve budget for recursive residual states instead of exhausting
                # every first-level quantile before reaching depth two.
                per_state = (4 if depth == 0 else 2) if search_stage == "defect" else (6 if depth == 0 else 4)
                trace["states_pruned"] += max(0,len(proposals)-per_state)
                chosen = proposals[:min(per_state, max(0, candidate_limit-diagnostics["candidates_tested"]))]
                verified_plans = [None]*len(chosen)
                remaining = search_deadline-(time.monotonic()-started)
                if preservation and len(chosen) > 1 and remaining > 5:
                    from stone_preservation_parallel import verify_batch
                    arguments = []
                    try:
                        for _, index, left, right, normal, kind in chosen:
                            proposed = chunks[:index]+[left,right]+chunks[index+1:]
                            envelopes = [_envelope(points[c[safe_raw[c]]] if safe_raw[c].sum() >= 4 else points[c], pitch) for c in proposed]
                            arguments.append((rough, envelopes, dict(blade_kerf_mm=cfg["blade_kerf_mm"],
                                preform_margin_mm=cfg["preform_mm"], max_cut_depth_mm=cfg["max_cut_depth_mm"],
                                mm_per_mesh_unit=scale, pitch=pitch, time_limit_seconds=min(1.5, remaining),
                                gem_values=[len(c)*per_cell for c in proposed], gem_ids=[f"G{i+1}" for i in range(len(proposed))],
                                defect_points=defect_points, defect_radius_mesh=half_diagonal, preferred_normals=[normal])))
                        verified_plans = verify_batch(arguments, diagnostics.setdefault("performance", {}))
                    except (QhullError, ValueError, ZeroDivisionError):
                        pass  # The ordinary assessment rejects invalid envelopes.
                for proposal_index, (_,index,left,right,normal,kind) in enumerate(chosen):
                    if diagnostics["candidates_tested"] >= candidate_limit or time.monotonic()-started >= search_deadline:
                        break
                    candidate = chunks[:index]+[left,right]+chunks[index+1:]
                    diagnostics["candidates_tested"] += 1
                    trace["candidate_kinds"][kind] = trace["candidate_kinds"].get(kind,0)+1
                    trace["evaluation_order"].append(kind)
                    score = assess(candidate, preferred_normal=normal, preverified=verified_plans[proposal_index])
                    successors.append((score,candidate,assessment_unusable.copy()))
                if diagnostics["candidates_tested"] >= candidate_limit or time.monotonic()-started >= search_deadline:
                    break
            successors.sort(key=lambda item:item[0],reverse=True)
            trace["states_pruned"] += max(0,len(successors)-beam_width)
            frontier = [(candidate,needs) for _,candidate,needs in successors[:beam_width]]
            if diagnostics["candidates_tested"] >= candidate_limit or time.monotonic()-started >= search_deadline:
                break
        trace["stage_runtime_seconds"][search_stage] = time.monotonic() - phase_started
    trace["termination_reason"] = (
        "time_limit" if time.monotonic()-started >= time_limit else
        "candidate_limit" if diagnostics["candidates_tested"] >= candidate_limit else
        "region_depth_limit" if frontier and cfg["max_regions"] > 1 else "queue_exhausted")
    if preservation:
        performance = diagnostics["performance"]
        performance["candidate_count"] = diagnostics["candidates_tested"]
        performance["search_seconds"] = time.monotonic()-started
        performance["manufacturing_seconds"] += performance["parallel_evaluation_seconds"]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    regions, discarded_regions = [], []
    physical = best[1] if best else uncut_physical
    if best is None and annotations:
        # No selected cut means the full original stock remains ONE physical leaf.
        # Confirmed safety mass is an overlapping exclusion, not removed material.
        piece = {
            "piece_id": "rough_piece_1", "region_id": "R1", "mesh": stock,
            "defect_points": physical_defects,
            "weight_ct": rough_weight_ct, "confirmed_defect_loss_ct": defect_weight,
            "retained": True, "discard_reason": None,
            "physical_retained_weight_ct": rough_weight_ct - defect_weight,
            "parent_piece_id": None, "created_by_cut_step": None,
        }
        physical = {
            "pieces": {"rough_piece_1": piece},
            "plan": {"status": "no_verified_plan", "sequence": []},
            "retained": rough_weight_ct - defect_weight, "kerf": 0.0,
            "defects": defect_weight,
            "discarded": 0.0, "partitions": [], "natural_components": [],
            "balance": mass_balance(
                rough_weight_ct, retained=rough_weight_ct - defect_weight,
                defects=defect_weight),
        }
        physical["usability"] = classify(physical, cfg, scale, morphology)
        piece["usability"]["usability_status"] = "needs_further_separation"
        for row in physical["usability"]["nonusable_regions"]:
            row["usability_status"] = "needs_further_separation"
    if preservation:
        from stone_preservation_closeout import close_out
        physical, closeout = close_out(physical, rough_weight_ct, defect_weight,
            cfg, snapshot, scale, per_cell, half_diagonal, defect_points, safe_raw,
            morphology, _envelope, seconds=closeout_seconds)
        diagnostics.update(closeout)
        if physical and physical["usability"]["usable_preform_weight_ct"] > 0:
            best = (preservation_score(physical, physical["usability"]), physical)
    plan = physical["plan"] if best else None
    if physical:
        for piece in physical["pieces"].values():
            public_piece = {
                "region_id": piece["region_id"], "physical_piece_id": piece["piece_id"],
                "physical_weight_ct": piece["weight_ct"],
                "physical_retained_weight_ct": piece.get(
                    "physical_retained_weight_ct", piece["weight_ct"]),
                "canonical_source_component_id": piece.get("canonical_source_component_id"),
                "physical_parent_piece_id": piece.get("parent_piece_id"),
                "created_by_cut_step": piece.get("created_by_cut_step"),
                "credited_to_usable_recovery": piece["usability"]["usable"],
                "separation_required": piece["usability"]["usability_status"] in
                    {"requires_separation", "requires_further_separation",
                     "needs_further_separation"},
                "volume_mesh_units": abs(float(piece["mesh"].volume)),
                "confirmed_defect_excluded_ct": piece["confirmed_defect_loss_ct"],
                "discard_reason": piece["discard_reason"],
                **piece["usability"],
                "suggested_finish_shapes": SHAPES[piece["usability"]["morphology"]],
            }
            # Derive after merging usability so the public flag always follows
            # the final normalized status.
            public_piece["separation_required"] = public_piece["usability_status"] in {
                "requires_separation", "requires_further_separation",
                "needs_further_separation"}
            if not piece["retained"]:
                public_piece["explicit_discarded_weight_ct"] = piece["weight_ct"] - piece["confirmed_defect_loss_ct"]
                discarded_regions.append(public_piece)
                continue
            region_id = piece["region_id"]
            piece["mesh"].export(output / (region_id + ".ply"))
            kind = piece["usability"]["morphology"]
            metrics = piece["usability"]["morphology_metrics"]
            regions.append({
                **public_piece, "retained_weight_ct": piece["usability"]["usable_preform_weight_ct"],
                "morphology": kind, "morphology_metrics": metrics,
                "shape_compatibility_score": None,
                "mesh_file": region_id + ".ply",
                "confirmed_defects_intersecting": (
                    [a.get("id") for a in annotations]
                    if piece["confirmed_defect_loss_ct"] > 0 else []),
                "usefulness_status": "geometric_screen_only_not_workshop_validated",
            })
        retained, kerf_loss = physical["retained"], physical["kerf"]
        defect_loss, explicit_discard = physical["defects"], physical["discarded"]
        physically_discarded_defect = math.fsum(
            p["confirmed_defect_loss_ct"] for p in physical["pieces"].values()
            if not p["retained"])
        unresolved = 0.0
        balance = physical["balance"]
    else:
        # A failed search is not evidence that the healthy physical stock vanished.
        retained = kerf_loss = explicit_discard = physically_discarded_defect = 0.0
        defect_loss = defect_weight
        unresolved = rough_weight_ct - defect_weight
        balance = mass_balance(rough_weight_ct, defects=defect_loss, unresolved=unresolved)
    accounting = {
        "model_version": PHYSICAL_MODEL_VERSION, "original_rough_weight_ct": rough_weight_ct,
        "confirmed_defect_loss_ct": defect_loss, "kerf_loss_ct": kerf_loss,
        "explicit_discarded_weight_ct": explicit_discard, "numerical_loss_ct": 0.0,
        "virtual_safety_excluded_ct": virtual_exclusion,
        "retained_physical_weight_ct": retained, "unresolved_weight_ct": unresolved,
        "partitions": physical["partitions"] if physical else [],
        "natural_component_partitions": physical.get("natural_components",[]) if physical else [], **balance,
    }
    usability = physical["usability"] if physical else {
        "usable_preform_weight_ct": 0.0, "nonusable_physical_weight_ct": 0.0,
        "usable_region_count": 0, "usable_regions": [], "nonusable_regions": [],
        "classification_balance_error_ct": 0.0,
    }
    usable_weight = usability["usable_preform_weight_ct"]
    metrics = recovery_metrics(rough_weight_ct, usable_weight, cfg["target_recovery_percent"], target_constrained)
    usable_accounting = {
        "physical_retained_weight_ct": retained, **usability,
        "usable_preform_recovery_percent": metrics["preform_recovery_percent"],
        "whole_rough_usable": whole_validation["usable"],
        "validation_basis": "geometric_resolution_and_existing_manufacturing_constraints",
        "workshop_validated": False,
    }
    limited = diagnostics["candidates_tested"] >= candidate_limit or time.monotonic() - started >= time_limit
    diagnostics.update({
        "runtime_seconds": time.monotonic() - started, "resource_limit_reached": limited,
        "defect_candidates_generated": trace["defect_candidates_generated"],
        "defect_candidates_verified": trace["defect_candidates_verified"],
        "generic_candidates_generated": trace["generic_candidates_generated"],
        "states_explored": trace["states_explored"],
        "termination_reason": trace["termination_reason"],
        "kerf_estimation": "closed physical blade-slab volume on each verified parent solid",
        "confirmed_safety_zone_weight_ct": defect_weight,
        "validation_defect_zone_weight_ct": int(validation_blocked_raw.sum()) * per_cell,
        "undersized_or_defect_containing_leaves_ct": explicit_discard,
        "remaining_protection_and_discard_loss_ct": explicit_discard + unresolved,
        "single_region_can_rank_highest": True,
    })
    # Inspect every reconstruction component, even one with NO safety core.
    # This is candidate evidence, never an automatic independent usable preform.
    components = raw_components
    largest_component = max(range(len(components)), key=lambda i: len(components[i]))
    component_diagnostics = []
    for index, active in enumerate(components):
        component = stock.subset(active)
        ct = rough_weight_ct * component.volume / reference_volume
        candidate = {"mesh": component, "weight_ct": ct,
                     "confirmed_defect_loss_ct": int(blocked_raw[active].sum()) * per_cell,
                     "retained": True, "discard_reason": None}
        check = validate_region(candidate, cfg, scale, morphology,
                                manufacturing_valid=None,
                                minimum_weight=0.0 if index == largest_component else cfg["min_secondary_carat"])
        component_ids = set(int(i) for i in active)
        contributions = []
        if physical:
            for piece in physical["pieces"].values():
                shared = [int(i) for i in piece["mesh"].active if int(i) in component_ids]
                if not shared:
                    continue
                mass = rough_weight_ct * piece["mesh"].subset(shared).volume / reference_volume
                contributions.append({
                    "region_id": piece["region_id"], "physical_piece_id": piece["piece_id"],
                    "physical_weight_ct": mass,
                    "usable_preform_weight_ct": mass if piece["usability"]["usable"] else 0.0,
                    "usability_status": piece["usability"]["usability_status"],
                    "rejection_reasons": piece["usability"]["rejection_reasons"],
                })
        component_diagnostics.append({
            "component_id": index + 1, "candidate_region_id": f"C{index+1}",
            "entity_type": "candidate_region", "voxel_count": len(active), "physical_weight_ct": ct,
            "physical_piece_ids": [c["physical_piece_id"] for c in contributions],
            "credit_basis": "overlap_with_validated_physical_pieces_only",
            "credited_to_usable_recovery": any(c["usable_preform_weight_ct"] > 0 for c in contributions),
            "separation_required": any(c["usability_status"] in
                {"requires_separation", "requires_further_separation",
                 "needs_further_separation"} for c in contributions),
            "safety_voxel_count": int(usable[tuple(raw_indices[active].T)].sum()),
            "morphology": check["morphology"], "suggested_finish_shapes": SHAPES[check["morphology"]],
            "candidate_geometry_eligible": check["geometry_eligible"],
            "candidate_usability_status": check["usability_status"],
            "candidate_rejection_reasons": check["rejection_reasons"],
            "usability_checks": check["usability_checks"],
            "selected_usable_weight_ct": sum(c["usable_preform_weight_ct"] for c in contributions),
            "selected_piece_contributions": contributions,
        })
    for candidate in component_diagnostics:
        candidate["candidate_status"] = (
            "candidate pointed preform requires separation"
            if candidate["morphology"] == "pointed" and candidate["separation_required"]
            else "requires_separation" if candidate["separation_required"]
            else "candidate_geometry_only")
    # Legacy diagnostic name retained; these are reconstruction components.
    diagnostics["physical_input_components"] = component_diagnostics
    diagnostics["whole_rough_validation"] = whole_validation
    message = "Highest-ranked geometrically usable preform plan found within implemented search limits."
    if not best:
        message = ("No usable preform plan passed the geometric and manufacturing checks; "
                   "retained physical stock remains available for Expert Review and further separation."
                   if annotations else
                   "No usable preform plan passed the geometric and manufacturing checks; "
                   "physical stock accounting is reported separately.")
    elif metrics["target_met"] is False:
        message = "Expert-defined usable-preform recovery target not reached under current bounded search."
    physical_piece_count = len(physical["pieces"]) if physical else 1
    requires_separation = sum(r["physical_retained_weight_ct"] for r in regions
                              if r["separation_required"])
    selected_cuts = plan["sequence"] if plan else []
    if physical_piece_count != len(selected_cuts) + 1:
        raise ValueError("Physical piece count disagrees with selected cut tree.")
    return {
        "physical_piece_count": physical_piece_count,
        "candidate_region_count": len(component_diagnostics),
        "reconstruction_component_count": topology["canonical_component_count"],
        "requires_separation_weight_ct": requires_separation,
        "needs_further_separation_weight_ct": requires_separation,
        "candidate_regions": component_diagnostics,
        "physical_piece_graph": {"root_piece_id":"rough_piece_1", "initial_piece_count":1,
            "partitions": accounting["partitions"],
            "leaf_piece_ids":[p["piece_id"] for p in physical["pieces"].values()] if physical else ["rough_piece_1"]},
        "mode": "preform_recovery", "optimization_mode": "preform_recovery",
        "recovery_basis": "retained_preform_mass", "recovery_model_version": MODEL_VERSION,
        **metrics, "recovery_accounting": accounting,
        "physical_retained_weight_ct": retained,
        "physical_retention_percent": retained / rough_weight_ct * 100,
        "usable_preform_weight_ct": usable_weight,
        "usable_preform_recovery_percent": metrics["preform_recovery_percent"],
        "whole_rough_usable": whole_validation["usable"],
        "whole_rough_validation": whole_validation,
        "stock_partition_basis": "one_original_physical_stock_and_verified_cuts_only",
        "usable_region_count": usability["usable_region_count"],
        "usable_preform_accounting": usable_accounting,
        "estimated_kerf_loss_ct": kerf_loss, "confirmed_defect_excluded_ct": defect_weight,
        "confirmed_defect_excluded_percent": defect_weight / rough_weight_ct * 100,
        "confirmed_defect_physically_discarded_ct": physically_discarded_defect,
        "unresolved_retained_weight_ct": usability["nonusable_physical_weight_ct"],
        "confirmed_exclusion_accounting": "Safety exclusion is separate from healthy retained mass; dirty-piece geometry remains intact until physical isolation",
        "usable_after_defects_ct": rough_weight_ct - defect_weight,
        "recovery_of_usable_percent": (usable_weight / (rough_weight_ct - defect_weight) * 100
                                       if rough_weight_ct > defect_weight else None),
        "regions": regions, "discarded_regions": discarded_regions,
        "cuts": plan["sequence"] if plan else [],
        "manufacturing_status": plan["status"] if plan else "no_verified_plan",
        "manufacturing_plan": plan,
        "search_state": "resource_limit_reached" if limited else "bounded_search_complete",
        "message": message, "diagnostics": diagnostics,
        "geometric_comparisons": comparisons[:20], "confirmed_defect_constraints": annotations,
        "coordinate_frame": {"name": "centered_rough_mesh", "mesh_units": "mesh_units",
                             "annotation_units": "mm", "mm_per_mesh_unit": scale},
        "limitations": [
            "Retained preform mass is not polished final gemstone weight.",
            "Physical retention and geometric usability are separate from workshop usefulness and optical quality.",
            "Two-voxel PCA width is a resolution screen, not an expert-approved handling-thickness standard.",
            "PCA bounding widths do not prove weakest-neck strength, fixturing, or polishing suitability.",
            "Expert-defined recovery target is configurable and is not a universal industry constant.",
            "Filled voxel geometry is approximate; all original occupancy is calibrated to measured mass.",
            "Canonical source ownership uses nearest surface vertices; boundary attribution is approximate.",
            "Reconstruction components are candidate geometry; only verified cuts create physical children.",
            "Mesh-calibrated display scale and coarse voxel geometric volume can differ; inspect volume ratio.",
            "Confirmed geometries exclude conservative voxel safety cells; placement allowances are virtual.",
            "Dirty physical leaves remain retained for further separation; confirmed safety mass is never credited as healthy or usable.",
            "Disconnected physical stock cannot automatically count as one usable preform.",
            "Candidate component eligibility is not a selected manufacturing-valid useful region.",
            "The cut sequence requires physical workshop validation; kite_diamond_preform is advisory.",
        ],
    }
