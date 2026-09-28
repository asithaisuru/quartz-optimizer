"""Extend a verified physical cut tree using only its pending leaves."""
import copy
import math
import os
import time

import numpy as np
from scipy.spatial import QhullError

from cut_sequence import plan_cut_sequence
from preform_defect_candidates import defect_candidates
from preform_features import component_candidates, neck_candidates, plane_key
from preform_material import evaluate_plan, mass_balance
from stone_preservation import classify_preservation
from stone_preservation_parallel import verify_batch


def configured_seconds(value=None):
    seconds = float(os.environ.get("STONE_PRESERVATION_CLOSEOUT_SECONDS", "120")
                    if value is None else value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("STONE_PRESERVATION_CLOSEOUT_SECONDS must be finite and nonnegative.")
    return seconds


def target_gap(clean, saved, target_percent=85.):
    target = clean*target_percent/100.
    return target, max(0., target-saved)


def score(state, target):
    usable = state["usability"]
    saved = usable["usable_preform_weight_ct"]
    return (saved, saved >= target, 1, -state["kerf"], -state["discarded"],
            -len(state["plan"]["sequence"]), -usable["nonusable_physical_weight_ct"])


def protected_tangents(stock, validation_defects, cfg, scale, defect_radius, features):
    """Propose planes at the SAME verifier guard, without stacking extra planning pads.

    This is only candidate generation: the full unchanged manufacturing verifier
    and physical defect-cell intersection check must still approve each cut.
    """
    if not len(validation_defects):
        return []
    _, axes = np.linalg.eigh(np.cov(stock.centres[stock.active].T))
    normals = list(np.eye(3))+list(axes.T)+[f["normal"] for f in features]
    candidates, seen = [], set()
    guard = cfg["blade_kerf_mm"]/scale/2+defect_radius
    for direction in normals:
        normal = np.asarray(direction, dtype=float)
        normal /= np.linalg.norm(normal)
        projected = validation_defects @ normal
        for support, sign in ((projected.min(), -1), (projected.max(), 1)):
            for slack in (.05, .30):
                offset = float(support+sign*(guard+stock.pitch*slack))
                key = plane_key(normal, offset, stock.pitch)
                if key not in seen:
                    seen.add(key)
                    candidates.append({"normal": normal.tolist(), "offset": offset,
                        "rank": 1., "kind": "protected_zone_tangent"})
    return candidates


def _relabel(value, names):
    if isinstance(value, dict):
        return {key: _relabel(child, names) for key, child in value.items()}
    if isinstance(value, list):
        return [_relabel(child, names) for child in value]
    return names.get(value, value) if isinstance(value, str) else value


def append_verified(state, parent_id, local, rough_weight):
    """Splice locally verified children; existing pieces/planes are immutable."""
    local_plan = local["plan"]
    if (local_plan["status"] != "complete" or
            local_plan.get("diagnostics", {}).get("exact_sequence_verified") is not True):
        raise ValueError("Cannot append an unverified local cut.")
    parent = state["pieces"][parent_id]
    if parent["usability"]["usability_status"] != "needs_further_separation":
        raise ValueError("Close-out may only split a pending physical leaf.")
    offset = len(state["plan"]["sequence"])
    next_piece = max(int(k.rsplit("_", 1)[1]) for k in state["pieces"])+1
    next_region = max(int(p["region_id"][1:]) for p in state["pieces"].values())+1
    local_ids = sorted({k for cut in local_plan["sequence"]
                        for k in cut["result_piece_ids"]}, key=lambda k: int(k.rsplit("_", 1)[1]))
    names = {"rough_piece_1": parent_id,
             **{k: f"rough_piece_{next_piece+i}" for i, k in enumerate(local_ids)},
             **{p["region_id"]: f"R{next_region+i}" for i, p in enumerate(local["pieces"].values())}}
    extension = _relabel(local_plan, names)
    for cut in extension["sequence"]:
        cut["step"] += offset
    def number(node):
        if node["type"] == "cut":
            node["step"] += offset
            number(node["left"])
            number(node["right"])
    number(extension["cut_tree"])
    plan = copy.deepcopy(state["plan"] if state["plan"].get("cut_tree") else {
        **local_plan, "sequence": [],
        "cut_tree": {"piece_id": parent_id, "type": "leaf", "gem_ids": [parent["region_id"]]}})
    replaced = []
    def splice(node):
        if node["type"] == "leaf":
            if node["piece_id"] == parent_id:
                replaced.append(parent_id)
                return extension["cut_tree"]
            return node
        node["left"], node["right"] = splice(node["left"]), splice(node["right"])
        return node
    plan["cut_tree"] = splice(plan["cut_tree"])
    if replaced != [parent_id]:
        raise ValueError("Pending physical parent must occur exactly once in the cut tree.")
    plan["sequence"].extend(extension["sequence"])
    plan.update(status="complete", diagnostics={**plan.get("diagnostics", {}),
        "exact_sequence_verified": True, "closeout_local_sequences_verified": True})
    pieces = {k: p for k, p in state["pieces"].items() if k != parent_id}
    for local_id, child in local["pieces"].items():
        new = {**child, "piece_id": names[local_id], "region_id": names[child["region_id"]],
               "parent_piece_id": names[child["parent_piece_id"]],
               "created_by_cut_step": child["created_by_cut_step"]+offset}
        pieces[new["piece_id"]] = new
    child_regions = [names[p["region_id"]] for p in local["pieces"].values()]
    def expand_regions(value):
        if isinstance(value, dict):
            return {key: expand_regions(item) for key, item in value.items()}
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, str) and item == parent["region_id"]:
                    result.extend(child_regions)
                else:
                    result.append(expand_regions(item))
            return result
        return value
    # Rebind earlier cut highlights to the terminal children of their old region.
    # Physical parent/child IDs, planes, order and already clean meshes do not move.
    plan = expand_regions(plan)
    plan["resulting_piece_ids"] = list(pieces)
    plan["retained_gems"] = plan["retained_preform_region_ids"] = [p["region_id"] for p in pieces.values() if p["retained"]]
    plan["discarded_region_ids"] = [p["region_id"] for p in pieces.values() if not p["retained"]]
    plan["gem_count"] = len(pieces)  # Legacy serializer metadata, never a score.
    for name, field, operation in (
        ("minimum_envelope_clearance_mm", "minimum_envelope_clearance_mm", min),
        ("maximum_required_depth_mm", "required_depth_mm", max),
        ("estimated_total_saw_area_mm2", "estimated_saw_area_mm2", sum),
        ("estimated_total_kerf_loss_mm3", "estimated_kerf_loss_mm3", sum)):
        plan[name] = operation(c[field] for c in plan["sequence"])
    plan["reorientation_count"] = len(plan["sequence"])-1
    partitions = _relabel(local["partitions"], names)
    for partition in partitions:
        partition["step"] += offset
    retained = math.fsum(p["physical_retained_weight_ct"] for p in pieces.values())
    saved = math.fsum(p["usability"]["usable_preform_weight_ct"] for p in pieces.values())
    pending = retained-saved
    kerf, discarded = state["kerf"]+local["kerf"], state["discarded"]+local["discarded"]
    balance = mass_balance(rough_weight, retained=retained, defects=state["defects"],
                           kerf=kerf, discarded=discarded)
    if not balance["mass_balance_valid"] or len(pieces) != len(plan["sequence"])+1:
        raise ValueError("Extended physical tree failed mass balance or leaf count.")
    usable_ids = [p["region_id"] for p in pieces.values() if p["usability"]["usable"]]
    return {**state, "pieces": pieces, "plan": plan, "retained": retained,
        "kerf": kerf, "discarded": discarded, "balance": balance,
        "partitions": state["partitions"]+partitions,
        "usability": {"usable_preform_weight_ct": saved, "nonusable_physical_weight_ct": pending,
            "classification_balance_error_ct": retained-saved-pending,
            "usable_region_count": len(usable_ids), "usable_regions": usable_ids,
            "nonusable_regions": [{"region_id": p["region_id"],
                "usability_status": p["usability"]["usability_status"],
                "physical_retained_weight_ct": p["physical_retained_weight_ct"]}
                for p in pieces.values() if p["retained"] and not p["usability"]["usable"]]}}


def close_out(physical, rough_weight, defect_weight, cfg, snapshot, scale,
              cell_weight, defect_radius, validation_defects, safe_raw,
              morphology_fn, envelope_fn, *, seconds=None):
    started = time.monotonic()
    budget = configured_seconds(seconds)
    saved_before = physical["usability"]["usable_preform_weight_ct"] if physical else 0.
    pending_before = physical["usability"]["nonusable_physical_weight_ct"] if physical else rough_weight-defect_weight
    target, gap = target_gap(rough_weight-defect_weight, saved_before, cfg["target_recovery_percent"])
    diagnostic = {"target_weight_ct": target, "target_gap_before_closeout_ct": gap,
        "target_gap_after_closeout_ct": gap, "closeout_attempted": False,
        "closeout_runtime_seconds": 0., "closeout_states_explored": 0,
        "closeout_cuts_added": 0, "closeout_termination_reason": None,
        "closeout_budget_seconds": budget, "saved_before_closeout_ct": saved_before,
        "pending_before_closeout_ct": pending_before,
        "kerf_before_closeout_ct": physical["kerf"] if physical else 0.,
        "cuts_before_closeout": len(physical["plan"]["sequence"]) if physical else 0,
        "closeout_rejections": {}, "closeout_pending_piece_ids_searched": [],
        "closeout_upper_bound_pruned": 0, "closeout_performance": {}}
    # The normal search's region cap limits its global beam. Close-out has a
    # separate finite extension allowance so that cap cannot consume its budget
    # before it starts. This changes search breadth, never machine constraints.
    max_added_cuts = int(os.environ.get("STONE_PRESERVATION_CLOSEOUT_MAX_CUTS", "12"))
    if max_added_cuts < 0:
        raise ValueError("STONE_PRESERVATION_CLOSEOUT_MAX_CUTS must be nonnegative.")
    diagnostic["closeout_max_added_cuts"] = max_added_cuts
    best = physical
    deadline = started+budget
    tolerance = max(1e-8, rough_weight*1e-8)
    def finish(reason):
        diagnostic["closeout_termination_reason"] = reason
        diagnostic["closeout_runtime_seconds"] = time.monotonic()-started
        if best:
            diagnostic["target_gap_after_closeout_ct"] = max(0., target-best["usability"]["usable_preform_weight_ct"])
            diagnostic["closeout_cuts_added"] = len(best["plan"]["sequence"])-diagnostic["cuts_before_closeout"]
        return best, diagnostic
    def reject(reason, count=1):
        diagnostic["closeout_rejections"][reason] = diagnostic["closeout_rejections"].get(reason, 0)+count
    if gap <= tolerance:
        return finish("target_already_met")
    if not physical:
        return finish("no_physical_plan")
    if budget == 0:
        return finish("disabled")
    diagnostic["closeout_attempted"] = True
    frontier = [physical]
    region_limit = False
    while frontier and time.monotonic() < deadline:
        successors = []
        for state in frontier:
            saved = state["usability"]["usable_preform_weight_ct"]
            pending = state["usability"]["nonusable_physical_weight_ct"]
            if saved+pending < target-tolerance:
                diagnostic["closeout_upper_bound_pruned"] += 1
                continue
            if len(state["plan"]["sequence"])-diagnostic["cuts_before_closeout"] >= max_added_cuts:
                region_limit = True
                continue
            leaves = sorted((p for p in state["pieces"].values() if p["retained"] and
                p["usability"]["usability_status"] == "needs_further_separation"),
                key=lambda p: (-p["physical_retained_weight_ct"], p["piece_id"]))
            for parent in leaves:
                if time.monotonic() >= deadline:
                    break
                diagnostic["closeout_pending_piece_ids_searched"].append(parent["piece_id"])
                stock = parent["mesh"]
                points = stock.centres
                features = (defect_candidates(stock, snapshot, scale, cfg)
                    + neck_candidates(stock, 0., cell_weight)
                    + component_candidates(stock, 0., cell_weight))
                features += protected_tangents(stock, validation_defects, cfg, scale, defect_radius, features)
                proposals, seen = [], set()
                for feature in features:
                    normal, offset = np.asarray(feature["normal"]), feature["offset"]
                    key = plane_key(normal, offset, stock.pitch)
                    if key in seen:
                        continue
                    seen.add(key)
                    half = cfg["blade_kerf_mm"]/scale/2
                    if len(validation_defects) and np.any(abs(validation_defects @ normal-offset) <= half+defect_radius):
                        reject("approved_defect_safety_zone")
                        continue
                    projection = points[stock.active] @ normal-offset
                    corridor = half+cfg["preform_mm"]/scale+defect_radius+stock.pitch*.05
                    chunks = [stock.active[projection < -corridor], stock.active[projection > corridor]]
                    if min(map(len, chunks)) < 4:
                        reject("insufficient_child_geometry")
                        continue
                    try:
                        envelopes = [envelope_fn(points[c[safe_raw[c]]] if safe_raw[c].sum() >= 4 else points[c], stock.pitch) for c in chunks]
                    except (ValueError, QhullError, ZeroDivisionError):
                        reject("invalid_validation_envelope")
                        continue
                    dirty_sides = [np.any((parent["defect_points"] @ normal-offset)*side > 0) for side in (-1, 1)]
                    expected = sum(len(c)*cell_weight for c, dirty in zip(chunks, dirty_sides) if not dirty)
                    proposals.append((expected, feature["rank"], normal, envelopes, chunks))
                proposals.sort(key=lambda row: (row[0], row[1]), reverse=True)
                # Small local beam reserves time for opposite-side/recursive cuts.
                for start in range(0, len(proposals), 4):
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        break
                    arguments = [(stock, envelopes, dict(blade_kerf_mm=cfg["blade_kerf_mm"],
                        preform_margin_mm=cfg["preform_mm"], max_cut_depth_mm=cfg["max_cut_depth_mm"],
                        mm_per_mesh_unit=scale, pitch=stock.pitch,
                        time_limit_seconds=min(5., remaining),
                        gem_values=[len(c)*cell_weight for c in chunks], gem_ids=["G1", "G2"],
                        defect_points=validation_defects, defect_radius_mesh=defect_radius,
                        preferred_normals=[normal]))
                        for _, _, normal, envelopes, chunks in proposals[start:start+4]]
                    plans = verify_batch(arguments, diagnostic["closeout_performance"]) if len(arguments) > 1 and remaining > 5 else [None]*len(arguments)
                    improved = False
                    for args, plan in zip(arguments, plans):
                        if time.monotonic() >= deadline:
                            break
                        diagnostic["closeout_states_explored"] += 1
                        if plan is None:
                            plan = plan_cut_sequence(*args[:2], **args[2])
                        if plan["status"] != "complete" or not plan.get("diagnostics", {}).get("exact_sequence_verified"):
                            for reason, count in plan.get("rejection_reasons", {}).items():
                                reject(reason, count)
                            reject(plan["status"])
                            continue
                        try:
                            local = evaluate_plan(stock, plan, parent["weight_ct"], parent["defect_points"],
                                cell_weight, cfg["blade_kerf_mm"]/scale, 0., defect_radius)
                            local["usability"] = classify_preservation(local, cfg, scale, morphology_fn)
                            candidate = append_verified(state, parent["piece_id"], local, rough_weight)
                        except (ValueError, QhullError, ZeroDivisionError) as exc:
                            reject("physical_partition_failed: "+str(exc))
                            continue
                        if score(candidate, target) > score(best, target):
                            best = candidate
                        improved |= candidate["usability"]["usable_preform_weight_ct"] > saved+tolerance
                        if candidate["usability"]["usable_preform_weight_ct"] >= target:
                            best = candidate
                            return finish("target_met")
                        successors.append(candidate)
                    if improved:
                        break  # Spend the next batch on the smaller dirty child.
        successors.sort(key=lambda state: score(state, target), reverse=True)
        frontier = successors[:3]
    return finish("time_limit" if time.monotonic() >= deadline else
                  "closeout_cut_limit" if region_limit else
                  "upper_bound_below_target" if diagnostic["closeout_upper_bound_pruned"] and not diagnostic["closeout_states_explored"] else
                  "candidate_verification_time_limit" if diagnostic["closeout_rejections"].get("timeout") else
                  "no_verified_extension")
