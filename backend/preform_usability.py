"""Geometric preform eligibility, separate from conserved physical stock.

This is an automatic geometric screen, not workshop/optical certification.
The two-cell dimension rule is a disclosed resolution check, not an industry
handling-size threshold. Neither virtual inset nor preform allowance removes mass.
"""
from __future__ import annotations

import math

import numpy as np
from preform_features import neck_candidates

MODEL_VERSION = "v2_usable_preform"


def physical_dimensions(stock):
    """PCA-aligned physical widths, including exact clipped cell vertices."""
    centres = stock.centres[stock.active]
    if not len(centres):
        return np.zeros(3)
    axes = np.eye(3)
    if len(centres) > 2:
        _, axes = np.linalg.eigh(np.cov(centres.T))
    full = np.array([i for i in stock.active if int(i) not in stock.fragments], dtype=int)
    lows, highs = [], []
    if len(full):
        projected = stock.centres[full] @ axes
        radius = stock.pitch / 2 * np.abs(axes).sum(axis=0)
        lows.append(projected.min(axis=0) - radius)
        highs.append(projected.max(axis=0) + radius)
    if stock.fragments:
        vertices = np.vstack([v[0] for v in stock.fragments.values()]) @ axes
        lows.append(vertices.min(axis=0))
        highs.append(vertices.max(axis=0))
    return np.sort(np.max(highs, axis=0) - np.min(lows, axis=0))


def validate_region(piece, cfg, scale, morphology_fn, *, manufacturing_valid,
                    minimum_weight=0.0):
    """Validate one entire physical leaf; never silently keep only its core."""
    stock = piece["mesh"]
    weight = float(piece["weight_ct"])
    reasons, rejected = [], []
    components = stock.connected_components()
    dimensions = physical_dimensions(stock) * scale
    points = stock.centres[stock.active]
    kind, metrics = morphology_fn(points)
    resolution_width = 2 * stock.pitch * scale
    features = neck_candidates(stock, cfg["min_secondary_carat"],
                               weight * stock.pitch**3 / max(stock.volume,1e-30), maximum=12)
    strong_necks = [f for f in features if f["strong"]]
    tol = stock.pitch * scale * 1e-8
    resolved = bool(len(dimensions) == 3 and dimensions[0] + tol >= resolution_width)
    reachable = bool(len(dimensions) == 3 and dimensions[0] <= cfg["max_cut_depth_mm"] + tol)
    if not math.isfinite(weight) or weight <= 0 or stock.volume <= 0:
        status = "numerical_debris"
        rejected.append("nonpositive_or_invalid_physical_volume")
    elif piece["confirmed_defect_loss_ct"] > 0:
        status = "needs_further_separation"
        rejected.append("confirmed_defect_in_physical_piece")
    elif weight < minimum_weight or piece.get("discard_reason") == "below_minimum_secondary_mass":
        status = "too_small"
        rejected.append("below_configured_minimum_secondary_mass")
    elif len(components) != 1:
        status = "requires_separation"
        rejected.append("multiple_face_connected_physical_components")
    elif not resolved:
        status = "numerical_debris"
        rejected.append("thickness_not_resolved_across_two_voxel_widths")
    elif not reachable:
        status = "manufacturing_invalid"
        rejected.append("no_processing_orientation_within_configured_saw_depth")
    elif strong_necks:
        status = "requires_further_separation"
        rejected.append("strong_geometric_neck_requires_partition_test")
    elif not manufacturing_valid:
        status = "requires_separation" if manufacturing_valid is None else "manufacturing_invalid"
        rejected.append("component_not_accepted_as_a_selected_physical_leaf"
                        if manufacturing_valid is None else "no_verified_separation_sequence")
    else:
        status = "irregular_preform" if kind == "irregular" else "usable_preform"
        reasons = [
            "one_canonical_supported_physical_component",
            "positive_physical_mass_and_resolved_three_dimensional_size",
            "processing_width_within_configured_saw_depth",
            "no_confirmed_defect_in_physical_piece",
            "existing_manufacturing_verification_passed",
            "finish_shape_matching_not_required",
        ]
        reasons.append("primary_exempt_from_secondary_mass_minimum"
                       if minimum_weight == 0 else "configured_secondary_mass_minimum_met")
    usable = status in {"usable_preform", "irregular_preform"} and piece["retained"]
    return {
        "usable": usable, "usability_status": status,
        "neck_features": strong_necks,
        "geometry_eligible": bool(weight > 0 and weight >= minimum_weight
                                  and piece["confirmed_defect_loss_ct"] == 0
                                  and len(components) == 1 and resolved and reachable and not strong_necks),
        "usable_preform_weight_ct": weight if usable else 0.0,
        "usability_reasons": reasons, "rejection_reasons": rejected,
        "morphology": kind, "morphology_metrics": metrics,
        "usability_checks": {
            "physical_component_count": len(components),
            "physical_dimensions_mm": dimensions.tolist(),
            "minimum_resolved_width_mm": resolution_width,
            "resolution_rule": "minimum_PCA_bounding_width_at_least_two_voxel_widths",
            "resolution_check_passed": resolved,
            "minimum_weight_ct": minimum_weight,
            "max_processing_depth_mm": cfg["max_cut_depth_mm"],
            "processing_width_check_passed": reachable,
            "manufacturing_verified": bool(manufacturing_valid),
            "workshop_handling_validated": False,
            "optical_or_polished_suitability_validated": False,
        },
    }


def evaluate_usability(physical, cfg, scale, morphology_fn):
    plan = physical["plan"]
    verified = (not plan["sequence"] and plan["status"] == "no_separation_required"
                or (bool(plan["sequence"]) and plan["status"] == "complete"
                    and plan.get("diagnostics", {}).get("exact_sequence_verified") is True))
    retained = [p for p in physical["pieces"].values() if p["retained"]]
    primary = max(retained, key=lambda p: p["weight_ct"]) if retained else None
    for piece in physical["pieces"].values():
        piece["usability"] = validate_region(
            piece, cfg, scale, morphology_fn, manufacturing_valid=verified,
            minimum_weight=0.0 if piece is primary else cfg["min_secondary_carat"])
    usable = math.fsum(p["usability"]["usable_preform_weight_ct"]
                       for p in physical["pieces"].values())
    nonusable = math.fsum(
        p.get("physical_retained_weight_ct", p["weight_ct"])
        for p in retained if not p["usability"]["usable"])
    error = physical["retained"] - math.fsum((usable, nonusable))
    tolerance = physical["balance"]["mass_balance_tolerance_ct"]
    if usable < 0 or nonusable < 0 or abs(error) > tolerance:
        raise ValueError("Usability classification does not reconcile to physical retention.")
    return {
        "usable_preform_weight_ct": usable,
        "nonusable_physical_weight_ct": nonusable,
        "classification_balance_error_ct": error,
        "usable_region_count": sum(p["usability"]["usable"] for p in retained),
        "usable_regions": [p["region_id"] for p in retained if p["usability"]["usable"]],
        "nonusable_regions": [
            {"region_id": p["region_id"], "physical_weight_ct": p["weight_ct"],
             "physical_retained_weight_ct": p.get("physical_retained_weight_ct", p["weight_ct"]),
             "usability_status": p["usability"]["usability_status"],
             "rejection_reasons": p["usability"]["rejection_reasons"]}
            for p in retained if not p["usability"]["usable"]],
    }


def plan_score(physical, usability):
    """Usable mass dominates physical retention, kerf, shape and cut count."""
    tolerance = physical["balance"]["mass_balance_tolerance_ct"]
    return (usability["usable_preform_weight_ct"],
            -round(physical["kerf"]/tolerance),
            round(physical["retained"]/tolerance),
            -len(physical["plan"]["sequence"]),
            float(physical["plan"].get("minimum_envelope_clearance_mm") or 0))
