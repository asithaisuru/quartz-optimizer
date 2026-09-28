"""Healthy physical stock preservation, separate from faceted/preform usability."""
import math


def classify_preservation(physical, cfg, scale, morphology_fn):
    plan = physical["plan"]
    verified = (not plan["sequence"] and plan["status"] == "no_separation_required" or
                plan["status"] == "complete" and plan.get("diagnostics", {}).get("exact_sequence_verified") is True)
    saved, pending, ids, nonusable = 0., 0., [], []
    for piece in physical["pieces"].values():
        healthy = piece["weight_ct"] - piece["confirmed_defect_loss_ct"]
        clean = piece["confirmed_defect_loss_ct"] == 0
        preserved = bool(piece["retained"] and clean and verified and healthy > 0)
        kind, metrics = morphology_fn(piece["mesh"].centres[piece["mesh"].active])
        status = "preserved_clean_preform" if preserved else (
            "needs_further_separation" if piece["retained"] else "explicit_discard")
        piece["usability"] = {
            "usable": preserved, "usability_status": status,
            "usable_preform_weight_ct": healthy if preserved else 0.,
            "usability_reasons": ["healthy_physical_stock_retained_by_verified_plan"] if preserved else [],
            "rejection_reasons": [] if preserved else [status],
            "morphology": kind, "morphology_metrics": metrics,
            "usability_checks": {"manufacturing_verified": bool(verified),
                "finished_gem_suitability_claimed": False, "workshop_handling_validated": False},
        }
        if preserved:
            saved += healthy
            ids.append(piece["region_id"])
        elif piece["retained"]:
            pending += healthy
            nonusable.append({"region_id": piece["region_id"], "usability_status": status,
                              "physical_retained_weight_ct": healthy})
    error = physical["retained"] - saved - pending
    if abs(error) > physical["balance"]["mass_balance_tolerance_ct"]:
        raise ValueError("Preservation classification does not reconcile.")
    return {"usable_preform_weight_ct": saved, "nonusable_physical_weight_ct": pending,
            "classification_balance_error_ct": error, "usable_region_count": len(ids),
            "usable_regions": ids, "nonusable_regions": nonusable}


def preservation_score(physical, classification):
    """Mass first. Piece count and finish shapes never increase this score."""
    return (classification["usable_preform_weight_ct"], 1,
            -physical["kerf"], -physical["discarded"], -len(physical["plan"]["sequence"]))


def recovery_metrics(rough, defects, saved, pending, kerf, discarded):
    values = (rough, defects, saved, pending, kerf, discarded)
    tolerance = max(1e-8, rough*1e-8)
    if not all(math.isfinite(v) and v >= 0 for v in values) or rough <= 0:
        raise ValueError("Preservation masses must be finite and nonnegative.")
    clean = rough-defects
    error = rough-math.fsum((defects, saved, pending, kerf, discarded))
    if clean < 0 or abs(error) > tolerance:
        raise ValueError("Stone preservation mass balance failed.")
    percent = 100*saved/clean if clean > tolerance else None
    met = percent is not None and percent >= 85
    status = "met" if met else "pending_separation" if pending > tolerance else "not_met"
    # Physical retention includes unsafe stock still present in the cut leaves;
    # unsafe material is excluded from the clean denominator, not erased physically.
    retained = rough-kerf-discarded
    return {"rough_weight_ct": rough, "confirmed_defect_excluded_ct": defects,
        "clean_material_available_ct": clean, "saved_clean_material_ct": saved,
        "clean_material_recovery_percent": percent, "target_recovery_percent": 85.,
        "target_source": "expert_defined", "target_met": met, "target_status": status,
        "target_weight_ct": .85*clean, "kerf_loss_ct": kerf, "explicit_discard_ct": discarded,
        "pending_further_separation_ct": pending, "recoverable_pending_separation_weight_ct": pending,
        "physical_retention_ct": retained, "physical_retention_percent": 100*retained/rough,
        "mass_balance_error_ct": error, "mass_balance_valid": abs(error) <= tolerance}


def optimize_preservation(rough, weight, request, snapshot, output, **kwargs):
    from preform_recovery import optimize_preforms
    result = optimize_preforms(rough, weight, {**request, "min_secondary_carat": 0.},
                               snapshot, output, preservation=True, **kwargs)
    saved = result["usable_preform_weight_ct"]
    accounting = result["recovery_accounting"]
    pending = result["unresolved_retained_weight_ct"] + accounting.get("unresolved_weight_ct", 0.)
    metrics = recovery_metrics(weight, result["confirmed_defect_excluded_ct"], saved, pending,
                               result["estimated_kerf_loss_ct"], accounting["explicit_discarded_weight_ct"])
    for region in result["regions"]:
        region["preservation_status"] = region["usability_status"]
        region["healthy_weight_ct"] = region["physical_weight_ct"]-region["confirmed_defect_excluded_ct"]
        region["saved_clean_weight_ct"] = region["healthy_weight_ct"] if region["usable"] else 0.
    result.update(metrics, mode="stone_preservation", optimization_mode="stone_preservation",
        recovery_basis="saved_clean_material_over_rough_minus_confirmed_defects",
        target_applicable=True, runtime=result["diagnostics"]["runtime_seconds"],
        runtime_seconds=result["diagnostics"]["runtime_seconds"],
        message={"met": "Expert-defined clean-material recovery target met.",
                 "pending_separation": "Healthy material remains retained and needs further separation; the target is pending.",
                 "not_met": "The selected preservation plan did not meet the expert-defined target."}[metrics["target_status"]],
        limitations=["Saved stock is not finished or polished gemstone yield.",
            "Confirmed exclusion uses the existing conservative calibrated voxel safety model.",
            "Search is bounded; pending healthy material is retained, not waste.",
            "Physical cut verification is geometric; workshop validation remains required."])
    result.update({key: result["diagnostics"][key] for key in (
        "target_gap_before_closeout_ct", "target_gap_after_closeout_ct", "closeout_attempted",
        "closeout_runtime_seconds", "closeout_states_explored", "closeout_cuts_added", "closeout_termination_reason")})
    return result
