import trimesh
import numpy as np
import json
import os
import glob
import shutil
from scipy.spatial import cKDTree
from optimizer import (
    DEFAULT_BLADE_KERF_MM,
    DEFAULT_EXTRA_GEM_POLICY,
    DEFAULT_MIN_SECONDARY_CARAT,
    DEFAULT_ROUGH_CLEARANCE_MM,
    MAX_GEMS as DEFAULT_MAX_GEMS,
    optimize_cut,
)
from ray_tracer import calculate_light_performance
from research_completion import (
    build_manufacturing_plan,
    build_research_completion,
)

DENSITY_QUARTZ = 2.65   # g/cm³
CARATS_PER_GRAM = 5.0
DEFAULT_PREFORM_MARGIN_MM = 0.5
DEFAULT_MAX_CUT_DEPTH_MM = 60.0


def _traditional_waste_baseline():
    try:
        return float(os.getenv("TRADITIONAL_WASTE_BASELINE", "35"))
    except (TypeError, ValueError):
        return 35.0


TRADITIONAL_WASTE_BASELINE = _traditional_waste_baseline()


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _optional_float(value, default=None):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value, default=None):
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optimizer_config(overrides=None):
    overrides = overrides or {}
    policy = os.getenv("EXTRA_GEM_POLICY", DEFAULT_EXTRA_GEM_POLICY)
    config = {
        "blade_kerf_mm": _optional_float(
            overrides.get("blade_kerf_mm"),
            _env_float("BLADE_KERF_MM", DEFAULT_BLADE_KERF_MM),
        ),
        "rough_clearance_mm": _optional_float(
            overrides.get("rough_clearance_mm"),
            _env_float("ROUGH_CLEARANCE_MM", DEFAULT_ROUGH_CLEARANCE_MM),
        ),
        "preform_margin_mm": _optional_float(
            overrides.get("preform_margin_mm"),
            DEFAULT_PREFORM_MARGIN_MM,
        ),
        "max_cut_depth_mm": _optional_float(
            overrides.get("max_cut_depth_mm"),
            DEFAULT_MAX_CUT_DEPTH_MM,
        ),
        "extra_gem_policy": overrides.get("extra_gem_policy") or policy,
        "min_secondary_carat": _optional_float(
            overrides.get("min_secondary_carat"),
            _env_float(
                "MIN_SECONDARY_CARAT",
                DEFAULT_MIN_SECONDARY_CARAT,
            ),
        ),
        "max_gems": _optional_int(
            overrides.get("max_gems"),
            _env_int("MAX_GEMS", DEFAULT_MAX_GEMS),
        ),
    }
    config["blade_kerf_mm"] = min(max(config["blade_kerf_mm"], 0.2), 2.0)
    config["rough_clearance_mm"] = min(max(config["rough_clearance_mm"], 0.2), 5.0)
    config["min_secondary_carat"] = min(max(config["min_secondary_carat"], 0.1), 10.0)
    config["max_gems"] = min(max(config["max_gems"], 1), 20)
    if config["extra_gem_policy"] not in {"saleable", "maximum_count"}:
        config["extra_gem_policy"] = DEFAULT_EXTRA_GEM_POLICY
    if config["preform_margin_mm"] is not None:
        if not np.isfinite(config["preform_margin_mm"]):
            raise ValueError("Preform margin must be finite.")
        if not 0 < config["preform_margin_mm"] <= 5:
            raise ValueError("Preform margin must be greater than 0 and at most 5 mm.")
    if config["max_cut_depth_mm"] is not None:
        if not np.isfinite(config["max_cut_depth_mm"]):
            raise ValueError("Maximum cut depth must be finite.")
        if not 0 < config["max_cut_depth_mm"] <= 500:
            raise ValueError("Maximum cut depth must be greater than 0 and at most 500 mm.")
    if (config["preform_margin_mm"] is not None
            and config["rough_clearance_mm"] < config["preform_margin_mm"]):
        raise ValueError("Rough clearance must be at least the preform margin.")
    return config


def _mesh_surface_points(mesh, limit=1800):
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(verts) == 0 or len(faces) == 0:
        return verts

    tris = verts[faces]
    centers = tris.mean(axis=1)
    edges = np.concatenate([
        (tris[:, 0] + tris[:, 1]) * 0.5,
        (tris[:, 1] + tris[:, 2]) * 0.5,
        (tris[:, 2] + tris[:, 0]) * 0.5,
    ])
    pts = np.vstack([verts, centers, edges])
    if len(pts) > limit:
        stride = max(1, len(pts) // limit)
        pts = pts[::stride][:limit]
    return pts


def _actual_gap_diagnostics(mesh, mm_per_mesh_unit, target_gap_mm):
    try:
        mesh = mesh.copy()
        mesh.merge_vertices()
        parts = mesh.split(only_watertight=False)
    except Exception:
        parts = []

    parts = [p for p in parts if len(getattr(p, "vertices", []))]
    if len(parts) < 2:
        return {
            "actual_min_gap_mesh_units": None,
            "actual_min_gap_mm": None,
            "target_gap_mm": round(float(target_gap_mm), 3),
            "meets_target": True,
        }

    samples = [_mesh_surface_points(part) for part in parts]
    best = float("inf")
    for i in range(len(samples)):
        if len(samples[i]) == 0:
            continue
        tree_i = cKDTree(samples[i])
        for j in range(i + 1, len(samples)):
            if len(samples[j]) == 0:
                continue
            d_ji = tree_i.query(samples[j], k=1)[0].min()
            tree_j = cKDTree(samples[j])
            d_ij = tree_j.query(samples[i], k=1)[0].min()
            best = min(best, float(d_ij), float(d_ji))

    if not np.isfinite(best):
        return {
            "actual_min_gap_mesh_units": None,
            "actual_min_gap_mm": None,
            "target_gap_mm": round(float(target_gap_mm), 3),
            "meets_target": True,
        }

    actual_mm = float(best * mm_per_mesh_unit)
    return {
        "actual_min_gap_mesh_units": round(best, 6),
        "actual_min_gap_mm": round(actual_mm, 3),
        "target_gap_mm": round(float(target_gap_mm), 3),
        "meets_target": actual_mm + max(0.01, float(target_gap_mm) * 0.02) >= float(target_gap_mm),
    }


def _load_defect_summary(job_folder):
    summary = {
        "has_defects": False,
        "point_count": 0,
        "no_cut_point_count": 0,
        "source": "none",
        "evidence_status": "unavailable",
        "claim_boundary": (
            "No accepted defect evidence does not prove a defect-free stone."
        ),
    }
    if not job_folder:
        return summary

    associations_path = os.path.join(
        job_folder, "dense", "defect_associations.json"
    )
    if os.path.exists(associations_path):
        try:
            with open(associations_path, encoding="utf-8") as handle:
                associations = json.load(handle)
            summary.update({
                "has_defects": int(
                    associations.get("mapping_point_count", 0)
                ) > 0,
                "point_count": int(
                    associations.get("mapping_point_count", 0)
                ),
                "no_cut_point_count": int(
                    associations.get("no_cut_point_count", 0)
                ),
                "source": "policy_approved_sparse_associations",
                "evidence_status": associations.get(
                    "status", "unavailable"
                ),
                "mapping_method": associations.get("method"),
            })
        except Exception:
            pass

    policy = _load_defect_detection(job_folder)
    summary["mask_count"] = policy.get("mapping_count", 0)
    summary["no_cut_mask_count"] = policy.get("no_cut_count", 0)
    summary["source_counts"] = policy.get("counts_by_source", {})
    summary["policy"] = policy.get("policy", "unavailable")

    ai_path = os.path.join(job_folder, "ai_results.json")
    if os.path.exists(ai_path):
        try:
            with open(ai_path, encoding="utf-8") as f:
                ai = json.load(f)
            summary["images_scanned"] = ai.get("total_images_scanned", 0)
        except Exception:
            pass
    return summary


def _load_defect_detection(job_folder):
    result = {
        "policy": "unavailable",
        "strict_research_mode": None,
        "taxonomy_version": None,
        "model_sha256": None,
        "raw_prediction_count": 0,
        "visualization_count": 0,
        "mapping_count": 0,
        "no_cut_count": 0,
        "rejected_count": 0,
        "counts_by_class": {},
        "counts_by_source": {},
        "legacy_input_used": False,
        "provisional_input_used": False,
        "claim_status": "unavailable",
        "reason": "No structured detector-policy result was supplied.",
        "warnings": [],
    }
    if not job_folder:
        return result

    path = os.path.join(job_folder, "detections", "policy_summary.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle)
            if isinstance(value, dict):
                result.update(value)
        except (OSError, ValueError) as exc:
            result["warnings"].append(
                f"Detector policy summary could not be read: {exc}"
            )
    elif os.path.isdir(os.path.join(job_folder, "fractures")):
        result["legacy_input_used"] = True
        result["warnings"].append(
            "Legacy mask folder exists without structured policy metadata."
        )

    associations_path = os.path.join(
        job_folder, "dense", "defect_associations.json"
    )
    if os.path.exists(associations_path):
        try:
            with open(associations_path, encoding="utf-8") as handle:
                associations = json.load(handle)
            result["mapped_point_count"] = int(
                associations.get("mapping_point_count", 0)
            )
            result["no_cut_point_count"] = int(
                associations.get("no_cut_point_count", 0)
            )
            result["mapping_method"] = associations.get("method")
        except (OSError, ValueError):
            pass
    return result


def _heuristic_facet_recommendation(mesh, defects):
    candidates = []
    try:
        facets = mesh.facets
        ranked = sorted(
            facets,
            key=lambda face_ids: mesh.area_faces[face_ids].sum(),
            reverse=True
        )[:8]
        for face_ids in ranked:
            candidates.append(mesh.face_normals[face_ids[0]])
            candidates.append(-mesh.face_normals[face_ids[0]])
    except Exception:
        pass

    candidates.extend([
        np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]),
    ])

    center = mesh.bounds.mean(0)
    radius = max(float(np.max(mesh.extents)) / 2.0, 1e-6)
    best = None
    for n in candidates:
        normal = n / max(np.linalg.norm(n), 1e-9)
        visible_estimate = 0
        score = 85.0
        if len(defects):
            rel = defects - center
            depth = rel @ normal
            perpendicular = np.linalg.norm(rel - np.outer(depth, normal), axis=1)
            in_silhouette = perpendicular < radius * 0.72
            table_side = depth > 0
            visible_estimate = int(np.count_nonzero(in_silhouette & table_side))
            visible_ratio = visible_estimate / max(len(defects), 1)
            depth_spread = float(np.std(depth) / radius)
            score = max(0.0, 100.0 - visible_ratio * 75.0 + min(depth_spread, 1.0) * 10.0)
        else:
            visible_estimate = 0
            score = 92.0

        item = {
            "method": "heuristic_defect_visibility",
            "score": round(float(score), 1),
            "normal": [round(float(v), 4) for v in normal],
            "visible_defect_estimate": visible_estimate,
            "reason": "minimizes table-side defect visibility" if len(defects)
                      else "no mapped defects; keep strongest table orientation",
        }
        if best is None or item["score"] > best["score"]:
            best = item
    return best or {
        "method": "heuristic",
        "score": 0,
        "normal": [0, 0, 1],
        "visible_defect_estimate": 0,
        "reason": "no candidate normals",
    }


def _facet_recommendation(cut_path, job_folder=None):
    try:
        mesh = trimesh.load(cut_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        mesh.fix_normals()
    except Exception:
        return {
            "method": "heuristic",
            "score": 0,
            "normal": [0, 0, 1],
            "visible_defect_estimate": 0,
            "reason": "cut mesh unavailable",
        }

    defects = np.empty((0, 3))
    if job_folder:
        defects_path = os.path.join(job_folder, "dense", "defects.ply")
        if os.path.exists(defects_path):
            try:
                defects = np.asarray(trimesh.load(defects_path).vertices)
            except Exception:
                defects = np.empty((0, 3))

    heuristic = _heuristic_facet_recommendation(mesh, defects)
    if len(defects):
        try:
            from facet_ml import recommend_facet_orientation_ml

            ml_recommendation = recommend_facet_orientation_ml(mesh, defects)
            if ml_recommendation:
                ml_recommendation["heuristic_fallback"] = heuristic
                return ml_recommendation
        except Exception:
            pass
    return heuristic


def _build_gem_details(strat, option_index, output_dir, scale_factor,
                       target_weight, vol_original, plan_vol):
    details = []
    placements = strat.get("placements", [])
    mm_scale = float(scale_factor * 10.0)

    for gem_index, gem in enumerate(strat.get("gems", []), start=1):
        placement = (
            placements[gem_index - 1]
            if gem_index - 1 < len(placements)
            else {}
        )
        gem_volume = abs(float(placement.get("volume", 0.0) or gem.volume or 0.0))
        volume_ratio = gem_volume / max(float(vol_original), 1e-9)
        plan_ratio = gem_volume / max(float(plan_vol), 1e-9)
        gem_weight = target_weight * volume_ratio
        bounds_center = gem.bounds.mean(axis=0)
        dims_mm = [round(float(e) * mm_scale, 2) for e in gem.extents]
        center_mm = [round(float(v) * mm_scale, 2) for v in bounds_center]
        center_mesh = [round(float(v), 5) for v in bounds_center]

        gem_filename = f"option_{option_index}_gem_{gem_index}.ply"
        try:
            gem_export = gem.copy()
            gem_export.visual.face_colors = [255, 0, 0, 255]
            gem_export.export(os.path.join(output_dir, gem_filename))
        except Exception:
            gem_filename = None

        details.append({
            "index": gem_index,
            "shape": placement.get("shape") or strat.get("shape", "Gem"),
            "weight_ct": round(float(gem_weight), 2),
            "yield_percent": round(float(volume_ratio * 100), 2),
            "plan_share_percent": round(float(plan_ratio * 100), 1),
            "volume_mesh_units": round(float(gem_volume), 6),
            "dimensions_mm": dims_mm,
            "center_mesh_units": center_mesh,
            "center_mm": center_mm,
            "scale": placement.get("scale"),
            "surface_clearance_mesh_units": placement.get("surface_clearance"),
            "file": gem_filename,
        })

    return details


def _clear_generated_gem_exports(output_dir):
    removed = 0
    for stale_path in glob.glob(os.path.join(output_dir, "option_*_gem_*.ply")):
        try:
            os.remove(stale_path)
            removed += 1
        except OSError:
            pass
    return removed


def calculate_gem_stats(
    mesh_path,
    known_carats=None,
    preferred_shape=None,
    cut_mode="multi",
    job_folder=None,
    progress_callback=None,
    optimizer_settings=None
):
    """
    Parameters
    ----------
    mesh_path       : str   - path to the rough stone PLY (final_textured_model.ply)
    known_carats    : str   - user-provided carat weight (optional)
    preferred_shape : str   - gem shape name chosen by user (None = auto)
    cut_mode        : str   - "single" or "multi"
    job_folder      : str   - job root folder (used to locate defects.ply)
    progress_callback: callable(current, total, message)
    """
    if not os.path.exists(mesh_path):
        return {"error": "Mesh not found"}

    try:
        # --- Load & center rough stone ---
        mesh = trimesh.load(mesh_path)
        mesh.apply_translation(-mesh.bounds.mean(axis=0))

        # Save the centered stone for the 3D viewer
        aligned_path = mesh_path.replace("final_textured_model.ply", "visual_aligned_stone.ply")
        mesh.export(aligned_path)

        vol_mesh = mesh if mesh.is_watertight else mesh.convex_hull
        vol_original = vol_mesh.volume

        # --- Scale calibration ---
        known_weight_supplied = bool(known_carats and float(known_carats) > 0)
        if known_weight_supplied:
            target_weight = float(known_carats)
            target_grams = target_weight / CARATS_PER_GRAM
            target_vol = target_grams / DENSITY_QUARTZ
            scale_factor = (target_vol / vol_original) ** (1 / 3)
        else:
            scale_factor = 2.0 / np.max(vol_mesh.extents)
            target_vol = vol_original * (scale_factor ** 3)
            target_weight = target_vol * DENSITY_QUARTZ * CARATS_PER_GRAM

        mm_per_mesh_unit = float(scale_factor * 10.0)
        opt_config = _optimizer_config(optimizer_settings)
        min_secondary_volume = None
        if known_weight_supplied and target_weight > 0:
            min_secondary_volume = (
                vol_original * opt_config["min_secondary_carat"] / target_weight
            )
        carats_per_mesh_volume = (
            float(target_weight / vol_original)
            if known_weight_supplied and vol_original > 0 else None
        )

        # Export unscaled mesh for optimizer (optimizer works in mesh units)
        temp_opt_path = mesh_path.replace(".ply", "_opt_input.ply")
        mesh.export(temp_opt_path)

        # --- Run optimizer ---
        # Pass preferred_shape, cut_mode, and job_folder so the optimizer can:
        #   a) filter to the user's chosen shape
        #   b) run single or multi-gem packing
        #   c) load defects.ply and avoid fractures
        strategies = optimize_cut(
            temp_opt_path,
            mode=cut_mode,
            preferred_shape=preferred_shape,
            job_folder=job_folder,
            progress_callback=progress_callback,
            blade_kerf_mm=opt_config["blade_kerf_mm"],
            rough_clearance_mm=opt_config["rough_clearance_mm"],
            mm_per_mesh_unit=mm_per_mesh_unit,
            min_secondary_volume=min_secondary_volume,
            min_secondary_carat=(
                opt_config["min_secondary_carat"] if known_weight_supplied else None
            ),
            carats_per_mesh_volume=carats_per_mesh_volume,
            extra_gem_policy=opt_config["extra_gem_policy"],
            max_gems=opt_config["max_gems"],
            preform_margin_mm=opt_config["preform_margin_mm"],
            max_cut_depth_mm=opt_config["max_cut_depth_mm"],
        )

        options_data = []
        output_dir = os.path.dirname(mesh_path)
        _clear_generated_gem_exports(output_dir)
        for i, strat in enumerate(strategies):
            plan_vol = strat['total_volume']
            yield_ratio = plan_vol / vol_original if vol_original > 0 else 0
            plan_weight = target_weight * yield_ratio
            yield_pct = yield_ratio * 100
            diagnostics = dict(strat.get("diagnostics", {}))

            # Save this option's PLY
            opt_filename = f"option_{i}.ply"
            opt_path = os.path.join(output_dir, opt_filename)
            combined = trimesh.util.concatenate(strat['gems'])
            combined.visual.face_colors = [255, 0, 0, 255]
            combined.export(opt_path)
            gem_details = _build_gem_details(
                strat,
                i,
                output_dir,
                scale_factor,
                target_weight,
                vol_original,
                plan_vol,
            )
            protected_corridor_mm = float(opt_config["blade_kerf_mm"])
            if opt_config["preform_margin_mm"] is not None:
                protected_corridor_mm += 2.0 * float(
                    opt_config["preform_margin_mm"]
                )
            gap_diag = _actual_gap_diagnostics(
                combined,
                mm_per_mesh_unit,
                protected_corridor_mm,
            )
            diagnostics.setdefault("blade_clearance", {}).update(gap_diag)
            diagnostics.setdefault("blade_gap", {}).update(gap_diag)
            diagnostics.setdefault("clearance_model", {}).update({
                "blade_kerf_mm": round(float(opt_config["blade_kerf_mm"]), 3),
                "preform_margin_mm": (
                    round(float(opt_config["preform_margin_mm"]), 3)
                    if opt_config["preform_margin_mm"] is not None else None
                ),
                "protected_corridor_mm": round(protected_corridor_mm, 3),
                "actual_exported_gap_mm": gap_diag.get("actual_min_gap_mm"),
                "meets_exported_gap": bool(gap_diag.get("meets_target", True)),
            })
            diagnostics["optimizer_settings"] = {
                "blade_kerf_mm": round(float(opt_config["blade_kerf_mm"]), 3),
                "protected_corridor_mm": round(protected_corridor_mm, 3),
                "rough_clearance_mm": round(float(opt_config["rough_clearance_mm"]), 3),
                "preform_margin_mm": (
                    round(float(opt_config["preform_margin_mm"]), 3)
                    if opt_config["preform_margin_mm"] is not None else None
                ),
                "max_cut_depth_mm": (
                    round(float(opt_config["max_cut_depth_mm"]), 3)
                    if opt_config["max_cut_depth_mm"] is not None else None
                ),
                "extra_gem_policy": opt_config["extra_gem_policy"],
                "min_secondary_carat": round(
                    float(opt_config["min_secondary_carat"]),
                    3,
                ) if known_weight_supplied else None,
                "min_secondary_volume_mesh_units": (
                    round(float(min_secondary_volume), 6)
                    if min_secondary_volume is not None else None
                ),
                "max_gems": int(opt_config["max_gems"]),
                "mm_per_mesh_unit": round(mm_per_mesh_unit, 6),
            }

            # Dimensions of the cut gem (scaled to real mm)
            cut_extents_mm = [
                round(float(e) * scale_factor * 10, 2)
                for e in combined.extents
            ]

            # Light performance
            light = calculate_light_performance(opt_path)
            facet = _facet_recommendation(opt_path, job_folder)
            space_utilization = diagnostics.get("space_utilization", {})
            manufacturing_plan = strat.get("manufacturing_plan")
            manufacturing_eligible = bool(
                strat.get("manufacturing_eligible", True)
            )
            if (len(strat["gems"]) > 1
                    and manufacturing_eligible
                    and not gap_diag.get("meets_target", False)):
                manufacturing_eligible = False
                if isinstance(manufacturing_plan, dict):
                    manufacturing_plan["status"] = "invalid_exported_clearance"
                    manufacturing_plan.setdefault("warnings", []).append(
                        "Exported gem geometry does not meet the protected corridor."
                    )

            options_data.append({
                "name": strat['shape'],
                "type": strat['strategy'],
                "weight": round(plan_weight, 2),
                "yield": round(yield_pct, 1),
                "file": opt_filename,
                "light_score": light['score'],
                "light_grade": light['grade'],
                "cut_dims": cut_extents_mm,
                "space_utilization": space_utilization,
                "optimizer_diagnostics": diagnostics,
                "facet_recommendation": facet,
                "gem_details": gem_details,
                "manufacturing_plan": manufacturing_plan,
                "manufacturing_eligible": manufacturing_eligible,
                # Number of gems in this strategy
                "gem_count": len(strat['gems'])
            })

        geometric_options = [
            option for option in options_data
            if option.get("type") == "Geometric Comparison"
        ]
        geometric_yield = max(
            (option.get("yield", 0.0) for option in geometric_options),
            default=max((option.get("yield", 0.0) for option in options_data),
                        default=0.0),
        )
        for option in options_data:
            plan = option.get("manufacturing_plan")
            if not isinstance(plan, dict):
                continue
            plan["cuttable_yield_percent"] = option.get("yield")
            plan["unconstrained_geometric_yield_percent"] = geometric_yield
            plan["yield_difference_percentage_points"] = round(
                float(option.get("yield", 0.0)) - float(geometric_yield),
                1,
            )

        settings_present = (
            opt_config["preform_margin_mm"] is not None
            and opt_config["max_cut_depth_mm"] is not None
        )
        options_data.sort(
            key=lambda opt: (
                int(
                    not settings_present
                    or (
                        opt.get("manufacturing_eligible", True)
                        and opt.get("manufacturing_plan", {}).get("status")
                        in {"complete", "no_separation_required"}
                    )
                ),
                opt.get("weight", 0),
                opt.get("gem_count", 0),
                opt.get("manufacturing_plan", {}).get(
                    "minimum_envelope_clearance_mm", 0
                ) or 0,
                -(opt.get("manufacturing_plan", {}).get(
                    "maximum_required_depth_mm", 0
                ) or 0),
                opt.get("light_score", 0),
                opt.get("facet_recommendation", {}).get("score", 0),
            ),
            reverse=True
        )
        best = options_data[0]

        # `best_cut.ply` is what the 3D viewer and PDF report treat as
        # "the" result by default (served as cut_url / used for the PDF
        # blueprint). Export it from the exact strategy this sort just
        # picked as options[0], so the viewer, the sidebar, and the PDF
        # can never disagree about which cut plan is being shown.
        if best.get("file"):
            try:
                shutil.copyfile(
                    os.path.join(output_dir, best["file"]),
                    os.path.join(output_dir, "best_cut.ply"),
                )
            except OSError:
                pass

        defect_detection = _load_defect_detection(job_folder)
        defect_summary = _load_defect_summary(job_folder)
        projected_waste = round(100 - best['yield'], 1)
        waste_reduction = {
            "traditional_waste_baseline_percent": round(TRADITIONAL_WASTE_BASELINE, 1),
            "projected_waste_percent": projected_waste,
            "reduction_vs_baseline_percent": round(TRADITIONAL_WASTE_BASELINE - projected_waste, 1),
            "note": (
                "Positive values mean lower waste than the assumed internal "
                "reference. Not a validated traditional-cutting comparison."
            )
        }

        # Rough stone dimensions in mm
        rough_extents_mm = [round(float(e) * scale_factor * 10, 2) for e in vol_mesh.extents]

        stats = {
            "volume_cm3": round(target_weight / CARATS_PER_GRAM / DENSITY_QUARTZ, 2),
            "rough_dimensions_mm": rough_extents_mm,
            "raw_carats": round(target_weight, 2),
            "estimated_cut_carats": best['weight'],
            "yield_percent": best['yield'],
            "recommended_shape": best['name'],
            "cut_mode": cut_mode,
            "preferred_shape": preferred_shape or "Auto",
            "space_utilization": best.get("space_utilization", {}),
            "waste_reduction": waste_reduction,
            "optimizer_diagnostics": best.get("optimizer_diagnostics", {}),
            "defect_summary": defect_summary,
            "defect_detection": defect_detection,
            "facet_recommendation": best.get("facet_recommendation", {}),
            "gem_details": best.get("gem_details", []),
            "options": options_data,
            "light_analysis": {
                "score": best['light_score'],
                "grade": best['light_grade']
            }
        }
        stats["manufacturing_plan"] = build_manufacturing_plan(stats)
        stats["research_completion"] = build_research_completion(
            stats,
            job_folder=job_folder,
        )
        return stats

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
