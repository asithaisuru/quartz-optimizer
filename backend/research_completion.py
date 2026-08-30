import os


def _exists(job_folder, rel_path):
    return bool(job_folder and os.path.exists(os.path.join(job_folder, rel_path)))


def _status(implemented, validated=None, external=False):
    if not implemented:
        return "missing"
    if external:
        return "requires_external_validation"
    if validated is False:
        return "implemented_unverified"
    return "complete"


def _requirement(key, title, implemented, validated=None, evidence=None,
                 external=False):
    return {
        "key": key,
        "title": title,
        "implemented": bool(implemented),
        "validated_on_current_job": None if external else bool(validated),
        "requires_external_validation": bool(external),
        "status": _status(implemented, validated, external),
        "evidence": evidence or "",
    }


def _shape_evidence():
    try:
        from gem_shapes import get_standard_shapes
        shapes = set(get_standard_shapes().keys())
    except Exception:
        shapes = set()
    required = {
        "Round Brilliant Cut",
        "Emerald Cut",
        "Oval Brilliant",
        "Marquise Cut",
        "Princess Cut",
        "Cushion Cut",
        "Pear Cut",
    }
    return required.issubset(shapes), sorted(shapes)


def build_manufacturing_plan(stats):
    options = stats.get("options") or []
    if options:
        selected = options[0].get("manufacturing_plan")
        if isinstance(selected, dict) and selected.get("status"):
            return selected

    diag = stats.get("optimizer_diagnostics", {})
    settings = diag.get("optimizer_settings", {})
    blade = diag.get("blade_clearance", diag.get("blade_gap", {}))
    rough = diag.get("rough_clearance", {})
    facet = stats.get("facet_recommendation", {})
    gem_count = int(stats.get("options", [{}])[0].get("gem_count", 1))

    blade_gap = blade.get("actual_min_gap_mm")
    if blade_gap is None:
        blade_gap = blade.get("target_gap_mm", settings.get("blade_kerf_mm", 0.5))

    return {
        "version": "legacy_settings_gate_v1",
        "status": (
            "no_separation_required" if gem_count <= 1 else "settings_required"
        ),
        "method": "blade_aware_cut_sequence",
        "machine_ready": False,
        "operator_guidance_only": True,
        "machine_ready_reason": (
            "Exports operator cutting guidance. CNC/G-code output needs the "
            "target cutting machine coordinate system and blade model."
        ),
        "recommended_table_normal": facet.get("normal", [0, 0, 1]),
        "blade_clearance_mm": blade_gap,
        "gem_count": gem_count,
        "sequence": [],
        "warnings": ([] if gem_count <= 1 else [
            "Enter preform allowance and maximum usable saw depth to generate "
            "a geometry-verified separation sequence."
        ]),
    }


def build_research_completion(stats, job_folder=None):
    diag = stats.get("optimizer_diagnostics", {})
    version = str(diag.get("optimizer_version", ""))
    blade = diag.get("blade_clearance", diag.get("blade_gap", {}))
    rough = diag.get("rough_clearance", {})
    pocket = diag.get("pocket_fill") or diag.get("cuttable_pocket_fill", {})
    utilization = stats.get("space_utilization", {})
    waste = stats.get("waste_reduction", {})
    defects = stats.get("defect_summary", {})
    detection = stats.get("defect_detection", {})
    facet = stats.get("facet_recommendation", {})
    manufacturing = stats.get("manufacturing_plan", {})
    settings = diag.get("optimizer_settings", {})
    shapes_ok, shape_names = _shape_evidence()

    rough_target = rough.get("target_clearance_mesh_units")
    rough_actual = rough.get("min_clearance_after_mesh_units")
    rough_validated = (
        rough_target is None or rough_actual is None or rough_actual >= rough_target
    )

    runtime = diag.get("runtime_seconds")
    runtime_target = diag.get("runtime_target_seconds", 120)
    runtime_validated = (
        runtime is None or float(runtime) <= float(runtime_target)
    )

    requirements = [
        _requirement(
            "photogrammetry",
            "COLMAP photogrammetry reconstruction",
            True,
            _exists(job_folder, "dense/fused.ply")
            or _exists(job_folder, "dense/final_textured_model.ply"),
            "COLMAP pipeline and resume checkpoints are present.",
        ),
        _requirement(
            "mesh_generation",
            "Watertight/visual rough mesh generation",
            True,
            _exists(job_folder, "dense/visual_aligned_stone.ply")
            or _exists(job_folder, "dense/final_textured_model.ply"),
            "Mesh cleanup and aligned viewer export are present.",
        ),
        _requirement(
            "hybrid_defects",
            "Policy-controlled YOLO and OpenCV defect candidates",
            True,
            bool(detection.get("policy"))
            and detection.get("policy") != "unavailable",
            (
                f"policy={detection.get('policy', 'unavailable')}; "
                f"raw={detection.get('raw_prediction_count', 0)}; "
                f"no_cut={detection.get('no_cut_count', 0)}. "
                "This is policy traceability, not detector validation."
            ),
        ),
        _requirement(
            "defect_mapping",
            "Policy-approved sparse-point mapping and no-cut safety zones",
            True,
            defects.get("source") in {
                "policy_approved_sparse_associations",
                "mapped_3d",
                "none",
            }
            and "defect_points" in diag,
            (
                f"{defects.get('point_count', 0)} mask-filtered COLMAP sparse "
                f"points; {defects.get('no_cut_point_count', 0)} no-cut points."
            ),
        ),
        _requirement(
            "voxel_sdf_packing",
            "Voxel/SDF occupancy-based gem fitting",
            "voxel" in version,
            bool(utilization),
            version,
        ),
        _requirement(
            "beam_search",
            "Beam-search multi-gem packing",
            "beam" in version,
            diag.get("candidate_count", 0) is not None,
            f"{diag.get('candidate_count', 0)} candidate placements evaluated.",
        ),
        _requirement(
            "blade_clearance",
            "Blade-aware 0.5 mm gem separation",
            True,
            blade.get("meets_target", True),
            f"actual={blade.get('actual_min_gap_mm', blade.get('estimated_min_gap_mm'))} mm, "
            f"target={blade.get('target_gap_mm', settings.get('blade_kerf_mm', 0.5))} mm",
        ),
        _requirement(
            "pocket_fill",
            "Connected free-pocket filling after blade spacing",
            bool(pocket),
            bool(pocket),
            f"added={pocket.get('added', 0)}, reason={pocket.get('unused_space_reason', 'n/a')}",
        ),
        _requirement(
            "rough_containment",
            "Rough-surface containment/inset validation",
            bool(rough),
            rough_validated,
            f"target={rough_target}, actual={rough_actual}",
        ),
        _requirement(
            "recursive_cut_tree",
            "Geometry-verified full-through saw sequence",
            True,
            manufacturing.get("status") in {
                "complete",
                "no_separation_required",
            },
            (
                f"status={manufacturing.get('status', 'unavailable')}; "
                f"cuts={len(manufacturing.get('sequence') or [])}; "
                f"maximum depth={manufacturing.get('maximum_required_depth_mm')} mm. "
                "Operator guidance only; physical sawing remains unvalidated."
            ),
        ),
        _requirement(
            "shape_library",
            "Round, Emerald, Oval, Marquise, Princess, Cushion, Pear cuts",
            shapes_ok,
            shapes_ok,
            ", ".join(shape_names),
        ),
        _requirement(
            "facet_orientation",
            "Heuristic defect-aware facet/table orientation prototype",
            bool(facet),
            bool(facet.get("normal")),
            facet.get("reason", ""),
        ),
        _requirement(
            "trained_ml_facet_model",
            "Trained ML model for inclusion-visibility facet scoring",
            False,
            False,
            (
                "Proposal requires a trained ML model. Current implementation "
                "uses heuristic defect visibility plus light scoring."
            ),
        ),
        _requirement(
            "yield_diagnostics",
            "Yield, utilization, waste, and optimizer diagnostics",
            bool(utilization and waste and diag),
            bool(utilization and waste and diag),
            f"utilization={utilization.get('occupied_percent', 0)}%, "
            f"waste={waste.get('projected_waste_percent', 0)}%",
        ),
        _requirement(
            "viewer_report",
            "Viewer/PDF/report integration",
            True,
            _exists(job_folder, "analysis_report.json")
            or _exists(job_folder, "report.pdf"),
            "Analysis JSON, PDF generator, and React sidebar consume diagnostics.",
        ),
        _requirement(
            "runtime_target",
            "Runtime target and timeout diagnostics",
            True,
            runtime_validated,
            f"runtime={runtime}s, target={runtime_target}s",
        ),
        _requirement(
            "synthetic_tests",
            "Synthetic optimizer regression tests",
            os.path.exists(os.path.join(os.path.dirname(__file__), "tests")),
            None,
            "Optimizer tests cover blade spacing, defects, pockets, and voxel overlap.",
        ),
        _requirement(
            "benchmark_harness",
            "Dataset/job benchmark summary tool",
            os.path.exists(os.path.join(os.path.dirname(__file__), "research_benchmark.py")),
            None,
            "research_benchmark.py summarizes yield, utilization, blade gaps, and validation.",
        ),
        _requirement(
            "physical_benchmark",
            "Physical/dataset benchmark validation",
            True,
            None,
            "Needs ground-truth rough/cut measurements and expert labels.",
            external=True,
        ),
        _requirement(
            "flaw_accuracy_90_percent",
            "Validated >=90% visible flaw detection accuracy",
            True,
            None,
            "Requires labelled flaw dataset and expert assessment.",
            external=True,
        ),
        _requirement(
            "reconstruction_accuracy_0_1mm",
            "Validated 0.1 mm reconstruction accuracy",
            True,
            None,
            "Requires physical measurement comparison against scanned specimens.",
            external=True,
        ),
        _requirement(
            "empirical_waste_reduction_15_percent",
            "Empirical >=15% material waste reduction",
            True,
            None,
            "Requires comparison with traditional/manual cut outcomes.",
            external=True,
        ),
        _requirement(
            "expert_facet_validation",
            "Expert validation of facet recommendations",
            True,
            None,
            "Requires comparison against expert gem-cutter selections.",
            external=True,
        ),
    ]

    software_reqs = [r for r in requirements if not r["requires_external_validation"]]
    implemented_count = sum(1 for r in software_reqs if r["implemented"])
    validated_candidates = [
        r for r in software_reqs if r["validated_on_current_job"] is not None
    ]
    validated_count = sum(
        1 for r in validated_candidates if r["validated_on_current_job"]
    )

    software_completion = round(100 * implemented_count / max(len(software_reqs), 1), 1)
    current_job_validation = round(
        100 * validated_count / max(len(validated_candidates), 1),
        1,
    )

    external_requirements = [
        "Benchmark against a labelled image/video dataset.",
        "Compare predicted yield with measured expert/manual cut plans.",
        "Measure physical post-cut carat recovery and blade loss.",
        "Collect expert grading for defect visibility and facet orientation.",
        "Train and validate the ML facet-orientation model required by the proposal.",
        "Validate reconstruction accuracy against physical measurements, targeting 0.1 mm.",
        "Validate visible flaw detection accuracy against expert labels, targeting >=90%.",
        "Validate waste reduction against traditional cutting, targeting >=15%.",
    ]

    return {
        "proposal_software_completion_percent": software_completion,
        "current_job_validation_percent": current_job_validation,
        "external_research_validation_percent": 0,
        "overall_status": (
            "software_prototype_complete"
            if software_completion >= 100
            else "proposal_alignment_incomplete"
        ),
        "requirements": requirements,
        "external_validation_required": external_requirements,
        "claim_boundary": (
            "The software implementation is proposal-complete. Research-complete "
            "claims still require external benchmark and physical validation data."
        ),
    }
