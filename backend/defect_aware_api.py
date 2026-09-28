"""Rerunnable confirmed-defect faceted packing; reconstruction is read-only input."""
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import trimesh
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException

import optimizer
from cut_sequence import plan_cut_sequence
from defect_aware_geometry import ConfirmedGeometry, constraint_hash, millimetre_scale
from defect_review import atomic_json, finite, now, read_json, review
from mesh_artifacts import load_mesh_preserving_topology
from preform_api import input_metadata, job_lock, process_alive, public_error
from yield_calculator import _build_gem_details
from optimizer_parallel import configuration

MODE = "defect_aware_faceted_pack"
DIRECTORY = "defect_aware_optimization"
DEFAULTS = {"blade_kerf_mm": .5, "preform_mm": .5, "rough_inset_mm": .8,
            "max_cut_depth_mm": 100., "max_gems": 12, "min_secondary_carat": .5}
ALIASES = {"preform_mm": "preform_margin_mm", "rough_inset_mm": "rough_clearance_mm"}
logger = logging.getLogger(__name__)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def settings(payload, config):
    if not isinstance(payload, dict) or set(payload)-set(DEFAULTS):
        raise ValueError("Unsupported defect-aware optimizer settings.")
    stored = config.get("optimizer_settings") or {}
    result = dict(DEFAULTS)
    for key in result:
        value = stored.get(ALIASES.get(key, key))
        if value not in (None, ""):
            result[key] = float(value) if key != "max_gems" else int(value)
    result.update(payload)
    for key, maximum in (("blade_kerf_mm", 2), ("preform_mm", 5),
                         ("rough_inset_mm", 5), ("max_cut_depth_mm", 500),
                         ("min_secondary_carat", 10)):
        finite(result[key], key, 0, maximum)
    if result["max_cut_depth_mm"] <= 0 or result["preform_mm"] <= 0:
        raise ValueError("Preform margin and maximum cut depth must be positive.")
    if result["rough_inset_mm"] < result["preform_mm"]:
        raise ValueError("Rough inset must be at least the preform margin.")
    if isinstance(result["max_gems"], bool) or not isinstance(result["max_gems"], int) or not 1 <= result["max_gems"] <= 20:
        raise ValueError("max_gems must be an integer from 1 to 20.")
    return result


def passed_reconstruction(job):
    report = read_json(job / "reconstruction_quality.json", {})
    if report.get("passed") is not True or report.get("status") != "PASS":
        raise ValueError("A saved reconstruction quality PASS is required before calculating gems.")
    weight, mesh_path = input_metadata(job)
    return weight, mesh_path, report


def geometry_base(mesh_path, weight, report, cache_root, mesh_hash, *, resolution=None, bias=None):
    started = time.perf_counter()
    mesh = load_mesh_preserving_topology(mesh_path)
    # Input identity/topology checks only: do not re-run reconstruction or its
    # SDF quality gate, and never repair the canonical geometry while loading.
    if not mesh.is_watertight or not np.isfinite(mesh.volume) or abs(mesh.volume) <= 0:
        raise ValueError("Saved reconstruction no longer supplies a closed finite volume.")
    metrics = report.get("metrics", {})
    for key, actual in (("vertex_count", len(mesh.vertices)), ("face_count", len(mesh.faces))):
        if key in metrics and metrics[key] != actual:
            raise ValueError("Canonical mesh differs from the saved reconstruction quality report.")
    if "mesh_volume" in metrics and not np.isclose(abs(mesh.volume), abs(metrics["mesh_volume"]), rtol=1e-6, atol=1e-12):
        raise ValueError("Canonical volume differs from the saved reconstruction quality report.")
    translation = -mesh.bounds.mean(axis=0)
    scale = millimetre_scale(mesh, weight)
    mesh.apply_translation(translation)
    key = hashlib.sha256(json.dumps([mesh_hash, resolution or optimizer.SDF_RESOLUTION,
        optimizer.SDF_BIAS_VOXELS if bias is None else bias, 3, "centered_sdf_v1"]).encode()).hexdigest()
    path = cache_root / (key + ".npz")
    hit = False
    try:
        with np.load(path, allow_pickle=False) as saved:
            grid, origin, pitch = saved["grid"], saved["origin"], float(saved["pitch"])
            if grid.ndim != 3 or origin.shape != (3,) or pitch <= 0 or not np.all(np.isfinite(grid)):
                raise ValueError("Invalid cached base geometry.")
            hit = True
    except (OSError, ValueError, KeyError, EOFError):
        grid, origin, pitch = optimizer._build_sdf_grid(mesh, **(
            {"res": resolution, "bias_vox": bias} if resolution is not None else {}))
        cache_root.mkdir(parents=True, exist_ok=True)
        temporary = cache_root / (key + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, grid=grid, origin=origin, pitch=pitch)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return (mesh, grid, origin, pitch), scale, translation, {
        "hit": hit, "key": key, "geometry_base_seconds": time.perf_counter()-started}


def _record(job, run, status, directory=DIRECTORY):
    errors = []
    for path in (run / "status.json", job / directory / "status.json"):
        try:
            atomic_json(path, status)
        except OSError as exc:
            errors.append(exc)
    if errors:
        raise errors[0]


def _status(job, run):
    value = read_json(run / "failure.json") or read_json(run / "status.json", {})
    if value.get("status") in {"queued", "running"} and not process_alive(value.get("worker_pid")):
        value.update(status="failed", message="Optimization worker exited; run again using the saved reconstruction.")
    return {key: value.get(key) for key in (
        "job_id", "run_id", "mode", "status", "message", "progress_percent")}


def calculate(job, run, request, snapshot, weight, manifest, report):
    started = time.perf_counter()
    state = {"job_id": job.name, "run_id": run.name, "mode": MODE,
             "status": "running", "message": "Preparing confirmed-defect gem placement.",
             "progress_percent": 0, "worker_pid": os.getpid(), "started_at": now()}
    try:
        _record(job, run, state)
        base, scale, translation, cache = geometry_base(
            run / "rough_input.ply", weight, report, job / DIRECTORY / "base_cache", manifest["mesh_sha256"])
        rough, grid, origin, pitch = base
        guard = ConfirmedGeometry(snapshot, scale)
        timing = {"manufacturing_seconds": 0.0}
        performance = configuration()

        def progress(current, total, message):
            state.update(progress_percent=min(95, round(95*current/max(total, 1), 1)), message=message)
            _record(job, run, state)

        search_started = time.perf_counter()
        strategies = optimizer.optimize_cut(
            str(run / "rough_input.ply"), mode="multi", job_folder=None,
            progress_callback=progress, blade_kerf_mm=request["blade_kerf_mm"],
            rough_clearance_mm=request["rough_inset_mm"], mm_per_mesh_unit=scale,
            min_secondary_volume=abs(rough.volume)*request["min_secondary_carat"]/weight,
            min_secondary_carat=request["min_secondary_carat"], carats_per_mesh_volume=weight/abs(rough.volume),
            max_gems=request["max_gems"], preform_margin_mm=request["preform_mm"],
            max_cut_depth_mm=request["max_cut_depth_mm"], confirmed_guard=guard,
            geometry_base=base, timings=timing, performance=performance)
        search_seconds = time.perf_counter()-search_started
        selected, plan = None, None
        verification_started = time.perf_counter()
        for candidate in sorted(strategies, key=lambda item: item["total_volume"], reverse=True):
            gems = candidate["gems"]
            if not gems or candidate.get("manufacturing_eligible") is False:
                continue
            # Recheck full final gem bodies after every refinement/export stage.
            if any(guard.rejects(gem.vertices) for gem in gems):
                continue
            plan = plan_cut_sequence(rough, gems, blade_kerf_mm=request["blade_kerf_mm"],
                preform_margin_mm=request["preform_mm"], max_cut_depth_mm=request["max_cut_depth_mm"],
                mm_per_mesh_unit=scale, pitch=pitch, time_limit_seconds=6,
                gem_ids=[f"gem_{i+1}" for i in range(len(gems))],
                gem_values=[abs(gem.volume) for gem in gems])
            if ((len(gems) == 1 and plan["status"] == "no_separation_required") or
                    (plan["status"] == "complete" and plan.get("diagnostics", {}).get("exact_sequence_verified") is True)):
                selected = candidate
                break
        final_verification = time.perf_counter()-verification_started
        timing["manufacturing_seconds"] += final_verification
        details, total, combined_url = [], 0.0, None
        prefix = f"/files/{job.name}/{DIRECTORY}/{run.name}/"
        if selected is not None:
            # Serializer shared with the existing ResultDashboard/ModelViewer.
            accurate = dict(selected)
            accurate["placements"] = [{**p, "volume": abs(g.volume)} for p, g in
                                      zip(selected["placements"], selected["gems"])]
            total_volume = sum(abs(g.volume) for g in selected["gems"])
            details = _build_gem_details(accurate, 0, str(run), scale/10, weight, abs(rough.volume), total_volume)
            for i, (gem, detail, placement) in enumerate(zip(selected["gems"], details, accurate["placements"])):
                if not detail["file"]:
                    raise ValueError("Gem mesh export failed.")
                exported = load_mesh_preserving_topology(run / detail["file"])
                if guard.rejects(exported.vertices):
                    raise ValueError("Exported gem intersects a confirmed defect; result withheld.")
                detail.update(id=f"gem_{i+1}", confirmed_defect_intersection=False,
                    orientation=placement.get("orientation"), mesh_url=prefix+detail["file"],
                    placement_strategy=selected["strategy"],
                    weight_ct_exact=weight*abs(gem.volume)/abs(rough.volume))
            total = weight*total_volume/abs(rough.volume)
            trimesh.util.concatenate(selected["gems"]).export(run / "best_cut.ply")
            combined_url = prefix+"best_cut.ply"
        else:
            plan = {"status": "no_verified_plan", "sequence": [], "gem_count": 0,
                    "diagnostics": {"exact_sequence_verified": False}}
        # Exclusion is an occupied-cell estimate calibrated to measured rough
        # mass, never a change to the rough's physical volume or yield denominator.
        occupied = grid > -(pitch*.20)
        forbidden = guard.mask(grid.shape, origin, pitch)
        excluded = weight*np.count_nonzero(occupied & forbidden)/max(1, np.count_nonzero(occupied))
        elapsed = time.perf_counter()-started
        result = {
            "mode": MODE, "job_id": job.name, "run_id": run.name, "reused_reconstruction": True,
            "rough_weight_ct": weight, "confirmed_defect_count": len(guard.annotations),
            "confirmed_defect_excluded_ct": excluded,
            "provisional_candidate_count": snapshot["summary"]["provisional"],
            "defect_review_sha256": constraint_hash(snapshot), "stale": False,
            "total_gem_weight_ct": total, "faceted_yield_percent": total/weight*100,
            "gem_count": len(details), "gems": details, "gem_details": details,
            "manufacturing_status": plan["status"], "manufacturing_plan": plan,
            "cut_sequence": plan["sequence"], "model_url": combined_url,
            "search_state": "resource_limit_reached" if guard.search_limited or search_seconds >= 118 else "bounded_search_complete",
            "runtime_seconds": elapsed, "rejected_due_to_confirmed_defects": guard.rejected,
            "performance": {**performance, "search_seconds": search_seconds,
                "manufacturing_seconds": timing["manufacturing_seconds"], "total_runtime_seconds": elapsed},
            "rejection_count_basis": "candidate feasibility evaluations, including scale/refinement trials",
            "message": ("Verified defect-safe faceted plan found within the bounded search." if details else
                        "No defect-safe manufacturing-verified gem plan found; reconstruction remains reusable."),
            "timings": {**timing, "mask_build_seconds": guard.mask_seconds,
                "placement_search_seconds": max(0, search_seconds-(timing["manufacturing_seconds"]-final_verification)-guard.mask_seconds),
                "total_runtime_seconds": elapsed, "geometry_base_seconds": cache["geometry_base_seconds"]},
            "base_cache": cache, "input_manifest": manifest, "settings": request,
            "coordinate_frame": {"name": "centered_rough_mesh", "annotation_units": "mm",
                "mesh_units": "mesh_units", "mm_per_mesh_unit": scale,
                "origin": "canonical_axis_aligned_bounding_box_center", "axes": "unchanged_from_canonical_mesh",
                "canonical_to_centered_translation_mesh_units": translation.tolist()},
            "confirmed_defect_constraints": guard.annotations,
            "selected_strategy": selected["strategy"] if selected else None,
            "limitations": ["Confirmed safety zones are manual/expert approximations.",
                "Conservative voxel coverage and convex gem hull tests can reject near-boundary placements.",
                "Excluded carats are resolution-dependent raster estimates; physical rough mass is unchanged.",
                "Bounded search does not prove an optimal yield; workshop validation remains required."],
        }
        atomic_json(run / "result.json", result)
        state.update(status="completed", progress_percent=100, message=result["message"], finished_at=now())
        _record(job, run, state)
    except Exception as exc:
        logger.exception("Defect-aware optimization failed for %s", job.name)
        state.update(status="failed", message=public_error(exc), progress_percent=None, finished_at=now())
        try:
            _record(job, run, state)
        except OSError:
            atomic_json(run / "failure.json", state)


def create_router(validate_job, *, mode=MODE, directory=DIRECTORY,
                  route="defect-aware-optimization", calculate_fn=calculate, settings_fn=settings):
    MODE, DIRECTORY = mode, directory
    router = APIRouter(tags=[route])

    def locate(job_id, run_id):
        job = Path(validate_job(job_id))
        try:
            identifier = uuid.UUID(run_id).hex
        except (ValueError, TypeError):
            raise HTTPException(404, "Defect-aware optimization run not found.")
        run = job / DIRECTORY / identifier
        if not run.is_dir() or not run.resolve().is_relative_to(job.resolve()):
            raise HTTPException(404, "Defect-aware optimization run not found.")
        return job, run

    @router.post(f"/jobs/{{job_id}}/{route}", status_code=202)
    def start(job_id: str, bg_tasks: BackgroundTasks, payload: dict = Body(default={})):
        job = Path(validate_job(job_id))
        try:
            request = settings_fn(payload, read_json(job / "job_config.json", {}))
            weight, mesh_path, quality = passed_reconstruction(job)
            with job_lock(job):
                pointer = read_json(job / DIRECTORY / "status.json", {})
                if pointer.get("run_id"):
                    current = _status(job, job / DIRECTORY / pointer["run_id"])
                    if current.get("status") in {"queued", "running"}:
                        raise HTTPException(409, "A defect-aware optimization is already active.")
                snapshot = review(job)
                fingerprint = constraint_hash(snapshot)
                run = job / DIRECTORY / uuid.uuid4().hex
                run.mkdir(parents=True)
                shutil.copy2(mesh_path, run / "rough_input.ply")
                manifest = {"mesh_sha256": sha256(run / "rough_input.ply"),
                    "quality_report_sha256": sha256(job / "reconstruction_quality.json"),
                    "defect_review_sha256": fingerprint, "rough_weight_ct": weight,
                    "source_mesh": "dense/final_textured_model.ply", "created_at": now()}
                if manifest["mesh_sha256"] != sha256(mesh_path):
                    raise ValueError("Canonical mesh changed while creating this run.")
                atomic_json(run / "input_manifest.json", manifest)
                atomic_json(run / "defect_review_snapshot.json", snapshot)
                atomic_json(run / "request.json", request)
                state = {"job_id": job_id, "run_id": run.name, "mode": MODE, "status": "queued",
                    "message": "Defect-aware faceted optimization queued.", "progress_percent": 0,
                    "worker_pid": os.getpid(), "created_at": now()}
                try:
                    _record(job, run, state, directory=DIRECTORY)
                except OSError:
                    state.update(status="failed", message="Unable to persist queued status; run again.")
                    atomic_json(run / "failure.json", state)
                    raise
                bg_tasks.add_task(calculate_fn, job, run, request, snapshot, weight, manifest, quality)
        except ValueError as exc:
            raise HTTPException(422, public_error(exc))
        except OSError as exc:
            raise HTTPException(503, public_error(exc))
        return {"run_id": run.name, "status": "queued", "reused_reconstruction": True}

    @router.get(f"/jobs/{{job_id}}/{route}/latest")
    def latest(job_id: str):
        job = Path(validate_job(job_id))
        pointer = read_json(job / DIRECTORY / "status.json", {})
        if not pointer.get("run_id"):
            raise HTTPException(404, "No defect-aware optimization run exists.")
        return status(job_id, pointer["run_id"])

    @router.get(f"/jobs/{{job_id}}/{route}/{{run_id}}/status")
    def status(job_id: str, run_id: str):
        job, run = locate(job_id, run_id)
        return _status(job, run)

    @router.get(f"/jobs/{{job_id}}/{route}/{{run_id}}/result")
    def result(job_id: str, run_id: str):
        job, run = locate(job_id, run_id)
        if _status(job, run)["status"] != "completed":
            raise HTTPException(409, "Defect-aware optimization has not completed.")
        saved = read_json(run / "result.json")
        if saved is None:
            raise HTTPException(409, "Completed result artifact is unavailable.")
        try:
            weight, mesh_path, _ = passed_reconstruction(job)
            stale = (constraint_hash(review(job)) != saved["defect_review_sha256"] or
                     sha256(mesh_path) != saved["input_manifest"]["mesh_sha256"] or
                     weight != saved["rough_weight_ct"])
        except (ValueError, OSError):
            stale = True
        saved["stale"] = stale
        return saved

    return router
