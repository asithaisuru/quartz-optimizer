"""Additive Stone Preservation runs on saved reconstruction and reviewed defects."""
import logging
import os
import time

from defect_aware_api import create_router as run_router, geometry_base, _record
from defect_review import atomic_json, now
from preform_api import canonical_result, public_error
from preform_recovery import settings as preform_settings
from stone_preservation import optimize_preservation

DIRECTORY = "stone_preservation"
MODE = "stone_preservation"


def settings(payload, config):
    allowed = {"blade_kerf_mm", "preform_mm", "rough_inset_mm", "max_cut_depth_mm", "max_regions"}
    if set(payload)-allowed:
        raise ValueError("Unsupported Stone Preservation settings.")
    stored = config.get("optimizer_settings") or {}
    values = {"max_cut_depth_mm": 100., "max_regions": 12, "min_secondary_carat": 0.}
    for key in allowed-{"max_regions"}:
        legacy = {"preform_mm": "preform_margin_mm", "rough_inset_mm": "rough_clearance_mm"}.get(key, key)
        if stored.get(legacy) not in (None, ""):
            values[key] = float(stored[legacy])
    return preform_settings({**values, **payload, "target_recovery_percent": 85.})


def calculate(job, run, request, snapshot, weight, manifest, report):
    started = time.perf_counter()
    state = {"job_id": job.name, "run_id": run.name, "mode": MODE, "status": "running",
             "worker_pid": os.getpid(), "progress_percent": None, "started_at": now(),
             "message": "Isolating confirmed defects while preserving healthy physical stock."}
    try:
        _record(job, run, state, DIRECTORY)
        base, scale, translation, cache = geometry_base(run / "rough_input.ply", weight, report,
            job / DIRECTORY / "base_cache", manifest["mesh_sha256"], resolution=56, bias=.5)
        result = optimize_preservation(base[0], weight, request, snapshot, run,
                                       geometry_base=base[1:], time_limit=60.)
        for region in result["regions"]:
            region["mesh_file"] = f"/files/{job.name}/{DIRECTORY}/{run.name}/{region['mesh_file']}"
        result["coordinate_frame"]["canonical_to_centered_translation_mesh_units"] = translation.tolist()
        result = canonical_result(result, job, run, directory=DIRECTORY)
        result.update(job_id=job.name, run_id=run.name, reused_reconstruction=True,
            defect_review_sha256=manifest["defect_review_sha256"], stale=False,
            input_manifest=manifest, settings=request, base_cache=cache,
            runtime_seconds=time.perf_counter()-started, runtime=time.perf_counter()-started,
            confirmed_defect_count=snapshot["summary"]["confirmed"])
        result["diagnostics"]["performance"]["total_runtime_seconds"] = result["runtime_seconds"]
        atomic_json(run / "result.json", result)
        state.update(status="completed", message=result["message"], progress_percent=100, finished_at=now())
        _record(job, run, state, DIRECTORY)
    except Exception as exc:
        logging.getLogger(__name__).exception("Stone Preservation failed")
        state.update(status="failed", message=public_error(exc), finished_at=now())
        try:
            _record(job, run, state, DIRECTORY)
        except OSError:
            atomic_json(run / "failure.json", state)


def create_router(validate_job):
    return run_router(validate_job, mode=MODE, directory=DIRECTORY, route="stone-preservation",
                      calculate_fn=calculate, settings_fn=settings)
