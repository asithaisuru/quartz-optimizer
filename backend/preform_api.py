"""Isolated defect review and preform recovery HTTP workflow."""
from __future__ import annotations

from effective_result import resolve_effective_result
import copy
import math
import re
import hashlib
import logging
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException

from defect_review import atomic_json, finite, json_file_lock, mutate, now, read_json, review, vector
from mesh_artifacts import CANONICAL_MESH_FILENAME, load_mesh_preserving_topology
from preform_recovery import optimize_preforms, settings

logger = logging.getLogger(__name__)


@contextmanager
def job_lock(job):
    """OS file lock serializes review edits and run admission across workers."""
    path = Path(job) / ".preform-workflow.lock"
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        deadline = time.monotonic() + 5
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise HTTPException(409, "Preform workflow is busy; retry shortly.")
                time.sleep(.02)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def input_metadata(job):
    config = read_json(job / "job_config.json", {})
    metadata = read_json(job / "job_metadata.json", {})
    weight = config.get("known_weight") or config.get("rough_weight_ct") or metadata.get("rough_weight_ct")
    try:
        weight = float(weight)
    except (TypeError, ValueError):
        raise ValueError("A measured positive rough weight is required.")
    from preform_recovery import recovery_metrics
    recovery_metrics(weight, 0)
    mesh_path = job / "dense" / CANONICAL_MESH_FILENAME
    if not mesh_path.is_file():
        raise ValueError("Canonical final_textured_model.ply is required.")
    return weight, mesh_path


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


STATUS_MESSAGES = {
    "idle": "Preform recovery has not been started.",
    "queued": "Preform recovery is queued.",
    "running": "Preform recovery is running.",
    "completed": "Preform recovery completed.",
    "failed": "Preform recovery failed.",
}


def safe_message(message, fallback):
    text = str(message)
    # Treat drive paths, UNC paths, and absolute POSIX paths as private.
    if re.search(r"[A-Za-z]:[\\/]|\\\\|(?<!\w)/(?!/)[\w.~]", text):
        return fallback
    return text[:500] or fallback


def public_error(exc):
    if isinstance(exc, FileNotFoundError):
        return "A required preform recovery file is missing. Please retry."
    if isinstance(exc, PermissionError):
        return "Preform recovery could not access its files. Please retry."
    if isinstance(exc, OSError):
        return "Preform recovery could not read or save its files. Please retry."
    return safe_message(exc, "Preform recovery could not validate or process its inputs. Check the server logs.")


def status_response(status):
    """One lifecycle field; retain existing timestamps without exposing worker internals."""
    value = {key: status[key] for key in
             ("run_id", "status", "mode", "created_at", "started_at", "finished_at", "search_state")
             if key in status}
    value.setdefault("run_id", None)
    value.setdefault("mode", "preform_recovery")
    value["message"] = safe_message(status.get("message") or STATUS_MESSAGES[value["status"]], STATUS_MESSAGES[value["status"]])
    return value


def canonical_result(result, job, run):
    """Serialize public physical fields without changing the verified internal plan."""
    value = copy.deepcopy(result)
    frame = value["coordinate_frame"]
    scale = finite(frame["mm_per_mesh_unit"], "mm_per_mesh_unit", 0)
    if scale <= 0:
        raise ValueError("Physical mesh scale must be positive.")
    frame.update(origin="canonical_axis_aligned_bounding_box_center",
                 axes="unchanged_from_canonical_mesh")
    plan = value.get("manufacturing_plan") or {}
    verified = (value.get("manufacturing_status") == "complete"
                and plan.get("status") == "complete"
                and plan.get("diagnostics", {}).get("exact_sequence_verified") is True)
    cuts = []
    for index, cut in enumerate(value.get("cuts", []), 1):
        plane = cut["plane"]
        normal = vector(plane["normal"], "normal")
        length = math.sqrt(sum(v * v for v in normal))
        if length <= 0:
            raise ValueError("Cut plane normal cannot be zero.")
        origin = (vector(plane["origin_mm"], "origin_mm") if "origin_mm" in plane else
                  [v * scale for v in vector(plane["origin"], "origin")])
        sequence = cut.get("sequence", cut.get("step", index))
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("Cut sequence must be a positive integer.")
        region_ids = cut.get("region_ids", cut.get("retained_gems", []))
        depth = cut.get("required_depth_mm")
        kerf = cut.get("kerf_mm", (cut.get("kerf_slab") or {}).get("thickness_mm"))
        cuts.append({
            "cut_id": f"C{sequence}",
            "recommendation_status": "selected_verified" if verified else "geometric_comparison_only",
            "manufacturing_verified": verified,
            "sequence": sequence,
            "plane": {"origin_mm": origin, "normal": [v / length for v in normal]},
            "required_depth_mm": None if depth is None else finite(depth, "required_depth_mm", 0),
            "kerf_mm": None if kerf is None else finite(kerf, "kerf_mm", 0),
            "parent_piece_id": cut.get("parent_piece_id"),
            "result_piece_ids": cut.get("result_piece_ids", []),
            "region_ids": [identifier for identifier in region_ids if identifier.startswith("R")],
            "discarded_region_ids": cut.get("discarded_region_ids",
                                            [identifier for identifier in region_ids if identifier.startswith("W")]),
        })
    value["cuts"] = cuts
    for region in value["regions"]:
        region_id = region["region_id"]
        if not isinstance(region_id, str) or not re.fullmatch(r"R[1-9][0-9]*", region_id):
            raise ValueError("Invalid retained region ID.")
        resource = run / (region_id + ".ply")
        if not resource.resolve().is_relative_to(run.resolve()) or not resource.is_file():
            raise ValueError("Retained region mesh artifact is unavailable.")
        region["mesh_file"] = f"/files/{job.name}/preform_recovery/{run.name}/{region_id}.ply"
    return value


_VOLATILE_FAILURES = {}


def current_status(job):
    path = job / "preform_recovery" / "status.json"
    with json_file_lock(path):
        emergency = _VOLATILE_FAILURES.get(str(job.resolve()))
        if emergency:
            return copy.deepcopy(emergency)
        value = read_json(path, {"run_id": None, "status": "idle", "mode": "preform_recovery"})
        try:
            run_id = uuid.UUID(value["run_id"]).hex
        except (ValueError, TypeError, KeyError):
            return value
        run = job / "preform_recovery" / run_id
        # The job file is also the run pointer. Prefer its matching per-run
        # terminal state if replacement of the pointer was temporarily denied.
        failure = read_json(run / "failure.json")
        if failure and failure.get("run_id") == value["run_id"]:
            return failure
        latest = read_json(run / "status.json")
        if latest and latest.get("run_id") == value["run_id"]:
            return latest
        return value


def record_status(job, run, value):
    value = {**value, "message": value.get("message") or STATUS_MESSAGES[value["status"]]}
    errors = []
    # Attempt both copies even if the first cannot be replaced.
    for path in (run / "status.json", job / "preform_recovery" / "status.json"):
        try:
            atomic_json(path, value)
        except OSError as exc:
            errors.append(exc)
    if errors:
        raise errors[0]


def publish_failure(job, run, status):
    """Publish a terminal failure even if the two normal status files are locked."""
    key = str(job.resolve())
    pointer = job / "preform_recovery" / "status.json"
    try:
        with job_lock(job):
            with json_file_lock(pointer):
                _VOLATILE_FAILURES[key] = copy.deepcopy(status)
            try:
                record_status(job, run, status)
            except Exception:
                logger.exception("Could not replace normal failure status for %s", job.name)
                # A different filename bypasses a persistent lock on status.json.
                atomic_json(run / "failure.json", status)
            with json_file_lock(pointer):
                _VOLATILE_FAILURES.pop(key, None)
    except Exception:
        logger.exception("Could not persist preform failure for %s", job.name)
        # A completely unwritable filesystem cannot promise durable status.
        # Keep this worker's GET response terminal instead of stale queued.
        with json_file_lock(pointer):
            _VOLATILE_FAILURES[key] = copy.deepcopy(status)


def run_recovery(job, run, request, snapshot, weight, legacy_yield):
    status = {"run_id": run.name, "status": "running", "mode": "preform_recovery",
              "started_at": now(), "worker_pid": os.getpid()}
    try:
        with job_lock(job):
            record_status(job, run, status)
        mesh = load_mesh_preserving_topology(run / "rough_input.ply")
        translation = -mesh.bounds.mean(axis=0)
        mesh.apply_translation(translation)
        result = optimize_preforms(mesh, weight, request, snapshot, run)
        for region in result["regions"]:
            region["mesh_file"] = f"/files/{job.name}/preform_recovery/{run.name}/{region['mesh_file']}"
        result.update(run_id=run.name, review_revision=snapshot["revision"],
                      legacy_faceted_yield_percent=legacy_yield,
                      settings=request, input_manifest=read_json(run / "input_manifest.json"))
        result["coordinate_frame"]["canonical_to_centered_translation_mesh_units"] = translation.tolist()
        result = canonical_result(result, job, run)
        atomic_json(run / "result.json", result)
        status.update(status="completed", finished_at=now(), search_state=result["search_state"],
                      message=result["message"])
        with job_lock(job):
            record_status(job, run, status)
    except Exception as exc:
        logger.exception("Preform recovery failed for %s", job.name)
        status.update(status="failed", finished_at=now(), message=public_error(exc))
        publish_failure(job, run, status)


def create_router(validate_job, jobs_root=None):
    router = APIRouter(tags=["preform-recovery"])

    @router.get("/optimization-modes")
    def modes():
        return {
            "default": "legacy_faceted_pack",
            "optimization_modes": ["legacy_faceted_pack", "preform_recovery"],
            "legacy_faceted_pack": {"endpoint": "/jobs/{job_id}/recalculate",
                                    "metric": "legacy_faceted_yield_percent",
                                    "existing_metric": "yield_percent"},
            "preform_recovery": {"endpoint": "/jobs/{job_id}/preform-recovery",
                                 "metric": "preform_recovery_percent"},
        }

    @router.get("/jobs/{job_id}/defect-review")
    def get_review(job_id: str):
        job = validate_job(job_id)
        try:
            value = review(job)
            # Match the existing centered optimizer/viewer frame, in millimetres.
            try:
                weight, mesh_path = input_metadata(job)
                mesh = load_mesh_preserving_topology(mesh_path)
                if mesh.is_watertight and abs(mesh.volume) > 0:
                    value["coordinate_frame"].update(
                        mm_per_mesh_unit=(weight / (5 * 2.65) / abs(float(mesh.volume))) ** (1 / 3) * 10,
                        canonical_to_centered_translation_mesh_units=(-mesh.bounds.mean(axis=0)).tolist(),
                    )
            except ValueError:
                pass
            return value
        except (OSError, ValueError, KeyError) as exc:
            logger.exception("Defect review data could not be read for %s", job.name)
            raise HTTPException(409, public_error(exc))

    def edit(job_id, operation, payload=None, annotation_id=None):
        job = validate_job(job_id)
        try:
            with job_lock(job):
                return mutate(job, operation, payload, annotation_id)
        except KeyError:
            raise HTTPException(404, "Annotation not found.")
        except ValueError as exc:
            raise HTTPException(422, public_error(exc))

    @router.post("/jobs/{job_id}/defect-review/annotations", status_code=201)
    def create_annotation(job_id: str, payload: dict = Body(...)):
        return edit(job_id, "create", payload)

    @router.patch("/jobs/{job_id}/defect-review/annotations/{annotation_id}")
    def patch_annotation(job_id: str, annotation_id: str, payload: dict = Body(...)):
        return edit(job_id, "patch", payload, annotation_id)

    @router.delete("/jobs/{job_id}/defect-review/annotations/{annotation_id}")
    def delete_annotation(job_id: str, annotation_id: str):
        edit(job_id, "delete", annotation_id=annotation_id)
        return {"deleted": annotation_id}

    @router.post("/jobs/{job_id}/preform-recovery", status_code=202)
    def start_recovery(job_id: str, bg_tasks: BackgroundTasks, payload: dict = Body(default={})):
        job = validate_job(job_id)
        try:
            cfg = settings(payload)
            weight, mesh_path = input_metadata(job)
        except ValueError as exc:
            raise HTTPException(422, public_error(exc))
        with job_lock(job):
            existing = current_status(job)
            if existing["status"] in {"queued", "running"}:
                raise HTTPException(409, {"message": "A preform run is already active.", "run_id": existing["run_id"]})
            snapshot = review(job)
            run = job / "preform_recovery" / uuid.uuid4().hex
            run.mkdir(parents=True)
            shutil.copy2(mesh_path, run / "rough_input.ply")
            atomic_json(run / "request.json", cfg)
            atomic_json(run / "defect_review_snapshot.json", snapshot)
            digest = hashlib.sha256()
            with (run / "rough_input.ply").open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            effective = resolve_effective_result(job)
            legacy = read_json(effective["report_path"], {})
            legacy_yield = legacy.get("yield_percent")
            atomic_json(run / "input_manifest.json", {
                "schema_version": "1.0", "created_at": now(), "rough_weight_ct": weight,
                "mesh_sha256": digest.hexdigest(), "source_mesh": "dense/final_textured_model.ply",
                "defect_review_revision": snapshot["revision"],
                "legacy_comparison_source": {"result_id": effective["result_id"],
                                             "report_sha256": effective["report_hash"]},
            })
            status = {"run_id": run.name, "status": "queued", "mode": "preform_recovery",
                      "created_at": now(), "worker_pid": os.getpid()}
            with json_file_lock(job / "preform_recovery" / "status.json"):
                _VOLATILE_FAILURES.pop(str(job.resolve()), None)
            try:
                record_status(job, run, status)
            except OSError as exc:
                logger.exception("Could not queue preform recovery for %s", job.name)
                status.update(status="failed", finished_at=now(), message=public_error(exc))
                # We already hold job_lock here, so do not reacquire it.
                with json_file_lock(job / "preform_recovery" / "status.json"):
                    _VOLATILE_FAILURES[str(job.resolve())] = copy.deepcopy(status)
                try:
                    atomic_json(run / "failure.json", status)
                except OSError:
                    logger.exception("Could not persist queue failure for %s", job.name)
                raise HTTPException(503, status_response(status))
            bg_tasks.add_task(run_recovery, job, run, cfg, snapshot, weight, legacy_yield)
        return {"run_id": run.name, "status": "queued"}

    @router.get("/jobs/{job_id}/preform-recovery/status")
    def get_status(job_id: str):
        return status_response(current_status(validate_job(job_id)))

    @router.get("/jobs/{job_id}/preform-recovery/result")
    def get_result(job_id: str):
        job = validate_job(job_id)
        status = current_status(job)
        if status["status"] == "idle":
            raise HTTPException(404, "No preform recovery run exists.")
        if status["status"] != "completed":
            raise HTTPException(409, status_response(status))
        # Only server-generated UUID run directories are accepted.
        try:
            run_id = uuid.UUID(status["run_id"]).hex
        except (ValueError, KeyError):
            raise HTTPException(409, "Invalid stored preform run ID.")
        path = job / "preform_recovery" / run_id / "result.json"
        if not path.is_file():
            raise HTTPException(409, "Completed result artifact is unavailable.")
        try:
            return canonical_result(read_json(path), job, path.parent)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            logger.exception("Invalid preform result artifact for %s", job.name)
            raise HTTPException(409, public_error(exc))

    if jobs_root is not None:
        @router.on_event("startup")
        def recover_interrupted_runs():
            root = Path(jobs_root())
            if not root.is_dir():
                return
            for job in root.iterdir():
                if not job.is_dir():
                    continue
                status = current_status(job)
                if status["status"] not in {"queued", "running"}:
                    continue
                # Do not interrupt another live worker when using multiple processes.
                if process_alive(status.get("worker_pid")):
                    continue
                with job_lock(job):
                    latest = current_status(job)
                    if latest != status:
                        continue
                    try:
                        run_id = uuid.UUID(status["run_id"]).hex
                    except (ValueError, KeyError):
                        continue
                    status.update(status="failed", finished_at=now(),
                                  message="Preform worker exited before completion; start a new run.")
                    record_status(job, job / "preform_recovery" / run_id, status)

    from preform_expert_review import create_router as create_expert_review_router
    router.include_router(create_expert_review_router(validate_job, job_lock))
    return router
