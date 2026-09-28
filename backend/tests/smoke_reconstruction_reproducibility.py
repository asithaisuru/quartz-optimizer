"""Opt-in fresh QZ-05 reconstruction smoke; writes only a new tmp scratch tree.

Uses the same production capture, meshing, gate, and Phase 4 functions as a UI job.
No checkpoints are copied; no optimization is run. Exit 2 means the gate rejected
the new mesh, with measurements retained in validation_summary.json.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import traceback
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def inventory(roots):
    return {
        str(path): [path.stat().st_size, path.stat().st_mtime_ns]
        for root in roots for path in root.rglob("*") if path.is_file()
    }


def mesh_statistics(path):
    import numpy as np
    import trimesh

    mesh = trimesh.load(path, process=False)
    counts = np.bincount(mesh.edges_unique_inverse)
    return {
        "process": False,
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edges": int((counts == 1).sum()),
        "non_manifold_edges": int((counts > 2).sum()),
        "bad_edge_face_multiplicity": counts[counts > 2].tolist(),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "absolute_signed_volume_mesh_units_cubed": abs(float(mesh.volume)),
        "extents": mesh.extents.tolist(),
    }


def ply_point_count(path):
    with Path(path).open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("Missing PLY vertex count")
            if line.startswith(b"element vertex "):
                return int(line.split()[-1])


def run(scratch, summary):
    os.environ["STORAGE_PATH"] = str(scratch)
    os.environ["RECONSTRUCTION_REPRODUCIBLE_MODE"] = "true"

    from capture_quality import analyze_capture_set, frontend_report
    from video_utils import extract_best_frames
    from masking_utils import remove_backgrounds
    from colmap_runner import run_photogrammetry_pipeline, get_largest_sparse_model
    from cleanup_module import post_process_point_cloud
    from main import _require_reconstruction_quality
    from ai_runner import run_ai_pipeline
    from fracture_mapper import map_fractures_to_3d

    reference = REPO / "jobs/61620db4-2b2a-44b5-a704-eb7449d8611d"
    config = json.loads((reference / "job_config.json").read_text())
    sources = Path(config["source_folder"]) / "videos"
    job = scratch / "jobs" / str(uuid.uuid4())
    (job / "images").mkdir(parents=True)
    summary.update(job_path=str(job), rough_weight_ct=244.02, fresh=True,
                   checkpoint_reuse=False, phase_3_5_passed=False,
                   phase_4_completed=False, input_videos=[])
    write_json(job / "job_config.json", config)
    videos = {}
    for index, record in enumerate(config["video_hashes"]):
        source = sources / record["original_filename"]
        actual = digest(source)
        if actual != record["sha256"]:
            raise ValueError(f"Video hash mismatch: {source}")
        destination = job / config["processing_order"][index]["stored_filename"]
        shutil.copy2(source, destination)
        if digest(destination) != actual:
            raise ValueError(f"Copied video hash mismatch: {destination}")
        videos[f"{index + 1:02d}"] = destination
        summary["input_videos"].append({
            "source": str(source), "scratch_copy": str(destination), "sha256": actual})
    write_json(job / "job_metadata.json", {
        "specimen_id": "QZ-05", "rough_weight_ct": 244.02,
        "validation_kind": "fresh manual-equivalent reconstruction",
        "videos": summary["input_videos"],
    })
    print(f"Fresh job: {job}", flush=True)
    capture = frontend_report(analyze_capture_set(videos, metadata={"scan_mode": "turntable"}))
    write_json(job / "capture_quality.json", capture)
    summary["capture_quality"] = capture
    print(f"Capture quality: {capture['status']}", flush=True)

    counts = []
    total = 0
    for path in videos.values():
        count = extract_best_frames(str(path), str(job / "images"),
                                    target_frames=40, start_index=total)
        counts.append(count)
        total += count
    summary["extracted_frames_per_video"] = counts
    summary["extracted_frames"] = total
    if total < 10:
        raise RuntimeError("Too few extracted frames")
    remove_backgrounds(str(job / "images"))
    (job / "masking_done").touch()
    write_json(scratch / "validation_summary.json", summary)

    fused = run_photogrammetry_pipeline(str(job), scan_mode="turntable")
    summary["dense_points"] = ply_point_count(fused)
    model = job / "sparse" / get_largest_sparse_model(str(job / "sparse"))
    for name, field in (("points3D", "sparse_points"), ("images", "registered_images"),
                        ("cameras", "camera_count")):
        with (model / f"{name}.bin").open("rb") as handle:
            summary[field] = struct.unpack("<Q", handle.read(8))[0]
    mesh_path = job / "dense/final_textured_model.ply"
    result = post_process_point_cloud(fused, str(mesh_path))
    if not result:
        raise RuntimeError("Poisson failed to produce a mesh")
    summary["mesh_path"] = str(mesh_path)
    summary["mesh"] = mesh_statistics(mesh_path)
    summary["environment"] = json.loads((job / "reconstruction_environment.json").read_text())
    try:
        gate = _require_reconstruction_quality(str(mesh_path), str(job))
    except RuntimeError:
        summary["quality_gate"] = json.loads((job / "reconstruction_quality.json").read_text())
        summary["outcome"] = "FAIL_PHASE_3_5"
        print(json.dumps(summary["mesh"]), flush=True)
        return 2
    summary["quality_gate"] = gate
    summary["phase_3_5_passed"] = True
    print("Phase 3.5 PASS; running production Phase 4.", flush=True)
    write_json(scratch / "validation_summary.json", summary)
    run_ai_pipeline(str(job))
    map_fractures_to_3d(str(job))
    required = [job / "detections/policy_decisions.json", job / "detections/policy_summary.json"]
    if not all(path.is_file() for path in required):
        raise RuntimeError("Phase 4 did not produce required policy artifacts")
    summary["phase_4_completed"] = True
    summary["outcome"] = "PASS_THROUGH_PHASE_4"
    summary["phase_4_artifacts"] = [str(p.relative_to(job)) for p in
                                   sorted((job / "detections").glob("*.json"))]
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    scratch = REPO / "tmp" / (
        "qz05_reproducibility_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    scratch.mkdir(parents=True, exist_ok=False)
    protected = [
        REPO / "jobs" / job_id for job_id in (
            "af62338e-1706-4d6b-9816-d61856d03e22",
            "6b1a0149-3939-4e28-9501-f671fe884a39",
            "61620db4-2b2a-44b5-a704-eb7449d8611d",
        )
    ] + [REPO / "final_research_evidence"]
    before = inventory(protected)
    write_json(scratch / "protected_inventory_before.json", before)
    protected_code = {str(REPO / "backend" / name): digest(REPO / "backend" / name)
                      for name in ("reconstruction_quality.py", "cleanup_module.py", "optimizer.py")}
    summary = {"scratch_path": str(scratch), "started_at_epoch": time.time()}
    print(f"SCRATCH={scratch}", flush=True)
    with (scratch / "run.log").open("w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                result = run(scratch, summary)
            except Exception as exc:
                summary["outcome"] = "ERROR"
                summary["error"] = str(exc)
                traceback.print_exc()
                result = 1
            finally:
                after = inventory(protected)
                changed = [p for p in set(before) | set(after) if before.get(p) != after.get(p)]
                code_changed = [p for p, value in protected_code.items() if digest(p) != value]
                summary["integrity"] = {"protected_file_count": len(before),
                                        "changed_files": changed, "changed_protected_code": code_changed}
                summary["elapsed_seconds"] = time.time() - summary["started_at_epoch"]
                write_json(scratch / "validation_summary.json", summary)
    if changed or code_changed:
        result = 1
    print(json.dumps({"scratch": str(scratch), "outcome": summary.get("outcome"),
                      "integrity": summary["integrity"], "exit_code": result}), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
