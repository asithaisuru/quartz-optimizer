"""Opt-in real HTTP smoke on private scratch copies; never writes source jobs.

Run: conda run --no-capture-output -n quartz python -B backend/tests/smoke_preform_real.py
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid

import requests
import trimesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))
from mesh_artifacts import load_mesh_preserving_topology
from reconstruction_quality import assess_reconstruction_quality

SOURCES = {
    "QZ-05": Path("D:/project-quartz/jobs/61620db4-2b2a-44b5-a704-eb7449d8611d"),
    "QZ-01": Path("D:/project-quartz/jobs/e0d3af52-0f92-4862-8988-eac74bda9a0a"),
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def assert_public(value):
    if isinstance(value, dict):
        for child in value.values():
            assert_public(child)
    elif isinstance(value, list):
        for child in value:
            assert_public(child)
    elif isinstance(value, str):
        assert not re.search(r"[A-Za-z]:[\\/]|\\\\|(?:^|[\s'\"\"])/(?!files/)", value), value


def main():
    scratch = REPO / "tmp" / ("preform_real_smoke_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    scratch.mkdir(parents=True)
    jobs, hashes, summaries = {}, {}, {}
    inventories = {str(source): {str(p.relative_to(source)): (p.stat().st_size, p.stat().st_mtime_ns)
                               for p in source.rglob("*") if p.is_file()}
                   for source in SOURCES.values()}
    for specimen, source in SOURCES.items():
        canonical = source / "dense" / "final_textured_model.ply"
        if not canonical.is_file():
            raise FileNotFoundError("Required real source job is unavailable: " + specimen)
        target = scratch / "jobs" / str(uuid.uuid4())
        (target / "dense").mkdir(parents=True)
        files = ["job_config.json", "job_metadata.json", "analysis_report.json",
                 "dense/final_textured_model.ply", "dense/defect_associations.json",
                 "detections/policy_decisions.json", "detections/policy_summary.json",
                 "defect_review.json", "extended_search/results.json",
                 "extended_search/result_v2/analysis_report.json"]
        for relative in files:
            original = source / relative
            if not original.is_file():
                continue
            hashes[str(original)] = digest(original)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
        # One real source frame is sufficient to verify served evidence links.
        frame = next(iter(sorted((source / "images").glob("*.jpg"))), None)
        if frame:
            (target / "images").mkdir()
            shutil.copy2(frame, target / "images" / frame.name)
        copied = target / "dense" / "final_textured_model.ply"
        preserved = load_mesh_preserving_topology(copied)
        processed = trimesh.load(copied, force="mesh")
        quality = assess_reconstruction_quality(copied)
        assert preserved.is_watertight, specimen
        assert quality["passed"], (specimen, quality["failure_reasons"])
        jobs[specimen] = target
        summaries[specimen] = {
            "scratch_job_id": target.name,
            "process_true_watertight": bool(processed.is_watertight),
            "process_false_watertight": bool(preserved.is_watertight),
            "vertex_count": len(preserved.vertices), "face_count": len(preserved.faces),
            "quality_gate": quality["status"],
        }
        print(specimen + ": original topology preserved; quality gate PASS", flush=True)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    environment = dict(os.environ, STORAGE_PATH=str(scratch), API_BASE_URL=base)
    with (scratch / "server.log").open("w", encoding="utf-8") as logs:
        process = subprocess.Popen(
            [sys.executable, "-B", "-m", "uvicorn", "main:app", "--app-dir", "backend",
             "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
            cwd=REPO, env=environment, stdout=logs, stderr=logs,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            ready = time.monotonic() + 40
            while time.monotonic() < ready:
                try:
                    if requests.get(base + "/optimization-modes", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                if process.poll() is not None:
                    raise RuntimeError("Smoke server exited; see scratch server.log.")
                time.sleep(.1)
            else:
                raise TimeoutError("Smoke server did not become ready.")
            with requests.Session() as client:
                for specimen, job in jobs.items():
                    prefix = base + "/jobs/" + job.name
                    response = client.get(prefix + "/defect-review", timeout=30)
                    assert response.status_code == 200, response.text
                    review = response.json()
                    assert_public(review)
                    scale = review["coordinate_frame"]["mm_per_mesh_unit"]
                    assert scale > 0
                    assert review["summary"]["confirmed"] == 0, specimen
                    summaries[specimen].update(mm_per_mesh_unit=scale,
                                              candidate_count=len(review["candidates"]))
                    started = time.monotonic()
                    response = client.post(prefix + "/preform-recovery", json={}, timeout=30)
                    assert response.status_code == 202, response.text
                    assert_public(response.json())
                    run_id = response.json()["run_id"]
                    states, polls = [response.json()["status"]], 0
                    deadline = time.monotonic() + 180
                    while time.monotonic() < deadline:
                        response = client.get(prefix + "/preform-recovery/status", timeout=15)
                        assert response.status_code == 200, response.text
                        state = response.json()
                        assert_public(state)
                        assert state["run_id"] == run_id
                        polls += 1
                        if states[-1] != state["status"]:
                            states.append(state["status"])
                        if state["status"] in {"completed", "failed"}:
                            break
                        time.sleep(.01)
                    else:
                        raise TimeoutError(specimen + " did not reach a terminal state.")
                    summaries[specimen].update(status_history=states, poll_count=polls,
                                              elapsed_seconds=round(time.monotonic() - started, 3))
                    assert state["status"] == "completed", state
                    response = client.get(prefix + "/preform-recovery/result", timeout=30)
                    assert response.status_code == 200, response.text
                    result = response.json()
                    assert_public(result)
                    assert result["regions"], "No retained mesh to verify for " + specimen
                    for cut in result["cuts"]:
                        assert cut["recommendation_status"] == "selected_verified"
                        assert cut["manufacturing_verified"] is True
                        assert len(cut["plane"]["origin_mm"]) == 3
                        assert len(cut["plane"]["normal"]) == 3
                    accounting = result["recovery_accounting"]
                    assert result["recovery_model_version"] == "v2_usable_preform"
                    assert accounting["mass_balance_valid"], accounting
                    assert abs(accounting["mass_balance_error_ct"]) <= accounting["mass_balance_tolerance_ct"]
                    assert result["target_applicable"] is True
                    assert result["confirmed_defect_excluded_ct"] == 0
                    assert abs(sum(r["usable_preform_weight_ct"] for r in result["regions"])
                               - result["usable_preform_weight_ct"]) < 1e-6
                    assert abs(sum(r["physical_weight_ct"] for r in result["regions"])
                               - result["physical_retained_weight_ct"]) < 1e-6
                    assert result["retained_preform_weight_ct"] == result["usable_preform_weight_ct"]
                    assert result["preform_recovery_percent"] == result["usable_preform_recovery_percent"]
                    assert result["target_met"] == (
                        result["usable_preform_recovery_percent"] >= result["target_recovery_percent"])
                    usable_accounting = result["usable_preform_accounting"]
                    assert abs(result["physical_retained_weight_ct"]
                               - result["usable_preform_weight_ct"]
                               - usable_accounting["nonusable_physical_weight_ct"]) < 1e-6
                    assert result["usable_region_count"] == sum(r["usable"] for r in result["regions"])
                    assert result["whole_rough_validation"]["usable"] == result["whole_rough_usable"]
                    for region in result["regions"]:
                        mesh_response = client.get(base + region["mesh_file"], timeout=30)
                        assert mesh_response.status_code == 200 and mesh_response.content
                        downloaded = scratch / (specimen + "_" + region["region_id"] + "_downloaded.ply")
                        downloaded.write_bytes(mesh_response.content)
                        region_mesh = load_mesh_preserving_topology(downloaded)
                        assert len(region_mesh.faces) > 0
                        assert abs(abs(region_mesh.volume) - region["volume_mesh_units"]) < 1e-7
                    if not result["cuts"] and accounting["explicit_discarded_weight_ct"] == 0:
                        assert abs(result["physical_retention_percent"] - 100) < 1e-6
                        if not result["whole_rough_usable"]:
                            assert result["usable_preform_weight_ct"] == 0
                    from effective_result import resolve_effective_result
                    effective = resolve_effective_result(SOURCES[specimen])
                    report = json.loads(effective["report_path"].read_text())
                    assert result["legacy_faceted_yield_percent"] == report["yield_percent"]
                    assert result["input_manifest"]["legacy_comparison_source"]["report_sha256"] == effective["report_hash"]
                    summaries[specimen]["legacy_comparison_source"] = result["input_manifest"]["legacy_comparison_source"]
                    summaries[specimen]["topology"] = result["diagnostics"]["topology"]
                    summaries[specimen]["search_trace"] = result["diagnostics"]["search_trace"]
                    summaries[specimen]["discarded_regions"] = result["discarded_regions"]
                    assert result["physical_piece_count"] == len(result["cuts"])+1
                    assert not accounting["natural_component_partitions"]
                    if not result["cuts"]:
                        assert result["physical_piece_count"] == 1
                        if not result["whole_rough_usable"]:
                            assert result["usable_preform_weight_ct"] == 0
                    assert sum(len(p["result_piece_ids"])-1 for p in accounting["partitions"])+1 == result["physical_piece_count"]
                    for key in ("physical_piece_count","candidate_region_count","reconstruction_component_count",
                                "requires_separation_weight_ct","physical_piece_graph","candidate_regions"):
                        summaries[specimen][key] = result[key]
                    components = result["diagnostics"]["physical_input_components"]
                    assert abs(sum(c["physical_weight_ct"] for c in components)
                               - result["rough_weight_ct"]) < 1e-6
                    if specimen == "QZ-05":
                        pointed = next(c for c in components if c["voxel_count"] == 673)
                        assert abs(pointed["physical_weight_ct"] - 10.412468932285062) < 1e-6
                        assert pointed["safety_voxel_count"] == 0
                        assert "candidate_geometry_eligible" in pointed
                        assert "selected_piece_contributions" in pointed
                        summaries[specimen]["pointed_component"] = pointed
                    summaries[specimen].update(
                        rough_weight_ct=result["rough_weight_ct"],
                        physical_retained_weight_ct=result["physical_retained_weight_ct"],
                        physical_retention_percent=result["physical_retention_percent"],
                        usable_preform_weight_ct=result["usable_preform_weight_ct"],
                        usable_preform_recovery_percent=result["usable_preform_recovery_percent"],
                        usable_preform_accounting=usable_accounting,
                        whole_rough_usable=result["whole_rough_usable"],
                        whole_rough_validation=result["whole_rough_validation"],
                        usable_region_count=result["usable_region_count"],
                        legacy_faceted_yield_percent=result.get("legacy_faceted_yield_percent"),
                        region_usability=[{key: r[key] for key in (
                            "region_id", "physical_weight_ct", "usable_preform_weight_ct",
                            "usable", "usability_status", "morphology", "suggested_finish_shapes",
                            "usability_reasons", "rejection_reasons", "usability_checks")}
                            for r in result["regions"]],
                        confirmed_defect_excluded_ct=result["confirmed_defect_excluded_ct"],
                        retained_preform_weight_ct=result["retained_preform_weight_ct"],
                        estimated_kerf_loss_ct=result["estimated_kerf_loss_ct"],
                        target_applicable=result["target_applicable"],
                        recovery_model_version=result["recovery_model_version"],
                        recovery_accounting=accounting, physical_input_components=components,
                        region_meshes_downloaded=len(result["regions"]),
                    )
                    summaries[specimen].update(
                        run_id=run_id, regions=len(result["regions"]), cuts=len(result["cuts"]),
                        manufacturing_status=result["manufacturing_status"],
                        preform_recovery_percent=result["preform_recovery_percent"],
                        target_met=result["target_met"], result_http=200, mesh_http=200,
                        no_private_paths=True,
                    )
                    print(specimen + ": " + json.dumps({k:summaries[specimen][k] for k in
                        ("usable_preform_recovery_percent","physical_retention_percent","cuts",
                         "legacy_faceted_yield_percent","target_met")}), flush=True)
            for original, before in hashes.items():
                assert digest(Path(original)) == before, "Original input changed."
            for source in SOURCES.values():
                after = {str(p.relative_to(source)): (p.stat().st_size, p.stat().st_mtime_ns)
                         for p in source.rglob("*") if p.is_file()}
                assert after == inventories[str(source)], "Original job inventory changed."
            (scratch / "smoke_summary.json").write_text(
                json.dumps({"source_artifacts_unchanged": True, "original_job_inventories_unchanged": True,
                            "specimens": summaries}, indent=2),
                encoding="utf-8",
            )
            print("SMOKE_REPORT=" + str(scratch / "smoke_summary.json"), flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
