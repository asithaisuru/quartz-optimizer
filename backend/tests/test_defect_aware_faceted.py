"""Hard-volume exclusions and the additive, reconstruction-reusing API contract."""
import copy
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import defect_aware_api as api
import optimizer
from defect_aware_geometry import ConfirmedGeometry, constraint_hash, millimetre_scale
from defect_review import atomic_json, read_json


def annotation(status="confirmed", tube=False, center=(0, 0, 0), radii=(1, 1, 1)):
    return {"id": "test", "status": status, "type": "fracture" if tube else "inclusion",
            "source": "manual_3d", "notes": "", "source_frames": [],
            "geometry_type": "tube_polyline" if tube else "ellipsoid",
            "geometry": {"points_mm": [[-3, 0, 0], [3, 0, 0]], "radius_mm": .2} if tube else
                        {"center_mm": list(center), "radii_mm": list(radii)}}


def guard(*items, scale=1):
    return ConfirmedGeometry({"annotations": list(items)}, scale)


@pytest.mark.parametrize("tube", [False, True])
def test_confirmed_geometry_is_hard_mask_and_whole_volume_exclusion(tube):
    constraint = guard(annotation(tube=tube))
    mask = constraint.mask((21, 21, 21), np.full(3, -5.), .5)
    assert mask[10, 10, 10] and not mask[20, 20, 20]
    gem = trimesh.creation.box(extents=(8, 8, 8))
    # All box vertices lie outside the tiny defect, but the body encloses it.
    assert constraint.rejects(gem.vertices)
    gem.apply_translation([0, 5.001, 0])
    assert not constraint.rejects(gem.vertices)


@pytest.mark.parametrize("status", ["provisional", "rejected"])
def test_unconfirmed_has_zero_optimization_effect(status):
    constraint = guard(annotation(status))
    assert not constraint.mask((11, 11, 11), np.full(3, -5.), 1.).any()
    assert not constraint.rejects(trimesh.creation.box().vertices)


def test_multiple_defects_and_anisotropic_ellipsoid():
    constraint = guard(annotation(center=(-3, 0, 0)), annotation(center=(3, 0, 0), radii=(.1, 2, .2)))
    for x in [-3, 3]:
        gem = trimesh.creation.box(extents=(.2, .2, .2))
        gem.apply_translation([x, 0, 0])
        assert constraint.rejects(gem.vertices)
    assert not constraint.rejects(trimesh.creation.box().vertices)


def test_tube_crossing_face_and_parallel_edge_are_rejected():
    item = annotation(tube=True)
    item["geometry"]["points_mm"] = [[-2, .6, 0], [2, .6, 0]]
    assert guard(item).rejects(trimesh.creation.box().vertices)
    item["geometry"]["points_mm"] = [[-2, .701, 0], [2, .701, 0]]
    assert not guard(item).rejects(trimesh.creation.box().vertices)


def test_hash_ignores_notes_order_provisional_and_equivalent_numeric_types():
    first = {"annotations": [annotation(), annotation(center=(4, 0, 0))]}
    same = copy.deepcopy(first)
    same["annotations"].reverse()
    same["annotations"][0]["notes"] = "Reviewed again"
    same["annotations"].extend([annotation("provisional"), annotation("rejected"), annotation()])
    same["annotations"][1]["geometry"]["center_mm"] = [-0., 0., 0.]
    assert constraint_hash(first) == constraint_hash(same)
    same["annotations"][0]["geometry"]["radii_mm"][0] = 2
    assert constraint_hash(first) != constraint_hash(same)


@pytest.fixture
def service(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from preform_api import create_router
    job = tmp_path / "testjob"
    (job / "dense").mkdir(parents=True)
    mesh = trimesh.creation.box(extents=(20, 20, 20))
    mesh.apply_translation([45, -13, 7])
    mesh.export(job / "dense" / "final_textured_model.ply")
    atomic_json(job / "job_config.json", {"known_weight": 106.})  # scale exactly 1 mm/unit
    atomic_json(job / "reconstruction_quality.json", {"passed": True, "status": "PASS",
        "metrics": {"vertex_count": 8, "face_count": 12, "mesh_volume": 8000.}})
    atomic_json(job / "analysis_report.json", {"sentinel": "legacy result"})
    atomic_json(job / "defect_review.json", {"schema_version": "1.0", "annotations": [annotation()], "revision": 1})
    monkeypatch.setattr(optimizer, "_build_sdf_grid", lambda mesh:
                        (np.ones((23, 23, 23)), np.full(3, -11.), 1.))
    app = FastAPI()
    app.include_router(api.create_router(lambda _: job))
    app.include_router(create_router(lambda _: job))
    with TestClient(app) as client:
        yield job, client


def strategy(centers=((4, 0, 0),), eligible=True):
    gems = []
    for center in centers:
        gem = trimesh.creation.box(extents=(2, 2, 2))
        gem.apply_translation(center)
        gems.append(gem)
    return {"strategy": "test faceted placements", "gems": gems, "shapes": ["Emerald"]*len(gems),
            "total_volume": sum(g.volume for g in gems), "manufacturing_eligible": eligible,
            "placements": [{"shape": "Emerald", "center": list(center), "scale": 1.,
                "volume": g.volume, "orientation": np.eye(3).tolist(), "surface_clearance": 1.}
                           for g, center in zip(gems, centers)]}


def run(client):
    response = client.post("/jobs/testjob/defect-aware-optimization", json={})
    assert response.status_code == 202, response.text
    queued = response.json()
    assert queued["reused_reconstruction"] is True and queued["status"] == "queued"
    root = "/jobs/testjob/defect-aware-optimization/" + queued["run_id"]
    status = client.get(root + "/status").json()
    assert status["status"] == "completed", status
    result = client.get(root + "/result")
    assert result.status_code == 200, result.text
    return root, result.json()


def test_coordinate_frame_matches_existing_defect_review(service):
    job, client = service
    frame = client.get("/jobs/testjob/defect-review").json()["coordinate_frame"]
    mesh = trimesh.load(job / "dense" / "final_textured_model.ply", process=False)
    assert millimetre_scale(mesh, 106) == pytest.approx(frame["mm_per_mesh_unit"])
    constraint = guard(annotation(center=(4, 0, 0)), scale=2)
    gem = trimesh.creation.box()
    gem.apply_translation([2, 0, 0])
    assert constraint.rejects(gem.vertices)


def test_run_reuses_mesh_cache_keeps_legacy_and_tracks_staleness(service):
    job, client = service
    before = {p: p.read_bytes() for p in (job / "analysis_report.json", job / "dense" / "final_textured_model.ply")}
    with patch.object(optimizer, "optimize_cut", return_value=[strategy()]):
        root, result = run(client)
        assert result["gem_count"] == 1 and result["total_gem_weight_ct"] > 0
        assert result["manufacturing_status"] == "no_separation_required"
        assert not result["gems"][0]["confirmed_defect_intersection"]
        assert result["gems"][0]["orientation"] == np.eye(3).tolist()
        assert not result["base_cache"]["hit"]
        _, second = run(client)
        assert second["base_cache"]["hit"]
    assert all(p.read_bytes() == value for p, value in before.items())
    assert not client.get(root + "/result").json()["stale"]
    value = read_json(job / "defect_review.json")
    value["annotations"].append(annotation("provisional"))
    value["annotations"][0]["notes"] = "unchanged geometry"
    atomic_json(job / "defect_review.json", value)
    assert not client.get(root + "/result").json()["stale"]
    value["annotations"][0]["geometry"]["radii_mm"][0] = 2
    atomic_json(job / "defect_review.json", value)
    assert client.get(root + "/result").json()["stale"]


@pytest.mark.parametrize("candidate", [strategy(((0, 0, 0),)), strategy(eligible=False)])
def test_no_safe_verified_plan_returns_zero_without_fabrication(service, candidate):
    _, client = service
    with patch.object(optimizer, "optimize_cut", return_value=[candidate]):
        _, result = run(client)
    assert result["gem_count"] == result["total_gem_weight_ct"] == result["faceted_yield_percent"] == 0
    assert result["cut_sequence"] == []


def test_new_gems_get_new_exactly_verified_manufacturing_plan(service):
    _, client = service
    with patch.object(optimizer, "optimize_cut", return_value=[strategy(((-4, 0, 0), (4, 0, 0)))]):
        with patch.object(api, "plan_cut_sequence", wraps=api.plan_cut_sequence) as planner:
            _, result = run(client)
    assert planner.call_count == 1
    assert result["gem_count"] == 2 and len(result["cut_sequence"]) == 1
    assert result["manufacturing_plan"]["diagnostics"]["exact_sequence_verified"] is True
    assert planner.call_args.kwargs["gem_ids"] == [g["id"] for g in result["gems"]]
    assert all(not g["confirmed_defect_intersection"] for g in result["gems"])


def test_manufacturing_failure_withholds_multigem_plan(service):
    _, client = service
    with patch.object(optimizer, "optimize_cut", return_value=[strategy(((-4, 0, 0), (4, 0, 0)))]), \
         patch.object(api, "plan_cut_sequence", return_value={"status": "infeasible", "sequence": []}):
        _, result = run(client)
    assert result["gem_count"] == 0


@pytest.mark.parametrize("quality", [{}, {"passed": False, "status": "FAIL"}])
def test_saved_quality_pass_is_required(service, quality):
    job, client = service
    atomic_json(job / "reconstruction_quality.json", quality)
    assert client.post("/jobs/testjob/defect-aware-optimization", json={}).status_code == 422


def test_windows_transient_atomic_status_write_and_read(service):
    import defect_review
    job, client = service
    original = defect_review.os.replace
    seen = []
    def sharing_violation_once(source, destination):
        if Path(destination).name == "status.json" and not seen:
            seen.append(True)
            raise PermissionError("Windows sharing violation")
        return original(source, destination)
    with patch.object(defect_review.os, "replace", side_effect=sharing_violation_once), \
         patch.object(optimizer, "optimize_cut", return_value=[]):
        root, _ = run(client)
    for _ in range(10):
        assert client.get(root + "/status").json()["status"] == "completed"
    assert read_json(job / api.DIRECTORY / "status.json")["status"] == "completed"


def test_worker_failure_is_terminal_and_reconstruction_reusable(service):
    job, client = service
    with patch.object(optimizer, "optimize_cut", side_effect=RuntimeError("search failed")):
        queued = client.post("/jobs/testjob/defect-aware-optimization", json={}).json()
    root = "/jobs/testjob/defect-aware-optimization/" + queued["run_id"]
    assert client.get(root + "/status").json()["status"] == "failed"
    assert client.get(root + "/result").status_code == 409
    assert api.passed_reconstruction(job)[0] == 106


def test_failed_queue_status_does_not_leave_job_permanently_busy(service):
    job, client = service
    original = api.atomic_json
    def locked_run_status(path, value):
        if path.name == "status.json" and path.parent.name != api.DIRECTORY:
            raise PermissionError("Persistent sharing violation")
        return original(path, value)
    with patch.object(api, "atomic_json", side_effect=locked_run_status):
        assert client.post("/jobs/testjob/defect-aware-optimization", json={}).status_code == 503
    assert client.get("/jobs/testjob/defect-aware-optimization/latest").json()["status"] == "failed"
    with patch.object(optimizer, "optimize_cut", return_value=[]):
        run(client)


@pytest.mark.parametrize("defer", [True, False, None])
def test_pipeline_defer_stops_before_legacy_optimizer_and_default_still_runs(tmp_path, defer):
    from backend.tests.test_job_progress_extended import load_main
    main = load_main()
    job = tmp_path / "pipeline"
    (job / "dense").mkdir(parents=True)
    modules = {}
    for module_name, function in (("colmap_runner", "run_photogrammetry_pipeline"),
        ("video_utils", "extract_best_frames"), ("cleanup_module", "post_process_point_cloud"),
        ("masking_utils", "remove_backgrounds"), ("ai_runner", "run_ai_pipeline"),
        ("fracture_mapper", "map_fractures_to_3d")):
        module = types.ModuleType(module_name)
        setattr(module, function, Mock())
        modules[module_name] = module
    with patch.dict(sys.modules, modules), \
         patch.object(main, "_colmap_done", return_value=True), \
         patch.object(main, "_mesh_done", return_value=True), \
         patch.object(main, "_ai_done", return_value=False), \
         patch.object(main, "_require_reconstruction_quality", return_value={"metrics": {"usable_sdf_voxel_count": 100}}), \
         patch.object(main, "calculate_gem_stats", return_value={"options": []}) as optimize, \
         patch.object(main, "_generate_pdf_report", return_value=None), \
         patch.object(main, "update_job_status") as status:
        kwargs = {} if defer is None else {"defer_optimization_until_defect_review": defer}
        main.process_full_pipeline("pipeline", str(job), False, "handheld", "106", **kwargs)
    modules["colmap_runner"].run_photogrammetry_pipeline.assert_not_called()
    modules["ai_runner"].run_ai_pipeline.assert_called_once()
    modules["fracture_mapper"].map_fractures_to_3d.assert_called_once()
    if defer:
        optimize.assert_not_called()
        assert status.call_args.kwargs["status"] == "awaiting_defect_review"
        assert (job / "defect_review_preparation.json").is_file()
        assert not (job / "analysis_report.json").exists()
    else:
        optimize.assert_called_once()
        assert status.call_args.kwargs["status"] == "Completed"


def test_defect_free_guard_preserves_candidate_feasibility():
    rough = trimesh.creation.box(extents=(10, 10, 10))
    grid = np.ones((23, 23, 23))*5
    ctx = optimizer.FitContext(grid, np.full(3, -11.), 1., grid>0, np.zeros(grid.shape, bool),
        optimizer._fit_settings(rough.extents, 1., mm_per_mesh_unit=1.))
    variant = optimizer._make_variants({"Box": trimesh.creation.box()})[0]
    baseline = optimizer._candidate_fit(variant, np.zeros(3), 1., ctx)
    ctx.confirmed_guard = guard(annotation("provisional"))
    checked = optimizer._candidate_fit(variant, np.zeros(3), 1., ctx)
    np.testing.assert_array_equal(checked[0], baseline[0])
    assert checked[1:] == baseline[1:]


@pytest.mark.parametrize("defer", [True, False])
def test_upload_and_resume_preserve_deferred_option(tmp_path, defer):
    import asyncio
    from backend.tests.test_job_progress_extended import load_main, RecordingBackgroundTasks
    from backend.tests.test_job_upload_metadata import FakeUpload
    main = load_main()
    main.JOBS_DIR = str(tmp_path)
    tasks = RecordingBackgroundTasks()
    value = asyncio.run(main.create_job(bg_tasks=tasks, files=[FakeUpload("rough.mov", b"video")],
        specimen_id="QZ-test", source_folder="test capture", known_weight="106", is_video=True,
        defer_optimization_until_defect_review=defer))
    job = tmp_path / value["job_id"]
    assert read_json(job / "job_config.json")["defer_optimization_until_defect_review"] == defer
    assert tasks.tasks[0][2].get("defer_optimization_until_defect_review", False) == defer
    resumed = RecordingBackgroundTasks()
    asyncio.run(main.resume_job(value["job_id"], resumed))
    assert resumed.tasks[0][2].get("defer_optimization_until_defect_review", False) == defer


def test_awaiting_review_status_exposes_model_without_completed_result(tmp_path):
    import asyncio
    from backend.tests.test_job_progress_extended import load_main
    main = load_main()
    main.JOBS_DIR = str(tmp_path)
    job = tmp_path / "waiting"
    atomic_json(job / "status.json", {"status": "awaiting_defect_review", "progress": 85})
    result = asyncio.run(main.get_status("waiting"))
    assert result["status"] == "awaiting_defect_review"
    assert result["model_url"].endswith("/dense/final_textured_model.ply")
    assert result["defect_aware_optimization_url"].endswith("/defect-aware-optimization")
