import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defect_review import atomic_json, candidates, confirmed_annotations, mutate, read_json, review
from preform_recovery import (DEFAULTS, morphology, optimize_preforms, recovery_metrics,
                              safety_mask, settings, voxel_surface)


def ellipsoid(status="confirmed"):
    return {
        "type": "inclusion", "source": "expert", "status": status,
        "geometry_type": "ellipsoid", "geometry": {"center_mm": [0, 0, 0], "radii_mm": [1, 1, 1]},
        "notes": "", "confidence": None, "source_frames": [],
    }


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.job = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_ellipsoid_and_fracture_persist_with_history(self):
        first = mutate(self.job, "create", ellipsoid())
        fracture = dict(ellipsoid(), type="fracture", source="manual_3d",
                        geometry_type="tube_polyline",
                        geometry={"points_mm": [[0, 0, 0], [1, 2, 3]], "radius_mm": .2})
        second = mutate(self.job, "create", fracture)
        data = review(self.job)
        self.assertEqual(data["summary"]["confirmed"], 2)
        self.assertEqual(data["annotations"][0]["geometry"], first["geometry"])
        self.assertEqual(data["annotations"][1]["geometry"], second["geometry"])
        persisted = read_json(self.job / "defect_review.json")
        self.assertEqual(persisted["schema_version"], "1.0")
        self.assertEqual(len(persisted["history"]), 2)
        self.assertIn("safety-zone approximations", data["geometry_claim"])

    def test_confirm_reject_delete_and_immutable_provenance(self):
        item = mutate(self.job, "create", ellipsoid("provisional"))
        self.assertEqual(confirmed_annotations(review(self.job)), [])
        item = mutate(self.job, "patch", {"status": "confirmed"}, item["id"])
        self.assertEqual(len(confirmed_annotations(review(self.job))), 1)
        self.assertEqual(item["confirmation"]["method"], "manual_review")
        with self.assertRaises(ValueError):
            mutate(self.job, "patch", {"source": "ai_yolo"}, item["id"])
        mutate(self.job, "patch", {"status": "rejected"}, item["id"])
        self.assertEqual(confirmed_annotations(review(self.job)), [])
        mutate(self.job, "delete", annotation_id=item["id"])
        self.assertEqual(review(self.job)["annotations"], [])
        self.assertEqual(len(read_json(self.job / "defect_review.json")["history"]), 4)

    def test_ai_provenance_sparse_association_and_review(self):
        detector = {"records": [{"prediction_id": "p1", "source": "yolo", "class_name": "fracture",
                                "confidence": .95, "image_id": "frame1", "created_at_utc": "t",
                                "accepted_for_no_cut_zone": True}]}
        path = self.job / "detections" / "policy_decisions.json"
        atomic_json(path, detector)
        original = path.read_bytes()
        item = candidates(self.job)[0]
        self.assertEqual(item["status"], "provisional")
        self.assertEqual(item["geometry"]["points_mesh_units"], [])
        atomic_json(self.job / "dense" / "defect_associations.json",
                    {"status": "complete", "warnings": [], "associations": [
                        {"xyz_aligned": [1, 2, 3], "mapping": {"prediction_ids": ["p1"]}}]})
        item = candidates(self.job)[0]
        self.assertEqual(item["geometry"]["points_mesh_units"], [[1., 2., 3.]])
        with self.assertRaises(ValueError):
            mutate(self.job, "patch", {"status": "confirmed"}, item["id"])
        mutate(self.job, "patch", {"status": "confirmed", "geometry_type": "tube_polyline",
                                  "geometry": {"points_mm": [[0, 0, 0], [1, 1, 1]], "radius_mm": .3}}, item["id"])
        data = review(self.job)
        self.assertEqual(data["candidates"], [])
        self.assertEqual(len(confirmed_annotations(data)), 1)
        self.assertEqual(data["annotations"][0]["provenance"]["prediction_id"], "p1")
        mutate(self.job, "delete", annotation_id=item["id"])
        self.assertEqual(review(self.job)["candidates"], [])
        self.assertEqual(path.read_bytes(), original)

    def test_invalid_geometry_and_nonfinite_values(self):
        for geometry in ({"center_mm": [0, 0], "radii_mm": [1, 1, 1]},
                         {"center_mm": [0, 0, 0], "radii_mm": [-1, 1, 1]},
                         {"center_mm": [float("nan"), 0, 0], "radii_mm": [1, 1, 1]}):
            with self.assertRaises(ValueError):
                mutate(self.job, "create", dict(ellipsoid(), geometry=geometry))

    def test_tube_and_ellipsoid_safety_masks(self):
        points = np.array([[0, 0, 0], [1.5, 0, 0], [5, 5, 5]])
        self.assertEqual(safety_mask(points, [ellipsoid()], 0).tolist(), [True, False, False])
        tube = dict(ellipsoid(), type="fracture", geometry_type="tube_polyline",
                    geometry={"points_mm": [[0, 0, 0], [2, 0, 0]], "radius_mm": .2})
        self.assertEqual(safety_mask(points, [tube], 0).tolist(), [True, True, False])


class RecoveryTests(unittest.TestCase):
    def test_target_math_percentage_and_configurability(self):
        value = recovery_metrics(100., 85.)
        self.assertEqual(value["target_weight_ct"], 85.)
        self.assertEqual(value["preform_recovery_percent"], 85.)
        self.assertTrue(value["target_met"])
        self.assertFalse(recovery_metrics(100., 84.)["target_met"])
        self.assertTrue(recovery_metrics(100., 84., 80.)["target_met"])
        self.assertEqual(recovery_metrics(200., 50.)["preform_recovery_percent"], 25.)
        self.assertIsNone(recovery_metrics(100., 20., constrained=True)["target_met"])
        for cfg in ({"target_recovery_percent": 101}, {"blade_kerf_mm": float("nan")},
                    {"max_regions": 1.5}, {"defect_policy": "union"}):
            with self.assertRaises(ValueError):
                settings(cfg)

    def test_morphology(self):
        rng = np.random.default_rng(42)
        cube = rng.uniform(-1, 1, (5000, 3))
        # Independent equal-variance PCA rotations would make random cubes noisy;
        # use a dense regular grid for a blocky shape.
        grid = np.array(np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-.8, .8, 15),
                                   np.linspace(-.7, .7, 14))).T.reshape(-1, 3)
        sphere = cube[np.linalg.norm(cube, axis=1) < 1]
        cone = cube[(cube[:, 0] > -.9) &
                    (np.linalg.norm(cube[:, 1:], axis=1) < (1 - cube[:, 0]) * .4)]
        cone[:, 0] *= 1.8
        for points, expected in ((grid, "blocky"), (sphere, "rounded"),
                                 (sphere * [3, 1, 1], "elongated"), (cone, "pointed")):
            self.assertEqual(morphology(points)[0], expected)

    def run_box(self, tmp, annotations=None, **kwargs):
        cfg = {**DEFAULTS, "rough_inset_mm": 0, "preform_mm": .1,
               "max_regions": 3, "min_secondary_carat": .5}
        cfg.update(kwargs.pop("request", {}))
        return optimize_preforms(trimesh.creation.box(extents=[2., 1., 1.]), 100.,
                                 cfg, {"annotations": annotations or []}, tmp,
                                 resolution=18, candidate_limit=6, time_limit=8, **kwargs)

    def test_provisional_defects_do_not_change_result_and_confirmed_do(self):
        with tempfile.TemporaryDirectory() as tmp:
            normal = self.run_box(tmp)
            provisional = self.run_box(tmp, [ellipsoid("provisional")])
            self.assertEqual(normal["retained_preform_weight_ct"], provisional["retained_preform_weight_ct"])
            self.assertEqual(provisional["confirmed_defect_excluded_ct"], 0)
            confirmed = self.run_box(tmp, [ellipsoid()])
            self.assertFalse(confirmed["target_applicable"])
            self.assertGreater(confirmed["confirmed_defect_excluded_ct"], 0)
            self.assertLess(confirmed["retained_preform_weight_ct"], normal["retained_preform_weight_ct"])
            self.assertNotIn("yield_percent", confirmed)
            self.assertIn("not polished final", " ".join(confirmed["limitations"]))

    def test_mesh_export_mass_and_inset(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_box(tmp, request={"max_regions": 1})
            self.assertEqual(result["manufacturing_status"], "no_separation_required")
            self.assertLessEqual(result["preform_recovery_percent"], 100)
            self.assertAlmostEqual(sum(r["retained_weight_ct"] for r in result["regions"]),
                                   result["retained_preform_weight_ct"])
            for region in result["regions"]:
                mesh = trimesh.load(Path(tmp) / region["mesh_file"], process=False)
                self.assertTrue(mesh.is_watertight)
                self.assertAlmostEqual(abs(mesh.volume), region["volume_mesh_units"], places=5)
            inset = self.run_box(tmp, request={"rough_inset_mm": 2, "max_regions": 1})
            self.assertAlmostEqual(inset["preform_recovery_percent"], result["preform_recovery_percent"])
            self.assertGreater(inset["recovery_accounting"]["virtual_safety_excluded_ct"],
                               result["recovery_accounting"]["virtual_safety_excluded_ct"])

    def test_manufacturing_invalid_splits_never_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_box(tmp, request={"max_cut_depth_mm": .01})
            self.assertEqual(result["cuts"], [])
            self.assertEqual(result["manufacturing_status"], "no_verified_plan")
            self.assertEqual(result["physical_retention_percent"], 100)
            self.assertEqual(result["usable_preform_recovery_percent"], 0)
            self.assertTrue(result["geometric_comparisons"])
            self.assertTrue(result["diagnostics"]["manufacturing_rejections"])

    def test_cut_line_plan_includes_subvoxel_kerf_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = optimize_preforms(
                trimesh.creation.box(extents=[2., 1., 1.]), 100.,
                {"rough_inset_mm": 0, "preform_mm": .1, "max_regions": 4},
                {"annotations": [ellipsoid()]}, tmp,
                resolution=24, candidate_limit=100, time_limit=15)
            self.assertEqual(result["manufacturing_status"], "complete")
            self.assertGreater(len(result["cuts"]), 0)
            self.assertTrue(result["manufacturing_plan"]["diagnostics"]["exact_sequence_verified"])
            self.assertGreater(result["estimated_kerf_loss_ct"], 0)
            self.assertGreater(result["diagnostics"]["usable_plan_count"], 0)
            self.assertEqual(result["discarded_regions"], [])
            dirty = [region for region in result["regions"]
                     if region["confirmed_defects_intersecting"]]
            self.assertTrue(dirty)
            for unresolved in dirty:
                self.assertIn("suggested_finish_shapes", unresolved)
                self.assertFalse(unresolved["usable"])
                self.assertEqual(unresolved["usable_preform_weight_ct"], 0)
                self.assertEqual(unresolved["usability_status"], "needs_further_separation")
                self.assertTrue(unresolved["rejection_reasons"])
                self.assertIsNone(unresolved["discard_reason"])
            self.assertEqual(result["recovery_accounting"]["explicit_discarded_weight_ct"], 0)
            self.assertLess(result["settings"]["blade_kerf_mm"] if "settings" in result else .5,
                            result["diagnostics"]["pitch_mm"])
            self.assertLessEqual(result["retained_preform_weight_ct"] + result["estimated_kerf_loss_ct"], 100)
            scale = result["coordinate_frame"]["mm_per_mesh_unit"]
            for region in (r for r in result["regions"] if r["usable"]):
                mesh = trimesh.load(Path(tmp) / region["mesh_file"], process=False)
                self.assertFalse(safety_mask(mesh.vertices * scale, [ellipsoid()], 0).any())
            self.assertIsNone(result["target_met"])

    def test_confirmed_inclusion_disables_target_even_outside_rough(self):
        annotation = ellipsoid()
        annotation["geometry"]["center_mm"] = [10000, 10000, 10000]
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_box(tmp, [annotation], request={"max_regions": 1})
            self.assertFalse(result["target_applicable"])
            self.assertIsNone(result["target_met"])
            self.assertEqual(result["confirmed_defect_excluded_ct"], 0)

    def test_disconnected_safety_core_does_not_cut_connected_physical_rough(self):
        occupied = np.zeros((15, 5, 5), dtype=bool)
        occupied[:5] = True
        occupied[10:] = True
        occupied[5:10, 2, 2] = True
        rough = voxel_surface(np.argwhere(occupied), np.zeros(3), 1.)
        self.assertTrue(rough.is_watertight)
        with tempfile.TemporaryDirectory() as tmp:
            result = optimize_preforms(rough, 100,
                                      {"rough_inset_mm": 2.0, "max_regions": 1},
                                      {"annotations": []}, tmp, resolution=30, candidate_limit=4)
            # Erosion does not remove physical material. The strong physical neck
            # requires partition testing, unavailable at max_regions=1.
            self.assertAlmostEqual(result["physical_retention_percent"], 100)
            self.assertEqual(result["whole_rough_validation"]["usability_status"], "requires_further_separation")
            self.assertEqual(result["preform_recovery_percent"], 0)
            self.assertEqual(result["cuts"], [])

    def test_nonwatertight_input_rejected(self):
        mesh = trimesh.creation.box()
        mesh.update_faces(np.arange(len(mesh.faces) - 1))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                optimize_preforms(mesh, 100, {}, {"annotations": []}, tmp)

    def test_disjoint_voxel_surface_does_not_fill_gap(self):
        indices = np.array([[0, 0, 0], [1, 0, 0], [4, 0, 0]])
        mesh = voxel_surface(indices, np.zeros(3), 1.)
        self.assertAlmostEqual(mesh.volume, 3.)


class ApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.testclient import TestClient
        from preform_api import create_router
        self.temp = tempfile.TemporaryDirectory()
        self.job = Path(self.temp.name) / "test-job"
        (self.job / "dense").mkdir(parents=True)
        trimesh.creation.box().export(self.job / "dense" / "final_textured_model.ply")
        atomic_json(self.job / "job_config.json", {"known_weight": "100"})
        self.legacy = b'{"yield_percent":26.8,"estimated_cut_carats":115.38}'
        (self.job / "analysis_report.json").write_bytes(self.legacy)
        def validate(job_id):
            if job_id != self.job.name:
                raise HTTPException(404, "Job not found.")
            return self.job
        app = FastAPI()
        app.include_router(create_router(validate))
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def test_api_crud_and_candidate_rejection(self):
        prefix = "/jobs/test-job/defect-review"
        self.assertEqual(self.client.get(prefix).json()["policy"], "confirmed_only")
        response = self.client.post(prefix + "/annotations", json=ellipsoid("provisional"))
        self.assertEqual(response.status_code, 201)
        annotation_id = response.json()["id"]
        url = prefix + "/annotations/" + annotation_id
        self.assertEqual(self.client.patch(url, json={"status": "confirmed"}).status_code, 200)
        self.assertEqual(self.client.get(prefix).json()["summary"]["confirmed"], 1)
        self.assertEqual(self.client.patch(url, json={"status": "rejected"}).status_code, 200)
        self.assertEqual(self.client.delete(url).status_code, 200)
        self.assertEqual(self.client.patch(url, json={"notes": "missing"}).status_code, 404)
        self.assertEqual(self.client.get("/jobs/missing/defect-review").status_code, 404)
        self.assertEqual(self.client.post(prefix + "/annotations", json={"type": "invalid"}).status_code, 422)

    def test_api_comparison_uses_effective_promoted_report_without_rewriting_saved_data(self):
        ext = self.job / "extended_search"
        (ext / "result_v2").mkdir(parents=True)
        atomic_json(ext / "results.json", {"best_result":"result_v2"})
        promoted = ext / "result_v2" / "analysis_report.json"
        atomic_json(promoted, {"yield_percent":34.7})
        before = promoted.read_bytes()
        response = self.client.post("/jobs/test-job/preform-recovery", json={"max_regions":1})
        self.assertEqual(response.status_code, 202)
        result = self.client.get("/jobs/test-job/preform-recovery/result").json()
        self.assertEqual(result["legacy_faceted_yield_percent"], 34.7)
        self.assertEqual(result["input_manifest"]["legacy_comparison_source"]["result_id"], "result_v2")
        self.assertEqual(promoted.read_bytes(), before)
        self.assertEqual((self.job / "analysis_report.json").read_bytes(), self.legacy)

    def test_real_run_polling_persistence_and_legacy_unchanged(self):
        prefix = "/jobs/test-job/preform-recovery"
        self.assertEqual(self.client.get(prefix + "/status").json()["status"], "idle")
        self.assertEqual(self.client.get(prefix + "/result").status_code, 404)
        response = self.client.post(prefix, json={"max_regions": 1, "rough_inset_mm": 0})
        self.assertEqual(response.status_code, 202)
        run_id = response.json()["run_id"]
        self.assertEqual(self.client.get(prefix + "/status").json()["status"], "completed")
        result = self.client.get(prefix + "/result").json()
        self.assertEqual(result["run_id"], run_id)
        self.assertEqual(result["legacy_faceted_yield_percent"], 26.8)
        self.assertEqual(result["target_recovery_source"], "expert_defined")
        self.assertTrue(result["regions"][0]["mesh_file"].startswith("/files/test-job/preform_recovery/"))
        self.assertEqual(result["recovery_model_version"], "v2_usable_preform")
        self.assertEqual(result["retained_preform_weight_ct"], result["usable_preform_weight_ct"])
        self.assertEqual(result["preform_recovery_percent"], result["usable_preform_recovery_percent"])
        self.assertIn("physical_retention_percent", result)
        self.assertIn("whole_rough_validation", result)
        self.assertIn("usable_preform_accounting", result)
        run = self.job / "preform_recovery" / run_id
        for name in ("request.json", "result.json", "status.json", "defect_review_snapshot.json", "input_manifest.json"):
            self.assertTrue((run / name).is_file())
        self.assertEqual((self.job / "analysis_report.json").read_bytes(), self.legacy)
        self.assertEqual(read_json(self.job / "job_config.json"), {"known_weight": "100"})
        modes = self.client.get("/optimization-modes").json()
        self.assertEqual(modes["default"], "legacy_faceted_pack")
        original_result = (run / "result.json").read_bytes()
        response2 = self.client.post(prefix, json={"max_regions": 1, "target_recovery_percent": 75})
        self.assertNotEqual(run_id, response2.json()["run_id"])
        self.assertEqual((run / "result.json").read_bytes(), original_result)
        self.assertEqual(self.client.get(prefix + "/result").json()["target_recovery_percent"], 75)

    def test_startup_marks_dead_worker_failed(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from preform_api import create_router
        import uuid
        run_id = uuid.uuid4().hex
        atomic_json(self.job / "preform_recovery" / "status.json",
                    {"status": "running", "run_id": run_id, "worker_pid": -1})
        app = FastAPI()
        app.include_router(create_router(lambda _: self.job, lambda: self.job.parent))
        with TestClient(app) as client:
            status = client.get("/jobs/test-job/preform-recovery/status").json()
            self.assertEqual(status["status"], "failed")
            self.assertTrue((self.job / "preform_recovery" / run_id / "status.json").is_file())

    def test_busy_validation_and_no_stale_result(self):
        prefix = "/jobs/test-job/preform-recovery"
        self.assertEqual(self.client.post(prefix, json={"defect_policy": "union"}).status_code, 422)
        atomic_json(self.job / "preform_recovery" / "status.json", {"status": "running", "run_id": "busy"})
        self.assertEqual(self.client.post(prefix, json={}).status_code, 409)
        self.assertEqual(self.client.get(prefix + "/result").status_code, 409)

    def test_failure_is_reported_without_overwriting_legacy(self):
        prefix = "/jobs/test-job/preform-recovery"
        with self.assertLogs("preform_api", level="ERROR"), patch("preform_api.optimize_preforms", side_effect=ValueError("synthetic failure")):
            self.assertEqual(self.client.post(prefix, json={}).status_code, 202)
        self.assertEqual(self.client.get(prefix + "/status").json()["status"], "failed")
        self.assertEqual(self.client.get(prefix + "/result").status_code, 409)
        self.assertEqual((self.job / "analysis_report.json").read_bytes(), self.legacy)


if __name__ == "__main__":
    unittest.main()
