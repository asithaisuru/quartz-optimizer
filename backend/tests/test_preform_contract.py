"""Reconciliation tests for the public preform and defect-review contract."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defect_review import atomic_json, candidates, confirmed_annotations, read_json, review, source_frames
from preform_api import canonical_result, create_router, status_response
from preform_recovery import optimize_preforms

REQUIRED_RESULT = set("""mode recovery_basis rough_weight_ct target_recovery_percent
target_recovery_source target_applicable target_context retained_preform_weight_ct
preform_recovery_percent target_met estimated_kerf_loss_ct confirmed_defect_excluded_ct
regions cuts manufacturing_status search_state message""".split())
REGION_FIELDS = set("""region_id retained_weight_ct volume_mesh_units morphology
suggested_finish_shapes shape_compatibility_score mesh_file confirmed_defects_intersecting""".split())
CUT_FIELDS = set("""cut_id recommendation_status manufacturing_verified sequence plane
required_depth_mm kerf_mm parent_piece_id result_piece_ids region_ids discarded_region_ids""".split())


def ellipsoid():
    return {"type": "inclusion", "source": "manual_3d", "status": "provisional",
            "geometry_type": "ellipsoid",
            "geometry": {"center_mm": [0, 0, 0], "radii_mm": [1, 1, 1]}}


class ContractTests(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from fastapi.testclient import TestClient
        self.temp = tempfile.TemporaryDirectory()
        self.job = Path(self.temp.name) / "job1"
        (self.job / "dense").mkdir(parents=True)
        (self.job / "images").mkdir()
        (self.job / "images" / "frame 1.jpg").write_bytes(b"test image artifact")
        mesh = trimesh.creation.box()
        mesh.apply_translation([8, -3, 12])
        mesh.export(self.job / "dense" / "final_textured_model.ply")
        atomic_json(self.job / "job_config.json", {"known_weight": "100"})
        app = FastAPI()
        app.mount("/files", StaticFiles(directory=self.job.parent))
        app.include_router(create_router(lambda _: self.job))
        self.client = TestClient(app)
        self.review_url = "/jobs/job1/defect-review"
        self.recovery_url = "/jobs/job1/preform-recovery"

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def test_flat_geometry_required_and_fracture_round_trip(self):
        response = self.client.post(self.review_url + "/annotations", json=ellipsoid())
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["geometry"], ellipsoid()["geometry"])
        for bad in (
            {"type": "inclusion", "geometry": {"ellipsoid": ellipsoid()["geometry"]}},
            {**ellipsoid(), "geometry": {"ellipsoid": ellipsoid()["geometry"]}},
            {"type": "inclusion", "geometry": ellipsoid()["geometry"]},
        ):
            self.assertEqual(self.client.post(self.review_url + "/annotations", json=bad).status_code, 422)
        fracture = {**ellipsoid(), "type": "fracture", "geometry_type": "tube_polyline",
                    "geometry": {"points_mm": [[0, 0, 0], [2, 0, 0]], "radius_mm": .5}}
        response = self.client.post(self.review_url + "/annotations", json=fracture)
        self.assertEqual(response.status_code, 201)
        url = self.review_url + "/annotations/" + response.json()["id"]
        self.assertEqual(self.client.patch(url, json={"status": "confirmed"}).status_code, 200)
        self.assertEqual(self.client.patch(url, json={"status": "rejected"}).status_code, 200)
        self.assertEqual(self.client.delete(url).status_code, 200)

    def test_candidate_materialization_preserves_provenance_without_inventing_depth(self):
        artifact = self.job / "detections" / "policy_decisions.json"
        atomic_json(artifact, {"records": [
            {"prediction_id": source, "source": source, "class_name": "inclusion",
             "confidence": .7, "image_id": "frame 1.jpg", "source_image_path": "images/frame 1.jpg"}
            for source in ("yolo", "opencv")]})
        original = artifact.read_bytes()
        payload = self.client.get(self.review_url).json()
        self.assertEqual(payload["summary"]["provisional"], 2)
        for item in payload["candidates"]:
            self.assertEqual(item["geometry"]["points_mesh_units"], [])
            self.assertEqual(item["source_frames"], [{"frame": "frame 1.jpg",
                              "url": "/files/job1/images/frame%201.jpg"}])
            self.assertEqual(self.client.get(item["source_frames"][0]["url"]).status_code, 200)
            url = self.review_url + "/annotations/" + item["id"]
            self.assertEqual(self.client.patch(url, json={"status": "confirmed"}).status_code, 422)
            self.assertEqual(self.client.patch(url, json={"status": "rejected"}).status_code, 200)
            # The same ID can subsequently be confirmed with explicit human geometry.
            response = self.client.patch(url, json={"status": "confirmed", "geometry_type": "ellipsoid",
                                                    "geometry": ellipsoid()["geometry"]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["id"], item["id"])
            self.assertEqual(response.json()["source"], item["source"])
            self.assertEqual(response.json()["provenance"], item["provenance"])
        payload = self.client.get(self.review_url).json()
        self.assertEqual(payload["candidates"], [])
        self.assertEqual(len(confirmed_annotations(payload)), 2)
        self.assertEqual(artifact.read_bytes(), original)

    def test_source_frames_never_fabricate_or_expose_outside_urls(self):
        outside = Path(self.temp.name) / "outside.jpg"
        outside.write_bytes(b"outside")
        entries = source_frames(self.job, [
            "images/frame 1.jpg", "missing.jpg", str(outside),
            {"frame": "remote.jpg", "url": "https://example.com/remote.jpg"},
            {"frame": "traversal.jpg", "url": "/files/job1/../outside.jpg"}])
        self.assertEqual(entries[0]["url"], "/files/job1/images/frame%201.jpg")
        self.assertTrue(all(x["url"] is None for x in entries[1:]))
        self.assertTrue(all(set(x) == {"frame", "url"} for x in entries))
        self.assertNotIn(str(self.temp.name), json.dumps(entries))
        response = self.client.post(self.review_url + "/annotations",
                                    json={**ellipsoid(), "source_frames": ["images/frame 1.jpg"]})
        self.assertEqual(response.json()["source_frames"], entries[:1])

    def test_status_messages_and_queued_response(self):
        for state in ("idle", "queued", "running", "completed", "failed"):
            value = status_response({"status": state, "run_id": None if state == "idle" else "r",
                                     "worker_pid": 123})
            self.assertTrue(value["message"])
            self.assertNotIn("running", value)
            self.assertNotIn("worker_pid", value)
        self.assertEqual(self.client.get(self.recovery_url + "/status").json()["status"], "idle")
        with patch("preform_api.run_recovery"):
            response = self.client.post(self.recovery_url, json={"max_regions": 1})
        self.assertEqual(response.status_code, 202)
        status = self.client.get(self.recovery_url + "/status").json()
        self.assertEqual(status["status"], "queued")
        self.assertTrue(status["message"])

    def test_result_region_coordinate_and_asset_contract(self):
        frame = self.client.get(self.review_url).json()["coordinate_frame"]
        self.assertEqual(frame["canonical_to_centered_translation_mesh_units"], [-8, 3, -12])
        scale = frame["mm_per_mesh_unit"]
        centred = np.array([.1, .2, -.3])
        millimetres = centred * scale
        # Inverse viewer rotation and physical scaling returns the same local point.
        viewer = np.array([centred[0], centred[2], -centred[1]])
        unrotated = np.array([viewer[0], -viewer[2], viewer[1]])
        np.testing.assert_allclose(unrotated * scale, millimetres)
        self.assertEqual(self.client.post(self.recovery_url, json={"max_regions": 1}).status_code, 202)
        self.assertEqual(self.client.get(self.recovery_url + "/status").json()["status"], "completed")
        value = self.client.get(self.recovery_url + "/result").json()
        self.assertTrue(REQUIRED_RESULT <= set(value))
        self.assertEqual(value["coordinate_frame"]["mm_per_mesh_unit"], scale)
        for region in value["regions"]:
            self.assertTrue(REGION_FIELDS <= set(region))
            self.assertNotIn("file", region)
            self.assertNotIn("shape_compatibility", region)
            self.assertEqual(self.client.get(region["mesh_file"]).status_code, 200)
        self.assertEqual(value["cuts"], [])

    def test_real_cut_canonical_schema_and_unverified_gate(self):
        run = self.job / "preform_recovery" / "test-run"
        annotation = {**ellipsoid(), "status": "confirmed", "notes": "", "confidence": None, "source_frames": []}
        result = optimize_preforms(trimesh.creation.box(extents=[2, 1, 1]), 100,
                                   {"rough_inset_mm": 0, "preform_mm": .1, "max_regions": 4},
                                   {"annotations": [annotation]}, run,
                                   resolution=24, candidate_limit=100, time_limit=15)
        raw = copy.deepcopy(result)
        output = canonical_result(result, self.job, run)
        self.assertTrue(output["cuts"])
        for cut, original in zip(output["cuts"], raw["cuts"]):
            self.assertEqual(set(cut), CUT_FIELDS)
            self.assertEqual(set(cut["plane"]), {"origin_mm", "normal"})
            self.assertEqual(cut["recommendation_status"], "selected_verified")
            self.assertTrue(cut["manufacturing_verified"])
            np.testing.assert_allclose(cut["plane"]["origin_mm"],
                                       np.array(original["plane"]["origin"]) * output["coordinate_frame"]["mm_per_mesh_unit"])
            self.assertAlmostEqual(np.linalg.norm(cut["plane"]["normal"]), 1)
        self.assertEqual(canonical_result(output, self.job, run), output)
        self.assertEqual(raw["cuts"], result["cuts"])
        result["manufacturing_plan"]["diagnostics"]["exact_sequence_verified"] = False
        rejected = canonical_result(result, self.job, run)["cuts"]
        self.assertTrue(all(c["recommendation_status"] == "geometric_comparison_only" and
                            c["manufacturing_verified"] is False for c in rejected))


class ActualAppSmokeTests(unittest.TestCase):
    def test_actual_app_crud_candidate_recovery_and_resources(self):
        script = r"""
import json, os, sys, uuid
from pathlib import Path
sys.path.insert(0, "backend")
import main
import trimesh
from fastapi.testclient import TestClient
job_id = str(uuid.uuid4())
job = Path(main.JOBS_DIR) / job_id
(job / "dense").mkdir(parents=True)
(job / "images").mkdir()
(job / "detections").mkdir()
trimesh.creation.box().export(job / "dense" / "final_textured_model.ply")
(job / "images" / "frame.jpg").write_bytes(b"frame")
(job / "job_config.json").write_text(json.dumps({"known_weight": "100"}))
legacy = b'{"yield_percent":26.8}'
(job / "analysis_report.json").write_bytes(legacy)
(job / "detections" / "policy_decisions.json").write_text(json.dumps({"records": [
    {"source": "yolo", "prediction_id": "p1", "class_name": "inclusion", "confidence": .9,
     "image_id": "frame.jpg", "source_image_path": "images/frame.jpg"}]}))
prefix = "/jobs/" + job_id
with TestClient(main.app) as client:
    review = client.get(prefix + "/defect-review")
    assert review.status_code == 200, review.text
    candidate = review.json()["candidates"][0]
    assert client.get(candidate["source_frames"][0]["url"]).status_code == 200
    for kind, defect_type, geometry in (
        ("ellipsoid", "inclusion", {"center_mm": [0,0,0], "radii_mm": [.1,.1,.1]}),
        ("tube_polyline", "fracture", {"points_mm": [[0,0,0],[1,0,0]], "radius_mm": .1})):
        response = client.post(prefix + "/defect-review/annotations", json={
            "type": defect_type, "source": "manual_3d", "status": "provisional",
            "geometry_type": kind, "geometry": geometry})
        assert response.status_code == 201, response.text
        url = prefix + "/defect-review/annotations/" + response.json()["id"]
        assert client.patch(url, json={"status":"confirmed"}).status_code == 200
        assert client.patch(url, json={"status":"rejected"}).status_code == 200
        assert client.delete(url).status_code == 200
    url = prefix + "/defect-review/annotations/" + candidate["id"]
    assert client.patch(url, json={"status":"confirmed"}).status_code == 422
    converted = client.patch(url, json={"status":"confirmed", "geometry_type":"ellipsoid",
                             "geometry":{"center_mm":[0,0,0], "radii_mm":[.1,.1,.1]}})
    assert converted.status_code == 200, converted.text
    assert converted.json()["source"] == "ai_yolo"
    assert client.patch(url, json={"status":"rejected"}).status_code == 200
    assert client.delete(url).status_code == 200
    response = client.post(prefix + "/preform-recovery", json={"max_regions":1})
    assert response.status_code == 202, response.text
    status = client.get(prefix + "/preform-recovery/status")
    assert status.status_code == 200 and status.json()["status"] == "completed", status.text
    assert status.json()["message"]
    result = client.get(prefix + "/preform-recovery/result")
    assert result.status_code == 200, result.text
    assert result.json()["mode"] == "preform_recovery"
    assert result.json()["regions"]
    for region in result.json()["regions"]:
        assert client.get(region["mesh_file"]).status_code == 200
    assert (job / "analysis_report.json").read_bytes() == legacy
print("Actual app: defect CRUD, AI materialization, frame GET, 202 start -> 200 status -> 200 result -> mesh GET passed.")
"""
        with tempfile.TemporaryDirectory() as tmp:
            environment = dict(os.environ, STORAGE_PATH=tmp)
            result = subprocess.run([sys.executable, "-B", "-c", script],
                                    cwd=Path(__file__).resolve().parents[2], env=environment,
                                    text=True, capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("mesh GET passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
