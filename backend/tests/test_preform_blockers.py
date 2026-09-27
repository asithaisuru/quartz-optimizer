"""Regression tests for Windows polling races, topology and private errors."""
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defect_review import atomic_json, json_file_lock, read_json
import preform_api
from preform_api import create_router, current_status, record_status, run_recovery
from reconstruction_quality import assess_reconstruction_quality
from mesh_artifacts import load_mesh_preserving_topology


class PreformBlockerTests(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        self.temp = tempfile.TemporaryDirectory()
        self.job = Path(self.temp.name) / str(uuid.uuid4())
        (self.job / "dense").mkdir(parents=True)
        self.mesh_path = self.job / "dense" / "final_textured_model.ply"
        trimesh.creation.box().export(self.mesh_path)
        atomic_json(self.job / "job_config.json", {"known_weight": "100"})
        app = FastAPI()
        app.include_router(create_router(lambda _: self.job))
        self.client = TestClient(app)
        self.url = "/jobs/" + self.job.name + "/preform-recovery"

    def tearDown(self):
        self.client.close()
        preform_api._VOLATILE_FAILURES.pop(str(self.job.resolve()), None)
        self.temp.cleanup()

    def test_concurrent_http_polling_and_replacements(self):
        run = self.job / "preform_recovery" / uuid.uuid4().hex
        state = {"run_id": run.name, "status": "queued"}
        record_status(self.job, run, state)
        done = threading.Event()
        real_replace = os.replace
        attempts, retries = [0], [0]
        def flaky_replace(source, destination):
            attempts[0] += 1
            if attempts[0] % 11 == 0:
                retries[0] += 1
                error = PermissionError(errno.EACCES, "simulated Windows sharing violation")
                error.winerror = 5
                raise error
            return real_replace(source, destination)
        def writer():
            try:
                for i in range(60):
                    record_status(self.job, run, {**state, "status": "running", "message": f"Update {i}"})
                    time.sleep(.002)
                record_status(self.job, run, {**state, "status": "completed"})
            finally:
                done.set()
        def reader():
            states = []
            while not done.is_set() or len(states) < 15:
                response = self.client.get(self.url + "/status")
                self.assertEqual(response.status_code, 200)
                value = response.json()
                self.assertIsInstance(value["message"], str)
                states.append(value["status"])
            return states
        with patch("defect_review.os.replace", side_effect=flaky_replace), ThreadPoolExecutor(max_workers=4) as pool:
            readers = [pool.submit(reader) for _ in range(3)]
            writer_future = pool.submit(writer)
            writer_future.result(timeout=20)
            observed = [future.result(timeout=20) for future in readers]
        self.assertGreater(retries[0], 0)
        self.assertTrue(all(len(states) >= 15 for states in observed))
        self.assertEqual(self.client.get(self.url + "/status").json()["status"], "completed")
        self.assertEqual(read_json(run / "status.json")["status"], "completed")
        self.assertEqual(read_json(self.job / "preform_recovery" / "status.json")["status"], "completed")

    def test_atomic_retry_is_bounded_and_keeps_original_json(self):
        path = self.job / "preform_recovery" / "status.json"
        atomic_json(path, {"status": "queued"})
        with patch("defect_review.os.replace", side_effect=PermissionError("locked")) as replace:
            with self.assertRaises(PermissionError):
                atomic_json(path, {"status": "running"})
        self.assertEqual(replace.call_count, 8)
        self.assertEqual(read_json(path), {"status": "queued"})
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_unrelated_status_files_are_not_serialized(self):
        first = self.job / "one.json"
        second = self.job / "two.json"
        with ThreadPoolExecutor(max_workers=1) as pool:
            with json_file_lock(first):
                future = pool.submit(atomic_json, second, {"status": "completed"})
                future.result(timeout=2)
        self.assertEqual(read_json(second)["status"], "completed")

    def test_exhausted_running_transition_becomes_failed(self):
        real_replace = os.replace
        def deny_running(source, destination):
            if Path(destination).name == "status.json" and json.loads(Path(source).read_text())["status"] == "running":
                raise PermissionError(5, "access denied", str(destination))
            return real_replace(source, destination)
        with self.assertLogs("preform_api", level="ERROR"), patch("defect_review.os.replace", side_effect=deny_running):
            response = self.client.post(self.url, json={"max_regions": 1})
        self.assertEqual(response.status_code, 202)
        status = self.client.get(self.url + "/status").json()
        self.assertEqual(status["status"], "failed")
        self.assertNotIn(str(self.job), status["message"])
        self.assertEqual(read_json(self.job / "preform_recovery" / "status.json")["status"], "failed")

    def test_failed_terminal_replacement_uses_durable_failure_file(self):
        real_replace = os.replace
        def deny_terminal(source, destination):
            if Path(destination).name == "status.json" and json.loads(Path(source).read_text())["status"] in {"completed", "failed"}:
                raise PermissionError(5, "locked", str(destination))
            return real_replace(source, destination)
        with self.assertLogs("preform_api", level="ERROR"), patch("defect_review.os.replace", side_effect=deny_terminal):
            response = self.client.post(self.url, json={"max_regions": 1})
        self.assertEqual(response.status_code, 202)
        run_id = response.json()["run_id"]
        self.assertTrue((self.job / "preform_recovery" / run_id / "failure.json").is_file())
        # Prove another process can observe failure, not only the in-memory fallback.
        preform_api._VOLATILE_FAILURES.clear()
        self.assertEqual(self.client.get(self.url + "/status").json()["status"], "failed")
        self.assertEqual(self.client.get(self.url + "/result").status_code, 409)

    def test_total_storage_failure_is_terminal_in_worker(self):
        run = self.job / "preform_recovery" / uuid.uuid4().hex
        record_status(self.job, run, {"run_id": run.name, "status": "queued"})
        with self.assertLogs("preform_api", level="ERROR"), patch("preform_api.atomic_json", side_effect=PermissionError("storage unavailable")):
            run_recovery(self.job, run, {}, {"revision": 0, "annotations": []}, 100, None)
        self.assertEqual(current_status(self.job)["status"], "failed")

    def test_private_paths_are_logged_but_never_returned_in_failure_status(self):
        paths = [r"D:\private server\quartz\status.json", "/srv/private/quartz/status.json",
                 r"\\server\share\quartz\status.json"]
        for private_path in paths:
            with self.subTest(path=private_path):
                error = RuntimeError("Cannot process:" + private_path)
                with self.assertLogs("preform_api", level="ERROR") as logs, patch("preform_api.optimize_preforms", side_effect=error):
                    self.assertEqual(self.client.post(self.url, json={"max_regions": 1}).status_code, 202)
                self.assertIn(private_path, "\n".join(logs.output))
                status = self.client.get(self.url + "/status").json()
                self.assertEqual(status["status"], "failed")
                self.assertNotIn(private_path, json.dumps(status))
                self.assertNotIn(private_path, json.dumps(self.client.get(self.url + "/result").json()))
        # Historical status messages are also sanitized at read time.
        preform_api._VOLATILE_FAILURES.clear()
        atomic_json(self.job / "preform_recovery" / "status.json",
                    {"run_id": "historic", "status": "failed", "message": "Failed at /srv/private/job"})
        self.assertNotIn("/srv/private", self.client.get(self.url + "/status").json()["message"])

    def test_zero_area_centroid_uses_existing_voxel_corner_envelope(self):
        from preform_recovery import _envelope
        points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
        real_hull = trimesh.convex.convex_hull
        calls = [0]
        def hull(value):
            calls[0] += 1
            if calls[0] == 1:
                raise ZeroDivisionError("Weights sum to zero, can't be normalized")
            return real_hull(value)
        with patch("preform_recovery.trimesh.convex.convex_hull", side_effect=hull):
            envelope = _envelope(points, .2)
        self.assertTrue(envelope.is_watertight)
        self.assertAlmostEqual(envelope.volume, 3.2 * .2 * .2)

    def test_topology_preserving_load_reaches_recovery_and_calibrates_frame(self):
        left, right = trimesh.creation.box(), trimesh.creation.box()
        right.apply_translation([1, 0, 0])
        mesh = trimesh.util.concatenate([left, right])
        mesh.export(self.mesh_path)
        before = self.mesh_path.read_bytes()
        processed = trimesh.load(self.mesh_path, force="mesh")
        canonical = load_mesh_preserving_topology(self.mesh_path)
        self.assertFalse(processed.is_watertight)
        self.assertTrue(canonical.is_watertight)
        self.assertEqual(len(canonical.vertices), len(mesh.vertices))
        self.assertEqual(len(canonical.faces), len(mesh.faces))
        np.testing.assert_array_equal(canonical.faces, mesh.faces)
        self.assertTrue(assess_reconstruction_quality(self.mesh_path, sdf_resolution=24)["passed"])
        review = self.client.get("/jobs/" + self.job.name + "/defect-review")
        self.assertEqual(review.status_code, 200)
        self.assertGreater(review.json()["coordinate_frame"]["mm_per_mesh_unit"], 0)
        response = self.client.post(self.url, json={"max_regions": 1})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.client.get(self.url + "/status").json()["status"], "completed")
        self.assertEqual(self.client.get(self.url + "/result").status_code, 200)
        self.assertEqual(self.mesh_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
