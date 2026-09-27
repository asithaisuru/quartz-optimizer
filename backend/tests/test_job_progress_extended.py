import asyncio
import importlib.util
import json
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
def load_main():
    module_names = (
        "yield_calculator",
        "capture_quality_api",
        "report_generator",
        "dotenv",
        "fastapi",
        "fastapi.middleware",
        "fastapi.middleware.cors",
        "fastapi.staticfiles",
        "fastapi.responses",
    )
    saved = {name: sys.modules.get(name) for name in module_names}

    class DummyApp:
        def __init__(self, *args, **kwargs):
            pass

        def _decorator(self, *args, **kwargs):
            return lambda function: function

        get = post = on_event = _decorator

        def add_middleware(self, *args, **kwargs):
            pass

        def mount(self, *args, **kwargs):
            pass

        def include_router(self, *args, **kwargs):
            pass

    class DummyHTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = DummyApp
    fastapi.UploadFile = object
    fastapi.File = lambda default=None, **_kwargs: default
    fastapi.BackgroundTasks = object
    fastapi.Form = lambda default=None, **_kwargs: default
    fastapi.HTTPException = DummyHTTPException
    sys.modules["fastapi"] = fastapi

    middleware = types.ModuleType("fastapi.middleware")
    cors = types.ModuleType("fastapi.middleware.cors")
    cors.CORSMiddleware = object
    staticfiles = types.ModuleType("fastapi.staticfiles")
    staticfiles.StaticFiles = lambda *args, **kwargs: object()
    responses = types.ModuleType("fastapi.responses")
    responses.FileResponse = object
    responses.JSONResponse = lambda value, status_code=200: value
    sys.modules["fastapi.middleware"] = middleware
    sys.modules["fastapi.middleware.cors"] = cors
    sys.modules["fastapi.staticfiles"] = staticfiles
    sys.modules["fastapi.responses"] = responses

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv

    capture = types.ModuleType("capture_quality_api")
    capture.router = object()
    sys.modules["capture_quality_api"] = capture

    reports = types.ModuleType("report_generator")
    reports.create_pdf = lambda *args, **kwargs: None
    reports.find_existing_pdf = lambda *args, **kwargs: None
    reports.is_valid_pdf = lambda *args, **kwargs: False
    sys.modules["report_generator"] = reports

    calculator = types.ModuleType("yield_calculator")
    calculator.calculate_gem_stats = lambda *args, **kwargs: None
    sys.modules["yield_calculator"] = calculator
    sys.path.insert(0, str(BACKEND_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "job_progress_extended_test_main",
            BACKEND_DIR / "main.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if sys.path[0] == str(BACKEND_DIR):
            sys.path.pop(0)
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def report(
    yield_percent=20.0,
    weight=10.0,
    candidate_count=12,
    search_diagnostics=None,
):
    diagnostics = {"candidate_count": candidate_count}
    if search_diagnostics is not None:
        diagnostics["preserve_fill"] = search_diagnostics
    return {
        "yield_percent": yield_percent,
        "estimated_cut_carats": weight,
        "gem_details": [{"gem_id": 1}, {"gem_id": 2}],
        "manufacturing_plan": {"status": "complete"},
        "optimizer_diagnostics": diagnostics,
        "options": [],
    }


class RecordingBackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, function, *args, **kwargs):
        self.tasks.append((function, args, kwargs))


class JobProgressTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(BACKEND_DIR))
        import job_progress

        self.progress = job_progress
        self.temporary = tempfile.TemporaryDirectory()
        self.job = Path(self.temporary.name) / str(uuid.uuid4())
        self.job.mkdir()

    def tearDown(self):
        self.temporary.cleanup()
        if sys.path and sys.path[0] == str(BACKEND_DIR):
            sys.path.pop(0)

    def test_progress_percentage_uses_completed_stages_only(self):
        self.progress.initialize_progress(self.job)
        self.progress.update_stage(
            self.job, "capture_quality", "completed", "Capture ready."
        )
        self.progress.update_stage(
            self.job, "reconstruction", "completed", "Mesh ready."
        )
        self.progress.update_stage(
            self.job, "scale_calibration", "running", "Calibrating."
        )

        value = self.progress.progress_response(self.job.name, self.job)

        self.assertEqual(value["progress_percent"], 25.0)
        self.assertEqual(value["current_stage"], "scale_calibration")
        self.assertEqual(value["overall_status"], "running")
        self.assertEqual(len(value["stages"]), 8)

    def test_legacy_completed_job_has_compatible_progress_view(self):
        (self.job / "status.json").write_text(
            json.dumps({"status": "Completed", "progress": 100}),
            encoding="utf-8",
        )
        (self.job / "analysis_report.json").write_text("{}", encoding="utf-8")

        value = self.progress.progress_response(self.job.name, self.job)

        self.assertEqual(value["overall_status"], "completed")
        self.assertEqual(value["progress_percent"], 100.0)


class ExtendedSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.main.JOBS_DIR = self.temporary.name
        self.quality_gate = mock.patch.object(
            self.main,
            "_require_reconstruction_quality",
            return_value={"status": "PASS", "passed": True},
        )
        self.quality_gate.start()
        self.job_id = str(uuid.uuid4())
        self.job = Path(self.temporary.name) / self.job_id
        (self.job / "dense").mkdir(parents=True)
        (self.job / "dense" / "final_textured_model.ply").write_text(
            "ply\n", encoding="utf-8"
        )
        (self.job / "status.json").write_text(
            json.dumps({"status": "Completed", "progress": 100}),
            encoding="utf-8",
        )
        (self.job / "job_config.json").write_text(
            json.dumps(
                {
                    "known_weight": "50",
                    "preferred_shape": None,
                    "cut_mode": "multi",
                    "optimizer_settings": {},
                }
            ),
            encoding="utf-8",
        )
        self.original = report()
        (self.job / "analysis_report.json").write_text(
            json.dumps(self.original), encoding="utf-8"
        )

    def tearDown(self):
        self.quality_gate.stop()
        self.temporary.cleanup()

    def start(self, tasks):
        return asyncio.run(self.main.start_extended_search(self.job_id, tasks))

    def extended_status(self):
        return asyncio.run(self.main.get_extended_search_status(self.job_id))

    def test_progress_endpoint_uses_existing_status_lifecycle(self):
        self.main.initialize_progress(self.job)
        self.main.update_stage(
            self.job, "capture_quality", "completed", "Capture ready."
        )
        self.main.update_job_status(
            self.job, "Reconstruction", 30, "Reconstructing..."
        )

        value = asyncio.run(self.main.get_calculation_progress(self.job_id))
        legacy = json.loads((self.job / "status.json").read_text(encoding="utf-8"))

        self.assertEqual(value["job_id"], self.job_id)
        self.assertEqual(value["progress_percent"], 12.5)
        self.assertEqual(legacy["step"], "Reconstruction")
        self.assertIn("stages", legacy)

    def test_legacy_status_endpoint_contract_remains_available(self):
        value = asyncio.run(self.main.get_status(self.job_id))

        self.assertEqual(value["status"], "Completed")
        self.assertEqual(value["progress"], 100)
        self.assertIn("analysis_report.json", value["report_url"])

    def test_extended_search_start_is_enqueued_and_non_blocking(self):
        tasks = RecordingBackgroundTasks()

        response = self.start(tasks)

        self.assertEqual(
            response,
            {
                "started": True,
                "job_id": self.job_id,
                "message": "Extended optimization started",
            },
        )
        self.assertEqual(len(tasks.tasks), 1)
        value = self.extended_status()
        stored = json.loads(
            (self.job / "extended_search" / "status.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(value["running"])
        self.assertEqual(value["search_mode"], "extended")
        self.assertEqual(value["statistics"]["search_state"], "running")
        self.assertNotIn("timeout_seconds", stored)
        self.assertGreater(stored["resource_limit_seconds"], 270)

    def test_status_endpoint_reports_existing_values(self):
        tasks = RecordingBackgroundTasks()
        self.start(tasks)

        value = self.extended_status()

        self.assertEqual(value["current_best"]["yield_percent"], 20.0)
        self.assertEqual(value["current_best"]["weight"], 10.0)
        self.assertEqual(value["current_best"]["gem_count"], 2)
        self.assertEqual(value["current_best"]["manufacturing_status"], "complete")
        self.assertEqual(value["statistics"]["candidates_tested"], 0)
        self.assertIn("elapsed_time", value["statistics"])
        self.assertEqual(value["statistics"]["search_state"], "running")
        self.assertEqual(
            value["remaining_space_metadata"],
            {
                "remaining_space_type": "geometric_diagnostic",
                "candidate_verified": False,
                "manufacturing_verified": False,
                "search_evaluation_status": "extended_search_running",
                "diagnostic_message": (
                    "Extended search is currently re-evaluating this remaining "
                    "geometric space."
                ),
            },
        )

    def test_idle_status_identifies_original_bounded_search_diagnostic(self):
        value = self.extended_status()

        self.assertEqual(value["status"], "idle")
        self.assertEqual(
            value["remaining_space_metadata"]["search_evaluation_status"],
            "initial_bounded_search",
        )
        self.assertEqual(
            value["remaining_space_metadata"]["diagnostic_message"],
            "Available geometric space remains. The initial bounded search "
            "ended before exhaustive rejection, so this space is diagnostic "
            "only and not a verified gemstone candidate.",
        )

    def test_status_polling_does_not_artificially_stop_the_worker(self):
        tasks = RecordingBackgroundTasks()
        self.start(tasks)
        status_path = self.job / "extended_search" / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["started_at_epoch"] = 0
        status["resource_limit_seconds"] = 1
        status_path.write_text(json.dumps(status), encoding="utf-8")

        value = self.extended_status()

        self.assertTrue(value["running"])
        self.assertEqual(value["status"], "running")

    def test_improvement_preserves_v1_and_stores_v2(self):
        tasks = RecordingBackgroundTasks()
        self.start(tasks)
        function, args, kwargs = tasks.tasks[0]
        improved = report(25.0, 12.5, 27)
        calculation_kwargs = {}

        def fake_calculate(mesh_path, **passed_kwargs):
            calculation_kwargs.update(passed_kwargs)
            Path(mesh_path).with_name("best_cut.ply").write_text(
                "ply\n", encoding="utf-8"
            )
            return improved

        with mock.patch.object(
            self.main, "calculate_gem_stats", side_effect=fake_calculate
        ):
            function(*args, **kwargs)

        original_after = json.loads(
            (self.job / "analysis_report.json").read_text(encoding="utf-8")
        )
        stored = json.loads(
            (
                self.job
                / "extended_search"
                / "result_v2"
                / "analysis_report.json"
            ).read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (self.job / "extended_search" / "results.json").read_text(
                encoding="utf-8"
            )
        )
        value = self.extended_status()
        job_status = asyncio.run(self.main.get_status(self.job_id))
        effective = self.main.get_effective_result(self.job_id)

        self.assertEqual(original_after, self.original)
        self.assertEqual(stored, improved)
        self.assertEqual(manifest["best_result"], "result_v2")
        self.assertEqual(
            calculation_kwargs["optimizer_search_budget_multiplier"],
            self.main.EXTENDED_SEARCH_BUDGET_MULTIPLIER,
        )
        self.assertFalse(value["running"])
        self.assertEqual(value["status"], "completed_improvement")
        self.assertEqual(value["statistics"]["candidates_tested"], 27)
        self.assertEqual(value["statistics"]["improvements_found"], 1)
        self.assertEqual(
            value["statistics"]["search_state"], "completed_improvement"
        )
        self.assertEqual(
            value["remaining_space_metadata"]["search_evaluation_status"],
            "extended_search_completed_candidate_found",
        )
        self.assertIn(
            "superseded by the updated plan",
            value["remaining_space_metadata"]["diagnostic_message"],
        )
        self.assertEqual(value["effective_result_id"], "result_v2")
        self.assertEqual(job_status["effective_result_id"], "result_v2")
        self.assertIn(
            "/extended_search/result_v2/analysis_report.json",
            job_status["report_url"],
        )
        self.assertIn(
            "/extended_search/result_v2/best_cut.ply",
            job_status["cut_url"],
        )
        self.assertTrue(
            job_status["result_asset_base_url"].endswith(
                "/extended_search/result_v2"
            )
        )
        self.assertEqual(effective["result_id"], "result_v2")
        self.assertEqual(
            effective["report_path"],
            self.job / "extended_search" / "result_v2" / "analysis_report.json",
        )

    def test_missing_selected_v2_report_falls_back_to_root_bundle(self):
        extended = self.job / "extended_search"
        extended.mkdir()
        (extended / "results.json").write_text(
            json.dumps({"best_result": "result_v2"}),
            encoding="utf-8",
        )

        effective = self.main.get_effective_result(self.job_id)
        status = asyncio.run(self.main.get_status(self.job_id))

        self.assertEqual(effective["result_id"], "result_v1")
        self.assertEqual(effective["report_path"], self.job / "analysis_report.json")
        self.assertEqual(effective["artifact_path"], self.job / "dense")
        self.assertEqual(status["effective_result_id"], "result_v1")
        self.assertIn(f"/files/{self.job_id}/analysis_report.json", status["report_url"])

    def test_no_improvement_does_not_create_v2(self):
        tasks = RecordingBackgroundTasks()
        self.start(tasks)
        function, args, kwargs = tasks.tasks[0]
        with mock.patch.object(
            self.main,
            "calculate_gem_stats",
            return_value=report(19.0, 9.5, 18),
        ):
            function(*args, **kwargs)

        self.assertFalse(
            (self.job / "extended_search" / "result_v2").exists()
        )
        value = self.extended_status()
        self.assertEqual(value["status"], "completed_no_improvement")
        self.assertIn("No higher-yield verified manufacturing plan", value["message"])
        self.assertEqual(
            value["remaining_space_metadata"]["search_evaluation_status"],
            "extended_search_completed_no_candidate",
        )
        self.assertEqual(
            value["remaining_space_metadata"]["diagnostic_message"],
            "Extended search completed. No higher-yield verified gemstone "
            "placement was found in this remaining geometric space.",
        )

    def test_existing_v2_is_the_incumbent_and_is_not_overwritten(self):
        result_root = self.job / "extended_search" / "result_v2"
        result_root.mkdir(parents=True)
        existing_v2 = report(25.0, 12.5, 21)
        (result_root / "analysis_report.json").write_text(
            json.dumps(existing_v2), encoding="utf-8"
        )
        tasks = RecordingBackgroundTasks()
        self.start(tasks)
        function, args, kwargs = tasks.tasks[0]

        with mock.patch.object(
            self.main,
            "calculate_gem_stats",
            return_value=report(24.0, 12.0, 19),
        ):
            function(*args, **kwargs)

        stored = json.loads(
            (result_root / "analysis_report.json").read_text(encoding="utf-8")
        )
        value = self.extended_status()
        self.assertEqual(stored, existing_v2)
        self.assertEqual(value["current_best"]["yield_percent"], 25.0)
        self.assertEqual(value["status"], "completed_no_improvement")

    def test_worker_failure_is_reported_without_changing_v1(self):
        tasks = RecordingBackgroundTasks()
        self.start(tasks)
        function, args, kwargs = tasks.tasks[0]
        with mock.patch.object(
            self.main,
            "calculate_gem_stats",
            return_value={"error": "synthetic worker failure"},
        ):
            function(*args, **kwargs)

        value = self.extended_status()
        original_after = json.loads(
            (self.job / "analysis_report.json").read_text(encoding="utf-8")
        )
        self.assertFalse(value["running"])
        self.assertEqual(value["status"], "failed")
        self.assertEqual(original_after, self.original)
        self.assertEqual(
            value["remaining_space_metadata"]["search_evaluation_status"],
            "extended_search_failed",
        )
        self.assertIn(
            "failed before final evaluation",
            value["remaining_space_metadata"]["diagnostic_message"],
        )

    def test_cancel_request_is_reported_by_worker(self):
        tasks = RecordingBackgroundTasks()
        self.start(tasks)
        response = asyncio.run(self.main.cancel(self.job_id))
        function, args, kwargs = tasks.tasks[0]
        with mock.patch.object(
            self.main,
            "calculate_gem_stats",
            return_value=report(30.0, 15.0, 30),
        ):
            function(*args, **kwargs)

        self.assertEqual(
            response["message"], "Extended optimization cancellation requested"
        )
        self.assertEqual(self.extended_status()["status"], "cancelled")
        self.assertEqual(
            self.extended_status()["remaining_space_metadata"][
                "search_evaluation_status"
            ],
            "extended_search_cancelled",
        )
        self.assertIn(
            "stopped before final evaluation",
            self.extended_status()["remaining_space_metadata"][
                "diagnostic_message"
            ],
        )
        self.assertFalse(
            (self.job / "extended_search" / "result_v2").exists()
        )

    def test_system_resource_limit_is_reported_without_storing_candidate(self):
        tasks = RecordingBackgroundTasks()
        self.start(tasks)
        status_path = self.job / "extended_search" / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["started_at_epoch"] = 0
        status["resource_limit_seconds"] = 1
        status_path.write_text(json.dumps(status), encoding="utf-8")
        function, args, kwargs = tasks.tasks[0]
        with mock.patch.object(
            self.main,
            "calculate_gem_stats",
            return_value=report(30.0, 15.0, 30),
        ):
            function(*args, **kwargs)

        value = self.extended_status()
        self.assertEqual(value["status"], "resource_stopped")
        self.assertIn("system resource limit", value["message"])
        self.assertEqual(
            value["remaining_space_metadata"]["search_evaluation_status"],
            "extended_search_resource_stopped",
        )
        self.assertIn(
            "implemented resource boundary",
            value["remaining_space_metadata"]["diagnostic_message"],
        )
        self.assertFalse(
            (self.job / "extended_search" / "result_v2").exists()
        )

    def test_optimizer_resource_stop_diagnostic_is_reported(self):
        tasks = RecordingBackgroundTasks()
        self.start(tasks)
        function, args, kwargs = tasks.tasks[0]
        bounded = report(
            19.0,
            9.5,
            18,
            search_diagnostics={
                "timed_out": True,
                "stop_reason": "time_budget_exhausted",
            },
        )

        with mock.patch.object(
            self.main, "calculate_gem_stats", return_value=bounded
        ):
            function(*args, **kwargs)

        value = self.extended_status()
        self.assertEqual(value["status"], "resource_stopped")
        self.assertEqual(value["statistics"]["candidates_tested"], 18)
        self.assertEqual(value["statistics"]["improvements_found"], 0)
        self.assertEqual(
            value["remaining_space_metadata"]["search_evaluation_status"],
            "extended_search_resource_stopped",
        )
        self.assertIn(
            "implemented resource boundary",
            value["remaining_space_metadata"]["diagnostic_message"],
        )
        self.assertFalse(
            (self.job / "extended_search" / "result_v2").exists()
        )

    def test_resource_stopped_improvement_supersedes_old_diagnostic(self):
        tasks = RecordingBackgroundTasks()
        self.start(tasks)
        function, args, kwargs = tasks.tasks[0]
        improved = report(
            25.0,
            12.5,
            27,
            search_diagnostics={
                "timed_out": True,
                "stop_reason": "time_budget_exhausted",
            },
        )

        with mock.patch.object(
            self.main, "calculate_gem_stats", return_value=improved
        ):
            function(*args, **kwargs)

        value = self.extended_status()
        self.assertEqual(value["status"], "resource_stopped")
        self.assertEqual(value["statistics"]["improvements_found"], 1)
        self.assertEqual(
            value["remaining_space_metadata"]["search_evaluation_status"],
            "extended_search_completed_candidate_found",
        )

    def test_legacy_terminal_state_is_normalized_for_existing_consumers(self):
        status_root = self.job / "extended_search"
        status_root.mkdir(parents=True)
        (status_root / "status.json").write_text(
            json.dumps(
                {
                    "running": False,
                    "outcome": "no_improvement",
                    "started_at_epoch": 10,
                    "finished_at_epoch": 20,
                    "current_best": self.main._best_summary(self.original),
                    "candidates_tested": 12,
                    "improvements_found": 0,
                    "message": "Legacy result.",
                }
            ),
            encoding="utf-8",
        )

        value = self.extended_status()

        self.assertEqual(value["status"], "completed_no_improvement")
        self.assertEqual(
            value["statistics"]["search_state"], "completed_no_improvement"
        )
        self.assertEqual(
            value["remaining_space_metadata"]["search_evaluation_status"],
            "extended_search_completed_no_candidate",
        )
        self.assertEqual(
            value["remaining_space_metadata"]["diagnostic_message"],
            "Extended search completed. No higher-yield verified gemstone "
            "placement was found in this remaining geometric space.",
        )
        self.assertEqual(value["elapsed_seconds"], 10)

    def test_unavailable_feature_returns_contract(self):
        (self.job / "dense" / "final_textured_model.ply").unlink()
        tasks = RecordingBackgroundTasks()

        self.assertEqual(self.start(tasks), {"available": False})
        self.assertEqual(self.extended_status(), {"available": False})


if __name__ == "__main__":
    unittest.main()
