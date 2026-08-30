import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
HAS_REPORT_DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None
    for name in ("fpdf", "trimesh", "matplotlib", "scipy")
)
HAS_API_DEPENDENCIES = (
    HAS_REPORT_DEPENDENCIES
    and importlib.util.find_spec("fastapi") is not None
)


def minimal_stats():
    return {
        "raw_carats": 10.0,
        "estimated_cut_carats": 4.0,
        "yield_percent": 40.0,
        "rough_dimensions_mm": [10.0, 8.0, 6.0],
        "gem_details": [],
    }


def load_report_generator():
    sys.path.insert(0, str(BACKEND_DIR))
    try:
        import report_generator

        return report_generator
    finally:
        if sys.path[0] == str(BACKEND_DIR):
            sys.path.pop(0)


def load_main_without_pipeline_imports():
    dependency_functions = {
        "colmap_runner": "run_photogrammetry_pipeline",
        "video_utils": "extract_best_frames",
        "cleanup_module": "post_process_point_cloud",
        "masking_utils": "remove_backgrounds",
        "yield_calculator": "calculate_gem_stats",
        "ai_runner": "run_ai_pipeline",
        "fracture_mapper": "map_fractures_to_3d",
    }
    saved = {name: sys.modules.get(name) for name in dependency_functions}
    for module_name, function_name in dependency_functions.items():
        module = types.ModuleType(module_name)
        setattr(module, function_name, lambda *args, **kwargs: None)
        sys.modules[module_name] = module

    sys.path.insert(0, str(BACKEND_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "pdf_download_test_main",
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


@unittest.skipUnless(
    HAS_REPORT_DEPENDENCIES,
    "PDF report dependencies are not installed",
)
class PdfGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = load_report_generator()

    def test_generated_report_is_atomic_and_has_pdf_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.generator.create_pdf(tmp, str(uuid.uuid4()), minimal_stats())
            content = Path(path).read_bytes()
        self.assertEqual(Path(path).name, "report.pdf")
        self.assertTrue(content.startswith(b"%PDF-"))

    def test_failed_refresh_preserves_existing_valid_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.pdf"
            original = b"%PDF-1.7\nexisting"
            report.write_bytes(original)
            with mock.patch.object(
                self.generator.PDFReport,
                "output",
                side_effect=RuntimeError("synthetic output failure"),
            ):
                with self.assertRaises(RuntimeError):
                    self.generator.create_pdf(
                        tmp,
                        str(uuid.uuid4()),
                        minimal_stats(),
                    )
            self.assertEqual(report.read_bytes(), original)
            self.assertFalse(list(Path(tmp).glob(".report-*.pdf")))

    def test_legacy_report_filename_is_located(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "analysis_report.pdf"
            legacy.write_bytes(b"%PDF-1.7\nlegacy")
            found = self.generator.find_existing_pdf(tmp)
        self.assertEqual(Path(found).name, "analysis_report.pdf")

    def test_cut_sequence_adds_printable_step_page(self):
        stats = minimal_stats()
        stats["manufacturing_plan"] = {
            "version": "straight_full_through_v1",
            "status": "complete",
            "machine_ready": False,
            "operator_guidance_only": True,
            "machine_ready_reason": "Operator guidance only.",
            "gem_count": 2,
            "settings": {
                "blade_kerf_mm": 0.5,
                "preform_margin_mm": 0.5,
                "max_cut_depth_mm": 60.0,
            },
            "sequence": [{
                "step": 1,
                "operation": "straight_full_through_saw_cut",
                "parent_piece_id": "rough_piece_1",
                "result_piece_ids": ["rough_piece_2", "rough_piece_3"],
                "negative_side_gems": ["gem_1"],
                "positive_side_gems": ["gem_2"],
                "plane": {
                    "origin": [0, 0, 0],
                    "normal": [1, 0, 0],
                    "orientation": {"azimuth_deg": 0, "elevation_deg": 0},
                },
                "feed_direction": [0, 1, 0],
                "feed_orientation": {"azimuth_deg": 90, "elevation_deg": 0},
                "kerf_slab": {"thickness_mm": 0.5},
                "required_depth_mm": 12.0,
                "minimum_envelope_clearance_mm": 0.4,
                "section_contour": [
                    [0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1],
                ],
                "operator_checks": ["Confirm stable holding."],
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = self.generator.create_pdf(tmp, str(uuid.uuid4()), stats)
            content = Path(path).read_bytes()
        self.assertIn(b"Saw Cut 1", content)
        self.assertIn(b"Operator guidance only", content)


@unittest.skipUnless(
    HAS_API_DEPENDENCIES,
    "FastAPI PDF endpoint dependencies are not installed",
)
class PdfDownloadEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main_without_pipeline_imports()
        from fastapi import HTTPException

        cls.HTTPException = HTTPException

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.main.JOBS_DIR = self.temporary.name

    def tearDown(self):
        self.temporary.cleanup()

    def create_job(self, status="Completed", with_analysis=False):
        job_id = str(uuid.uuid4())
        job = Path(self.temporary.name) / job_id
        job.mkdir()
        (job / "status.json").write_text(
            json.dumps({"status": status, "progress": 100}),
            encoding="utf-8",
        )
        if with_analysis:
            (job / "analysis_report.json").write_text(
                json.dumps(minimal_stats()),
                encoding="utf-8",
            )
        return job_id, job

    @staticmethod
    def write_pdf(job, filename="report.pdf"):
        path = job / filename
        path.write_bytes(b"%PDF-1.7\nsynthetic report")
        return path

    def download(self, job_id):
        return asyncio.run(self.main.download_pdf_report(job_id))

    def status(self, job_id):
        return asyncio.run(self.main.get_status(job_id))

    def assert_http_error(self, status_code, callback):
        with self.assertRaises(self.HTTPException) as raised:
            callback()
        self.assertEqual(raised.exception.status_code, status_code)

    def test_existing_pdf_returns_200_pdf_attachment_and_signature(self):
        job_id, job = self.create_job()
        self.write_pdf(job)
        response = self.download(job_id)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.media_type, "application/pdf")
        self.assertIn("attachment", response.headers["content-disposition"])
        self.assertIn(job_id, response.headers["content-disposition"])
        self.assertEqual(response.headers["cache-control"], "no-store, no-cache, must-revalidate")
        self.assertTrue(Path(response.path).read_bytes().startswith(b"%PDF-"))

    def test_missing_pdf_and_analysis_returns_404(self):
        job_id, _ = self.create_job()
        self.assert_http_error(404, lambda: self.download(job_id))

    def test_processing_report_returns_409(self):
        job_id, _ = self.create_job(status="Processing")
        self.assert_http_error(409, lambda: self.download(job_id))

    def test_invalid_job_id_is_rejected(self):
        self.assert_http_error(400, lambda: self.download("not-a-uuid"))

    def test_path_traversal_is_rejected(self):
        self.assert_http_error(400, lambda: self.download("../outside"))

    def test_endpoint_cannot_access_pdf_outside_job_directory(self):
        job_id, _ = self.create_job()
        outside = Path(self.temporary.name).parent / "report.pdf"
        previous = outside.read_bytes() if outside.exists() else None
        try:
            outside.write_bytes(b"%PDF-1.7\noutside")
            self.assert_http_error(404, lambda: self.download(job_id))
        finally:
            if previous is None:
                outside.unlink(missing_ok=True)
            else:
                outside.write_bytes(previous)

    def test_status_reports_pdf_availability_and_relative_url(self):
        job_id, job = self.create_job()
        self.write_pdf(job)
        status = self.status(job_id)
        self.assertTrue(status["pdf_report_available"])
        self.assertEqual(
            status["pdf_report_url"],
            f"/api/jobs/{job_id}/report/pdf",
        )
        self.assertEqual(
            status["pdf_report_filename"],
            f"quartz-analysis-{job_id}.pdf",
        )
        self.assertIsNone(status["pdf_report_error"])
        self.assertNotIn(str(job), json.dumps(status))

    def test_status_does_not_publish_url_for_missing_pdf(self):
        job_id, _ = self.create_job()
        status = self.status(job_id)
        self.assertFalse(status["pdf_report_available"])
        self.assertIsNone(status["pdf_report_url"])

    def test_legacy_pdf_filename_is_served_without_renaming_source(self):
        job_id, job = self.create_job()
        legacy = self.write_pdf(job, "analysis_report.pdf")
        response = self.download(job_id)
        self.assertEqual(Path(response.path), legacy)
        self.assertIn(
            f"quartz-analysis-{job_id}.pdf",
            response.headers["content-disposition"],
        )

    def test_completed_legacy_job_can_generate_missing_pdf(self):
        job_id, job = self.create_job(with_analysis=True)
        response = self.download(job_id)
        content = Path(response.path).read_bytes()
        self.assertEqual(Path(response.path), job / "report.pdf")
        self.assertTrue(content.startswith(b"%PDF-"))

    def test_report_generation_error_is_visible_in_status(self):
        job_id, job = self.create_job(with_analysis=True)
        with mock.patch.object(
            self.main,
            "create_pdf",
            side_effect=RuntimeError("synthetic generation failure"),
        ):
            self.assert_http_error(500, lambda: self.download(job_id))
        status = self.status(job_id)
        self.assertFalse(status["pdf_report_available"])
        self.assertIn("generation failed", status["pdf_report_error"])
        self.assertTrue((job / "pdf_report_error.json").is_file())


if __name__ == "__main__":
    unittest.main()
