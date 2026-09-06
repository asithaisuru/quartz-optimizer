import asyncio
import io
import importlib
import importlib.util
import os
import sys
import types
import unittest
from unittest import mock


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import capture_quality_api as api
    from fastapi import HTTPException
except ImportError:
    api = None
    HTTPException = None

if importlib.util.find_spec("cv2") is not None:
    import capture_quality as cq
else:
    cq = None


class FakeUpload:
    def __init__(self, filename, data=b"video-bytes"):
        self.filename = filename
        self.file = io.BytesIO(data)


def _video(label, *, readable=True, median_sharpness=45.0,
           sharp_pass=92.0, fallback=False, duplicate=0.0,
           exposure="ok", occupancy="ok"):
    return {
        "label": label,
        "filename": f"{label}.mp4",
        "file_readable": readable,
        "expected_extracted_frames": 40 if readable else 0,
        "median_sharpness": median_sharpness if readable else None,
        "normal_sharp_gate_pass_percent": sharp_pass if readable else None,
        "fallback_triggered": fallback,
        "duplicate_rate": duplicate if readable else None,
        "sampled_decode_failures": 0 if readable else None,
        "median_occupancy": 0.12 if readable else None,
        "exposure_flag": exposure if readable else "unavailable",
        "occupancy_flag": occupancy if readable else "unavailable",
        "readable_candidate_count": 10 if readable else 0,
        "normal_sharp_gate_pass_count": int(sharp_pass / 10) if readable else 0,
        "duplicate_pairs": int(round(duplicate * 9)) if readable else 0,
        "adjacent_pairs": 9 if readable else 0,
    }


def _specimen(**overrides):
    videos = {
        "01": _video("01"),
        "02": _video("02"),
        "03": _video("03"),
        "04": _video("04"),
    }
    videos.update(overrides.pop("videos", {}))
    specimen = {
        "has_videos_folder": True,
        "missing_expected_files": [],
        "typo_folders": [],
        "videos": videos,
        "video_count": len(videos),
        "readable": all(video.get("file_readable") for video in videos.values()),
        "expected_frame_total": sum(
            int(video.get("expected_extracted_frames") or 0)
            for video in videos.values()
        ),
        "fallback_count": sum(
            1 for video in videos.values()
            if video.get("fallback_triggered")
        ),
        "sharp_gate_pass_percent": 92.0,
        "overall_median_sharpness": 45.0,
        "duplicate_rate": 0.0,
        "overlap": {
            "01_02": {"max": 24, "median": 7.0, "pairs": 64},
            "02_03": {"max": 18, "median": 5.0, "pairs": 64},
            "03_04": {"max": 16, "median": 3.0, "pairs": 64},
        },
    }
    specimen.update(overrides)
    status, reason, score = cq.classification_and_score(specimen)
    specimen["capture_screen"] = status
    specimen["reason"] = reason
    specimen["quality_score"] = score
    return specimen


def _pass_report():
    return cq.frontend_report(_specimen())


@unittest.skipIf(cq is None, "OpenCV dependencies are not installed")
class CaptureQualitySharedLogicTests(unittest.TestCase):
    def test_frontend_contract_shape_for_pass_response(self):
        report = cq.frontend_report(_specimen())

        self.assertEqual(
            set(report.keys()),
            {
                "status",
                "summary",
                "override_allowed",
                "acknowledgement_required",
                "videos",
                "continuity",
            },
        )
        self.assertEqual(report["status"], "pass")
        self.assertFalse(report["override_allowed"])
        self.assertFalse(report["acknowledgement_required"])
        self.assertEqual(len(report["videos"]), 4)
        self.assertEqual(
            set(report["videos"][0].keys()),
            {
                "index",
                "readable",
                "expected_frame_count",
                "median_sharpness",
                "sharp_gate_pass_percent",
                "fallback_triggered",
                "duplicate_rate_percent",
                "exposure_flag",
                "occupancy_flag",
                "foreground_occupancy_percent",
                "decode_failures",
                "status",
                "message",
            },
        )
        self.assertEqual(
            set(report["continuity"][0].keys()),
            {"pair", "state", "message"},
        )

    def test_pass_borderline_and_fail_gating(self):
        pass_report = cq.frontend_report(_specimen())
        borderline_report = cq.frontend_report(
            _specimen(overall_median_sharpness=20.0)
        )
        fail_report = cq.frontend_report(
            _specimen(expected_frame_total=8)
        )

        self.assertEqual(pass_report["status"], "pass")
        self.assertEqual(borderline_report["status"], "borderline")
        self.assertTrue(borderline_report["acknowledgement_required"])
        self.assertFalse(borderline_report["override_allowed"])
        self.assertEqual(fail_report["status"], "fail")
        self.assertFalse(fail_report["acknowledgement_required"])
        self.assertFalse(fail_report["override_allowed"])

    def test_unreadable_video_fails_with_useful_message(self):
        report = cq.frontend_report(
            _specimen(
                videos={"02": _video("02", readable=False)},
                readable=False,
            )
        )

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["videos"][1]["status"], "fail")
        self.assertIn("Could not read", report["videos"][1]["message"])

    def test_continuity_mapping_uses_expected_pairs_in_order(self):
        report = cq.frontend_report(_specimen(overlap={
            "01_02": {"max": 20, "median": 8.0, "pairs": 64},
            "02_03": {"max": 8, "median": 2.0, "pairs": 64},
            "03_04": {"max": 3, "median": 0.0, "pairs": 64},
        }))

        self.assertEqual(
            [(row["pair"], row["state"]) for row in report["continuity"]],
            [("01 \u2192 02", "good"), ("02 \u2192 03", "weak"), ("03 \u2192 04", "poor")],
        )

    def test_corrupt_video_file_is_not_reported_as_pass(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.mp4"
            bad.write_bytes(b"not a real video")
            settings = cq.production_settings()
            video = cq.analyze_video(bad, "01", settings)

        self.assertFalse(video["file_readable"])
        self.assertEqual(cq._video_status(video), "fail")


@unittest.skipIf(api is None or cq is None, "FastAPI/OpenCV dependencies are not installed")
class CaptureQualityEndpointTests(unittest.TestCase):
    def test_endpoint_accepts_four_valid_videos_and_cleans_temp_files(self):
        temp_dirs = []

        def fake_analysis(paths, scan_mode):
            self.assertEqual(scan_mode, "turntable")
            self.assertEqual(list(paths.keys()), ["01", "02", "03", "04"])
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertTrue(path.name.startswith("video_"))
                temp_dirs.append(path.parent)
            return _pass_report()

        files = [FakeUpload(f"unsafe-{index}.mp4") for index in range(4)]
        with mock.patch.object(
            api,
            "_run_capture_quality_analysis",
            side_effect=fake_analysis,
        ):
            response = asyncio.run(api.check_capture_quality(files, "turntable"))

        self.assertEqual(response["status"], "pass")
        self.assertTrue(temp_dirs)
        self.assertTrue(all(not path.exists() for path in temp_dirs))

    def test_wrong_video_count_is_rejected(self):
        with self.assertRaises(HTTPException) as err:
            asyncio.run(api.check_capture_quality(
                [FakeUpload("one.mp4"), FakeUpload("two.mp4")],
                "turntable",
            ))

        self.assertEqual(err.exception.status_code, 422)
        self.assertIn("exactly 4", err.exception.detail)

    def test_invalid_video_extension_is_rejected(self):
        files = [
            FakeUpload("01.mp4"),
            FakeUpload("02.mov"),
            FakeUpload("03.avi"),
            FakeUpload("../04.txt"),
        ]

        with self.assertRaises(HTTPException) as err:
            asyncio.run(api.check_capture_quality(files, "turntable"))

        self.assertEqual(err.exception.status_code, 422)
        self.assertIn(".mp4, .mov, or .avi", err.exception.detail)

    def test_unexpected_analysis_failure_is_not_silently_passed(self):
        files = [FakeUpload(f"{index}.mp4") for index in range(4)]
        with mock.patch.object(
            api,
            "_run_capture_quality_analysis",
            side_effect=RuntimeError("analysis broke"),
        ), mock.patch.object(api.logger, "exception"):
            with self.assertRaises(HTTPException) as err:
                asyncio.run(api.check_capture_quality(files, "turntable"))

        self.assertEqual(err.exception.status_code, 500)
        self.assertIn("could not complete", err.exception.detail)

    def test_endpoint_does_not_import_colmap(self):
        original_import = __import__
        files = [FakeUpload(f"{index}.mp4") for index in range(4)]

        def guarded_import(name, *args, **kwargs):
            if name == "colmap_runner" or name.startswith("colmap_runner."):
                raise AssertionError("capture-quality gate imported COLMAP")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            with mock.patch.object(
                api,
                "_run_capture_quality_analysis",
                return_value=_pass_report(),
            ):
                response = asyncio.run(api.check_capture_quality(
                    files,
                    "turntable",
                ))

        self.assertEqual(response["status"], "pass")

    def test_main_registers_capture_quality_without_removing_upload(self):
        fake_report = types.ModuleType("report_generator")
        fake_report.create_pdf = lambda *args, **kwargs: None
        fake_report.find_existing_pdf = lambda *args, **kwargs: None
        fake_report.is_valid_pdf = lambda *args, **kwargs: True
        prior_main = sys.modules.pop("main", None)

        try:
            with mock.patch.dict(sys.modules, {"report_generator": fake_report}):
                main = importlib.import_module("main")
            paths = {getattr(route, "path", None) for route in main.app.routes}
        finally:
            sys.modules.pop("main", None)
            if prior_main is not None:
                sys.modules["main"] = prior_main

        self.assertIn("/upload", paths)
        self.assertIn("/capture-quality/check", paths)


if __name__ == "__main__":
    unittest.main()
