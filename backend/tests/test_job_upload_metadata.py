import asyncio
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from backend.tests.test_job_progress_extended import (
    RecordingBackgroundTasks,
    load_main,
)


class FakeUpload:
    def __init__(self, filename, content):
        self.filename = filename
        self.file = io.BytesIO(content)


class JobUploadMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.main.JOBS_DIR = self.temporary.name

    def tearDown(self):
        self.temporary.cleanup()

    def _create_job(self, *, known_weight="18.75"):
        uploads = [
            FakeUpload("QZ-07_10.mov", b"ten"),
            FakeUpload("QZ-07_2.mov", b"two"),
            FakeUpload("QZ-07_1.mov", b"one"),
        ]
        tasks = RecordingBackgroundTasks()
        response = asyncio.run(
            self.main.create_job(
                bg_tasks=tasks,
                files=uploads,
                specimen_id="QZ-07",
                source_folder="captures/2026-09-26/QZ-07",
                known_weight=known_weight,
                is_video=True,
            )
        )
        return response, tasks

    def test_videos_are_sorted_hashed_and_recorded_before_processing(self):
        response, tasks = self._create_job()
        job = Path(self.temporary.name) / response["job_id"]
        metadata = json.loads((job / "job_metadata.json").read_text("utf-8"))
        config = json.loads((job / "job_config.json").read_text("utf-8"))

        self.assertEqual(
            [item["original_filename"] for item in metadata["videos"]],
            ["QZ-07_1.mov", "QZ-07_2.mov", "QZ-07_10.mov"],
        )
        self.assertEqual(
            [item["stored_filename"] for item in metadata["videos"]],
            ["video_0.mov", "video_1.mov", "video_2.mov"],
        )
        self.assertEqual(
            [item["sha256"] for item in metadata["videos"]],
            [
                hashlib.sha256(b"one").hexdigest(),
                hashlib.sha256(b"two").hexdigest(),
                hashlib.sha256(b"ten").hexdigest(),
            ],
        )
        self.assertEqual(metadata["specimen_id"], "QZ-07")
        self.assertEqual(metadata["source_folder"], "captures/2026-09-26/QZ-07")
        self.assertEqual(metadata["rough_weight_ct"], 18.75)
        self.assertEqual(config["known_weight"], "18.75")
        self.assertEqual(config["processing_order"], metadata["processing_order"])
        self.assertEqual(len(tasks.tasks), 1)

    def test_non_positive_weight_is_rejected_before_job_creation(self):
        with self.assertRaises(self.main.HTTPException) as raised:
            self._create_job(known_weight="0")

        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
