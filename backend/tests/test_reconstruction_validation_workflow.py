import shutil
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from research.reconstruction_validation import run_batch_validation
from research.reconstruction_validation import screen_all_captures


class ReconstructionValidationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path(tempfile.mkdtemp(prefix="qz_recon_validation_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _write_job(self, job_name, fused_header, final_mesh=True):
        job_dir = self.tmp_root / job_name
        dense_dir = job_dir / "dense"
        dense_dir.mkdir(parents=True)
        (dense_dir / "fused.ply").write_text(fused_header, encoding="ascii")
        if final_mesh:
            (dense_dir / "final_textured_model.ply").write_text(
                "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n",
                encoding="ascii",
            )
        return job_dir

    def test_reconstruction_provenance_uses_ply_artifacts_not_specimen_id(self):
        dense_header = (
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 3\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "end_header\n"
        )
        sparse_header = (
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 3\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        )
        dense_job = self._write_job("dense_job", dense_header)
        sparse_job = self._write_job("sparse_job", sparse_header)

        dense_result, _ = run_batch_validation.reconstruction_provenance({
            "status": "Completed",
            "specimen_id": "QZ-30",
            "job_folder": str(dense_job),
            "mesh_path": str(dense_job / "dense" / "final_textured_model.ply"),
        })
        sparse_result, _ = run_batch_validation.reconstruction_provenance({
            "status": "Completed",
            "specimen_id": "QZ-03",
            "job_folder": str(sparse_job),
            "mesh_path": str(sparse_job / "dense" / "final_textured_model.ply"),
        })

        self.assertEqual(dense_result, "TRUE_DENSE")
        self.assertEqual(sparse_result, "SPARSE_FALLBACK")

    def test_capture_discovery_reports_typo_folder_without_using_it(self):
        specimen = self.tmp_root / "QZ-22"
        typo_dir = specimen / "vidoes"
        typo_dir.mkdir(parents=True)
        for name in ("01.MOV", "02.MOV", "03.MOV", "04.MOV"):
            (typo_dir / name).write_bytes(b"not used")

        discovery = screen_all_captures.discover_specimen(specimen)

        self.assertFalse(discovery["has_videos_folder"])
        self.assertEqual(discovery["typo_folders"], ["vidoes"])
        self.assertEqual(discovery["inspected_files"], [])
        self.assertEqual(
            discovery["missing_expected_files"],
            ["01.mov", "02.mov", "03.mov", "04.mov"],
        )

    def test_qz03_methodology_metadata_is_not_weight_equivalent_replacement(self):
        supplemental = run_batch_validation.supplemental_validation_metadata()
        qz03 = next(item for item in supplemental if item["specimen_id"] == "QZ-03")

        self.assertEqual(qz03["known_weight_ct"], "90.76")
        self.assertEqual(qz03["capture_screen"], "PASS")
        self.assertFalse(qz03["weight_equivalent_replacement_for_qz30"])
        self.assertIn(
            "additional capture-qualified validation specimen rather than as a weight-equivalent replacement",
            run_batch_validation.METHODOLOGY_METADATA_WORDING,
        )


if __name__ == "__main__":
    unittest.main()
