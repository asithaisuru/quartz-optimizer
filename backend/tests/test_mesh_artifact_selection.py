import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import trimesh


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import mesh_artifacts
from mesh_artifacts import (
    VALIDATION_REPORT_FILENAME,
    load_mesh_preserving_topology,
    prepare_optimizer_mesh_artifacts,
    validate_mesh_artifact,
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MeshArtifactSelectionTests(unittest.TestCase):
    def test_prepared_optimizer_mesh_preserves_canonical_invariants(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            dense = job / "dense"
            dense.mkdir()
            canonical = dense / "final_textured_model.ply"
            trimesh.creation.box(extents=(7.0, 5.0, 3.0)).export(canonical)
            original_hash = file_sha256(canonical)

            prepared = prepare_optimizer_mesh_artifacts(canonical, job_folder=job)
            report = prepared["report"]

            self.assertEqual(file_sha256(canonical), original_hash)
            self.assertTrue(report["canonical_unchanged"])
            self.assertEqual(report["status"], "PASS")
            self.assertFalse(report["fallback_to_canonical"])
            self.assertEqual(
                Path(prepared["optimizer_mesh_path"]).name,
                "final_textured_model_opt_input.ply",
            )
            comparison = report["optimizer_validation"]["comparison"]
            self.assertTrue(comparison["safe_to_use"])
            runtime_comparison = report["optimizer_runtime_validation"][
                "comparison"
            ]
            self.assertTrue(runtime_comparison["safe_to_use"])
            self.assertFalse(prepared["optimizer_input"]["process"])
            self.assertLessEqual(
                comparison["volume_relative_difference"],
                comparison["volume_relative_tolerance"],
            )
            persisted = json.loads(
                (job / VALIDATION_REPORT_FILENAME).read_text("utf-8")
            )
            self.assertEqual(persisted["status"], "PASS")

    def test_validation_rejects_negative_topology_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical_path = root / "canonical.ply"
            candidate_path = root / "candidate.ply"
            canonical = trimesh.creation.box(extents=(4.0, 3.0, 2.0))
            canonical.export(canonical_path)
            candidate = canonical.copy()
            candidate.update_faces(np.arange(len(candidate.faces)) != 0)
            candidate.remove_unreferenced_vertices()
            candidate.export(candidate_path)

            validation = validate_mesh_artifact(
                canonical_path,
                candidate_path,
            )

            self.assertFalse(validation["comparison"]["safe_to_use"])
            codes = {
                reason["code"]
                for reason in validation["comparison"]["failure_reasons"]
            }
            self.assertIn("watertight_status_regressed", codes)
            self.assertIn("boundary_edge_count_increased", codes)

    def test_preparation_falls_back_to_canonical_when_candidate_regresses(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            dense = job / "dense"
            dense.mkdir()
            canonical = dense / "final_textured_model.ply"
            trimesh.creation.box(extents=(5.0, 4.0, 3.0)).export(canonical)
            real_validate = mesh_artifacts.validate_mesh_artifact

            def forced_regression(canonical_path, candidate_path):
                validation = real_validate(canonical_path, candidate_path)
                if Path(candidate_path).name == "final_textured_model_opt_input.ply":
                    validation["comparison"]["safe_to_use"] = False
                    validation["comparison"]["failure_reasons"].append(
                        {"code": "synthetic_topology_regression"}
                    )
                return validation

            with mock.patch.object(
                mesh_artifacts,
                "validate_mesh_artifact",
                side_effect=forced_regression,
            ):
                prepared = prepare_optimizer_mesh_artifacts(
                    canonical,
                    job_folder=job,
                )

            self.assertTrue(prepared["report"]["fallback_to_canonical"])
            self.assertEqual(prepared["report"]["status"], "FALLBACK_CANONICAL")
            self.assertEqual(Path(prepared["optimizer_mesh_path"]), canonical.resolve())
            runtime_scene = trimesh.load(prepared["optimizer_input"])
            runtime = trimesh.util.concatenate(
                tuple(runtime_scene.geometry.values())
            )
            self.assertTrue(runtime.is_watertight)
            self.assertEqual(
                len(runtime.vertices),
                len(load_mesh_preserving_topology(canonical).vertices),
            )

    def test_topology_preserving_loader_does_not_merge_vertices(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final_textured_model.ply"
            mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
            mesh.vertices = np.vstack([mesh.vertices, mesh.vertices[0]])
            mesh.export(path)

            loaded = load_mesh_preserving_topology(path)

            self.assertEqual(len(loaded.vertices), len(mesh.vertices))
            self.assertEqual(len(loaded.faces), len(mesh.faces))


if __name__ == "__main__":
    unittest.main()
