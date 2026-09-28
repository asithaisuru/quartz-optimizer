import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import trimesh


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from reconstruction_quality import assess_reconstruction_quality


class ReconstructionQualityGateTests(unittest.TestCase):
    def _assess(self, mesh):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mesh.ply"
            mesh.export(path)
            return assess_reconstruction_quality(path, sdf_resolution=32)

    def test_closed_manifold_mesh_with_sdf_volume_passes(self):
        report = self._assess(trimesh.creation.box(extents=(10.0, 8.0, 6.0)))

        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["passed"])
        self.assertTrue(report["checks"]["watertight"]["passed"])
        self.assertEqual(report["metrics"]["non_manifold_edge_count"], 0)
        self.assertGreater(report["metrics"]["usable_sdf_volume"], 0)

    def test_gate_loads_without_automatic_topology_processing(self):
        with patch("reconstruction_quality.trimesh.load", wraps=trimesh.load) as load:
            report = self._assess(trimesh.creation.box())
        self.assertTrue(report["passed"])
        self.assertEqual(load.call_args.kwargs, {"process": False})

    def test_open_mesh_fails_as_reconstruction_insufficient(self):
        mesh = trimesh.creation.box(extents=(10.0, 8.0, 6.0))
        mesh.update_faces(np.arange(len(mesh.faces)) != 0)
        mesh.remove_unreferenced_vertices()

        report = self._assess(mesh)

        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["checks"]["watertight"]["passed"])
        self.assertIn("Reconstruction insufficient", report["message"])
        self.assertIn(
            "mesh_not_watertight",
            {reason["code"] for reason in report["failure_reasons"]},
        )

    def test_non_manifold_edges_are_reported(self):
        box = trimesh.creation.box(extents=(10.0, 8.0, 6.0))
        faces = np.vstack([box.faces, box.faces[0]])
        mesh = trimesh.Trimesh(vertices=box.vertices.copy(), faces=faces, process=False)

        report = self._assess(mesh)

        self.assertEqual(report["status"], "FAIL")
        self.assertGreater(report["metrics"]["non_manifold_edge_count"], 0)
        self.assertIn(
            "non_manifold_edges_detected",
            {reason["code"] for reason in report["failure_reasons"]},
        )


if __name__ == "__main__":
    unittest.main()
