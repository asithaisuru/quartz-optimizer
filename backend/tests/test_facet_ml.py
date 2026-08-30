import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from facet_ml import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    hull_equations_for_mesh,
    recommend_facet_orientation_ml,
    simulation_visibility_score,
)
from yield_calculator import _facet_recommendation  # noqa: E402


EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "research" / "facet_ml"


class FacetMLTests(unittest.TestCase):
    def test_simulation_visibility_score_penalizes_near_surface_view(self):
        mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        defects = np.asarray([[0.0, 0.0, 0.86]], dtype=float)
        equations = hull_equations_for_mesh(mesh)

        toward_defect = simulation_visibility_score(
            mesh,
            defects,
            [0.0, 0.0, 1.0],
            equations=equations,
        )
        away_from_defect = simulation_visibility_score(
            mesh,
            defects,
            [0.0, 0.0, -1.0],
            equations=equations,
        )

        self.assertGreater(away_from_defect, toward_defect + 35.0)

    def test_evidence_uses_grouped_simulation_split_and_beats_heuristic(self):
        with (EVIDENCE_DIR / "facet_orientation_manifest.json").open(
            encoding="utf-8"
        ) as handle:
            manifest = json.load(handle)
        with (EVIDENCE_DIR / "facet_orientation_metrics.json").open(
            encoding="utf-8"
        ) as handle:
            metrics = json.load(handle)

        self.assertFalse(
            manifest["real_reconstruction_assessment"][
                "eligible_for_specimen_grouped_ml"
            ]
        )
        self.assertEqual(
            manifest["dataset_label"],
            "SIMULATION-DERIVED facet-orientation dataset",
        )

        scenario_splits = {}
        with (EVIDENCE_DIR / "facet_orientation_dataset.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                prior = scenario_splits.setdefault(row["scenario_id"], row["split"])
                self.assertEqual(prior, row["split"])

        self.assertEqual(
            len(scenario_splits),
            manifest["dataset"]["scenario_group_count"],
        )

        test_model = metrics["splits"]["test"]["model"]
        test_heuristic = metrics["splits"]["test"]["heuristic_baseline"]
        self.assertLess(test_model["mae"], test_heuristic["mae"])
        self.assertLess(test_model["rmse"], test_heuristic["rmse"])
        self.assertGreater(
            test_model["top_orientation_agreement"],
            test_heuristic["top_orientation_agreement"],
        )
        self.assertLess(
            test_model["mean_angular_error_degrees"],
            test_heuristic["mean_angular_error_degrees"],
        )

    def test_backend_uses_ml_recommendation_when_artifact_and_defects_exist(self):
        self.assertTrue(DEFAULT_MODEL_PATH.exists())
        mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        defects = np.asarray([
            [0.0, 0.0, 0.82],
            [0.2, 0.1, 0.35],
            [-0.15, -0.2, -0.25],
        ], dtype=float)

        direct = recommend_facet_orientation_ml(mesh, defects)
        self.assertIsNotNone(direct)
        self.assertEqual(
            direct["method"],
            "simulation_trained_facet_orientation_model",
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dense = root / "dense"
            dense.mkdir()
            cut_path = root / "cut.ply"
            defects_path = dense / "defects.ply"
            mesh.export(cut_path)
            trimesh.points.PointCloud(defects).export(defects_path)

            recommendation = _facet_recommendation(str(cut_path), str(root))

        self.assertEqual(
            recommendation["method"],
            "simulation_trained_facet_orientation_model",
        )
        self.assertEqual(recommendation["training_data"], "simulation-derived")
        self.assertIn("heuristic_fallback", recommendation)

    def test_missing_model_returns_none_for_safe_heuristic_fallback(self):
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        defects = np.asarray([[0.0, 0.0, 0.2]], dtype=float)

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing_model.json"
            self.assertIsNone(recommend_facet_orientation_ml(mesh, defects, missing))


if __name__ == "__main__":
    unittest.main()
