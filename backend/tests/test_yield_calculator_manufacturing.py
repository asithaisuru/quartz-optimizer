import os
import sys
import tempfile
import unittest
from unittest import mock

try:
    import trimesh
except ImportError:
    trimesh = None


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yield_calculator as yc  # noqa: E402


def _box_gem(center):
    gem = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    gem.apply_translation(center)
    return gem


def _strategy(label, shape, gems, status, total_volume):
    return {
        "strategy": label,
        "shape": shape,
        "gems": gems,
        "total_volume": total_volume,
        "diagnostics": {
            "space_utilization": {},
            "blade_clearance": {"meets_target": True},
        },
        "manufacturing_plan": {
            "version": "straight_full_through_v1",
            "status": status,
            "sequence": ([{"step": 1}] if status == "complete" else []),
            "settings": {},
        },
        "manufacturing_eligible": True,
        "placements": [
            {
                "shape": "Synthetic",
                "volume": gem.volume,
                "scale": 1.0,
                "surface_clearance": 0.1,
            }
            for gem in gems
        ],
    }


@unittest.skipIf(trimesh is None, "trimesh dependency is not installed")
class YieldCalculatorManufacturingTests(unittest.TestCase):
    def test_production_defaults_enable_exact_cut_sequence_search(self):
        config = yc._optimizer_config({})

        self.assertEqual(config["preform_margin_mm"], 0.5)
        self.assertEqual(config["max_cut_depth_mm"], 60.0)

    def test_explicit_machine_limits_override_production_defaults(self):
        config = yc._optimizer_config({
            "preform_margin_mm": "0.25",
            "max_cut_depth_mm": "42",
        })

        self.assertEqual(config["preform_margin_mm"], 0.25)
        self.assertEqual(config["max_cut_depth_mm"], 42.0)

    def test_stats_path_passes_machine_limits_to_optimizer(self):
        captured = {}

        def fake_optimize(_path, **kwargs):
            captured.update(kwargs)
            gems = [_box_gem((-0.8, 0, 0)), _box_gem((0.8, 0, 0))]
            return [
                _strategy(
                    "Preserve + Fill",
                    "2 gems (verified)",
                    gems,
                    "complete",
                    sum(gem.volume for gem in gems),
                )
            ]

        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = os.path.join(tmp, "final_textured_model.ply")
            trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(mesh_path)
            with mock.patch.object(yc, "optimize_cut", side_effect=fake_optimize):
                with mock.patch.object(
                    yc,
                    "calculate_light_performance",
                    return_value={"score": 0, "grade": "Test"},
                ):
                    stats = yc.calculate_gem_stats(
                        mesh_path,
                        known_carats="100",
                        cut_mode="multi",
                    )

        self.assertNotIn("error", stats)
        self.assertEqual(captured["preform_margin_mm"], 0.5)
        self.assertEqual(captured["max_cut_depth_mm"], 60.0)
        self.assertEqual(
            stats["optimizer_diagnostics"]["optimizer_settings"][
                "preform_margin_mm"
            ],
            0.5,
        )
        self.assertEqual(
            stats["optimizer_diagnostics"]["optimizer_settings"][
                "max_cut_depth_mm"
            ],
            60.0,
        )

    def test_unverified_multi_gem_layout_cannot_outrank_complete_plan(self):
        verified = [_box_gem((-0.8, 0, 0)), _box_gem((0.8, 0, 0))]
        unverified = [
            _box_gem((-1.0, 0, 0)),
            _box_gem((0, 0, 0)),
            _box_gem((1.0, 0, 0)),
        ]

        def fake_optimize(_path, **_kwargs):
            return [
                _strategy(
                    "Multi-Gem",
                    "3 gems (unverified)",
                    unverified,
                    "settings_required",
                    0.45,
                ),
                _strategy(
                    "Preserve + Fill",
                    "2 gems (verified)",
                    verified,
                    "complete",
                    0.25,
                ),
            ]

        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = os.path.join(tmp, "final_textured_model.ply")
            trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(mesh_path)
            with mock.patch.object(yc, "optimize_cut", side_effect=fake_optimize):
                with mock.patch.object(
                    yc,
                    "calculate_light_performance",
                    return_value={"score": 0, "grade": "Test"},
                ):
                    stats = yc.calculate_gem_stats(
                        mesh_path,
                        known_carats="100",
                        cut_mode="multi",
                    )

        self.assertNotIn("error", stats)
        self.assertEqual(stats["recommended_shape"], "2 gems (verified)")
        self.assertEqual(
            stats["manufacturing_plan"]["status"],
            "complete",
        )
        self.assertEqual(stats["options"][0]["type"], "Preserve + Fill")


if __name__ == "__main__":
    unittest.main()
