import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import trimesh  # noqa: E402
from yield_calculator import (  # noqa: E402
    _build_gem_details,
    _clear_generated_gem_exports,
)


class GemDetailsReportTests(unittest.TestCase):
    def test_stale_generated_gem_exports_are_removed_before_recalculation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stale = root / "option_0_gem_8.ply"
            unrelated = root / "rough.ply"
            stale.write_bytes(b"stale")
            unrelated.write_bytes(b"rough")

            removed = _clear_generated_gem_exports(str(root))

            self.assertEqual(removed, 1)
            self.assertFalse(stale.exists())
            self.assertTrue(unrelated.exists())

    def test_builds_one_detail_row_per_gem(self):
        gem_a = trimesh.creation.box(extents=(1.0, 0.8, 0.5))
        gem_b = trimesh.creation.box(extents=(0.5, 0.4, 0.3))
        strat = {
            "shape": "Multi-Gem",
            "gems": [gem_a, gem_b],
            "placements": [
                {
                    "shape": "Princess Cut",
                    "scale": 1.2,
                    "volume": 0.4,
                    "surface_clearance": 0.02,
                },
                {
                    "shape": "Round Brilliant Cut",
                    "scale": 0.7,
                    "volume": 0.06,
                    "surface_clearance": 0.04,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            details = _build_gem_details(
                strat,
                option_index=0,
                output_dir=tmp,
                scale_factor=2.0,
                target_weight=100.0,
                vol_original=2.0,
                plan_vol=0.46,
            )

            self.assertTrue(os.path.exists(os.path.join(tmp, "option_0_gem_1.ply")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "option_0_gem_2.ply")))

        self.assertEqual(len(details), 2)
        self.assertEqual(details[0]["shape"], "Princess Cut")
        self.assertEqual(details[0]["weight_ct"], 20.0)
        self.assertEqual(details[1]["weight_ct"], 3.0)
        self.assertEqual(details[0]["plan_share_percent"], 87.0)


if __name__ == "__main__":
    unittest.main()
