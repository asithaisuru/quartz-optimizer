import os
import sys
import tempfile
import unittest
import contextlib
import io
from types import SimpleNamespace

try:
    import trimesh
except ImportError:
    trimesh = None

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

if trimesh is not None:
    from optimizer import (
        optimize_cut,
        _beam_pack,
        _fit_settings,
        _make_no_cut_mask,
        _spacing_voxels,
    )
else:
    optimize_cut = None
    _beam_pack = None
    _fit_settings = None
    _make_no_cut_mask = None
    _spacing_voxels = None


@unittest.skipIf(trimesh is None, "trimesh dependency is not installed")
class OptimizerSyntheticTests(unittest.TestCase):
    def _run_optimizer(self, mesh, mode="multi", preferred_shape=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rough.ply")
            mesh.export(path)
            with contextlib.redirect_stdout(io.StringIO()):
                return optimize_cut(path, mode=mode, preferred_shape=preferred_shape)

    def test_box_prefers_large_rectangular_fit(self):
        rough = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        strategies = self._run_optimizer(
            rough,
            mode="single",
            preferred_shape="Princess Cut",
        )
        self.assertTrue(strategies)
        best = strategies[0]
        self.assertGreater(best["total_volume"], 0)
        util = best["diagnostics"]["space_utilization"]
        self.assertGreater(util["occupied_percent"], 20)

    def test_ellipsoid_multi_returns_valid_strategy(self):
        rough = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        rough.apply_scale((1.25, 0.85, 1.0))
        strategies = self._run_optimizer(rough, mode="multi")
        self.assertTrue(strategies)
        self.assertGreater(strategies[0]["total_volume"], 0)
        self.assertIn("diagnostics", strategies[0])
        multi = next((s for s in strategies if s["strategy"] == "Multi-Gem"), None)
        self.assertIsNotNone(multi)
        self.assertIn("pocket_fill", multi["diagnostics"])
        self.assertIn("free_space_components", multi["diagnostics"])

    def test_blade_gap_uses_real_world_scale(self):
        import numpy as np

        fit = _fit_settings(
            np.array([1.0, 1.0, 1.0]),
            0.001,
            blade_kerf_mm=0.5,
            mm_per_mesh_unit=100.0,
        )
        ctx = SimpleNamespace(fit=fit, pitch=0.001)

        self.assertAlmostEqual(fit["gem_spacing_buffer"], 0.005, places=6)
        self.assertAlmostEqual(fit["collision_radius_mesh_units"], 0.0025, places=6)
        self.assertEqual(_spacing_voxels(ctx), 3)
        self.assertAlmostEqual(
            fit["voxelized_protected_corridor_mesh_units"],
            0.006,
            places=6,
        )
        self.assertEqual(fit["blade_kerf_mm"], 0.5)

    def test_preform_corridor_is_split_between_symmetric_collision_masks(self):
        import numpy as np

        fit = _fit_settings(
            np.array([1.0, 1.0, 1.0]),
            0.001,
            blade_kerf_mm=0.5,
            preform_margin_mm=0.5,
            mm_per_mesh_unit=100.0,
        )
        ctx = SimpleNamespace(fit=fit, pitch=0.001)

        self.assertAlmostEqual(fit["protected_corridor_mm"], 1.5, places=6)
        self.assertAlmostEqual(fit["gem_spacing_buffer"], 0.015, places=6)
        self.assertAlmostEqual(fit["collision_radius_mesh_units"], 0.0075, places=6)
        self.assertEqual(_spacing_voxels(ctx), 8)
        self.assertAlmostEqual(fit["voxelized_protected_corridor_mm"], 1.6, places=6)
        self.assertLess(
            fit["voxelized_protected_corridor_mesh_units"],
            fit["gem_spacing_buffer"] * 1.2,
        )

    def test_disjoint_voxel_sets_can_share_same_aabb_center(self):
        left = SimpleNamespace(
            name="left",
            pos=(0.0, 0.0, 0.0),
            scale=1.0,
            volume=10.0,
            occ_set={1, 2, 3},
        )
        right = SimpleNamespace(
            name="right",
            pos=(0.0, 0.0, 0.0),
            scale=1.0,
            volume=9.0,
            occ_set={20, 21, 22},
        )

        packed = _beam_pack([left, right], max_gems=2)

        self.assertEqual(len(packed.placements), 2)
        self.assertEqual(packed.volume, 19.0)

    def test_collision_footprints_prevent_visual_merging(self):
        left = SimpleNamespace(
            name="left",
            pos=(0.0, 0.0, 0.0),
            scale=1.0,
            volume=10.0,
            occ_set={1},
            collision_set={1, 2, 3},
        )
        right = SimpleNamespace(
            name="right",
            pos=(0.0, 0.0, 0.0),
            scale=1.0,
            volume=9.0,
            occ_set={4},
            collision_set={3, 4, 5},
        )

        packed = _beam_pack([left, right], max_gems=2)

        self.assertEqual(len(packed.placements), 1)
        self.assertEqual(packed.volume, 10.0)

    def test_defect_voxel_mask_blocks_local_zone_only(self):
        import numpy as np

        mask = _make_no_cut_mask(
            (20, 20, 20),
            np.zeros(3),
            1.0,
            np.array([[10.0, 10.0, 10.0]]),
            radius=2.1,
        )

        self.assertTrue(mask[10, 10, 10])
        self.assertTrue(mask[12, 10, 10])
        self.assertFalse(mask[0, 0, 0])
        self.assertFalse(mask[19, 19, 19])


if __name__ == "__main__":
    unittest.main()
