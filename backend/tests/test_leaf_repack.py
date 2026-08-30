import os
import sys
import unittest

try:
    import numpy as np
    import trimesh
except ImportError:
    np = None
    trimesh = None

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

if np is not None and trimesh is not None:
    import optimizer as opt
    from cut_sequence import plan_cut_sequence
else:
    opt = None
    plan_cut_sequence = None


def _box(center, extents):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


@unittest.skipIf(opt is None, "numpy/trimesh dependencies are not installed")
class LeafPiecesTests(unittest.TestCase):
    def test_leaf_submeshes_partition_the_rough_along_the_real_cut_plane(self):
        rough = trimesh.creation.box(extents=(6.0, 4.0, 3.0))
        gems = [
            _box((-2.0, 0, 0), (1.0, 1.0, 1.0)),
            _box((2.0, 0, 0), (1.0, 1.0, 1.0)),
        ]
        plan = plan_cut_sequence(
            rough, gems,
            blade_kerf_mm=0.5, preform_margin_mm=0.3, max_cut_depth_mm=60.0,
            mm_per_mesh_unit=10.0, pitch=0.15, time_limit_seconds=5.0,
        )
        self.assertEqual(plan["status"], "complete")

        leaves = dict(opt._leaf_pieces(plan))
        self.assertEqual(set(leaves.keys()), {"gem_1", "gem_2"})

        left = opt._leaf_submesh(rough, leaves["gem_1"])
        right = opt._leaf_submesh(rough, leaves["gem_2"])

        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        # gem_1 sits at x=-2, so its piece should be the negative-x half;
        # gem_2 at x=2 should get the positive-x half.
        self.assertLessEqual(left.bounds[1][0], 0.01)
        self.assertGreaterEqual(right.bounds[0][0], -0.01)
        # The two pieces should partition the rough without losing volume.
        self.assertAlmostEqual(left.volume + right.volume, rough.volume, places=3)


@unittest.skipIf(opt is None, "numpy/trimesh dependencies are not installed")
class RepackLeafPiecesTests(unittest.TestCase):
    def _context_for(self, rough, blade_kerf_mm=0.5, preform_margin_mm=0.3,
                     max_cut_depth_mm=60.0, mm_per_mesh_unit=10.0,
                     min_secondary_carat=None, max_gems=12):
        grid, origin, pitch = opt._build_sdf_grid(rough)
        fit = opt._fit_settings(
            rough.extents, pitch,
            blade_kerf_mm=blade_kerf_mm,
            rough_clearance_mm=0.3,
            mm_per_mesh_unit=mm_per_mesh_unit,
            min_secondary_carat=min_secondary_carat,
            extra_gem_policy="maximum_count",
            max_gems=max_gems,
            preform_margin_mm=preform_margin_mm,
            max_cut_depth_mm=max_cut_depth_mm,
        )
        rough_mask = grid > -fit["voxel_tolerance"]
        no_cut_mask = np.zeros(grid.shape, dtype=bool)
        ctx = opt.FitContext(grid, origin, pitch, rough_mask, no_cut_mask, fit)
        return ctx

    def _small_placement_near(self, rough, ctx, pos, shapes, scale=0.12):
        variants = opt._make_variants(shapes)
        variant = next(v for v in variants if v.key.endswith(":0"))
        placement, reason = opt._max_scale_for(
            variant, np.asarray(pos, dtype=float), rough.bounds,
            float(np.max(rough.extents)), ctx,
            limit_scale=scale, strict=True,
        )
        self.assertIsNotNone(placement, f"setup failed to place seed gem: {reason}")
        return placement

    def test_never_regresses_or_drops_an_already_verified_gem(self):
        # A rough with barely more room than the one gem that already
        # fills it: nothing extra should ever be found or accepted, and
        # the original state/plan must come back completely unchanged.
        rough = trimesh.creation.box(extents=(1.4, 1.4, 1.4))
        shapes = opt.get_standard_shapes()
        ctx = self._context_for(rough)
        placement = self._small_placement_near(rough, ctx, (0, 0, 0), shapes, scale=1.0)
        state = opt.BeamState([placement], set(placement.collision_set), placement.volume)
        plan = {
            "status": "complete",
            "cut_tree": {"piece_id": "rough_piece_1", "type": "leaf", "gem_ids": ["gem_1"]},
            "sequence": [],
        }

        import time
        new_state, new_plan, diag = opt._repack_leaf_pieces(
            state, plan, rough, ctx, shapes, np.zeros((0, 3)),
            time.time() + 10.0,
        )

        self.assertIs(new_state, state)
        self.assertIs(new_plan, plan)
        self.assertEqual(diag["gems_added"], 0)

    def test_finds_and_verifies_an_extra_gem_in_spacious_leftover_space(self):
        # A long rough with one small gem parked at one end: the rest of
        # the piece is wide open, so the leaf-repack search should find a
        # second gem there and hand back a *verified* two-gem plan.
        rough = trimesh.creation.box(extents=(8.0, 3.0, 3.0))
        shapes = opt.get_standard_shapes()
        ctx = self._context_for(rough, min_secondary_carat=None)
        placement = self._small_placement_near(
            rough, ctx, (-3.2, 0, 0), shapes, scale=0.5,
        )
        state = opt.BeamState([placement], set(placement.collision_set), placement.volume)
        plan = {
            "status": "complete",
            "cut_tree": {"piece_id": "rough_piece_1", "type": "leaf", "gem_ids": ["gem_1"]},
            "sequence": [],
        }

        import time
        new_state, new_plan, diag = opt._repack_leaf_pieces(
            state, plan, rough, ctx, shapes, np.zeros((0, 3)),
            time.time() + 30.0,
        )

        self.assertEqual(new_plan["status"], "complete")
        self.assertEqual(len(new_state.placements), 2)
        self.assertEqual(diag["gems_added"], 1)
        # The originally-verified gem must still be present untouched.
        self.assertIn(placement, new_state.placements)


if __name__ == "__main__":
    unittest.main()
