import os
import sys
import unittest
from types import SimpleNamespace

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
class CuttableCavityFrontierTests(unittest.TestCase):
    def _placement(self, name, volume, flat):
        return SimpleNamespace(
            variant=SimpleNamespace(key=name),
            name=name,
            pos=np.array([float(flat), 0.0, 0.0]),
            scale=1.0,
            volume=float(volume),
            occ_set={flat},
            collision_set={flat},
        )

    def test_frontier_reserves_time_to_expand_candidates_found_at_timeout(self):
        seed = self._placement("seed", 10.0, 1)
        extra = self._placement("extra", 1.0, 2)
        state = opt.BeamState([seed], set(seed.collision_set), seed.volume)
        now = {"value": 100.0}
        deadline = 110.0
        local_deadlines = []
        accepted_lengths = []

        original_time = opt.time.time
        original_local = opt._local_pocket_candidates
        original_filter = opt._quick_cuttable_filter

        def fake_time():
            return now["value"]

        def fake_local(_state, _shapes, _fallback, _rough, _ctx,
                       local_deadline, structural_plan=None):
            local_deadlines.append(local_deadline)
            now["value"] = local_deadline
            diag = {
                "checked_fits": 1,
                "fit_candidates": 1,
                "below_saleable_size": 0,
                "rejected_reasons": {},
                "timed_out": True,
                "free_space_components": {},
                "verified_leaf_regions": 0,
                "leaf_aware_points": 0,
            }
            if len(local_deadlines) == 1:
                return [extra], diag
            return [], diag

        def fake_filter(_ctx, extra_normals=None):
            def accepts(candidate_state):
                accepted_lengths.append(len(candidate_state.placements))
                return True
            return accepts

        try:
            opt.time.time = fake_time
            opt._local_pocket_candidates = fake_local
            opt._quick_cuttable_filter = fake_filter

            states, diag = opt._cuttable_cavity_frontier(
                [state],
                {},
                [],
                None,
                SimpleNamespace(),
                2,
                deadline,
                "unit_test",
            )
        finally:
            opt.time.time = original_time
            opt._local_pocket_candidates = original_local
            opt._quick_cuttable_filter = original_filter

        self.assertLess(local_deadlines[0], deadline)
        self.assertIn(2, accepted_lengths)
        self.assertTrue(any(len(item.placements) == 2 for item in states))
        self.assertEqual(diag["accepted_additions"], 1)
        self.assertEqual(diag["stop_reason"], "time_budget_exhausted")

    def test_finalizer_can_verify_generated_state_without_refinement(self):
        seed = self._placement("seed", 10.0, 1)
        extra = self._placement("extra", 1.0, 2)
        state = opt.BeamState(
            [seed, extra],
            set(seed.collision_set) | set(extra.collision_set),
            seed.volume + extra.volume,
        )
        destroyed = opt.BeamState([seed], set(seed.collision_set), seed.volume)

        original_refine = opt._refine_state
        original_blade = opt._enforce_blade_clearance
        original_rough = opt._enforce_rough_clearance
        original_gap = opt._state_min_surface_gap
        original_plan_or_reduce = opt._plan_or_reduce_state

        def fake_plan_or_reduce(candidate_state, _rough, _ctx, _defect_points,
                                _deadline, preferred_normals=None):
            if candidate_state is state:
                return candidate_state, {"status": "complete"}
            return candidate_state, {"status": "not_cuttable"}

        try:
            opt._refine_state = lambda _state, _rough, _ctx: destroyed
            opt._enforce_blade_clearance = lambda candidate_state, _ctx: (
                candidate_state,
                {"adjustments": 0},
            )
            opt._enforce_rough_clearance = lambda candidate_state, _ctx: (
                candidate_state,
                {"adjustments": 0},
            )
            opt._state_min_surface_gap = lambda _placements: None
            opt._plan_or_reduce_state = fake_plan_or_reduce

            result, diag = opt._finalize_cuttable_frontier(
                [state],
                None,
                SimpleNamespace(fit={"gem_spacing_buffer": 0.0}, pitch=1.0),
                np.zeros((0, 3)),
                opt.time.time() + 5.0,
                refine_states=False,
            )
        finally:
            opt._refine_state = original_refine
            opt._enforce_blade_clearance = original_blade
            opt._enforce_rough_clearance = original_rough
            opt._state_min_surface_gap = original_gap
            opt._plan_or_reduce_state = original_plan_or_reduce

        self.assertIsNotNone(result)
        self.assertIs(result["state"], state)
        self.assertEqual(diag["states_verified"], 1)


@unittest.skipIf(opt is None, "numpy/trimesh dependencies are not installed")
class RepackLeafPiecesTests(unittest.TestCase):
    def _synthetic_placement(self, name, volume, flat):
        return SimpleNamespace(
            variant=SimpleNamespace(key=name, name=name),
            name=name,
            pos=np.array([float(flat), 0.0, 0.0]),
            scale=1.0,
            volume=float(volume),
            occ_set={flat},
            collision_set={flat},
        )

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

    def test_leaf_repack_reserves_time_for_exact_reverification(self):
        seed = self._synthetic_placement("seed", 10.0, 1)
        extra = self._synthetic_placement("extra", 1.0, 2)
        state = opt.BeamState([seed], set(seed.collision_set), seed.volume)
        plan = {"status": "complete", "sequence": []}
        now = {"value": 100.0}
        deadline = 110.0
        search_deadlines = []

        original_time = opt.time.time
        original_leaf_pieces = opt._leaf_pieces
        original_leaf_submesh = opt._leaf_submesh
        original_best = opt._best_extra_placement_in_leaf
        original_plan_or_reduce = opt._plan_or_reduce_state

        def fake_time():
            return now["value"]

        def fake_best(_submesh, _existing, _shapes, _fit, _defect_points,
                      _preferred_shape, search_deadline, _diag):
            search_deadlines.append(search_deadline)
            now["value"] = search_deadline
            return extra, None

        def fake_plan_or_reduce(candidate_state, _rough, _ctx, _defect_points,
                                verify_deadline, preferred_normals=None):
            if verify_deadline <= now["value"]:
                return state, {"status": "timeout"}
            return candidate_state, {"status": "complete"}

        try:
            opt.time.time = fake_time
            opt._leaf_pieces = lambda _plan: [("gem_1", [])]
            opt._leaf_submesh = lambda _rough, _constraints: SimpleNamespace(
                volume=20.0,
                extents=np.array([4.0, 2.0, 2.0]),
            )
            opt._best_extra_placement_in_leaf = fake_best
            opt._plan_or_reduce_state = fake_plan_or_reduce

            new_state, new_plan, diag = opt._repack_leaf_pieces(
                state,
                plan,
                None,
                SimpleNamespace(fit={}),
                {},
                np.zeros((0, 3)),
                deadline,
            )
        finally:
            opt.time.time = original_time
            opt._leaf_pieces = original_leaf_pieces
            opt._leaf_submesh = original_leaf_submesh
            opt._best_extra_placement_in_leaf = original_best
            opt._plan_or_reduce_state = original_plan_or_reduce

        self.assertLess(search_deadlines[0], deadline)
        self.assertEqual(new_plan["status"], "complete")
        self.assertEqual(len(new_state.placements), 2)
        self.assertEqual(diag["gems_added"], 1)

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
