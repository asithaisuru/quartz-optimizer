import os
import sys
import unittest

import numpy as np
import trimesh


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cut_sequence import (  # noqa: E402
    is_recursively_separable,
    plan_cut_sequence,
    settings_required_plan,
)


def box_at(center, extents=(0.55, 0.55, 0.55)):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


class CutSequenceTests(unittest.TestCase):
    def setUp(self):
        self.rough = trimesh.creation.box(extents=(5.0, 5.0, 5.0))
        self.common = {
            "blade_kerf_mm": 0.5,
            "preform_margin_mm": 0.5,
            "max_cut_depth_mm": 60.0,
            "mm_per_mesh_unit": 10.0,
            "pitch": 0.12,
            "time_limit_seconds": 4.0,
        }

    def test_missing_settings_disable_verified_sequence(self):
        plan = settings_required_plan(0.5, 10.0)
        self.assertEqual(plan["status"], "settings_required")
        self.assertFalse(plan["machine_ready"])
        self.assertTrue(plan["operator_guidance_only"])
        self.assertEqual(plan["sequence"], [])

    def test_two_gems_produce_verified_full_through_cut(self):
        gems = [box_at((-1.0, 0, 0)), box_at((1.0, 0, 0))]
        plan = plan_cut_sequence(self.rough, gems, **self.common)

        self.assertEqual(plan["status"], "complete")
        self.assertEqual(len(plan["sequence"]), 1)
        self.assertEqual(len(plan["resulting_piece_ids"]), 2)
        self.assertLessEqual(plan["maximum_required_depth_mm"], 60.0)
        self.assertTrue(plan["diagnostics"]["exact_sequence_verified"])

        step = plan["sequence"][0]
        normal = np.asarray(step["plane"]["normal"])
        offset = float(step["plane"]["offset"])
        half_kerf = self.common["blade_kerf_mm"] / 10.0 / 2.0
        preform = self.common["preform_margin_mm"] / 10.0
        for gem in gems:
            signed = np.asarray(gem.vertices) @ normal - offset
            self.assertGreater(np.min(np.abs(signed)), half_kerf + preform - 1e-5)

    def test_four_gems_produce_complete_binary_partition_tree(self):
        gems = [
            box_at((-1.2, -1.2, 0)),
            box_at((-1.2, 1.2, 0)),
            box_at((1.2, -1.2, 0)),
            box_at((1.2, 1.2, 0)),
        ]
        plan = plan_cut_sequence(self.rough, gems, **self.common)

        self.assertEqual(plan["status"], "complete")
        self.assertEqual(len(plan["sequence"]), 3)
        self.assertEqual(len(plan["resulting_piece_ids"]), 4)
        self.assertEqual(set(plan["retained_gems"]), {
            "gem_1", "gem_2", "gem_3", "gem_4",
        })

    def test_cut_order_is_deterministic_for_same_layout(self):
        gems = [
            box_at((-1.2, -1.2, 0)),
            box_at((-1.2, 1.2, 0)),
            box_at((1.2, -1.2, 0)),
            box_at((1.2, 1.2, 0)),
        ]
        first = plan_cut_sequence(self.rough, gems, **self.common)
        second = plan_cut_sequence(self.rough, gems, **self.common)

        self.assertEqual(first["status"], "complete")
        self.assertEqual(second["status"], "complete")
        first_steps = [
            (
                step["plane"]["normal"],
                step["plane"]["offset"],
                step["negative_side_gems"],
                step["positive_side_gems"],
            )
            for step in first["sequence"]
        ]
        second_steps = [
            (
                step["plane"]["normal"],
                step["plane"]["offset"],
                step["negative_side_gems"],
                step["positive_side_gems"],
            )
            for step in second["sequence"]
        ]
        self.assertEqual(first_steps, second_steps)

    def test_intersecting_envelopes_are_not_separable(self):
        gems = [box_at((0, 0, 0), (1.4, 1.4, 1.4)),
                box_at((0, 0, 0), (0.7, 0.7, 0.7))]

        self.assertFalse(is_recursively_separable(gems, 0.05, 0.05))
        plan = plan_cut_sequence(self.rough, gems, **self.common)
        self.assertEqual(plan["status"], "not_cuttable")
        self.assertIn("no_envelope_corridor", plan["rejection_reasons"])

    def test_approved_defect_zone_blocks_the_only_cut_corridor(self):
        gems = [
            box_at((-1.35, 0, 0), (2.0, 4.0, 4.0)),
            box_at((1.35, 0, 0), (2.0, 4.0, 4.0)),
        ]
        defect_plane = np.column_stack([
            np.zeros(25),
            np.linspace(-2.0, 2.0, 25),
            np.zeros(25),
        ])
        plan = plan_cut_sequence(
            self.rough,
            gems,
            defect_points=defect_plane,
            defect_radius_mesh=0.2,
            **self.common,
        )

        self.assertEqual(plan["status"], "not_cuttable")
        self.assertGreater(
            plan["rejection_reasons"].get("approved_defect_safety_zone", 0),
            0,
        )

    def test_maximum_depth_rejects_overdeep_cut(self):
        gems = [
            box_at((-1.35, 0, 0), (2.0, 4.0, 4.0)),
            box_at((1.35, 0, 0), (2.0, 4.0, 4.0)),
        ]
        settings = dict(self.common)
        settings["max_cut_depth_mm"] = 20.0
        plan = plan_cut_sequence(self.rough, gems, **settings)

        self.assertEqual(plan["status"], "not_cuttable")
        self.assertGreater(
            plan["rejection_reasons"].get("maximum_cut_depth_exceeded", 0),
            0,
        )


if __name__ == "__main__":
    unittest.main()
