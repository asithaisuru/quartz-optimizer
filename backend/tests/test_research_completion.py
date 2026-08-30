import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research_completion import (  # noqa: E402
    build_manufacturing_plan,
    build_research_completion,
)


class ResearchCompletionTests(unittest.TestCase):
    def _stats(self):
        return {
            "options": [{"gem_count": 3}],
            "space_utilization": {"occupied_percent": 42.0},
            "waste_reduction": {"projected_waste_percent": 58.0},
            "defect_summary": {
                "source": "mapped_3d",
                "point_count": 4,
                "source_counts": {"yolo": 2, "opencv": 2},
            },
            "facet_recommendation": {
                "normal": [0, 0, 1],
                "reason": "test",
            },
            "optimizer_diagnostics": {
                "optimizer_version": "voxel_beam_v9_blade_cavity_fill",
                "candidate_count": 12,
                "defect_points": 4,
                "runtime_seconds": 12.5,
                "runtime_target_seconds": 120,
                "optimizer_settings": {
                    "blade_kerf_mm": 0.5,
                    "rough_clearance_mm": 0.8,
                },
                "blade_clearance": {
                    "actual_min_gap_mm": 0.6,
                    "target_gap_mm": 0.5,
                    "meets_target": True,
                },
                "rough_clearance": {
                    "target_clearance_mesh_units": 0.1,
                    "min_clearance_after_mesh_units": 0.12,
                    "target_clearance_mm": 0.8,
                },
                "pocket_fill": {
                    "added": 2,
                    "unused_space_reason": "remaining pockets are below saleable size",
                },
            },
        }

    def test_completion_separates_software_from_external_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            dense = os.path.join(tmp, "dense")
            os.makedirs(dense)
            open(os.path.join(dense, "final_textured_model.ply"), "w").close()
            open(os.path.join(dense, "visual_aligned_stone.ply"), "w").close()
            open(os.path.join(tmp, "ai_results.json"), "w").close()
            open(os.path.join(tmp, "analysis_report.json"), "w").close()

            completion = build_research_completion(self._stats(), tmp)

        self.assertEqual(
            completion["overall_status"],
            "proposal_alignment_incomplete",
        )
        self.assertLess(
            completion["proposal_software_completion_percent"],
            100.0,
        )
        self.assertEqual(
            completion["external_research_validation_percent"],
            0,
        )
        self.assertTrue(completion["external_validation_required"])
        missing = [
            req["key"]
            for req in completion["requirements"]
            if req["status"] == "missing"
        ]
        self.assertIn("trained_ml_facet_model", missing)

    def test_manufacturing_plan_is_operator_guidance(self):
        plan = build_manufacturing_plan(self._stats())

        self.assertFalse(plan["machine_ready"])
        self.assertTrue(plan["operator_guidance_only"])
        self.assertEqual(plan["gem_count"], 3)
        self.assertEqual(plan["status"], "settings_required")
        self.assertEqual(len(plan["sequence"]), 0)
        self.assertEqual(plan["blade_clearance_mm"], 0.6)

    def test_verified_option_plan_is_preserved(self):
        stats = self._stats()
        verified = {
            "version": "straight_full_through_v1",
            "status": "complete",
            "machine_ready": False,
            "operator_guidance_only": True,
            "sequence": [{"step": 1}],
        }
        stats["options"][0]["manufacturing_plan"] = verified

        self.assertIs(build_manufacturing_plan(stats), verified)


if __name__ == "__main__":
    unittest.main()
