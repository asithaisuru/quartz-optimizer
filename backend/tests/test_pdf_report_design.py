import importlib.util
import sys
import tempfile
import unittest
import uuid
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
HAS_REPORT_DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None
    for name in ("fpdf", "trimesh", "matplotlib", "scipy")
)


def load_report_generator():
    sys.path.insert(0, str(BACKEND_DIR))
    try:
        import report_generator

        return report_generator
    finally:
        if sys.path[0] == str(BACKEND_DIR):
            sys.path.pop(0)


def rich_stats(gem_count=4):
    gems = []
    for index in range(1, gem_count + 1):
        gems.append(
            {
                "index": index,
                "shape": "Round Brilliant Cut" if index % 2 else "Princess Cut",
                "weight_ct": round(3.75 / index, 2),
                "yield_percent": round(14.2 / index, 2),
                "plan_share_percent": round(100 / gem_count, 1),
                "volume_mesh_units": round(0.0025 / index, 6),
                "dimensions_mm": [8.4, 7.1, 4.3],
                "center_mm": [index * 1.2, index * -0.8, index * 0.4],
                "scale": round(1.1 / index, 4),
                "surface_clearance_mesh_units": round(0.013 / index, 6),
                "file": (
                    f"option_0_gem_{index}_technical_export_reference.ply"
                ),
            }
        )

    sequence = [
        {
            "step": index,
            "operation": "separate_gem",
            "target": "main_gem" if index == 1 else f"pocket_gem_{index - 1}",
            "minimum_blade_clearance_mm": 0.5,
            "rough_surface_inset_mm": 0.8,
        }
        for index in range(1, gem_count + 1)
    ]

    return {
        "volume_cm3": 2.35,
        "raw_carats": 31.2,
        "estimated_cut_carats": 12.6,
        "yield_percent": 40.4,
        "rough_dimensions_mm": [22.4, 18.8, 15.2],
        "recommended_shape": f"{gem_count} gems (mixed)",
        "cut_mode": "multi",
        "preferred_shape": "Auto",
        "space_utilization": {
            "occupied_percent": 43.7,
            "occupied_volume_mesh_units": 0.01942,
        },
        "waste_reduction": {
            "traditional_waste_baseline_percent": 35.0,
            "projected_waste_percent": 59.6,
            "reduction_vs_baseline_percent": -24.6,
            "note": (
                "Positive values mean lower waste than the assumed internal "
                "reference. Not a validated traditional-cutting comparison."
            ),
        },
        "defect_summary": {
            "images_scanned": 36,
            "point_count": 14,
            "no_cut_point_count": 6,
            "source": "policy-approved masks",
            "evidence_status": "available",
            "claim_boundary": (
                "Mapped candidate evidence does not establish detector accuracy."
            ),
        },
        "defect_detection": {
            "policy": "strict_research",
            "taxonomy_version": "approved-v1",
            "claim_status": "not_evaluated",
            "raw_prediction_count": 9,
            "visualization_count": 7,
            "mapping_count": 5,
            "no_cut_count": 3,
            "rejected_count": 2,
            "reason": "Detector performance was not evaluated in this phase.",
            "warnings": [
                "Synthetic report warning used to verify the warning layout."
            ],
        },
        "optimizer_diagnostics": {
            "optimizer_version": "voxel_beam_test",
            "candidate_count": 120,
            "runtime_seconds": 24.8,
            "blade_clearance": {
                "actual_min_gap_mm": 0.61,
                "target_gap_mm": 0.5,
            },
            "rough_clearance": {"target_clearance_mm": 0.8},
            "pocket_fill": {
                "added": max(gem_count - 1, 0),
                "timed_out": False,
                "unused_space_reason": (
                    "Remaining pockets are below the configured saleable size."
                ),
            },
            "free_space_components": {"count": 8},
        },
        "facet_recommendation": {
            "method": "heuristic_defect_visibility",
            "score": 88.2,
            "normal": [0.0, 1.0, 0.0],
            "visible_defect_estimate": 1,
            "reason": "Minimizes estimated table-side defect visibility.",
        },
        "light_analysis": {"score": 74.5, "grade": "Good"},
        "gem_details": gems,
        "manufacturing_plan": {
            "method": "blade_aware_cut_sequence",
            "machine_ready": False,
            "machine_ready_reason": (
                "CNC output requires a target machine coordinate system."
            ),
            "recommended_table_normal": [0.0, 1.0, 0.0],
            "recommended_cut_order": "largest_first_then_pocket_gems",
            "blade_clearance_mm": 0.61,
            "gem_count": gem_count,
            "sequence": sequence,
        },
        "research_completion": {
            "proposal_software_completion_percent": 94.1,
            "current_job_validation_percent": 64.7,
            "external_research_validation_percent": 0,
            "overall_status": "proposal_alignment_incomplete",
            "claim_boundary": (
                "Research-complete claims require external benchmark and "
                "physical validation data."
            ),
            "requirements": [
                {
                    "title": "Voxel occupancy-based gem fitting",
                    "status": "complete",
                    "evidence": "Synthetic layout-test evidence.",
                },
                {
                    "title": "External physical benchmark",
                    "status": "requires_external_validation",
                    "evidence": "Independent measurements are not yet available.",
                },
            ],
            "external_validation_required": [
                "Compare predicted yield with measured expert cut plans.",
                "Measure physical post-cut carat recovery and blade loss.",
            ],
        },
    }


@unittest.skipUnless(
    HAS_REPORT_DEPENDENCIES,
    "PDF report dependencies are not installed",
)
class PdfReportDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = load_report_generator()

    def generate(self, stats):
        temporary = tempfile.TemporaryDirectory()
        path = self.generator.create_pdf(
            temporary.name,
            str(uuid.uuid4()),
            stats,
        )
        return temporary, Path(path).read_bytes(), Path(path).name

    def test_report_keeps_canonical_filename_and_signature(self):
        temporary, content, filename = self.generate(rich_stats())
        self.addCleanup(temporary.cleanup)

        self.assertEqual(filename, "report.pdf")
        self.assertTrue(content.startswith(b"%PDF-"))

    def test_expected_section_titles_are_present_in_pdf_text(self):
        temporary, content, _ = self.generate(rich_stats())
        self.addCleanup(temporary.cleanup)

        for title in (
            b"Executive Summary",
            b"Source and Analysis Overview",
            b"Defect Detection Summary",
            b"Optimization and Yield",
            b"Gem Details",
            b"Cuttable Rough-Separation Plan",
            b"Notes, Warnings and Limitations",
        ):
            self.assertIn(title, content)

    def test_research_proposal_metadata_is_excluded(self):
        # The PDF is a gem-cutting report for the customer/operator, not a
        # research-proposal progress tracker — this section used to leak
        # thesis/software-completion bookkeeping into it.
        temporary, content, _ = self.generate(rich_stats())
        self.addCleanup(temporary.cleanup)

        self.assertNotIn(b"Research Proposal Alignment", content)

    def test_waste_reference_is_not_labeled_as_traditional_baseline(self):
        temporary, content, _ = self.generate(rich_stats())
        self.addCleanup(temporary.cleanup)

        self.assertIn(b"Internal reference waste", content)
        self.assertIn(b"Difference vs reference", content)
        self.assertIn(b"Not a validated traditional-cutting comparison", content)
        self.assertNotIn(b"Traditional baseline", content)

    def test_missing_optional_data_does_not_break_rendering(self):
        temporary, content, filename = self.generate({})
        self.addCleanup(temporary.cleanup)

        self.assertEqual(filename, "report.pdf")
        self.assertTrue(content.startswith(b"%PDF-"))
        self.assertIn(b"Data unavailable", content)
        self.assertIn(b"Gem details unavailable", content)

    def test_many_gems_and_warnings_render_across_multiple_pages(self):
        temporary, content, _ = self.generate(rich_stats(gem_count=32))
        self.addCleanup(temporary.cleanup)

        self.assertGreaterEqual(content.count(b"/Type /Page\n"), 4)
        self.assertIn(b"Placement and export references", content)
        self.assertIn(b"Synthetic report warning", content)
        self.assertIn(b"Page 1 of", content)

    def test_blueprints_render_headlessly_when_meshes_are_available(self):
        import trimesh

        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            dense = job / "dense"
            dense.mkdir()
            rough = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            cut = trimesh.creation.box(extents=(1.2, 0.9, 0.7))
            rough.export(dense / "visual_aligned_stone.ply")
            cut.export(dense / "best_cut.ply")

            path = self.generator.create_pdf(
                str(job),
                str(uuid.uuid4()),
                rich_stats(),
            )
            content = Path(path).read_bytes()

        self.assertTrue(content.startswith(b"%PDF-"))
        self.assertIn(b"Optimization Blueprints", content)
        self.assertIn(b"Visual fit reference", content)
        self.assertNotIn(b"Blueprint unavailable", content)


if __name__ == "__main__":
    unittest.main()
