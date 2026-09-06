import csv
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from research.expert_validation import common as expert_common  # noqa: E402
from research.expert_validation import compare_system_vs_expert  # noqa: E402
from research.expert_validation import validate_expert_responses  # noqa: E402
from research.optimizer_validation import run_system_yield_validation  # noqa: E402


class Objective3ValidationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _patch_optimizer_paths(self):
        module = run_system_yield_validation
        originals = {
            name: getattr(module, name)
            for name in (
                "ROOT",
                "RECON_RESULTS",
                "OPTIMIZER_BASELINE",
                "QZ01_VISUAL_AUDIT",
                "OUTPUT_DIR",
                "RESULTS_CSV",
                "SUMMARY_JSON",
                "REPORT_MD",
            )
        }
        module.ROOT = self.root
        module.RECON_RESULTS = self.root / "reconstruction.csv"
        module.OPTIMIZER_BASELINE = self.root / "optimizer_baseline.csv"
        module.QZ01_VISUAL_AUDIT = self.root / "qz01_manifest.json"
        module.OUTPUT_DIR = self.root / "outputs"
        module.RESULTS_CSV = module.OUTPUT_DIR / "system_yield_results.csv"
        module.SUMMARY_JSON = module.OUTPUT_DIR / "system_yield_summary.json"
        module.REPORT_MD = module.OUTPUT_DIR / "system_yield_report.md"
        self.addCleanup(lambda: [setattr(module, k, v) for k, v in originals.items()])

    def _write_csv(self, path, fields, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _write_job_report(self, job_id, options):
        path = self.root / "jobs" / job_id
        path.mkdir(parents=True)
        (path / "analysis_report.json").write_text(
            json.dumps({"options": options}),
            encoding="utf-8",
        )

    def _single_option(self, weight, yield_percent, candidate_count=0):
        return {
            "type": "Single Large",
            "name": "Emerald Cut",
            "gem_count": 1,
            "weight": weight,
            "yield": yield_percent,
            "manufacturing_eligible": True,
            "manufacturing_plan": {
                "status": "no_separation_required",
                "sequence": [],
            },
            "diagnostics": {
                "candidate_count": candidate_count,
                "defect_points": 0,
                "no_cut_volume_mesh_units": 0.0,
            },
        }

    def _multi_option(self, strategy, weight, yield_percent, complete=True):
        status = "complete" if complete else "settings_required"
        return {
            "type": strategy,
            "name": "multi",
            "gem_count": 3 if complete else 8,
            "weight": weight,
            "yield": yield_percent,
            "manufacturing_eligible": True,
            "manufacturing_plan": {
                "status": status,
                "sequence": [{"step": 1}] if complete else [],
                "diagnostics": {"exact_sequence_verified": complete},
            },
            "diagnostics": {
                "candidate_count": 330,
                "runtime_seconds": 120.47 if complete else 63.04,
                "runtime_target_seconds": 120,
                "meets_runtime_target": complete is False,
                "defect_points": 0,
                "no_cut_volume_mesh_units": 0.0,
                "blade_clearance": {"meets_target": True},
                "rough_clearance": {
                    "min_clearance_after_mesh_units": 0.02,
                    "target_clearance_mesh_units": 0.01,
                },
            },
        }

    def _write_optimizer_fixture(self):
        self._patch_optimizer_paths()
        self._write_csv(
            run_system_yield_validation.OPTIMIZER_BASELINE,
            [
                "job_id",
                "specimen_id_if_confirmed",
                "counts_as_independent_physical_input",
                "analysis_report_path",
                "input_weight_ct",
                "evidence_status",
            ],
            [
                {
                    "job_id": "qz01-job",
                    "specimen_id_if_confirmed": "QZ-01",
                    "counts_as_independent_physical_input": "yes",
                    "analysis_report_path": "jobs/qz01-job/analysis_report.json",
                    "input_weight_ct": "431.25",
                    "evidence_status": "confirmed",
                },
                {
                    "job_id": "same-input",
                    "specimen_id_if_confirmed": "",
                    "counts_as_independent_physical_input": "no",
                    "analysis_report_path": "jobs/same-input/analysis_report.json",
                    "input_weight_ct": "453.25",
                    "evidence_status": "same_input_secondary_sensitivity_run_not_independent",
                },
            ],
        )
        self._write_csv(
            run_system_yield_validation.RECON_RESULTS,
            [
                "specimen_id",
                "job_id",
                "known_weight_ct",
                "reconstruction_provenance",
            ],
            [
                {"specimen_id": "QZ-03", "job_id": "qz03-job", "known_weight_ct": "90.76", "reconstruction_provenance": "TRUE_DENSE"},
                {"specimen_id": "QZ-05", "job_id": "qz05-job", "known_weight_ct": "244.02", "reconstruction_provenance": "TRUE_DENSE"},
                {"specimen_id": "QZ-08", "job_id": "qz08-job", "known_weight_ct": "142.54", "reconstruction_provenance": "TRUE_DENSE"},
                {"specimen_id": "QZ-14", "job_id": "qz14-job", "known_weight_ct": "50.11", "reconstruction_provenance": "TRUE_DENSE"},
                {"specimen_id": "QZ-09", "job_id": "qz09-job", "known_weight_ct": "120.50", "reconstruction_provenance": "SPARSE_FALLBACK"},
                {"specimen_id": "QZ-30", "job_id": "qz30-job", "known_weight_ct": "10.52", "reconstruction_provenance": "SPARSE_FALLBACK"},
            ],
        )
        self._write_job_report(
            "qz01-job",
            [
                self._multi_option("Preserve + Fill", 102.39, 23.7),
                self._multi_option("Repacked Cuttable Plan", 92.6, 21.5),
                self._single_option(85.14, 19.7, candidate_count=330),
                {
                    **self._multi_option("Geometric Comparison", 102.39, 23.7),
                    "manufacturing_eligible": False,
                    "manufacturing_plan": {
                        "status": "geometric_comparison_only",
                        "sequence": [],
                    },
                },
            ],
        )
        self._write_job_report("qz03-job", [self._single_option(1.21, 1.3)])
        self._write_job_report(
            "qz05-job",
            [self._multi_option("Multi-Gem", 89.28, 36.6, complete=False), self._single_option(47.83, 19.6, candidate_count=330)],
        )
        self._write_job_report("qz08-job", [self._single_option(1.12, 0.8)])
        self._write_job_report("qz14-job", [self._single_option(0.73, 1.4)])
        run_system_yield_validation.QZ01_VISUAL_AUDIT.write_text(
            json.dumps({
                "current_saved_plan_metrics": {
                    "saved_estimated_weight_ct_from_fit": 102.391,
                },
                "stored_report_diagnostics": {
                    "preserve_fill": {
                        "stop_reason": "time_budget_exhausted",
                        "timed_out": True,
                        "fit_candidates": 68,
                    },
                },
                "fourth_gem_probe_whole_stone_unconstrained": {
                    "fit_candidate_count": 24,
                    "top_candidates_checked": [
                        {
                            "estimated_weight_ct": 5.168,
                            "surface_clearance_mm": 0.872,
                            "exact_cut_sequence_check": None,
                        },
                        {
                            "estimated_weight_ct": 4.836,
                            "surface_clearance_mm": 0.884,
                            "exact_cut_sequence_check": {
                                "status": "complete",
                                "sequence_length": 3,
                                "minimum_envelope_clearance_mm": 0.024,
                                "diagnostics": {
                                    "exact_sequence_verified": True,
                                    "approved_defect_point_count": 0,
                                },
                            },
                        },
                    ],
                },
            }),
            encoding="utf-8",
        )

    def test_optimizer_workflow_preserves_verified_and_diagnostic_boundaries(self):
        self._write_optimizer_fixture()

        rows, unavailable = run_system_yield_validation.collect_rows()
        summary = run_system_yield_validation.summarize(rows, unavailable)

        self.assertEqual(
            summary["eligible_specimens"],
            ["QZ-01", "QZ-03", "QZ-05", "QZ-08", "QZ-14"],
        )
        self.assertEqual(
            summary["proposal_target"]["status"],
            "PENDING EXTERNAL TRADITIONAL/EXPERT COMPARISON",
        )
        diagnostic_four = next(
            row for row in rows
            if row["optimizer_strategy"].startswith("Repacked Cuttable Plan (diagnostic")
        )
        self.assertEqual(diagnostic_four["retained_cuttable_weight_ct"], 107.227)
        self.assertEqual(diagnostic_four["yield_percent"], 24.86)
        self.assertEqual(
            diagnostic_four["verification_scope"],
            "diagnostic_only_exact_cut_sequence_verified_not_production_option",
        )
        qz05_multi = next(
            row for row in rows
            if row["specimen_id"] == "QZ-05" and row["optimizer_strategy"] == "Multi-Gem"
        )
        self.assertEqual(
            qz05_multi["verification_scope"],
            "diagnostic_only_missing_exact_cut_sequence",
        )
        compared = [
            item["compared_strategy"]
            for item in summary["internal_strategy_comparisons"]
        ]
        self.assertNotIn("Geometric Comparison", compared)
        self.assertTrue(all(
            item["traditional_cutting_comparison"] == "NO"
            for item in summary["internal_strategy_comparisons"]
        ))

    def _patch_expert_system_results(self):
        original = expert_common.SYSTEM_YIELD_RESULTS
        path = self.root / "system_yield_results.csv"
        expert_common.SYSTEM_YIELD_RESULTS = path
        self.addCleanup(lambda: setattr(expert_common, "SYSTEM_YIELD_RESULTS", original))
        self._write_csv(
            path,
            [
                "specimen_id",
                "job_id",
                "known_weight_ct",
                "optimizer_strategy",
                "strategy_name",
                "gem_count",
                "retained_cuttable_weight_ct",
                "yield_percent",
                "waste_percent",
                "manufacturing_cut_sequence_complete",
                "verification_scope",
            ],
            [
                {"specimen_id": "QZ-01", "job_id": "qz01", "known_weight_ct": "431.25", "optimizer_strategy": "Preserve + Fill", "strategy_name": "3 gems", "gem_count": "3", "retained_cuttable_weight_ct": "102.39", "yield_percent": "23.7", "waste_percent": "76.3", "manufacturing_cut_sequence_complete": "YES", "verification_scope": "fully_verified_existing_report"},
                {"specimen_id": "QZ-01", "job_id": "qz01", "known_weight_ct": "431.25", "optimizer_strategy": "Diagnostic", "strategy_name": "4 gems", "gem_count": "4", "retained_cuttable_weight_ct": "107.227", "yield_percent": "24.86", "waste_percent": "75.14", "manufacturing_cut_sequence_complete": "YES", "verification_scope": "diagnostic_only_exact_cut_sequence_verified_not_production_option"},
                {"specimen_id": "QZ-03", "job_id": "qz03", "known_weight_ct": "90.76", "optimizer_strategy": "Single Large", "strategy_name": "Emerald Cut", "gem_count": "1", "retained_cuttable_weight_ct": "1.21", "yield_percent": "1.3", "waste_percent": "98.7", "manufacturing_cut_sequence_complete": "YES", "verification_scope": "fully_verified_existing_report"},
                {"specimen_id": "QZ-05", "job_id": "qz05", "known_weight_ct": "244.02", "optimizer_strategy": "Multi-Gem", "strategy_name": "8 gems", "gem_count": "8", "retained_cuttable_weight_ct": "89.28", "yield_percent": "36.6", "waste_percent": "63.4", "manufacturing_cut_sequence_complete": "NO", "verification_scope": "diagnostic_only_missing_exact_cut_sequence"},
                {"specimen_id": "QZ-05", "job_id": "qz05", "known_weight_ct": "244.02", "optimizer_strategy": "Single Large", "strategy_name": "Emerald Cut", "gem_count": "1", "retained_cuttable_weight_ct": "47.83", "yield_percent": "19.6", "waste_percent": "80.4", "manufacturing_cut_sequence_complete": "YES", "verification_scope": "fully_verified_existing_report"},
                {"specimen_id": "QZ-08", "job_id": "qz08", "known_weight_ct": "142.54", "optimizer_strategy": "Single Large", "strategy_name": "Emerald Cut", "gem_count": "1", "retained_cuttable_weight_ct": "1.12", "yield_percent": "0.8", "waste_percent": "99.2", "manufacturing_cut_sequence_complete": "YES", "verification_scope": "fully_verified_existing_report"},
                {"specimen_id": "QZ-14", "job_id": "qz14", "known_weight_ct": "50.11", "optimizer_strategy": "Single Large", "strategy_name": "Emerald Cut", "gem_count": "1", "retained_cuttable_weight_ct": "0.73", "yield_percent": "1.4", "waste_percent": "98.6", "manufacturing_cut_sequence_complete": "YES", "verification_scope": "fully_verified_existing_report"},
            ],
        )

    def _expert_row(self, specimen_id="QZ-01", **overrides):
        row = {
            "response_id": "r1",
            "submitted_at": "2026-09-06T00:00:00Z",
            "expert_id": "expert-a",
            "expert_name_or_code": "Expert A",
            "years_experience": "10",
            "consent_independent_blind": "YES",
            "form_version": expert_common.FORM_A_VERSION,
            "specimen_id": specimen_id,
            "recommended_cut_shape": "Emerald Cut",
            "gem_count": "1",
            "retained_weight_ct": "80",
            "orientation_description": "table along longest stable face",
            "orientation_vector_x": "",
            "orientation_vector_y": "",
            "orientation_vector_z": "",
            "defects_or_areas_to_avoid": "none visible",
            "rationale": "maximize retained weight while preserving stability",
            "confidence_1_to_5": "4",
            "manufacturable_with_standard_saw": "YES",
            "notes": "",
        }
        row.update(overrides)
        return row

    def test_expert_workflow_excludes_diagnostic_system_rows_from_primary_claim(self):
        self._patch_expert_system_results()

        primary = expert_common.load_primary_system_results()
        self.assertEqual(primary["QZ-01"]["retained_cuttable_weight_ct"], "102.39")
        self.assertEqual(primary["QZ-05"]["optimizer_strategy"], "Single Large")

        comparisons = compare_system_vs_expert.compare_rows([
            self._expert_row("QZ-01", retained_weight_ct="90")
        ])

        self.assertEqual(comparisons[0]["system_retained_weight_ct"], 102.39)
        self.assertEqual(comparisons[0]["relative_waste_reduction_percent"], 3.576923)
        self.assertEqual(comparisons[0]["proposal_target_status"], "NOT ACHIEVED")

    def test_expert_response_validation_rejects_removed_and_missing_values(self):
        self._patch_expert_system_results()

        removed = validate_expert_responses.validate_rows([self._expert_row("QZ-09")])
        missing = validate_expert_responses.validate_rows([
            self._expert_row(
                "QZ-01",
                recommended_cut_shape="",
                manufacturable_with_standard_saw="MAYBE",
            )
        ])

        self.assertTrue(any(
            error["field"] == "specimen_id"
            for error in removed["errors"]
        ))
        self.assertTrue(any(
            error["field"] == "recommended_cut_shape"
            for error in missing["errors"]
        ))
        self.assertTrue(any(
            error["field"] == "manufacturable_with_standard_saw"
            for error in missing["errors"]
        ))

    def test_form_a_script_uses_final_specimens_and_backend_schema(self):
        script = (
            BACKEND_DIR
            / "research"
            / "expert_validation"
            / "form_a_objective3_google_apps_script.gs"
        ).read_text(encoding="utf-8")

        specimen_block = re.search(
            r"const FINAL_SPECIMENS = Object\.freeze\(\[(.*?)\]\);",
            script,
            re.S,
        ).group(1)
        final_specimens = re.findall(r'"(QZ-\d+)"', specimen_block)
        self.assertEqual(
            final_specimens,
            ["QZ-01", "QZ-03", "QZ-05", "QZ-08", "QZ-14"],
        )
        self.assertNotIn("QZ-09", final_specimens)
        self.assertNotIn("QZ-30", final_specimens)

        header_block = re.search(
            r"const SCHEMA_HEADERS = Object\.freeze\(\[(.*?)\]\);",
            script,
            re.S,
        ).group(1)
        script_headers = re.findall(r'"([^"]+)"', header_block)
        self.assertEqual(script_headers, expert_common.EXPERT_RESPONSE_FIELDS)
        self.assertIn("before reviewing any system or optimizer result", script)
        self.assertNotIn("Form B", script)


if __name__ == "__main__":
    unittest.main()
