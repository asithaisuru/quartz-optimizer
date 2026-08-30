import json
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research_benchmark import main, summarize_reports  # noqa: E402


class ResearchBenchmarkTests(unittest.TestCase):
    def test_summarizes_completed_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            for idx, yield_pct in enumerate((30.0, 40.0)):
                job = os.path.join(tmp, f"job_{idx}")
                os.makedirs(job)
                report = {
                    "yield_percent": yield_pct,
                    "space_utilization": {"occupied_percent": 20.0 + idx},
                    "optimizer_diagnostics": {
                        "blade_clearance": {
                            "actual_min_gap_mm": 0.5 + idx * 0.1,
                        }
                    },
                    "research_completion": {
                        "proposal_software_completion_percent": 100.0,
                        "current_job_validation_percent": 90.0,
                    },
                }
                with open(os.path.join(job, "analysis_report.json"), "w") as f:
                    json.dump(report, f)

            summary = summarize_reports(tmp)

        self.assertEqual(summary["job_count"], 2)
        self.assertEqual(summary["average_yield_percent"], 35.0)
        self.assertEqual(summary["minimum_blade_gap_mm"], 0.5)
        self.assertEqual(summary["average_software_completion_percent"], 100.0)

    def test_legacy_cli_behavior_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(io.StringIO()):
                result = main([tmp])

        self.assertEqual(result, 0)

    def test_defect_policy_audit_reports_unconfirmed_taxonomy_without_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            taxonomy = os.path.join(tmp, "taxonomy.csv")
            with open(taxonomy, "w", newline="") as handle:
                handle.write(
                    "class_id,class_name,semantic_definition,taxonomy_status,"
                    "is_visible_defect,eligible_for_3d_mapping,"
                    "eligible_for_no_cut_zone,minimum_confidence,"
                    "expert_confirmed,confirmed_by,confirmed_date,notes\n"
                    "0,Blue,,unknown,false,false,false,,false,,,\n"
                )
            dataset_yaml = os.path.join(tmp, "data.yaml")
            with open(dataset_yaml, "w") as handle:
                handle.write("names: ['Blue']\n")
            output = os.path.join(tmp, "evidence")
            config = os.path.join(tmp, "policy.json")
            with open(config, "w") as handle:
                json.dump(
                    {
                        "policy": "yolo_only",
                        "taxonomy_path": taxonomy,
                        "dataset_yaml": dataset_yaml,
                        "output_directory": output,
                        "strict_research_mode": True,
                        "allow_provisional_for_visualization": True,
                        "allow_provisional_for_no_cut": False,
                        "unknown_class_action": "exclude",
                        "minimum_confidence_default": 0.25,
                        "minimum_confidence_by_class": {},
                        "opencv": {
                            "enabled": True,
                            "eligible_for_visualization": True,
                            "eligible_for_3d_mapping": False,
                            "eligible_for_no_cut": False,
                            "minimum_area_pixels": 0,
                            "gated_union_min_iou": 0.1,
                            "fallback_on_model_unavailable": True,
                            "fallback_on_inference_error": True,
                            "fallback_on_zero_detections": False,
                        },
                        "legacy": {"allow_legacy_union_all": False},
                    },
                    handle,
                )
            with redirect_stdout(io.StringIO()):
                normal = main(
                    ["defect-policy-audit", "--config", config]
                )
                strict = main(
                    ["defect-policy-audit", "--config", config, "--strict"]
                )
            with open(
                os.path.join(output, "defect_policy_audit.json")
            ) as handle:
                report = json.load(handle)

        self.assertEqual(normal, 0)
        self.assertEqual(strict, 2)
        self.assertFalse(report["performance_metrics_calculated"])
        self.assertEqual(report["claim_status"]["status"], "unavailable")
        self.assertEqual(report["summary"]["unknown_class_count"], 1)

    def test_environment_check_cli_completes_with_missing_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = os.path.join(tmp, "environment.json")
            output = os.path.join(tmp, "environment-output")
            with open(config, "w") as handle:
                json.dump(
                    {
                        "output_directory": output,
                        "minimum_python_version": "3.10",
                        "colmap_executable": "missing-colmap-test",
                        "dependencies": ["trimesh"],
                    },
                    handle,
                )
            with redirect_stdout(io.StringIO()):
                result = main(
                    ["environment-check", "--config", config]
                )
            with open(
                os.path.join(output, "environment_check.json")
            ) as handle:
                report = json.load(handle)
        self.assertEqual(result, 0)
        self.assertFalse(report["environment_variables_recorded"])

    def test_taxonomy_validate_strict_returns_two_for_example(self):
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(io.StringIO()):
                result = main(
                    [
                        "taxonomy-validate",
                        "--taxonomy",
                        os.path.join(root, "research", "templates", "defect_taxonomy.csv"),
                        "--approval",
                        os.path.join(
                            root,
                            "research",
                            "templates",
                            "taxonomy_approval.example.json",
                        ),
                        "--dataset-yaml",
                        os.path.join(root, "dataset", "data.yaml"),
                        "--output",
                        tmp,
                        "--strict",
                    ]
                )
        self.assertEqual(result, 2)

    def test_phase3b_gate_cli_reports_no_training_or_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = {
                "summary": {
                    "status": "blocked",
                    "authorized_for_clean_training": False,
                    "passed_gate_count": 1,
                    "blocked_gate_count": 12,
                }
            }
            files = {"report": Path(tmp) / "phase3b_gate.json"}
            with (
                patch(
                    "research_benchmark.load_phase3b_gate_config",
                    return_value={"_repo_root": tmp},
                ),
                patch(
                    "research_benchmark.run_phase3b_gate",
                    return_value=(report, files),
                ),
                redirect_stdout(io.StringIO()) as output,
            ):
                result = main(
                    [
                        "phase3b-gate",
                        "--config",
                        os.path.join(tmp, "gate.json"),
                        "--output",
                        tmp,
                    ]
                )
        value = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertFalse(value["authorized_for_clean_training"])
        self.assertFalse(value["training_occurred"])
        self.assertFalse(value["final_metrics_calculated"])

    def test_candidate_dataset_audit_cli_reports_no_training_or_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = {
                "summary": {
                    "status": "pass_with_warnings",
                    "registration_ready": True,
                },
                "dataset": {
                    "image_count": 173,
                    "label_count": 173,
                    "class_names": ["fracture", "inclusion"],
                },
                "lineage": {"unique_gem_count": 21},
            }
            files = {"report": Path(tmp) / "candidate_dataset_audit.json"}
            with (
                patch(
                    "research_benchmark.load_candidate_config",
                    return_value={"_repo_root": tmp},
                ),
                patch(
                    "research_benchmark.run_candidate_audit",
                    return_value=(report, files),
                ),
                redirect_stdout(io.StringIO()) as output,
            ):
                result = main(
                    [
                        "candidate-dataset-audit",
                        "--config",
                        os.path.join(tmp, "candidate.json"),
                    ]
                )

        value = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertTrue(value["registration_ready"])
        self.assertFalse(value["current_roboflow_split_trusted"])
        self.assertFalse(value["training_occurred"])
        self.assertFalse(value["final_metrics_calculated"])

    def test_development_split_plan_cli_keeps_current_images_out_of_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = {
                "summary": {
                    "status": "blocked_pending_human_approvals",
                    "structural_plan_generated": True,
                    "split_image_counts": {"train": 139, "valid": 34},
                    "split_gem_counts": {"train": 17, "valid": 4},
                    "gem_identity_rule_approved": False,
                    "split_approved": False,
                    "frozen": False,
                    "materialization_allowed": False,
                }
            }
            files = {"summary": Path(tmp) / "development_split_summary.json"}
            with (
                patch(
                    "research_benchmark.load_candidate_config",
                    return_value={"_repo_root": tmp},
                ),
                patch(
                    "research_benchmark.run_development_split_plan",
                    return_value=(plan, files),
                ),
                redirect_stdout(io.StringIO()) as output,
            ):
                result = main(
                    [
                        "development-split-plan",
                        "--config",
                        os.path.join(tmp, "candidate.json"),
                    ]
                )

        value = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertTrue(value["structural_plan_generated"])
        self.assertEqual(value["current_images_used_as_final_test"], 0)
        self.assertFalse(value["split_approved"])
        self.assertFalse(value["materialization_allowed"])
        self.assertFalse(value["training_occurred"])
        self.assertFalse(value["final_metrics_calculated"])

    def test_development_split_materialize_cli_defaults_to_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_payload = {
                "status": "preview",
                "applied": False,
                "training_occurred": False,
            }
            with (
                patch(
                    "research_benchmark.load_candidate_config",
                    return_value={
                        "_repo_root": tmp,
                        "development_output_directory": tmp,
                        "materialization": {"mode": "copy"},
                    },
                ),
                patch(
                    "research_benchmark.materialize_development_split",
                    return_value=result_payload,
                ) as materialize,
                redirect_stdout(io.StringIO()) as output,
            ):
                result = main(
                    [
                        "development-split-materialize",
                        "--config",
                        os.path.join(tmp, "candidate.json"),
                        "--output",
                        os.path.join(tmp, "derived"),
                    ]
                )

        value = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(value["status"], "preview")
        self.assertFalse(value["applied"])
        materialize.assert_called_once()
        self.assertFalse(materialize.call_args.kwargs["apply"])
        self.assertFalse(materialize.call_args.kwargs["confirm"])

    def test_gate2_approval_cli_keeps_training_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = {
                "summary": {
                    "status": "development_ready",
                    "development_dataset_approval_ready": True,
                    "external_test_concept_approved": True,
                }
            }
            files = {"report": Path(tmp) / "gate2_approval_validation.json"}
            with (
                patch(
                    "research_benchmark.load_gate2_config",
                    return_value={"_repo_root": tmp},
                ),
                patch(
                    "research_benchmark.run_gate2_approval_validation",
                    return_value=(report, files),
                ),
                redirect_stdout(io.StringIO()) as output,
            ):
                result = main(
                    [
                        "gate2-approvals-validate",
                        "--config",
                        os.path.join(tmp, "gate2.json"),
                    ]
                )
        value = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertTrue(value["development_dataset_approval_ready"])
        self.assertFalse(value["external_test_operational_ready"])
        self.assertFalse(value["training_authorized"])
        self.assertFalse(value["training_occurred"])
        self.assertFalse(value["weights_downloaded"])

    def test_materialized_dataset_audit_cli_reports_no_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = {
                "summary": {
                    "status": "pass",
                    "integrity_verified": True,
                },
                "split_counts": {"train": 138, "valid": 35, "test": 0},
                "class_names": ["fracture", "inclusion"],
                "leakage": {
                    "gem_overlap_count": 0,
                    "exact_duplicate_overlap_count": 0,
                    "source_frame_overlap_count": 0,
                    "flagged_perceptual_pair_overlap_count": 0,
                },
                "test_directory_exists": False,
            }
            files = {"report": Path(tmp) / "materialized_dataset_audit.json"}
            with (
                patch(
                    "research_benchmark.load_gate2_config",
                    return_value={
                        "_repo_root": tmp,
                        "development_output_directory": tmp,
                        "materialized_audit_output_directory": tmp,
                    },
                ),
                patch(
                    "research_benchmark.run_materialized_dataset_audit",
                    return_value=(report, files),
                ),
                redirect_stdout(io.StringIO()) as output,
            ):
                result = main(
                    [
                        "materialized-dataset-audit",
                        "--config",
                        os.path.join(tmp, "gate2.json"),
                        "--dataset",
                        os.path.join(tmp, "derived"),
                    ]
                )
        value = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertTrue(value["integrity_verified"])
        self.assertEqual(value["split_counts"]["test"], 0)
        self.assertFalse(value["training_occurred"])
        self.assertFalse(value["weights_downloaded"])
        self.assertFalse(value["final_metrics_calculated"])


if __name__ == "__main__":
    unittest.main()
