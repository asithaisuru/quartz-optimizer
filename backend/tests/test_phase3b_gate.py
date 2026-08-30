import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research.phase3b_gate import (  # noqa: E402
    CAPTURE_FIELDS,
    EXTERNAL_EXPERT_FIELDS,
    SAMPLE_FIELDS,
    SPECIMEN_FIELDS,
    _environment_status,
    duplicate_relationships,
    evaluate_phase3b_gate,
    run_phase3b_gate,
    validate_external_holdout,
    validate_initialization_policy,
    validate_training_configuration,
)


def write_csv(path, fields, rows):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Phase3BGateTests(unittest.TestCase):
    def _base(self, root):
        root = Path(root)
        taxonomy = root / "taxonomy.csv"
        taxonomy.write_text(
            "class_id,class_name,semantic_definition,taxonomy_status,"
            "is_visible_defect,eligible_for_visualization,"
            "eligible_for_3d_mapping,eligible_for_no_cut_zone,"
            "minimum_confidence,expert_confirmed,confirmed_by,"
            "confirmed_date,notes\n"
            "0,Defect,,unknown,false,false,false,false,,false,,,\n",
            encoding="utf-8",
        )
        dataset_yaml = root / "data.yaml"
        dataset_yaml.write_text("names: ['Defect']\n", encoding="utf-8")
        protocol = root / "protocol.md"
        protocol.write_text("# Protocol\n", encoding="utf-8")
        environment = root / "environment.json"
        environment.write_text(
            json.dumps(
                {
                    "environment_location_category": "external_isolated",
                    "environment_variables_recorded": False,
                    "summary": {"final_model_runtime_ready": False},
                }
            ),
            encoding="utf-8",
        )
        audit = root / "audit.json"
        audit.write_text(
            json.dumps({"near_duplicate_candidates": []}),
            encoding="utf-8",
        )
        review = root / "review.csv"
        write_csv(review, ("image_a", "image_b", "review_decision"), [])
        split = root / "split"
        split.mkdir()
        (split / "split_summary.json").write_text(
            json.dumps(
                {
                    "split_approved": False,
                    "manifest_hash": "manifest",
                    "metadata_coverage": {
                        "specimen_id_coverage": 0.0,
                        "recording_id_coverage": 0.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        (split / "split_lock.json").write_text(
            json.dumps({"frozen": False, "split_id": "split", "test_set_hash": None}),
            encoding="utf-8",
        )
        (split / "split_leakage_audit.json").write_text(
            json.dumps({"passes": True}),
            encoding="utf-8",
        )
        write_csv(
            split / "grouped_split_manifest.csv",
            ("sample_id", "file_path", "image_sha256", "assigned_split"),
            [],
        )
        training = root / "training.json"
        training.write_text(
            json.dumps(
                {
                    "enabled": False,
                    "dataset_route": None,
                    "base_model": None,
                    "image_size": None,
                    "batch_size": None,
                    "epochs": None,
                    "patience": None,
                    "device": None,
                }
            ),
            encoding="utf-8",
        )
        training_approval = root / "training-approval.json"
        training_approval.write_text(
            json.dumps({"example_only": True}),
            encoding="utf-8",
        )
        specimens = root / "specimens.csv"
        sessions = root / "sessions.csv"
        samples = root / "samples.csv"
        expert = root / "expert.csv"
        write_csv(specimens, SPECIMEN_FIELDS, [])
        write_csv(sessions, CAPTURE_FIELDS, [])
        write_csv(samples, SAMPLE_FIELDS, [])
        write_csv(expert, EXTERNAL_EXPERT_FIELDS, [])
        return {
            "_repo_root": str(root),
            "_config_path": str(root / "gate.json"),
            "output_directory": str(root / "gate-output"),
            "dataset_route": None,
            "dataset_route_approval": {"example_only": True},
            "duplicate_policy": None,
            "duplicate_policy_approval": {"example_only": True},
            "dataset_yaml": str(dataset_yaml),
            "dataset_audit": str(audit),
            "taxonomy": str(taxonomy),
            "taxonomy_approval": None,
            "metric_approval": None,
            "test_protocol": str(protocol),
            "near_duplicate_review": str(review),
            "grouped_split_directory": str(split),
            "ml_environment_report": str(environment),
            "training_config": str(training),
            "training_approval": str(training_approval),
            "external_holdout": {
                "specimens_manifest": str(specimens),
                "capture_sessions_manifest": str(sessions),
                "samples_manifest": str(samples),
                "expert_review": str(expert),
                "required_expert_review_coverage": 1.0,
                "perceptual_hamming_threshold": 5,
                "freeze_approval": {"example_only": True},
            },
        }

    def _external(self, root):
        root = Path(root)
        image = root / "external.png"
        Image.new("RGB", (10, 10), (20, 30, 40)).save(image)
        label = root / "external.txt"
        label.write_text("0 0 0 1 0 1 1\n", encoding="utf-8")
        specimens = root / "external-specimens.csv"
        sessions = root / "external-sessions.csv"
        samples = root / "external-samples.csv"
        expert = root / "external-expert.csv"
        write_csv(
            specimens,
            SPECIMEN_FIELDS,
            [
                {
                    "specimen_id": "specimen-new-1",
                    "specimen_source": "new collection",
                    "acquisition_date": "2026-07-29",
                    "custodian": "Research team",
                    "legacy_dataset_exclusion_confirmed": "true",
                    "exclusion_confirmed_by": "Custodian",
                    "notes": "",
                }
            ],
        )
        write_csv(
            sessions,
            CAPTURE_FIELDS,
            [
                {
                    "capture_session_id": "session-1",
                    "specimen_id": "specimen-new-1",
                    "capture_date": "2026-07-29",
                    "camera_id": "camera-1",
                    "capture_mode": "still",
                    "lighting": "controlled",
                    "polarization": "none",
                    "background": "neutral",
                    "image_count": "1",
                    "operator_id": "operator",
                    "notes": "",
                }
            ],
        )
        sample = {field: "" for field in SAMPLE_FIELDS}
        sample.update(
            {
                "sample_id": "external-1",
                "specimen_id": "specimen-new-1",
                "capture_session_id": "session-1",
                "source_image_path": str(image),
                "label_path": str(label),
                "expert_review_status": "approved",
                "expert_reviewer_id": "expert",
                "original_image_sha256": hashlib.sha256(
                    image.read_bytes()
                ).hexdigest(),
                "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
                "test_set_status": "frozen",
                "is_augmented": "false",
                "used_for_threshold_selection": "false",
                "used_for_architecture_selection": "false",
                "used_for_early_stopping": "false",
                "used_for_confidence_tuning": "false",
                "used_for_augmentation_tuning": "false",
                "used_for_model_selection": "false",
            }
        )
        write_csv(samples, SAMPLE_FIELDS, [sample])
        review = {field: "" for field in EXTERNAL_EXPERT_FIELDS}
        review.update(
            {
                "sample_id": "external-1",
                "specimen_id": "specimen-new-1",
                "expert_reviewer_id": "expert",
                "review_date": "2026-07-29",
                "review_status": "approved",
                "mask_correct": "true",
                "class_correct": "true",
                "visible_defect_present": "true",
                "review_confidence": "high",
                "class_ids_present": "0",
            }
        )
        write_csv(expert, EXTERNAL_EXPERT_FIELDS, [review])
        config = {
            "specimens_manifest": str(specimens),
            "capture_sessions_manifest": str(sessions),
            "samples_manifest": str(samples),
            "expert_review": str(expert),
            "required_expert_review_coverage": 1.0,
            "perceptual_hamming_threshold": 5,
            "freeze_approval": {
                "approved_by": "Supervisor",
                "approver_role": "Research supervisor",
                "approval_date": "2026-07-29",
                "example_only": False,
            },
        }
        return config, sample

    @staticmethod
    def _gate(report, name):
        return next(item for item in report["gates"] if item["gate"] == name)

    def test_gate_blocks_without_taxonomy_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_phase3b_gate(self._base(tmp))
        self.assertEqual(self._gate(report, "taxonomy_approval")["status"], "blocked")

    def test_gate_blocks_without_metric_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_phase3b_gate(self._base(tmp))
        self.assertEqual(self._gate(report, "primary_metric")["status"], "blocked")

    def test_gate_blocks_when_environment_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_phase3b_gate(self._base(tmp))
        self.assertEqual(
            self._gate(report, "environment_readiness")["status"], "blocked"
        )

    def test_gate_blocks_when_test_membership_not_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base(tmp)
            config["dataset_route"] = "external_specimen_holdout"
            report = evaluate_phase3b_gate(config)
        self.assertEqual(self._gate(report, "final_test_set")["status"], "blocked")

    def test_gate_blocks_when_expert_review_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base(tmp)
            config["dataset_route"] = "external_specimen_holdout"
            report = evaluate_phase3b_gate(config)
        self.assertEqual(
            self._gate(report, "expert_ground_truth")["status"], "blocked"
        )

    def test_backend_best_is_rejected_for_final_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            approval = root / "approval.json"
            approval.write_text(
                json.dumps(
                    {
                        "initialization_policy": "training_from_scratch",
                        "approved_by": "Supervisor",
                        "approver_role": "Supervisor",
                        "approval_date": "2026-07-29",
                        "example_only": False,
                    }
                ),
                encoding="utf-8",
            )
            report = validate_initialization_policy(
                approval, {"base_model": "backend/best.pt"}, root
            )
        self.assertFalse(report["approved"])
        self.assertFalse(report["legacy_model_excluded"])

    def test_external_route_allows_missing_legacy_ids_with_documented_limitation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base(tmp)
            config["dataset_route"] = "external_specimen_holdout"
            config["dataset_route_approval"] = {
                "dataset_route": "external_specimen_holdout",
                "approved_by": "Supervisor",
                "approver_role": "Supervisor",
                "approval_date": "2026-07-29",
                "limitations_acknowledged": [
                    "legacy_specimen_identity_unavailable"
                ],
                "example_only": False,
            }
            report = evaluate_phase3b_gate(config)
        self.assertEqual(self._gate(report, "dataset_route")["status"], "pass")

    def test_external_specimen_requires_legacy_exclusion_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            external, _ = self._external(tmp)
            rows, _ = self._read(external["specimens_manifest"])
            rows[0]["legacy_dataset_exclusion_confirmed"] = "false"
            write_csv(external["specimens_manifest"], SPECIMEN_FIELDS, rows)
            report = validate_external_holdout(external, Path(tmp), [])
        self.assertFalse(report["ready"])
        self.assertTrue(any("exclusion" in item for item in report["blockers"]))

    def test_external_image_cannot_occur_in_development(self):
        with tempfile.TemporaryDirectory() as tmp:
            external, sample = self._external(tmp)
            development = [
                {
                    "file_path": sample["source_image_path"],
                    "image_sha256": sample["original_image_sha256"],
                }
            ]
            report = validate_external_holdout(external, Path(tmp), development)
        self.assertTrue(report["duplicates"]["exact_cross_set_sample_ids"])
        self.assertFalse(report["ready"])

    def test_exact_duplicate_external_images_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            external, sample = self._external(tmp)
            rows, _ = self._read(external["samples_manifest"])
            duplicate = dict(rows[0])
            duplicate["sample_id"] = "external-2"
            write_csv(external["samples_manifest"], SAMPLE_FIELDS, [rows[0], duplicate])
            report = validate_external_holdout(external, Path(tmp), [])
        self.assertTrue(report["duplicates"]["exact_internal_groups"])

    def test_perceptual_cross_set_candidates_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            external, sample = self._external(root)
            development_image = root / "development.png"
            Image.new("RGB", (10, 10), (90, 100, 110)).save(development_image)
            development = [
                {
                    "file_path": str(development_image),
                    "image_sha256": hashlib.sha256(
                        development_image.read_bytes()
                    ).hexdigest(),
                }
            ]
            report = validate_external_holdout(external, root, development)
        self.assertTrue(
            report["duplicates"]["perceptual_cross_set_candidates"]
        )

    def test_augmented_external_image_is_rejected(self):
        self._assert_sample_flag_blocked("is_augmented", "true", "augmented")

    def test_test_image_cannot_be_used_for_threshold_tuning(self):
        self._assert_sample_flag_blocked(
            "used_for_threshold_selection", "true", "threshold selection"
        )

    def test_test_image_cannot_be_used_for_early_stopping(self):
        self._assert_sample_flag_blocked(
            "used_for_early_stopping", "true", "early stopping"
        )

    def test_test_image_cannot_be_used_for_model_selection(self):
        self._assert_sample_flag_blocked(
            "used_for_model_selection", "true", "model selection"
        )

    def _assert_sample_flag_blocked(self, field, value, phrase):
        with tempfile.TemporaryDirectory() as tmp:
            external, _ = self._external(tmp)
            rows, _ = self._read(external["samples_manifest"])
            rows[0][field] = value
            write_csv(external["samples_manifest"], SAMPLE_FIELDS, rows)
            report = validate_external_holdout(external, Path(tmp), [])
        self.assertTrue(any(phrase in item for item in report["blockers"]))

    def test_all_images_from_specimen_must_remain_in_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            external, _ = self._external(tmp)
            rows, _ = self._read(external["samples_manifest"])
            second = dict(rows[0])
            second["sample_id"] = "external-2"
            second["test_set_status"] = "development"
            write_csv(external["samples_manifest"], SAMPLE_FIELDS, [rows[0], second])
            report = validate_external_holdout(external, Path(tmp), [])
        self.assertTrue(
            any("All images from specimen" in item for item in report["blockers"])
        )

    def test_recovered_lineage_route_blocks_without_specimen_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base(tmp)
            config["dataset_route"] = "recovered_legacy_lineage"
            config["dataset_route_approval"] = {
                "dataset_route": "recovered_legacy_lineage",
                "approved_by": "Supervisor",
                "approver_role": "Supervisor",
                "approval_date": "2026-07-29",
                "example_only": False,
            }
            report = evaluate_phase3b_gate(config)
        self.assertEqual(self._gate(report, "legacy_lineage")["status"], "blocked")

    def test_recovered_lineage_reads_expert_coverage_from_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base(tmp)
            config["dataset_route"] = "recovered_legacy_lineage"
            config["dataset_route_approval"] = {
                "dataset_route": "recovered_legacy_lineage",
                "approved_by": "Supervisor",
                "approver_role": "Supervisor",
                "approval_date": "2026-07-29",
                "example_only": False,
            }
            config["duplicate_policy"] = "human_reviewed"
            config["duplicate_policy_approval"] = {
                "duplicate_policy": "human_reviewed",
                "approved_by": "Supervisor",
                "approval_date": "2026-07-29",
                "example_only": False,
            }
            split = Path(config["grouped_split_directory"])
            (split / "split_summary.json").write_text(
                json.dumps(
                    {
                        "split_approved": True,
                        "manifest_hash": "manifest",
                        "metadata_coverage": {
                            "specimen_id_coverage": 1.0,
                            "recording_id_coverage": 1.0,
                            "expert_review_coverage": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (split / "split_lock.json").write_text(
                json.dumps(
                    {
                        "frozen": True,
                        "split_id": "split",
                        "test_set_hash": "test-hash",
                    }
                ),
                encoding="utf-8",
            )
            report = evaluate_phase3b_gate(config)
        self.assertEqual(self._gate(report, "legacy_lineage")["status"], "pass")
        self.assertEqual(self._gate(report, "expert_ground_truth")["status"], "pass")

    def test_conservative_grouping_is_deterministic(self):
        candidates = [
            {"left": "a", "right": "b"},
            {"left": "b", "right": "c"},
        ]
        first = duplicate_relationships(candidates, [], "conservative_union")
        second = duplicate_relationships(candidates, [], "conservative_union")
        self.assertEqual(first["components"], second["components"])

    def test_conservative_grouping_is_labelled_provisional(self):
        report = duplicate_relationships(
            [{"left": "a", "right": "b"}], [], "conservative_union"
        )
        self.assertTrue(report["provisional"])
        self.assertEqual(
            report["method"], "conservative_near_duplicate_grouping"
        )

    def test_human_review_decision_overrides_candidate_status(self):
        candidate = [{"left": "a", "right": "b"}]
        review = [
            {
                "image_a": "a",
                "image_b": "b",
                "review_decision": "not_duplicate",
            }
        ]
        report = duplicate_relationships(candidate, review, "conservative_union")
        self.assertEqual(report["joined_pair_count"], 0)
        self.assertEqual(report["unresolved_count"], 0)

    def test_materialization_remains_disabled_and_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base(tmp)
            source = Path(config["dataset_yaml"])
            before = source.read_bytes()
            report, files = run_phase3b_gate(config)
            after = source.read_bytes()
            yaml_text = files["yaml"].read_text(encoding="utf-8")
        self.assertFalse(report["summary"]["authorized_for_clean_training"])
        self.assertEqual(before, after)
        self.assertIn("manifest only", yaml_text)

    def test_training_config_rejects_legacy_model_initialization(self):
        report = validate_training_configuration(
            {
                "enabled": False,
                "dataset_route": "external_specimen_holdout",
                "base_model": "backend/best.pt",
                "image_size": 640,
                "batch_size": 8,
                "epochs": 50,
                "patience": 10,
                "device": "cpu",
            },
            {
                "taxonomy_approved": True,
                "metric_approved": True,
                "environment_ready": True,
                "development_split_approved": True,
                "test_set_frozen": True,
                "initialization_approved": False,
                "test_excluded_from_training": True,
                "test_excluded_from_threshold_selection": True,
            },
        )
        self.assertFalse(report["ready"])

    def test_environment_summary_does_not_echo_secrets(self):
        status = _environment_status(
            {
                "environment_location_category": "external_isolated",
                "environment_variables_recorded": False,
                "summary": {"final_model_runtime_ready": True},
                "SECRET_TOKEN": "do-not-copy",
            }
        )
        self.assertNotIn("do-not-copy", json.dumps(status))

    def test_gate0_never_produces_final_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_phase3b_gate(self._base(tmp))
        self.assertFalse(report["summary"]["training_occurred"])
        self.assertFalse(report["summary"]["final_metrics_calculated"])
        self.assertFalse(report["summary"]["claim_90_percent_available"])

    @staticmethod
    def _read(path):
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader), list(reader.fieldnames or [])


if __name__ == "__main__":
    unittest.main()
