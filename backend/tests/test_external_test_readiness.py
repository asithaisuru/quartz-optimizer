import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research.external_test_readiness import (  # noqa: E402
    ANNOTATION_FIELDS,
    APPROVED_SPECIMEN_IDS,
    CAPTURE_FIELDS,
    EXPERT_REVIEW_FIELDS,
    SAMPLE_FIELDS,
    SPECIMEN_FIELDS,
    evaluate_external_test_readiness,
    run_external_test_readiness,
)
from research_benchmark import main as benchmark_main  # noqa: E402


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
)


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class ExternalTestReadinessTests(unittest.TestCase):
    def _base(self, root):
        root = Path(root)
        evidence_dir = root / "evidence-links"
        evidence_dir.mkdir()
        evidence_links = {}
        for index, name in enumerate(
            (
                "capture_protocol",
                "annotation_protocol",
                "expert_review_protocol",
                "freeze_protocol",
                "approved_taxonomy",
                "approved_metric",
                "approved_dataset_route",
                "frozen_development_split",
                "gate2_evidence_manifest",
            )
        ):
            path = evidence_dir / f"{index:02d}-{name}.txt"
            path.write_text(name, encoding="utf-8")
            evidence_links[name] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        approval = {
            "schema_version": "1.0",
            "approval_id": "external-test-operations-gate3-v1",
            "approval_reference_date": "2026-07-29",
            "approved_by": "Research Team",
            "approver_role": "Research Team",
            "external_holdout_scope": "small_sample_external_holdout",
            "specimen_count": 5,
            "specimen_ids": list(APPROVED_SPECIMEN_IDS),
            "capture_operator_role": "Research Team",
            "initial_annotation_creator_role": "Dataset Annotator",
            "expert_review_role": "Co-Supervisor / Gem Expert",
            "expected_capture_date": "2026-08-12",
            "expected_expert_review_completion_date": "2026-08-19",
            "source_development_exclusion_required": True,
            "test_data_prohibited_uses": [
                "training",
                "validation",
                "early_stopping",
                "threshold_selection",
                "architecture_selection",
                "augmentation_decisions",
                "model_selection",
            ],
            "small_sample_limitation": {
                "pilot_evaluation_supported": True,
                "generalizability_limited": True,
                "confidence_limited": True,
                "additional_specimens_recommended": True,
                "silent_specimen_count_increase_prohibited": True,
                "final_thesis_must_disclose_specimen_and_image_counts": True,
            },
            "operational_status": "planned",
            "data_availability_status": "unavailable",
            "final_test_frozen": False,
            "training_authorized": False,
            "final_evaluation_ready": False,
            "evidence_links": evidence_links,
            "example_only": False,
        }
        approval_path = root / "operations.json"
        approval_path.write_text(json.dumps(approval), encoding="utf-8")

        registries = root / "registries"
        specimen_path = registries / "specimens.csv"
        capture_path = registries / "captures.csv"
        sample_path = registries / "samples.csv"
        annotation_path = registries / "annotations.csv"
        review_path = registries / "reviews.csv"
        write_csv(
            specimen_path,
            SPECIMEN_FIELDS,
            [
                {
                    "specimen_id": specimen_id,
                    "specimen_status": "planned",
                }
                for specimen_id in APPROVED_SPECIMEN_IDS
            ],
        )
        write_csv(capture_path, CAPTURE_FIELDS, [])
        write_csv(sample_path, SAMPLE_FIELDS, [])
        write_csv(annotation_path, ANNOTATION_FIELDS, [])
        write_csv(review_path, EXPERT_REVIEW_FIELDS, [])

        external_root = root / "external"
        for specimen_id in APPROVED_SPECIMEN_IDS:
            (external_root / "source" / specimen_id).mkdir(parents=True)
        for name in (
            "annotations_original",
            "annotations_expert_corrected",
            "manifests",
            "reviews",
            "evidence",
            "derived",
        ):
            (external_root / name).mkdir(parents=True)

        development_root = root / "development"
        development_root.mkdir()
        development_manifest = development_root / "materialization_manifest.json"
        development_manifest.write_text(
            json.dumps({"records": [{"image_sha256": "f" * 64}]}),
            encoding="utf-8",
        )
        integrity = root / "integrity.json"
        integrity.write_text(
            json.dumps(
                {
                    "class_names": ["fracture", "inclusion"],
                    "split_counts": {"train": 138, "valid": 35, "test": 0},
                    "summary": {"integrity_verified": True},
                }
            ),
            encoding="utf-8",
        )
        split_lock = root / "split-lock.json"
        split_lock.write_text(
            json.dumps(
                {
                    "split_id": "development-test",
                    "frozen": True,
                    "split_approved": True,
                    "training_authorized": False,
                }
            ),
            encoding="utf-8",
        )
        config = {
            "_repo_root": str(root),
            "_config_path": str(root / "config.json"),
            "operations_approval": str(approval_path),
            "registries": {
                "specimens": str(specimen_path),
                "captures": str(capture_path),
                "samples": str(sample_path),
                "annotations": str(annotation_path),
                "expert_reviews": str(review_path),
            },
            "protocols": {},
            "external_data_root": str(external_root),
            "capture": {
                "target_images_per_specimen": 1,
                "allowed_image_extensions": [".png", ".jpg"],
            },
            "development": {
                "dataset_root": str(development_root),
                "materialization_manifest": str(development_manifest),
                "integrity_report": str(integrity),
                "split_lock": str(split_lock),
            },
            "external_membership_approval": None,
            "external_test_lock": None,
            "perceptual_duplicate_review": None,
            "strict_duplicate_exclusion": True,
            "output_directory": str(root / "output"),
        }
        Path(config["_config_path"]).write_text(
            json.dumps({key: value for key, value in config.items()
                        if not key.startswith("_")}),
            encoding="utf-8",
        )
        return config

    def _complete(self, root):
        config = self._base(root)
        paths = config["registries"]
        specimen_rows = read_csv(paths["specimens"])
        for row in specimen_rows:
            row.update(
                {
                    "specimen_status": "received",
                    "physically_new_confirmed": "true",
                    "development_exclusion_confirmed": "true",
                }
            )
        write_csv(paths["specimens"], SPECIMEN_FIELDS, specimen_rows)

        capture_rows = []
        sample_rows = []
        annotation_rows = []
        review_rows = []
        external_root = Path(config["external_data_root"])
        for index, specimen_id in enumerate(APPROVED_SPECIMEN_IDS, 1):
            image = external_root / "source" / specimen_id / f"view-{index}.png"
            image.write_bytes(PNG_BYTES + bytes([index]))
            image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
            annotation = (
                external_root
                / "annotations_original"
                / f"sample-{index}.txt"
            )
            annotation.write_text("0 0 0 1 0 1 1\n", encoding="utf-8")
            annotation_hash = hashlib.sha256(annotation.read_bytes()).hexdigest()
            capture_rows.append(
                {
                    "capture_session_id": f"capture-{index}",
                    "specimen_id": specimen_id,
                    "capture_date": "2026-08-12",
                    "capture_operator_role": "Research Team",
                    "image_target": "1",
                    "actual_image_count": "1",
                    "angular_increment_degrees": "10",
                    "capture_status": "completed",
                }
            )
            sample = {field: "" for field in SAMPLE_FIELDS}
            sample.update(
                {
                    "sample_id": f"sample-{index}",
                    "specimen_id": specimen_id,
                    "capture_session_id": f"capture-{index}",
                    "source_image_path": str(image),
                    "original_image_sha256": image_hash,
                    "annotation_path": str(annotation),
                    "annotation_sha256": annotation_hash,
                    "test_membership_status": "approved",
                }
            )
            for field in (
                "used_for_training",
                "used_for_validation",
                "used_for_early_stopping",
                "used_for_threshold_selection",
                "used_for_architecture_selection",
                "used_for_augmentation_decisions",
                "used_for_model_selection",
            ):
                sample[field] = "false"
            sample_rows.append(sample)
            annotation_rows.append(
                {
                    "sample_id": f"sample-{index}",
                    "specimen_id": specimen_id,
                    "annotator_role": "Dataset Annotator",
                    "annotation_date": "2026-08-13",
                    "fracture_present": "true",
                    "inclusion_present": "false",
                    "negative_image": "false",
                    "annotation_status": "complete",
                }
            )
            review_rows.append(
                {
                    "sample_id": f"sample-{index}",
                    "specimen_id": specimen_id,
                    "reviewer_role": "Co-Supervisor / Gem Expert",
                    "review_date": "2026-08-19",
                    "review_status": "approved",
                    "mask_correct": "true",
                    "class_correct": "true",
                    "review_confidence": "high",
                }
            )
        write_csv(paths["captures"], CAPTURE_FIELDS, capture_rows)
        write_csv(paths["samples"], SAMPLE_FIELDS, sample_rows)
        write_csv(paths["annotations"], ANNOTATION_FIELDS, annotation_rows)
        write_csv(paths["expert_reviews"], EXPERT_REVIEW_FIELDS, review_rows)
        perceptual = Path(root) / "perceptual.csv"
        write_csv(
            perceptual,
            ("external_sample_id", "status"),
            [
                {"external_sample_id": row["sample_id"], "status": "cleared"}
                for row in sample_rows
            ],
        )
        config["perceptual_duplicate_review"] = str(perceptual)
        membership = Path(root) / "membership.json"
        membership.write_text(
            json.dumps(
                {
                    "approved": True,
                    "specimen_ids": list(APPROVED_SPECIMEN_IDS),
                    "example_only": False,
                }
            ),
            encoding="utf-8",
        )
        config["external_membership_approval"] = str(membership)
        lock = Path(root) / "external-lock.json"
        lock.write_text(
            json.dumps(
                {
                    "external_test_frozen": True,
                    "final_evaluation_authorized": True,
                    "manifest_sha256": "a" * 64,
                    "lock_hash": "b" * 64,
                }
            ),
            encoding="utf-8",
        )
        config["external_test_lock"] = str(lock)
        return config

    @staticmethod
    def _rewrite_rows(config, registry, fields, edit):
        path = config["registries"][registry]
        rows = read_csv(path)
        edit(rows)
        write_csv(path, fields, rows)

    def test_five_planned_specimen_ids_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_external_test_readiness(self._base(tmp))
        self.assertEqual(report["statuses"]["specimen_count_planned"], "pass")

    def test_duplicate_specimen_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base(tmp)
            self._rewrite_rows(
                config,
                "specimens",
                SPECIMEN_FIELDS,
                lambda rows: rows.__setitem__(1, dict(rows[0])),
            )
            report = evaluate_external_test_readiness(config)
        self.assertEqual(report["statuses"]["specimen_count_planned"], "blocked")

    def test_specimen_ids_outside_pattern_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base(tmp)
            self._rewrite_rows(
                config,
                "specimens",
                SPECIMEN_FIELDS,
                lambda rows: rows[0].update({"specimen_id": "GEM-1"}),
            )
            report = evaluate_external_test_readiness(config)
        self.assertEqual(report["statuses"]["specimen_count_planned"], "blocked")

    def test_planned_status_is_not_received(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_external_test_readiness(self._base(tmp))
        self.assertEqual(
            report["statuses"]["specimens_physically_available"],
            "unavailable",
        )

    def test_missing_physical_new_confirmation_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_external_test_readiness(self._base(tmp))
        evidence = report["checks"]["specimens_physically_available"]["evidence"]
        self.assertEqual(evidence["confirmed_available_ids"], [])

    def test_missing_development_exclusion_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_external_test_readiness(self._base(tmp))
        self.assertEqual(
            report["statuses"]["specimen_independence_confirmed"],
            "unavailable",
        )

    def test_future_capture_date_is_schedule_not_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_external_test_readiness(self._base(tmp))
        self.assertEqual(report["statuses"]["capture_schedule_recorded"], "pass")
        self.assertEqual(report["statuses"]["capture_complete"], "blocked")

    def test_future_review_date_is_schedule_not_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_external_test_readiness(self._base(tmp))
        self.assertEqual(report["statuses"]["review_schedule_recorded"], "pass")
        self.assertEqual(report["statuses"]["expert_review_complete"], "blocked")

    def test_empty_source_directories_do_not_count_as_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_external_test_readiness(self._base(tmp))
        self.assertEqual(
            report["checks"]["capture_complete"]["evidence"]["actual_image_count"],
            0,
        )

    def test_placeholder_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base(tmp)
            path = (
                Path(config["external_data_root"])
                / "source"
                / APPROVED_SPECIMEN_IDS[0]
                / "placeholder.jpg"
            )
            path.write_bytes(PNG_BYTES)
            report = evaluate_external_test_readiness(config)
        rejected = report["checks"]["capture_complete"]["evidence"][
            "rejected_placeholder_or_invalid_files"
        ]
        self.assertEqual(len(rejected), 1)

    def test_capture_count_is_validated_when_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._complete(tmp)
            self._rewrite_rows(
                config,
                "captures",
                CAPTURE_FIELDS,
                lambda rows: rows[0].update({"actual_image_count": "2"}),
            )
            report = evaluate_external_test_readiness(config)
        self.assertEqual(report["statuses"]["capture_complete"], "blocked")

    def test_all_images_from_specimen_must_remain_grouped(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._complete(tmp)
            wrong_path = (
                Path(config["external_data_root"])
                / "source"
                / APPROVED_SPECIMEN_IDS[1]
                / "view-2.png"
            )
            self._rewrite_rows(
                config,
                "samples",
                SAMPLE_FIELDS,
                lambda rows: rows[0].update({"source_image_path": str(wrong_path)}),
            )
            report = evaluate_external_test_readiness(config)
        grouping = report["checks"]["capture_complete"]["evidence"][
            "grouping_errors"
        ]
        self.assertTrue(grouping)

    def test_exact_development_duplicate_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._complete(tmp)
            sample = read_csv(config["registries"]["samples"])[0]
            Path(
                config["development"]["materialization_manifest"]
            ).write_text(
                json.dumps(
                    {"records": [{"image_sha256": sample["original_image_sha256"]}]}
                ),
                encoding="utf-8",
            )
            report = evaluate_external_test_readiness(config)
        self.assertEqual(report["statuses"]["duplicate_exclusion"], "blocked")

    def test_perceptual_duplicate_candidate_blocks_strict_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._complete(tmp)
            write_csv(
                config["perceptual_duplicate_review"],
                ("external_sample_id", "status"),
                [{"external_sample_id": "sample-1", "status": "candidate"}],
            )
            report = evaluate_external_test_readiness(config)
        self.assertEqual(report["statuses"]["duplicate_exclusion"], "blocked")

    def test_missing_annotation_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._complete(tmp)
            rows = read_csv(config["registries"]["annotations"])[1:]
            write_csv(config["registries"]["annotations"], ANNOTATION_FIELDS, rows)
            report = evaluate_external_test_readiness(config)
        self.assertEqual(report["statuses"]["annotations_complete"], "blocked")

    def test_missing_expert_review_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._complete(tmp)
            rows = read_csv(config["registries"]["expert_reviews"])[1:]
            write_csv(
                config["registries"]["expert_reviews"],
                EXPERT_REVIEW_FIELDS,
                rows,
            )
            report = evaluate_external_test_readiness(config)
        self.assertEqual(report["statuses"]["expert_review_complete"], "blocked")

    def test_unresolved_disagreement_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._complete(tmp)
            self._rewrite_rows(
                config,
                "expert_reviews",
                EXPERT_REVIEW_FIELDS,
                lambda rows: rows[0].update(
                    {"disagreement_reason": "Boundary uncertain"}
                ),
            )
            report = evaluate_external_test_readiness(config)
        self.assertEqual(report["statuses"]["expert_review_complete"], "blocked")

    def test_original_labels_cannot_be_overwritten_by_corrections(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._complete(tmp)
            original = read_csv(config["registries"]["samples"])[0][
                "annotation_path"
            ]
            self._rewrite_rows(
                config,
                "annotations",
                ANNOTATION_FIELDS,
                lambda rows: rows[0].update({"correction_path": original}),
            )
            report = evaluate_external_test_readiness(config)
        self.assertEqual(report["statuses"]["annotations_complete"], "blocked")

    def _assert_use_blocks(self, field):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._complete(tmp)
            self._rewrite_rows(
                config,
                "samples",
                SAMPLE_FIELDS,
                lambda rows: rows[0].update({field: "true"}),
            )
            report = evaluate_external_test_readiness(config)
        self.assertEqual(report["statuses"]["training_isolation"], "blocked")

    def test_test_images_cannot_be_used_for_training(self):
        self._assert_use_blocks("used_for_training")

    def test_test_images_cannot_be_used_for_validation(self):
        self._assert_use_blocks("used_for_validation")

    def test_test_images_cannot_be_used_for_early_stopping(self):
        self._assert_use_blocks("used_for_early_stopping")

    def test_test_images_cannot_be_used_for_threshold_selection(self):
        self._assert_use_blocks("used_for_threshold_selection")

    def test_test_images_cannot_be_used_for_architecture_selection(self):
        self._assert_use_blocks("used_for_architecture_selection")

    def test_test_images_cannot_be_used_for_augmentation_decisions(self):
        self._assert_use_blocks("used_for_augmentation_decisions")

    def test_test_images_cannot_be_used_for_model_selection(self):
        self._assert_use_blocks("used_for_model_selection")

    def test_five_specimens_emit_small_sample_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_external_test_readiness(self._base(tmp))
        self.assertIn("limited external pilot", report["small_sample_warning"])

    def test_additional_specimen_requires_renewed_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base(tmp)
            rows = read_csv(config["registries"]["specimens"])
            rows.append(
                {
                    "specimen_id": "EXT-GEM-006",
                    "specimen_status": "planned",
                }
            )
            write_csv(config["registries"]["specimens"], SPECIMEN_FIELDS, rows)
            report = evaluate_external_test_readiness(config)
        self.assertEqual(report["statuses"]["specimen_count_planned"], "blocked")

    def test_final_freeze_requires_lock_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._complete(tmp)
            lock = Path(config["external_test_lock"])
            value = json.loads(lock.read_text(encoding="utf-8"))
            value["lock_hash"] = ""
            lock.write_text(json.dumps(value), encoding="utf-8")
            report = evaluate_external_test_readiness(config)
        self.assertFalse(report["statuses"]["test_set_frozen"])

    def test_final_freeze_requires_expert_review_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._complete(tmp)
            write_csv(
                config["registries"]["expert_reviews"],
                EXPERT_REVIEW_FIELDS,
                [],
            )
            report = evaluate_external_test_readiness(config)
        self.assertFalse(report["statuses"]["test_set_frozen"])

    def test_current_planned_readiness_remains_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_external_test_readiness(self._base(tmp))
        self.assertEqual(report["summary"]["status"], "blocked")
        self.assertFalse(report["summary"]["training_authorized"])
        self.assertFalse(report["summary"]["final_evaluation_authorized"])

    def test_readiness_never_trains_or_creates_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_external_test_readiness(self._complete(tmp))
        self.assertFalse(report["safety"]["training_occurred"])
        self.assertFalse(report["safety"]["model_weights_created"])
        self.assertFalse(report["safety"]["performance_metrics_calculated"])
        self.assertFalse(report["summary"]["training_authorized"])

    def test_prior_legacy_summary_command_remains_operational(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_main([tmp])
        self.assertEqual(result, 0)

    def test_run_generates_complete_pending_human_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base(tmp)
            report, files = run_external_test_readiness(config)
            names = {
                path.name
                for path in files["human_collection_pack"].iterdir()
                if path.is_file()
            }
        self.assertEqual(report["summary"]["status"], "blocked")
        self.assertEqual(len(names), 10)
        self.assertIn("capture_day_checklist.md", names)

    def test_external_test_readiness_cli_is_integrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base(tmp)
            result = benchmark_main(
                [
                    "external-test-readiness",
                    "--config",
                    config["_config_path"],
                    "--output",
                    str(Path(tmp) / "cli-output"),
                ]
            )
        self.assertEqual(result, 0)

    def test_role_only_approval_rejects_email(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base(tmp)
            path = Path(config["operations_approval"])
            value = json.loads(path.read_text(encoding="utf-8"))
            value["notes"] = "contact@example.com"
            path.write_text(json.dumps(value), encoding="utf-8")
            report = evaluate_external_test_readiness(config)
        self.assertEqual(report["statuses"]["operations_plan"], "blocked")

    def test_hash_link_mismatch_blocks_operations_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base(tmp)
            approval = json.loads(
                Path(config["operations_approval"]).read_text(encoding="utf-8")
            )
            linked = Path(
                approval["evidence_links"]["capture_protocol"]["path"]
            )
            linked.write_text("changed", encoding="utf-8")
            report = evaluate_external_test_readiness(config)
        self.assertEqual(report["statuses"]["operations_plan"], "blocked")

    def test_fully_complete_fixture_can_freeze_but_not_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_external_test_readiness(self._complete(tmp))
        self.assertTrue(report["statuses"]["test_set_frozen"])
        self.assertTrue(report["summary"]["final_evaluation_authorized"])
        self.assertFalse(report["summary"]["training_authorized"])


if __name__ == "__main__":
    unittest.main()
