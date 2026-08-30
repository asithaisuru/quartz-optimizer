import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research.candidate_dataset import (  # noqa: E402
    DEFAULT_LINEAGE_PATTERN,
    GEM_IDENTITY_RULE,
    audit_materialized_development_dataset,
    audit_candidate_dataset,
    assign_gem_groups,
    development_manifest_sha256,
    materialize_development_split,
    parse_lineage,
    plan_development_split,
    run_candidate_audit,
    run_development_split_plan,
)
from research.common import sha256_file  # noqa: E402


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


class CandidateDatasetTests(unittest.TestCase):
    def _workspace(self, root, samples=None, bad_yaml_paths=False):
        root = Path(root)
        repo = root / "repo"
        source = root / "source"
        repo.mkdir()
        source.mkdir()
        (source / "README.dataset.txt").write_text(
            "Synthetic candidate\n",
            encoding="utf-8",
        )
        (source / "README.roboflow.txt").write_text(
            "The dataset includes synthetic images.\n"
            "No pre-processing or augmentation was applied.\n",
            encoding="utf-8",
        )
        prefix = "../" if bad_yaml_paths else ""
        (source / "data.yaml").write_text(
            f"train: {prefix}train/images\n"
            f"val: {prefix}valid/images\n"
            f"test: {prefix}test/images\n"
            "nc: 2\n"
            "names: ['fracture', 'inclusion']\n",
            encoding="utf-8",
        )
        samples = samples or [
            ("train", 1, 0, 10, 0),
            ("valid", 1, 1, 20, 1),
            ("test", 2, 0, 30, 0),
            ("train", 3, 0, 40, 1),
            ("valid", 4, 0, 50, 0),
            ("test", 5, 0, 60, 1),
        ]
        for index, (split, gem, video, frame, class_id) in enumerate(samples):
            images = source / split / "images"
            labels = source / split / "labels"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            stem = (
                f"Gem_{gem}_Vid_{video}_Fr_{frame}_jpg.rf."
                f"export{index:02d}"
            )
            image = Image.new(
                "RGB",
                (24, 24),
                ((index * 37) % 255, (index * 71) % 255, (index * 19) % 255),
            )
            draw = ImageDraw.Draw(image)
            draw.line(
                (index % 20, 0, (index * 3 + 5) % 24, 23),
                fill=(255, 255, 255),
                width=2,
            )
            image.save(images / f"{stem}.jpg")
            (labels / f"{stem}.txt").write_text(
                f"{class_id} 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n",
                encoding="utf-8",
            )

        taxonomy = repo / "taxonomy.csv"
        taxonomy.write_text(
            "class_id,class_name,semantic_definition,taxonomy_status,"
            "is_visible_defect,eligible_for_visualization,"
            "eligible_for_3d_mapping,eligible_for_no_cut_zone,"
            "expert_confirmed,confirmed_by,confirmed_date,notes\n"
            "0,fracture,,unknown,false,false,false,false,false,,,\n"
            "1,inclusion,,unknown,false,false,false,false,false,,,\n",
            encoding="utf-8",
        )
        taxonomy_approval = repo / "taxonomy-approval.json"
        write_json(taxonomy_approval, {"example_only": True})
        identity = repo / "identity.json"
        write_json(
            identity,
            {
                "rule": GEM_IDENTITY_RULE,
                "approved": False,
                "example_only": True,
            },
        )
        route = repo / "route.json"
        write_json(
            route,
            {
                "dataset_route": "external_specimen_holdout",
                "approved": False,
                "example_only": True,
            },
        )
        metric = repo / "metric.json"
        write_json(metric, {"example_only": True})
        training = repo / "training.json"
        write_json(training, {"example_only": True})
        model = repo / "best.pt"
        model.write_bytes(b"legacy model")
        config = {
            "_repo_root": str(repo),
            "_config_path": str(repo / "candidate.json"),
            "candidate_id": "synthetic",
            "source_root": str(source),
            "candidate_zip": None,
            "dataset_yaml": "data.yaml",
            "lineage_pattern": DEFAULT_LINEAGE_PATTERN,
            "expected": {
                "image_count": len(samples),
                "label_count": len(samples),
                "polygon_count": len(samples),
                "class_names": ["fracture", "inclusion"],
            },
            "project_dataset_root": None,
            "legacy_model": str(model),
            "candidate_taxonomy": str(taxonomy),
            "candidate_taxonomy_approval": str(taxonomy_approval),
            "gem_identity_approval": str(identity),
            "dataset_route_approval": str(route),
            "metric_approval_template": str(metric),
            "training_approval_template": str(training),
            "development_split_approval": {
                "approved": False,
                "approved_by": None,
                "approver_role": None,
                "approval_date": None,
                "example_only": True,
            },
            "development_split": {
                "ratios": {"train": 0.8, "valid": 0.2},
                "seed": 42,
                "beam_width": 256,
                "perceptual_hamming_threshold": -1,
                "request_freeze": False,
            },
            "output_directory": str(repo / "audit-output"),
            "development_output_directory": str(repo / "plan-output"),
            "materialization": {"default": "manifest_only", "mode": "copy"},
        }
        write_json(repo / "candidate.json", config)
        return config, source, repo

    def _approve_split(self, config):
        audit = audit_candidate_dataset(config)
        write_json(
            config["gem_identity_approval"],
            {
                "rule": GEM_IDENTITY_RULE,
                "candidate_directory_manifest_sha256": audit["source"][
                    "directory_manifest_sha256"
                ],
                "approved": True,
                "approved_by": "Co-Supervisor / Gem Expert",
                "approver_role": "Co-Supervisor / Gem Expert",
                "approval_date": "2026-07-29",
                "exceptions": [],
                "evidence": "signed synthetic test approval",
                "example_only": False,
            },
        )
        write_json(
            config["dataset_route_approval"],
            {
                "dataset_route": "external_specimen_holdout",
                "approved": True,
                "approved_by": "Primary Supervisor",
                "approver_role": "Primary Supervisor",
                "approval_date": "2026-07-29",
                "example_only": False,
            },
        )
        preapproval = plan_development_split(config)
        config["development_split_approval"] = {
            "split_id": preapproval["development_split_lock"]["split_id"],
            "candidate_directory_manifest_sha256": audit["source"][
                "directory_manifest_sha256"
            ],
            "grouped_development_manifest_sha256": (
                development_manifest_sha256(preapproval["rows"])
            ),
            "seed": 42,
            "expected_image_counts": {
                "train": preapproval["summary"]["split_image_counts"].get(
                    "train",
                    0,
                ),
                "valid": preapproval["summary"]["split_image_counts"].get(
                    "valid",
                    0,
                ),
            },
            "expected_gem_counts": {
                "train": preapproval["summary"]["split_gem_counts"].get(
                    "train",
                    0,
                ),
                "valid": preapproval["summary"]["split_gem_counts"].get(
                    "valid",
                    0,
                ),
            },
            "required_overlap_counts": {
                "gem": 0,
                "exact_duplicate": 0,
                "source_frame": 0,
                "flagged_perceptual_pair": 0,
            },
            "approved": True,
            "approved_by": "Primary Supervisor",
            "approver_role": "Primary Supervisor",
            "approval_date": "2026-07-29",
            "example_only": False,
        }
        config["development_split"]["request_freeze"] = True

    def test_valid_candidate_yaml_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            report = audit_candidate_dataset(config)
        self.assertTrue(report["summary"]["registration_ready"])
        self.assertEqual(report["dataset"]["image_count"], 6)

    def test_invalid_declared_yaml_paths_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp, bad_yaml_paths=True)
            report = audit_candidate_dataset(config)
        self.assertTrue(
            any(not row["exists"] for row in report["dataset"]["yaml_declared_paths"])
        )
        self.assertTrue(report["summary"]["registration_ready"])

    def test_class_names_are_extracted_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            report = audit_candidate_dataset(config)
        self.assertEqual(report["dataset"]["class_names"], ["fracture", "inclusion"])

    def test_valid_yolo_segmentation_polygon_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            report = audit_candidate_dataset(config)
        self.assertEqual(report["integrity"]["label_issues"], [])

    def test_invalid_polygon_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, source, _ = self._workspace(tmp)
            label = next((source / "train" / "labels").glob("*.txt"))
            label.write_text("0 0.1 0.2 0.3\n", encoding="utf-8")
            config["expected"].pop("polygon_count")
            report = audit_candidate_dataset(config)
        self.assertFalse(report["summary"]["registration_ready"])
        self.assertTrue(report["integrity"]["label_issues"])

    def test_gem_id_parsing(self):
        lineage = parse_lineage(
            "Gem_13_Vid_2_Fr_476_jpg.rf.identifier.jpg"
        )
        self.assertEqual(lineage["gem_id"], "13")

    def test_video_id_parsing(self):
        lineage = parse_lineage(
            "Gem_13_Vid_2_Fr_476_jpg.rf.identifier.jpg"
        )
        self.assertEqual(lineage["video_id"], "2")

    def test_frame_id_parsing(self):
        lineage = parse_lineage(
            "Gem_13_Vid_2_Fr_476_jpg.rf.identifier.jpg"
        )
        self.assertEqual(lineage["frame_id"], "476")

    def test_unparseable_lineage_blocks_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, source, _ = self._workspace(tmp)
            image = next((source / "train" / "images").glob("*.jpg"))
            label = source / "train" / "labels" / f"{image.stem}.txt"
            image.rename(image.with_name("unknown.jpg"))
            label.rename(label.with_name("unknown.txt"))
            report = audit_candidate_dataset(config)
        self.assertFalse(report["summary"]["registration_ready"])

    def test_same_gem_never_crosses_development_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            plan = plan_development_split(config)
        memberships = {}
        for row in plan["rows"]:
            memberships.setdefault(row["gem_id"], set()).add(row["assigned_split"])
        self.assertTrue(all(len(values) == 1 for values in memberships.values()))

    def test_same_gem_can_retain_multiple_videos(self):
        samples = [
            ("train", 1, 0, 10, 0),
            ("valid", 1, 1, 20, 1),
            ("train", 2, 0, 30, 0),
            ("test", 3, 0, 40, 1),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp, samples=samples)
            plan = plan_development_split(config)
        gem_one = [row for row in plan["rows"] if row["gem_id"] == "1"]
        self.assertEqual({row["video_id"] for row in gem_one}, {"0", "1"})
        self.assertEqual(len({row["assigned_split"] for row in gem_one}), 1)

    def test_current_roboflow_split_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            report = audit_candidate_dataset(config)
        self.assertFalse(report["current_split_assessment"]["trusted"])
        self.assertFalse(report["current_split_assessment"]["usable_for_final_metrics"])

    def test_no_current_image_becomes_final_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            plan = plan_development_split(config)
        self.assertEqual(
            {row["assigned_split"] for row in plan["rows"]},
            {"train", "valid"},
        )
        self.assertTrue(all(not row["final_test_eligible"] for row in plan["rows"]))

    def test_grouped_plan_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            first = plan_development_split(config)
            second = plan_development_split(config)
        first_rows = [
            (row["sample_id"], row["assigned_split"]) for row in first["rows"]
        ]
        second_rows = [
            (row["sample_id"], row["assigned_split"]) for row in second["rows"]
        ]
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(
            first["development_split_lock"]["split_id"],
            second["development_split_lock"]["split_id"],
        )

    def test_class_distribution_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            plan = plan_development_split(config)
        self.assertEqual(len(plan["class_distribution"]), 4)
        self.assertEqual(
            {row["class_name"] for row in plan["class_distribution"]},
            {"fracture", "inclusion"},
        )

    def test_rare_class_group_imbalance_is_reported(self):
        samples = [
            ("train", 1, 0, 10, 1),
            ("train", 1, 1, 20, 0),
            ("valid", 2, 0, 30, 0),
            ("test", 3, 0, 40, 0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp, samples=samples)
            plan = plan_development_split(config)
        self.assertTrue(
            any("inclusion" in item for item in plan["summary"]["warnings"])
        )

    def test_source_files_remain_unchanged_after_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, source, _ = self._workspace(tmp)
            before = {
                path.relative_to(source).as_posix(): sha256_file(path)
                for path in source.rglob("*")
                if path.is_file()
            }
            plan_development_split(config)
            after = {
                path.relative_to(source).as_posix(): sha256_file(path)
                for path in source.rglob("*")
                if path.is_file()
            }
        self.assertEqual(before, after)

    def test_source_data_yaml_remains_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, source, _ = self._workspace(tmp, bad_yaml_paths=True)
            before = sha256_file(source / "data.yaml")
            plan_development_split(config)
            after = sha256_file(source / "data.yaml")
        self.assertEqual(before, after)

    def test_corrected_yaml_is_written_only_to_plan_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, source, _ = self._workspace(tmp, bad_yaml_paths=True)
            source_text = (source / "data.yaml").read_text(encoding="utf-8")
            _, files = run_development_split_plan(config)
            derived = files["yaml"].read_text(encoding="utf-8")
        self.assertIn("../train/images", source_text)
        self.assertIn("train: train/images", derived)
        self.assertIn("val: valid/images", derived)
        self.assertNotIn("\ntest:", derived)

    def test_manifest_only_is_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
        self.assertEqual(config["materialization"]["default"], "manifest_only")

    def test_materialization_requires_apply(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            config, _, repo = self._workspace(tmp)
            _, files = run_development_split_plan(config)
            destination = Path(out) / "derived"
            result = materialize_development_split(
                plan_directory=files["summary"].parent,
                output_directory=destination,
                repo_root=repo,
            )
        self.assertEqual(result["status"], "preview")
        self.assertFalse(result["output_created"])

    def test_materialization_requires_confirm(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            config, _, repo = self._workspace(tmp)
            self._approve_split(config)
            _, files = run_development_split_plan(config)
            destination = Path(out) / "derived"
            result = materialize_development_split(
                plan_directory=files["summary"].parent,
                output_directory=destination,
                repo_root=repo,
                apply=True,
                confirm=False,
            )
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(destination.exists())

    def test_materialization_never_moves_originals(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            config, source, repo = self._workspace(tmp)
            self._approve_split(config)
            source_files = {
                path.relative_to(source).as_posix(): sha256_file(path)
                for path in source.rglob("*")
                if path.is_file()
            }
            _, files = run_development_split_plan(config)
            result = materialize_development_split(
                plan_directory=files["summary"].parent,
                output_directory=Path(out) / "derived",
                repo_root=repo,
                apply=True,
                confirm=True,
            )
            after = {
                path.relative_to(source).as_posix(): sha256_file(path)
                for path in source.rglob("*")
                if path.is_file()
            }
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["source_files_moved"])
        self.assertEqual(source_files, after)

    def test_materialized_output_hashes_match_inputs(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            config, _, repo = self._workspace(tmp)
            self._approve_split(config)
            _, files = run_development_split_plan(config)
            destination = Path(out) / "derived"
            materialize_development_split(
                plan_directory=files["summary"].parent,
                output_directory=destination,
                repo_root=repo,
                apply=True,
                confirm=True,
            )
            manifest = json.loads(
                (destination / "materialization_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            hashes_match = all(
                sha256_file(row["image_path"]) == row["image_sha256"]
                and sha256_file(row["label_path"]) == row["label_sha256"]
                for row in manifest["records"]
            )
        self.assertTrue(hashes_match)

    def test_materialization_creates_no_test_directory(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            config, _, repo = self._workspace(tmp)
            self._approve_split(config)
            _, files = run_development_split_plan(config)
            destination = Path(out) / "derived"
            materialize_development_split(
                plan_directory=files["summary"].parent,
                output_directory=destination,
                repo_root=repo,
                apply=True,
                confirm=True,
            )
        self.assertFalse((destination / "test").exists())

    def test_materialized_dataset_has_resolving_yaml_and_provenance_structure(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            config, _, repo = self._workspace(tmp)
            self._approve_split(config)
            _, files = run_development_split_plan(config)
            destination = Path(out) / "derived"
            result = materialize_development_split(
                plan_directory=files["summary"].parent,
                output_directory=destination,
                repo_root=repo,
                apply=True,
                confirm=True,
            )
            yaml_text = (destination / "data.yaml").read_text(encoding="utf-8")
            train_images_exists = (destination / "train" / "images").is_dir()
            valid_labels_exists = (destination / "valid" / "labels").is_dir()
            derived_manifest_exists = (
                destination / "manifests" / "derived_file_manifest.csv"
            ).is_file()
            provenance_exists = (
                destination / "provenance" / "provenance.json"
            ).is_file()
        self.assertEqual(result["status"], "complete")
        self.assertIn("train: train/images", yaml_text)
        self.assertIn("val: valid/images", yaml_text)
        self.assertTrue(train_images_exists)
        self.assertTrue(valid_labels_exists)
        self.assertTrue(derived_manifest_exists)
        self.assertTrue(provenance_exists)

    def test_materialized_dataset_audit_retains_stable_sample_ids(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            config, _, repo = self._workspace(tmp)
            self._approve_split(config)
            _, files = run_development_split_plan(config)
            destination = Path(out) / "derived"
            materialize_development_split(
                plan_directory=files["summary"].parent,
                output_directory=destination,
                repo_root=repo,
                apply=True,
                confirm=True,
            )
            report = audit_materialized_development_dataset(
                destination,
                plan_directory=files["summary"].parent,
            )
        self.assertTrue(report["summary"]["integrity_verified"])
        self.assertEqual(report["leakage"]["gem_overlap_count"], 0)
        self.assertEqual(
            report["leakage"]["flagged_perceptual_pair_overlap_count"],
            0,
        )

    def test_split_approval_manifest_hash_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            self._approve_split(config)
            config["development_split_approval"][
                "grouped_development_manifest_sha256"
            ] = "0" * 64
            plan = plan_development_split(config)
        self.assertFalse(plan["summary"]["split_approved"])
        self.assertFalse(plan["summary"]["frozen"])
        self.assertTrue(
            any(
                "grouped_development_manifest_sha256" in blocker
                for blocker in plan["summary"]["blockers"]
            )
        )

    def test_frozen_split_cannot_change_without_renewed_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            self._approve_split(config)
            approved = plan_development_split(config)
            config["development_split"]["seed"] = 99
            changed = plan_development_split(config)
        self.assertTrue(approved["summary"]["frozen"])
        self.assertFalse(changed["summary"]["split_approved"])
        self.assertFalse(changed["summary"]["frozen"])

    def test_unapproved_gem_identity_blocks_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            config["development_split"]["request_freeze"] = True
            plan = plan_development_split(config)
        self.assertFalse(plan["summary"]["gem_identity_rule_approved"])
        self.assertFalse(plan["summary"]["frozen"])

    def test_unknown_taxonomy_does_not_block_structural_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            plan = plan_development_split(config)
        self.assertFalse(plan["summary"]["taxonomy_approved"])
        self.assertTrue(plan["summary"]["structural_plan_generated"])
        self.assertFalse(plan["summary"]["taxonomy_blocks_structural_plan"])

    def test_existing_best_is_rejected_as_final_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            report = audit_candidate_dataset(config)
        provenance = report["model_provenance"]
        self.assertEqual(provenance["model_status"], "legacy_or_exploratory_only")
        self.assertFalse(provenance["final_initialization_eligible"])

    def test_candidate_is_not_automatic_best_model_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            report = audit_candidate_dataset(config)
        self.assertFalse(
            report["model_provenance"]["candidate_dataset_attributed_to_model"]
        )

    def test_external_test_identities_do_not_overlap_development(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            plan = plan_development_split(config)
        policy = plan["external_test_identity_policy"]
        self.assertTrue(
            all(value.startswith("Gem_") for value in policy["development_gem_ids"])
        )
        self.assertTrue(policy["first_reserved_identity"].startswith("EXT-GEM-"))
        self.assertNotIn(
            policy["first_reserved_identity"],
            policy["development_gem_ids"],
        )

    def test_human_input_pack_contains_no_external_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            _, files = run_development_split_plan(config)
            with files["human_external"].open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(rows, [])

    def test_real_outputs_report_no_training_or_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._workspace(tmp)
            report, _ = run_candidate_audit(config)
            plan, _ = run_development_split_plan(config)
        self.assertFalse(report["summary"]["training_occurred"])
        self.assertFalse(report["summary"]["final_metrics_calculated"])
        self.assertFalse(plan["summary"]["training_occurred"])
        self.assertFalse(plan["summary"]["final_metrics_calculated"])

    def test_assignment_helper_keeps_atomic_groups(self):
        groups = {
            "gem-1": [
                {"class_counts": {0: 2}, "class_ids": [0]},
                {"class_counts": {1: 1}, "class_ids": [1]},
            ],
            "gem-2": [{"class_counts": {0: 1}, "class_ids": [0]}],
            "gem-3": [{"class_counts": {1: 2}, "class_ids": [1]}],
        }
        first = assign_gem_groups(
            groups,
            class_count=2,
            train_ratio=0.8,
            valid_ratio=0.2,
            seed=42,
            beam_width=64,
        )
        second = assign_gem_groups(
            groups,
            class_count=2,
            train_ratio=0.8,
            valid_ratio=0.2,
            seed=42,
            beam_width=64,
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(groups))


if __name__ == "__main__":
    unittest.main()
