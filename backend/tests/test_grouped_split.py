import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research.grouped_split import (  # noqa: E402
    build_atomic_groups,
    materialize_grouped_split,
    plan_grouped_split,
)


class GroupedSplitTests(unittest.TestCase):
    def _config(self, root, metadata=True):
        root = Path(root)
        dataset = root / "dataset"
        names = ["Defect", "Rare"]
        records = []
        for split in ("train", "valid", "test"):
            (dataset / split / "images").mkdir(parents=True)
            (dataset / split / "labels").mkdir(parents=True)
        for index in range(9):
            split = ("train", "valid", "test")[index % 3]
            parent = index // 2
            image = dataset / split / "images" / (
                f"sample_{index}.rf.parent{parent}.jpg"
            )
            array = np.ones((8, 8, 3), dtype=np.uint8) * index
            Image.fromarray(array).save(image)
            label = dataset / split / "labels" / f"{image.stem}.txt"
            class_id = 1 if index == 8 else 0
            label.write_text(f"{class_id} 0 0 1 0 1 1\n")
            records.append((image, label, split, index))
        yaml = dataset / "data.yaml"
        yaml.write_text(
            "train: ./train/images\n"
            "val: ./valid/images\n"
            "test: ./test/images\n"
            f"names: {names!r}\n"
        )
        manifest = root / "manifest.csv"
        if metadata:
            fields = [
                "sample_id", "file_path", "label_path", "specimen_id",
                "capture_session_id", "recording_id", "source_frame_id",
                "augmentation_parent_id", "split", "source_dataset",
                "license", "annotator_id", "annotation_status",
                "expert_review_status",
            ]
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for image, label, split, index in records:
                    writer.writerow(
                        {
                            "sample_id": f"s{index}",
                            "file_path": image,
                            "label_path": label,
                            "specimen_id": f"specimen-{index // 3}",
                            "capture_session_id": f"session-{index // 3}",
                            "recording_id": f"recording-{index // 3}",
                            "source_frame_id": f"frame-{index}",
                            "augmentation_parent_id": f"parent-{index // 2}",
                            "split": split,
                            "source_dataset": "synthetic",
                            "license": "test",
                            "annotator_id": "test",
                            "annotation_status": "complete",
                            "expert_review_status": "approved",
                        }
                    )
        taxonomy = root / "taxonomy.csv"
        taxonomy.write_text(
            "class_id,class_name,semantic_definition,taxonomy_status,"
            "is_visible_defect,eligible_for_3d_mapping,"
            "eligible_for_no_cut_zone,minimum_confidence,expert_confirmed,"
            "confirmed_by,confirmed_date,notes\n"
            "0,Defect,Visible,confirmed_defect,true,true,true,0.25,true,E,2026-01-01,\n"
            "1,Rare,Rare visible,confirmed_defect,true,true,true,0.25,true,E,2026-01-01,\n"
        )
        approval = root / "approval.json"
        approval.write_text(
            json.dumps(
                {
                    "taxonomy_id": "t",
                    "taxonomy_version": "1",
                    "taxonomy_file": str(taxonomy),
                    "taxonomy_sha256": hashlib.sha256(
                        taxonomy.read_bytes()
                    ).hexdigest(),
                    "approved_by": "E",
                    "approver_role": "Expert",
                    "approval_date": "2026-01-01",
                    "approval_scope": "test",
                    "primary_visible_defect_definition": "Visible",
                    "primary_claim_metric": "binary_visible_defect_f1",
                    "claim_threshold": 0.9,
                    "precision_floor": None,
                    "notes": "",
                    "example_only": False,
                }
            )
        )
        return {
            "_repo_root": str(root),
            "dataset_yaml": str(yaml),
            "metadata_manifest": str(manifest) if metadata else None,
            "near_duplicate_review": None,
            "taxonomy_path": str(taxonomy),
            "taxonomy_approval": str(approval),
            "deterministic_seed": 42,
            "ratios": {"train": 0.6, "valid": 0.2, "test": 0.2},
            "additional_grouping_fields": [],
            "request_freeze": False,
            "split_approval": {"approved_by": None, "approval_date": None},
        }, records

    def test_lineage_and_exact_duplicates_never_cross_splits(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            config, source = self._config(tmp)
            source[6][0].write_bytes(source[0][0].read_bytes())
            plan = plan_grouped_split(config)
        by_specimen = {}
        by_recording = {}
        by_parent = {}
        for row in plan["rows"]:
            by_specimen.setdefault(row["specimen_id"], set()).add(row["assigned_split"])
            by_recording.setdefault(row["recording_id"], set()).add(row["assigned_split"])
            by_parent.setdefault(row["augmentation_parent_id"], set()).add(row["assigned_split"])
        self.assertTrue(all(len(value) == 1 for value in by_specimen.values()))
        self.assertTrue(all(len(value) == 1 for value in by_recording.values()))
        self.assertTrue(all(len(value) == 1 for value in by_parent.values()))
        self.assertTrue(plan["leakage_audit"]["passes"])
        duplicate_rows = [
            row
            for row in plan["rows"]
            if row["file_path"] in {
                str(source[0][0].relative_to(Path(tmp))).replace("\\", "/"),
                str(source[6][0].relative_to(Path(tmp))).replace("\\", "/"),
            }
        ]
        self.assertEqual(len(duplicate_rows), 2)
        self.assertEqual(
            len({row["assigned_split"] for row in duplicate_rows}),
            1,
        )
        self.assertTrue(
            all(row["exact_duplicate_group_id"] for row in duplicate_rows)
        )

    def test_configured_missing_near_duplicate_review_blocks_approval(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            config, _ = self._config(tmp)
            config["near_duplicate_review"] = str(
                Path(tmp) / "missing-near-review.csv"
            )
            config["request_freeze"] = True
            config["split_approval"] = {
                "approved_by": "Reviewer",
                "approval_date": "2026-01-01",
            }
            plan = plan_grouped_split(config)
        self.assertFalse(plan["summary"]["split_approved"])
        self.assertFalse(plan["split_lock"]["frozen"])
        self.assertTrue(
            any(
                "near-duplicate review evidence is missing" in blocker
                for blocker in plan["summary"]["blockers"]
            )
        )

    def test_shared_source_frame_never_crosses_splits(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            config, _ = self._config(tmp)
            manifest = Path(config["metadata_manifest"])
            with manifest.open("r", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            rows[0]["source_frame_id"] = "shared-frame"
            rows[6]["source_frame_id"] = "shared-frame"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            plan = plan_grouped_split(config)
        selected = [
            row["assigned_split"]
            for row in plan["rows"]
            if row["sample_id"] in {"s0", "s6"}
        ]
        self.assertEqual(len(set(selected)), 1)

    def test_confirmed_same_source_near_duplicates_never_cross_splits(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            config, source = self._config(tmp)
            root = Path(tmp)
            review = root / "near-review.csv"
            row = {
                "candidate_group_id": "candidate-1",
                "image_a": str(source[0][0].relative_to(root)).replace("\\", "/"),
                "image_b": str(source[6][0].relative_to(root)).replace("\\", "/"),
                "review_decision": "same_source",
            }
            with review.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=row)
                writer.writeheader()
                writer.writerow(row)
            config["near_duplicate_review"] = str(review)
            plan = plan_grouped_split(config)
        selected = [
            row["assigned_split"]
            for row in plan["rows"]
            if row["sample_id"] in {"s0", "s6"}
        ]
        self.assertEqual(len(set(selected)), 1)
        self.assertEqual(plan["grouping"]["near_duplicate_join_count"], 1)

    def test_output_and_test_lock_are_deterministic(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            config, _ = self._config(tmp)
            first = plan_grouped_split(config)
            second = plan_grouped_split(config)
        assignments_a = [
            (row["sample_id"], row["assigned_split"]) for row in first["rows"]
        ]
        assignments_b = [
            (row["sample_id"], row["assigned_split"]) for row in second["rows"]
        ]
        self.assertEqual(assignments_a, assignments_b)
        self.assertEqual(
            first["split_lock"]["test_set_hash"],
            second["split_lock"]["test_set_hash"],
        )
        self.assertEqual(
            first["split_lock"]["lock_hash"],
            second["split_lock"]["lock_hash"],
        )

    def test_parent_only_grouping_is_provisional_and_not_approved(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            config, _ = self._config(tmp, metadata=False)
            plan = plan_grouped_split(config)
        self.assertEqual(plan["grouping"]["level"], "provisional_parent_grouped")
        self.assertTrue(plan["summary"]["provisional"])
        self.assertFalse(plan["summary"]["split_approved"])

    def test_rare_class_imbalance_is_reported(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            config, _ = self._config(tmp)
            plan = plan_grouped_split(config)
        self.assertTrue(
            any("Class 1" in warning for warning in plan["summary"]["warnings"])
        )

    def test_materialization_defaults_disabled_and_never_moves_sources(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            source_image = root / "source.jpg"
            source_label = root / "source.txt"
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(source_image)
            source_label.write_text("0 0 0 1 0 1 1\n")
            plan_dir = root / "plan"
            plan_dir.mkdir()
            row = {
                "sample_id": "s1",
                "file_path": str(source_image),
                "label_path": str(source_label),
                "image_sha256": hashlib.sha256(source_image.read_bytes()).hexdigest(),
                "label_sha256": hashlib.sha256(source_label.read_bytes()).hexdigest(),
                "assigned_split": "train",
            }
            with (plan_dir / "grouped_split_manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=row)
                writer.writeheader()
                writer.writerow(row)
            (plan_dir / "split_summary.json").write_text(
                json.dumps({"materialization_allowed": True})
            )
            (plan_dir / "split_lock.json").write_text(
                json.dumps({"frozen": True, "split_id": "split-test"})
            )
            preview = materialize_grouped_split(
                plan_directory=plan_dir,
                output_directory=root / "derived",
            )
            result = materialize_grouped_split(
                plan_directory=plan_dir,
                output_directory=root / "derived",
                apply=True,
                confirm=True,
            )
            source_hash_after = hashlib.sha256(source_image.read_bytes()).hexdigest()
        self.assertEqual(preview["status"], "preview")
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["source_files_moved"])
        self.assertEqual(source_hash_after, row["image_sha256"])


if __name__ == "__main__":
    unittest.main()
