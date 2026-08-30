import contextlib
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research.dataset_audit import (  # noqa: E402
    MANIFEST_FIELDS,
    audit_dataset,
    augmentation_parent_id,
)
from research_benchmark import main as benchmark_main  # noqa: E402


VALID_LABEL = "0 0.1 0.1 0.8 0.1 0.8 0.8\n"


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        for split in ("train", "valid", "test"):
            (self.dataset / split / "images").mkdir(parents=True)
            (self.dataset / split / "labels").mkdir(parents=True)
        self.yaml_path = self.dataset / "data.yaml"
        self.yaml_path.write_text(
            "\n".join(
                [
                    "train: ./train/images",
                    "val: ./valid/images",
                    "test: ./test/images",
                    "",
                    "nc: 4",
                    "names: ['Blue', 'Inclusion', 'Red', 'Yellow']",
                    "",
                    "roboflow:",
                    "  project: synthetic-quartz-test",
                    "  version: 1",
                    "  license: CC BY 4.0",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _image(self, split, name, color=(40, 80, 120), size=(16, 16)):
        path = self.dataset / split / "images" / name
        Image.new("RGB", size, color).save(path)
        return path

    def _label(self, split, stem, content=VALID_LABEL):
        path = self.dataset / split / "labels" / f"{stem}.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def _sample(self, split, name, label=VALID_LABEL, color=(40, 80, 120)):
        image = self._image(split, name, color=color)
        self._label(split, image.stem, label)
        return image

    def _config(self, manifest=None, near_duplicates=False):
        return {
            "_repo_root": str(self.root),
            "dataset_yaml": str(self.yaml_path),
            "metadata_manifest": str(manifest) if manifest else None,
            "supported_image_extensions": [".png", ".jpg", ".jpeg"],
            "near_duplicate_detection": {
                "enabled": near_duplicates,
                "hamming_distance_threshold": 5,
                "max_candidates": 100,
            },
            "deterministic_seed": 7,
            "proposed_split_ratios": {
                "train": 0.6,
                "valid": 0.2,
                "test": 0.2,
            },
            "source_metadata": {"provider": "synthetic-test"},
            "license_metadata": {"name": "test-only"},
        }

    def _audit(self, manifest=None, near_duplicates=False):
        return audit_dataset(self._config(manifest, near_duplicates))

    @staticmethod
    def _codes(report):
        return [issue["error_code"] for issue in report["issues"]]

    def _manifest(self, rows):
        path = self.root / "dataset_manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _manifest_row(self, image, split, sample_id, **overrides):
        row = {field: "" for field in MANIFEST_FIELDS}
        row.update(
            {
                "sample_id": sample_id,
                "file_path": str(image),
                "label_path": str(
                    self.dataset / split / "labels" / f"{image.stem}.txt"
                ),
                "split": split,
                "augmentation_parent_id": augmentation_parent_id(image.stem),
                "source_dataset": "synthetic-test",
                "license": "test-only",
            }
        )
        row.update(overrides)
        return row

    def test_correct_image_label_pairing(self):
        self._sample("train", "stone-a.png")

        report = self._audit()

        self.assertNotIn("image_without_label", self._codes(report))
        self.assertNotIn("label_without_image", self._codes(report))
        self.assertEqual(report["splits"][0]["image_count"], 1)

    def test_image_without_label(self):
        self._image("train", "missing-label.png")

        report = self._audit()

        self.assertIn("image_without_label", self._codes(report))

    def test_label_without_image(self):
        self._label("valid", "missing-image")

        report = self._audit()

        self.assertIn("label_without_image", self._codes(report))

    def test_empty_label(self):
        image = self._image("train", "empty.png")
        self._label("train", image.stem, "")

        report = self._audit()

        self.assertIn("empty_label", self._codes(report))
        self.assertEqual(report["totals"]["empty_labels"], 1)

    def test_invalid_class_id(self):
        self._sample("train", "bad-class.png", "8 0.1 0.1 0.8 0.1 0.8 0.8\n")

        report = self._audit()

        self.assertIn("class_id_out_of_range", self._codes(report))

    def test_invalid_coordinate_count(self):
        self._sample("train", "bad-count.png", "0 0.1 0.1 0.8 0.1 0.8\n")

        report = self._audit()

        self.assertIn("invalid_coordinate_count", self._codes(report))

    def test_out_of_range_coordinates(self):
        self._sample(
            "train",
            "bad-range.png",
            "0 0.1 0.1 1.2 0.1 0.8 0.8\n",
        )

        report = self._audit()

        self.assertIn("coordinate_out_of_range", self._codes(report))

    def test_exact_duplicate_images(self):
        first = self._sample("train", "duplicate-a.png")
        second = self.dataset / "train" / "images" / "duplicate-b.png"
        second.write_bytes(first.read_bytes())
        self._label("train", second.stem)

        report = self._audit()
        groups = [
            group for group in report["duplicate_groups"]
            if group["duplicate_type"] == "image_sha256"
        ]

        self.assertEqual(len(groups), 1)
        self.assertFalse(groups[0]["cross_split"])

    def test_exact_duplicate_across_splits(self):
        first = self._sample("train", "duplicate-train.png")
        second = self.dataset / "test" / "images" / "duplicate-test.png"
        second.write_bytes(first.read_bytes())
        self._label("test", second.stem)

        report = self._audit()

        self.assertEqual(report["totals"]["cross_split_exact_duplicate_groups"], 1)
        self.assertFalse(report["claim_readiness"]["claim_ready"])

    def test_conflicting_labels_for_identical_images(self):
        first = self._sample("train", "conflict-a.png")
        second = self.dataset / "valid" / "images" / "conflict-b.png"
        second.write_bytes(first.read_bytes())
        self._label("valid", second.stem, "1 0.2 0.2 0.7 0.2 0.7 0.7\n")

        report = self._audit()
        image_group = next(
            group for group in report["duplicate_groups"]
            if group["duplicate_type"] == "image_sha256"
        )

        self.assertTrue(image_group["conflicting_labels"])

    def test_roboflow_augmentation_parent_derivation(self):
        parent = augmentation_parent_id("stone_01_jpg.rf.0123456789abcdef")

        self.assertEqual(parent, "stone_01_jpg")

    def test_augmentation_parent_across_splits(self):
        self._sample("train", "stone_01_jpg.rf.aaaa.png")
        self._sample("test", "stone_01_jpg.rf.bbbb.png", color=(90, 10, 20))

        report = self._audit()

        self.assertEqual(report["totals"]["augmentation_parent_leakage_groups"], 1)
        self.assertEqual(report["totals"]["augmentation_leakage_affected_images"], 2)

    def test_class_distribution(self):
        self._sample(
            "train",
            "classes.png",
            VALID_LABEL + "1 0.2 0.2 0.7 0.2 0.7 0.7\n",
        )

        report = self._audit()
        rows = {
            (row["split"], row["class_id"]): row
            for row in report["class_distribution"]
        }

        self.assertEqual(rows[("train", 0)]["object_count"], 1)
        self.assertEqual(rows[("train", 1)]["object_count"], 1)
        self.assertEqual(rows[("train", 1)]["image_count"], 1)

    def test_missing_specimen_metadata_is_unavailable(self):
        self._sample("train", "no-manifest.png")

        report = self._audit()

        self.assertEqual(
            report["metadata_validation"]["specimen_check"]["status"],
            "unavailable",
        )
        self.assertIn(
            "Specimen identity metadata was not supplied.",
            report["claim_readiness"]["unavailable_checks"],
        )

    def test_specimen_overlap_with_manifest(self):
        train = self._sample("train", "specimen-train.png")
        test = self._sample("test", "specimen-test.png", color=(70, 50, 30))
        manifest = self._manifest(
            [
                self._manifest_row(
                    train,
                    "train",
                    "sample-train",
                    specimen_id="stone-1",
                ),
                self._manifest_row(
                    test,
                    "test",
                    "sample-test",
                    specimen_id="stone-1",
                ),
            ]
        )

        report = self._audit(manifest)

        self.assertTrue(
            any(group["level"] == "specimen" for group in report["leakage_groups"])
        )

    def test_recording_overlap_with_manifest(self):
        train = self._sample("train", "recording-train.png")
        valid = self._sample("valid", "recording-valid.png", color=(15, 35, 55))
        manifest = self._manifest(
            [
                self._manifest_row(
                    train,
                    "train",
                    "sample-train",
                    specimen_id="stone-1",
                    recording_id="recording-1",
                ),
                self._manifest_row(
                    valid,
                    "valid",
                    "sample-valid",
                    specimen_id="stone-2",
                    recording_id="recording-1",
                ),
            ]
        )

        report = self._audit(manifest)

        self.assertTrue(
            any(group["level"] == "recording" for group in report["leakage_groups"])
        )

    def test_recording_coverage_is_calculated(self):
        first = self._sample("train", "recording-covered.png")
        second = self._sample(
            "valid", "recording-unavailable.png", color=(15, 35, 55)
        )
        manifest = self._manifest(
            [
                self._manifest_row(
                    first,
                    "train",
                    "sample-covered",
                    recording_id="recording-1",
                ),
                self._manifest_row(second, "valid", "sample-unavailable"),
            ]
        )

        report = self._audit(manifest)

        self.assertEqual(
            report["metadata_validation"]["recording_id_coverage"],
            50.0,
        )

    def test_duplicate_manifest_sample_id_is_rejected(self):
        first = self._sample("train", "duplicate-id-a.png")
        second = self._sample(
            "valid", "duplicate-id-b.png", color=(15, 35, 55)
        )
        manifest = self._manifest(
            [
                self._manifest_row(first, "train", "duplicate-id"),
                self._manifest_row(second, "valid", "duplicate-id"),
            ]
        )

        report = self._audit(manifest)

        self.assertIn("duplicate_sample_id", self._codes(report))

    def test_missing_manifest_references_are_reported(self):
        missing_image_source = self._sample(
            "train", "manifest-image-reference.png"
        )
        missing_label_source = self._sample(
            "valid",
            "manifest-label-reference.png",
            color=(15, 35, 55),
        )
        missing_image = self._manifest_row(
            missing_image_source, "train", "missing-image-reference"
        )
        missing_image["file_path"] = str(self.root / "missing-image.png")
        missing_label = self._manifest_row(
            missing_label_source, "valid", "missing-label-reference"
        )
        missing_label["label_path"] = str(self.root / "missing-label.txt")
        manifest = self._manifest([missing_image, missing_label])

        report = self._audit(manifest)

        self.assertIn("manifest_image_missing", self._codes(report))
        self.assertIn("manifest_label_missing", self._codes(report))

    def test_expert_review_coverage_is_calculated(self):
        first = self._sample("train", "expert-approved.png")
        second = self._sample(
            "valid", "expert-unreviewed.png", color=(15, 35, 55)
        )
        manifest = self._manifest(
            [
                self._manifest_row(
                    first,
                    "train",
                    "expert-approved",
                    expert_review_status="approved",
                ),
                self._manifest_row(
                    second,
                    "valid",
                    "expert-unreviewed",
                    expert_review_status="unreviewed",
                ),
            ]
        )

        report = self._audit(manifest)

        self.assertEqual(
            report["metadata_validation"]["expert_review_coverage"],
            50.0,
        )

    def test_deterministic_proposed_split(self):
        for index, split in enumerate(("train", "train", "valid", "test")):
            self._sample(
                split,
                f"parent-{index}_jpg.rf.hash-{index}.png",
                color=(index * 20, 20, 100),
            )

        first = self._audit()["proposed_split"]["assignments"]
        second = self._audit()["proposed_split"]["assignments"]

        self.assertEqual(first, second)

    def test_dataset_files_remain_unchanged(self):
        self._sample("train", "unchanged.png")
        files = sorted(path for path in self.dataset.rglob("*") if path.is_file())
        before = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        }

        self._audit(near_duplicates=True)

        after = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        }
        self.assertEqual(before, after)

    def test_claim_readiness_false_for_leakage(self):
        self._sample("train", "leak_jpg.rf.a.png")
        self._sample("valid", "leak_jpg.rf.b.png", color=(1, 2, 3))

        report = self._audit()

        self.assertFalse(report["claim_readiness"]["claim_ready"])
        self.assertTrue(
            any(
                "augmentation-parent" in reason
                for reason in report["claim_readiness"]["blocking_reasons"]
            )
        )

    def test_benchmark_cli_integration(self):
        self._sample("train", "cli.png")
        config = self._config()
        output = self.root / "evidence"
        config["output_directory"] = str(output)
        config_path = self.root / "audit-config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        with contextlib.redirect_stdout(io.StringIO()):
            result = benchmark_main(
                ["dataset-audit", "--config", str(config_path)]
            )

        self.assertEqual(result, 0)
        for name in (
            "dataset_audit.json",
            "dataset_summary.csv",
            "dataset_issues.csv",
            "duplicate_groups.csv",
            "leakage_groups.csv",
            "class_distribution.csv",
            "proposed_split.csv",
            "audit_summary.md",
            "manifest.json",
        ):
            self.assertTrue((output / name).exists(), name)

    def test_benchmark_cli_strict_mode_returns_blocking_exit(self):
        self._sample("train", "strict_jpg.rf.a.png")
        self._sample("test", "strict_jpg.rf.b.png", color=(3, 4, 5))
        config = self._config()
        output = self.root / "strict-evidence"
        config["output_directory"] = str(output)
        config_path = self.root / "strict-audit-config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        with contextlib.redirect_stdout(io.StringIO()):
            result = benchmark_main(
                ["dataset-audit", "--config", str(config_path), "--strict"]
            )

        self.assertEqual(result, 2)
        self.assertTrue((output / "dataset_audit.json").exists())


if __name__ == "__main__":
    unittest.main()
