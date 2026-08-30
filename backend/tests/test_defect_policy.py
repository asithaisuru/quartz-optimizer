import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from defect_policy import (  # noqa: E402
    DetectionRecord,
    approved_no_cut_cloud_path,
    apply_policy,
    load_taxonomy,
    materialize_policy_outputs,
    policy_summary,
    validate_policy_config,
)


class DefectPolicyTests(unittest.TestCase):
    def _config(self, policy="yolo_only", strict=True):
        return {
            "policy": policy,
            "taxonomy_path": "taxonomy.csv",
            "strict_research_mode": strict,
            "allow_provisional_for_visualization": True,
            "allow_provisional_for_no_cut": not strict,
            "unknown_class_action": "exclude",
            "minimum_confidence_default": 0.25,
            "minimum_confidence_by_class": {},
            "opencv": {
                "enabled": True,
                "eligible_for_visualization": True,
                "eligible_for_3d_mapping": not strict,
                "eligible_for_no_cut": not strict,
                "minimum_area_pixels": 0,
                "gated_union_min_iou": 0.1,
                "fallback_on_model_unavailable": True,
                "fallback_on_inference_error": True,
                "fallback_on_zero_detections": False,
            },
            "legacy": {"allow_legacy_union_all": False},
        }

    def _taxonomy(self, root, rows):
        path = Path(root) / "taxonomy.csv"
        fields = [
            "class_id",
            "class_name",
            "semantic_definition",
            "taxonomy_status",
            "is_visible_defect",
            "eligible_for_3d_mapping",
            "eligible_for_no_cut_zone",
            "minimum_confidence",
            "expert_confirmed",
            "confirmed_by",
            "confirmed_date",
            "notes",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path, load_taxonomy(path)

    def _entry(self, class_id, name, status, **values):
        return {
            "class_id": class_id,
            "class_name": name,
            "semantic_definition": values.get("semantic_definition", "test"),
            "taxonomy_status": status,
            "is_visible_defect": values.get("visible", True),
            "eligible_for_3d_mapping": values.get("mapping", True),
            "eligible_for_no_cut_zone": values.get("no_cut", True),
            "minimum_confidence": values.get("threshold", ""),
            "expert_confirmed": values.get("confirmed", False),
            "confirmed_by": "",
            "confirmed_date": "",
            "notes": "",
        }

    def _mask(self, root, name, pixels=None):
        path = Path(root) / name
        array = np.zeros((8, 8), dtype=np.uint8)
        if pixels is None:
            array[2:6, 2:6] = 255
        else:
            for row, column in pixels:
                array[row, column] = 255
        Image.fromarray(array).save(path)
        return path

    def _record(
        self,
        root,
        source="yolo",
        class_id=0,
        name="Crack",
        confidence=0.9,
        mask_name="mask.png",
    ):
        path = self._mask(root, mask_name)
        return DetectionRecord(
            prediction_id=f"image-{source}-{mask_name}",
            image_id="image.jpg",
            source_image_path="images/image.jpg",
            source=source,
            class_id=class_id if source == "yolo" else None,
            class_name=name if source == "yolo" else "opencv_candidate",
            confidence=confidence if source == "yolo" else None,
            mask_path=str(path),
            mask_width=8,
            mask_height=8,
            mask_area_pixels=16,
        )

    def test_confirmed_defect_preserves_class_and_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, taxonomy = self._taxonomy(
                tmp,
                [
                    self._entry(
                        0,
                        "Crack",
                        "confirmed_defect",
                        confirmed=True,
                    )
                ],
            )
            record = self._record(tmp, class_id=0, name="raw-name")
            apply_policy([record], taxonomy, self._config())

        self.assertEqual(record.class_id, 0)
        self.assertEqual(record.class_name, "Crack")
        self.assertEqual(record.confidence, 0.9)
        self.assertTrue(record.accepted_for_visualization)
        self.assertTrue(record.accepted_for_3d_mapping)
        self.assertTrue(record.accepted_for_no_cut_zone)

    def test_non_defect_unknown_and_provisional_are_not_no_cut_in_strict_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, taxonomy = self._taxonomy(
                tmp,
                [
                    self._entry(
                        0,
                        "NonDefect",
                        "confirmed_non_defect",
                        confirmed=True,
                    ),
                    self._entry(1, "Candidate", "provisional_defect"),
                ],
            )
            non_defect = self._record(tmp, class_id=0, mask_name="non.png")
            provisional = self._record(tmp, class_id=1, mask_name="prov.png")
            unknown = self._record(tmp, class_id=9, mask_name="unknown.png")
            apply_policy(
                [non_defect, provisional, unknown],
                taxonomy,
                self._config(),
            )

        self.assertEqual(non_defect.rejection_reason, "confirmed_non_defect")
        self.assertTrue(provisional.accepted_for_visualization)
        self.assertFalse(provisional.accepted_for_3d_mapping)
        self.assertFalse(provisional.accepted_for_no_cut_zone)
        self.assertEqual(unknown.rejection_reason, "class_missing_from_taxonomy")

    def test_per_class_confidence_threshold_is_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, taxonomy = self._taxonomy(
                tmp,
                [
                    self._entry(
                        0,
                        "Crack",
                        "confirmed_defect",
                        confirmed=True,
                    )
                ],
            )
            config = self._config()
            config["minimum_confidence_by_class"] = {"Crack": 0.95}
            record = self._record(tmp, confidence=0.94)
            apply_policy([record], taxonomy, config)

        self.assertEqual(record.threshold, 0.95)
        self.assertEqual(record.rejection_reason, "below_confidence_threshold")

    def test_yolo_only_excludes_opencv_and_union_keeps_explicit_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, taxonomy = self._taxonomy(
                tmp,
                [
                    self._entry(
                        0,
                        "Crack",
                        "confirmed_defect",
                        confirmed=True,
                    )
                ],
            )
            yolo = self._record(tmp, mask_name="yolo.png")
            opencv = self._record(tmp, source="opencv", mask_name="cv.png")
            apply_policy([yolo, opencv], taxonomy, self._config())
            self.assertTrue(yolo.accepted_for_no_cut_zone)
            self.assertEqual(opencv.rejection_reason, "excluded_by_yolo_only")

            yolo = self._record(tmp, mask_name="yolo2.png")
            opencv = self._record(tmp, source="opencv", mask_name="cv2.png")
            apply_policy(
                [yolo, opencv],
                taxonomy,
                self._config("union", strict=False),
            )
            self.assertTrue(yolo.accepted_for_no_cut_zone)
            self.assertTrue(opencv.accepted_for_no_cut_zone)
            self.assertIsNone(opencv.confidence)

    def test_opencv_only_requires_explicit_channel_permission(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, taxonomy = self._taxonomy(tmp, [])
            record = self._record(tmp, source="opencv")
            config = self._config("opencv_only", strict=False)
            config["opencv"]["eligible_for_3d_mapping"] = False
            config["opencv"]["eligible_for_no_cut"] = False
            apply_policy([record], taxonomy, config)
            self.assertFalse(record.accepted_for_no_cut_zone)

            record = self._record(tmp, source="opencv", mask_name="cv2.png")
            config["opencv"]["eligible_for_3d_mapping"] = True
            config["opencv"]["eligible_for_no_cut"] = True
            apply_policy([record], taxonomy, config)
            self.assertTrue(record.accepted_for_no_cut_zone)

    def test_gated_union_records_overlap_acceptance_and_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, taxonomy = self._taxonomy(
                tmp,
                [
                    self._entry(
                        0,
                        "Crack",
                        "confirmed_defect",
                        confirmed=True,
                    )
                ],
            )
            config = self._config("gated_union", strict=False)
            yolo = self._record(tmp, mask_name="yolo.png")
            overlap = self._record(tmp, source="opencv", mask_name="overlap.png")
            apply_policy([yolo, overlap], taxonomy, config)
            self.assertTrue(overlap.accepted_for_no_cut_zone)
            self.assertGreaterEqual(
                overlap.policy_metadata["overlap_value"],
                overlap.policy_metadata["overlap_threshold"],
            )

            yolo = self._record(tmp, mask_name="yolo2.png")
            other_path = self._mask(tmp, "other.png", [(0, 0)])
            separate = self._record(
                tmp, source="opencv", mask_name="separate-original.png"
            )
            separate.mask_path = str(other_path)
            separate.mask_area_pixels = 1
            apply_policy([yolo, separate], taxonomy, config)
            self.assertFalse(separate.accepted_for_no_cut_zone)
            self.assertEqual(
                separate.rejection_reason,
                "gated_union_overlap_below_threshold",
            )

    def test_fallback_conditions_distinguish_failures_from_zero_detections(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, taxonomy = self._taxonomy(tmp, [])
            config = self._config("fallback", strict=False)
            for event in ("model_unavailable", "model_load_error", "inference_error"):
                record = self._record(
                    tmp,
                    source="opencv",
                    mask_name=f"{event}.png",
                )
                apply_policy(
                    [record],
                    taxonomy,
                    config,
                    image_events={"image.jpg": event},
                )
                self.assertTrue(record.accepted_for_no_cut_zone)

            zero = self._record(tmp, source="opencv", mask_name="zero.png")
            apply_policy(
                [zero],
                taxonomy,
                config,
                image_events={"image.jpg": "zero_detections"},
            )
            self.assertFalse(zero.accepted_for_no_cut_zone)
            self.assertEqual(zero.rejection_reason, "fallback_not_activated")

    def test_legacy_union_warns_and_strict_mode_rejects_configuration(self):
        config = self._config("legacy_union_all", strict=False)
        config["legacy"]["allow_legacy_union_all"] = True
        with tempfile.TemporaryDirectory() as tmp:
            _, taxonomy = self._taxonomy(tmp, [])
            record = self._record(tmp, source="opencv")
            apply_policy([record], taxonomy, config)
            summary = policy_summary([record], config)
        self.assertIn("unsafe_legacy_policy", summary["warnings"])

        config["strict_research_mode"] = True
        with self.assertRaises(ValueError):
            validate_policy_config(config)

    def test_raw_mask_is_unchanged_and_rejected_record_remains_traceable(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job"
            job.mkdir()
            _, taxonomy = self._taxonomy(tmp, [])
            record = self._record(tmp, class_id=99)
            before = Path(record.mask_path).read_bytes()
            apply_policy([record], taxonomy, self._config())
            materialize_policy_outputs([record], job)
            after = Path(record.mask_path).read_bytes()
        self.assertEqual(before, after)
        self.assertEqual(record.prediction_id, "image-yolo-mask.png")
        self.assertIsNotNone(record.rejection_reason)

    def test_summary_separates_raw_mapping_no_cut_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, taxonomy = self._taxonomy(
                tmp,
                [
                    self._entry(
                        0,
                        "Crack",
                        "confirmed_defect",
                        confirmed=True,
                    ),
                    self._entry(1, "Candidate", "provisional_defect"),
                ],
            )
            records = [
                self._record(tmp, class_id=0, mask_name="confirmed.png"),
                self._record(tmp, class_id=1, mask_name="provisional.png"),
            ]
            apply_policy(records, taxonomy, self._config())
            first = policy_summary(records, self._config())
            second = policy_summary(records, self._config())
        self.assertEqual(first, second)
        self.assertEqual(first["raw_prediction_count"], 2)
        self.assertEqual(first["mapping_count"], 1)
        self.assertEqual(first["no_cut_count"], 1)

    def test_missing_confirmed_taxonomy_reports_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, taxonomy = self._taxonomy(tmp, [])
            record = self._record(tmp, class_id=7)
            config = self._config()
            apply_policy([record], taxonomy, config)
            summary = policy_summary([record], config)
        self.assertEqual(summary["claim_status"], "unavailable")
        self.assertEqual(
            summary["reason"],
            "No confirmed defect taxonomy was supplied.",
        )

    def test_visualization_cloud_alone_cannot_reach_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            (job / "detections").mkdir()
            (job / "dense").mkdir()
            (job / "dense" / "defects.ply").write_text("visualization")
            (job / "detections" / "policy_summary.json").write_text(
                json.dumps(
                    {
                        "no_cut_count": 0,
                        "strict_research_mode": True,
                        "legacy_input_used": False,
                        "provisional_input_used": False,
                    }
                )
            )
            path, reason = approved_no_cut_cloud_path(job)
            self.assertIsNone(path)
            self.assertEqual(reason, "no_approved_no_cut_masks")

            (job / "dense" / "no_cut_defects.ply").write_text("approved")
            (job / "detections" / "policy_summary.json").write_text(
                json.dumps(
                    {
                        "no_cut_count": 1,
                        "strict_research_mode": True,
                        "legacy_input_used": False,
                        "provisional_input_used": False,
                    }
                )
            )
            path, reason = approved_no_cut_cloud_path(job)
            self.assertEqual(path, job / "dense" / "no_cut_defects.ply")
            self.assertEqual(reason, "approved")


if __name__ == "__main__":
    unittest.main()
