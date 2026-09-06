import os
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from research.defect_validation.evaluate_independent_final_test import (  # noqa: E402
    _mask_from_yolo_label,
    _metrics_from_counts,
    device_selection_report,
)
from research.defect_validation.independent_final_test import (  # noqa: E402
    EXPECTED_FINAL_METRICS,
    EXPECTED_NATIVE_MASK_MAP50,
    final_result_freeze_metadata,
    frozen_metric_configuration,
)


class IndependentDefectValidationTests(unittest.TestCase):
    def test_yolo_label_rasterization_uses_width_height_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "label.txt"
            label.write_text("0 0 0 0.5 0 0.5 1 0 1\n", encoding="utf-8")

            mask = _mask_from_yolo_label(label, width=10, height=4, class_id=0)
            other_class = _mask_from_yolo_label(label, width=10, height=4, class_id=1)

        self.assertEqual(mask.shape, (4, 10))
        self.assertTrue(mask[:, 0].all())
        self.assertFalse(mask[:, 9].any())
        self.assertFalse(other_class.any())

    def test_empty_label_file_is_empty_ground_truth_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "negative.txt"
            label.write_text("", encoding="utf-8")

            mask = _mask_from_yolo_label(label, width=10, height=4)

        self.assertEqual(mask.shape, (4, 10))
        self.assertFalse(mask.any())

    def test_final_metrics_recompute_from_frozen_counts(self):
        counts = {
            "tp": 520,
            "fp": 16736,
            "fn": 541081,
            "tn": 92753663,
        }
        metrics = _metrics_from_counts(counts)

        for key, expected in EXPECTED_FINAL_METRICS.items():
            self.assertAlmostEqual(metrics[key], expected)

    def test_freeze_metadata_records_independent_cpu_no_tuning_result(self):
        metadata = final_result_freeze_metadata()
        config = frozen_metric_configuration()
        device = device_selection_report()

        self.assertEqual(metadata["validation_role"], "independent_test")
        self.assertTrue(metadata["labels_frozen_before_inference_required"])
        self.assertTrue(metadata["no_final_test_threshold_tuning"])
        self.assertEqual(metadata["evaluation_device"], "cpu")
        self.assertIn("RTX 5050", metadata["cpu_device_reason"])
        self.assertEqual(metadata["post_evaluation_integrity_audit"], "PASS")
        self.assertEqual(metadata["proposal_target_status"], "NOT ACHIEVED")
        self.assertFalse(config["final_test_threshold_tuning_allowed"])
        self.assertEqual(config["native_ultralytics_validation_iou"], 0.7)
        self.assertEqual(EXPECTED_NATIVE_MASK_MAP50, 0.00043102434676857947)
        self.assertEqual(device["device"], "cpu")
        self.assertFalse(device["cuda_autodetect_used"])


if __name__ == "__main__":
    unittest.main()
