import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research.defect_evaluation import (  # noqa: E402
    binary_segmentation_metrics,
    build_evaluation_gates,
    instance_segmentation_metrics,
    per_class_segmentation_metrics,
    unavailable_yolo_native_metrics,
)


class DefectEvaluationTests(unittest.TestCase):
    def test_perfect_binary_masks_have_unit_overlap_metrics(self):
        truth = np.asarray([[0, 1], [1, 0]], dtype=bool)
        result = binary_segmentation_metrics(truth, truth)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["iou"], 1.0)
        self.assertEqual(result["dice"], 1.0)

    def test_empty_prediction_against_positive_truth_records_false_negatives(self):
        truth = np.asarray([[1, 0], [1, 0]], dtype=bool)
        prediction = np.zeros_like(truth)
        result = binary_segmentation_metrics(truth, prediction)
        self.assertEqual(result["false_negative_pixels"], 2)
        self.assertEqual(result["true_positive_pixels"], 0)
        self.assertEqual(result["f1"], 0.0)
        self.assertIsNone(result["precision"])

    def test_empty_truth_and_prediction_are_explicit(self):
        empty = np.zeros((2, 2), dtype=bool)
        result = binary_segmentation_metrics(empty, empty)
        self.assertEqual(result["status"], "empty_ground_truth_and_prediction")
        self.assertEqual(result["iou"], 1.0)
        self.assertIsNone(result["recall"])

    def test_per_class_macro_and_micro_are_correct(self):
        truth = np.asarray([[0, 0], [1, 1]])
        predicted = np.asarray([[0, 1], [1, 1]])
        result = per_class_segmentation_metrics(truth, predicted, [0, 1])
        self.assertAlmostEqual(result["macro_average"]["recall"], 0.75)
        self.assertAlmostEqual(result["micro_average"]["precision"], 0.75)
        self.assertAlmostEqual(result["micro_average"]["recall"], 0.75)

    def test_instance_matching_is_one_to_one(self):
        mask = np.asarray([[1, 0], [0, 0]], dtype=bool)
        result = instance_segmentation_metrics([mask], [mask, mask], 0.5)
        self.assertEqual(result["true_positive_objects"], 1)
        self.assertEqual(result["false_positive_objects"], 1)
        self.assertEqual(len(result["matches"]), 1)

    def test_unavailable_map_is_not_zero(self):
        result = unavailable_yolo_native_metrics("not installed")
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["mask_map_50"])
        self.assertIsNone(result["mask_map_50_95"])

    def _ready_context(self):
        return {
            key: True
            for key in (
                "taxonomy_approved",
                "grouped_split_approved",
                "grouped_split_frozen",
                "grouping_leakage_free",
                "test_set_locked",
                "expert_review_sufficient",
                "test_labels_available",
                "primary_metric_approved",
                "model_final_eligible",
                "runtime_dependencies_available",
                "model_classes_match_taxonomy",
                "test_data_excluded_from_training",
                "threshold_tuned_on_validation_only",
                "final_output_available",
                "deterministic_seed_recorded",
            )
        }

    def test_unapproved_taxonomy_and_unfrozen_split_block(self):
        context = self._ready_context()
        context["taxonomy_approved"] = False
        context["grouped_split_frozen"] = False
        gates = {gate["gate"]: gate for gate in build_evaluation_gates(context)}
        self.assertFalse(gates["taxonomy_approved"]["passed"])
        self.assertFalse(gates["grouped_split_frozen"]["passed"])

    def test_legacy_model_and_missing_expert_review_block(self):
        context = self._ready_context()
        context["model_final_eligible"] = False
        context["expert_review_sufficient"] = False
        gates = build_evaluation_gates(context)
        self.assertEqual(sum(not gate["passed"] for gate in gates), 2)

    def test_test_set_threshold_tuning_is_rejected(self):
        context = self._ready_context()
        context["threshold_tuned_on_validation_only"] = False
        gate = next(
            gate for gate in build_evaluation_gates(context)
            if gate["gate"] == "threshold_tuned_on_validation_only"
        )
        self.assertEqual(gate["status"], "blocked")

    def test_exploratory_mode_cannot_bypass_final_gates(self):
        context = self._ready_context()
        context["model_final_eligible"] = False
        gates = build_evaluation_gates(context)
        self.assertFalse(all(gate["passed"] for gate in gates))


if __name__ == "__main__":
    unittest.main()
