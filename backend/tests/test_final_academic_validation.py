import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from research.final_academic_validation import bootstrap_uncertainty  # noqa: E402


def reconstruction_row(specimen_id, provenance="TRUE_DENSE", included="yes"):
    return {
        "specimen_id": specimen_id,
        "reconstruction_provenance": provenance,
        "included_in_dense_aggregate": included,
        "physical_ranked_mm": "10.000 x 5.000 x 2.000",
        "principal_oriented_ranked_mm": "9.000 x 4.000 x 1.000",
        "principal_mae_mm": "1.000000",
        "principal_rmse_mm": "1.000000",
        "principal_median_error_mm": "1.000000",
        "principal_max_error_mm": "1.000000",
    }


def defect_result():
    per_specimen = {
        "Gem_1": {"tp": 1, "fp": 2, "fn": 3, "tn": 4},
        "Gem_2": {"tp": 5, "fp": 7, "fn": 11, "tn": 13},
    }
    per_image = []
    for specimen_id in per_specimen:
        per_image.extend({"specimen_id": specimen_id} for _ in range(5))
    aggregate = {
        field: sum(group[field] for group in per_specimen.values())
        for field in bootstrap_uncertainty.CONFUSION_FIELDS
    }
    return {
        "image_count": 10,
        "specimen_group_count": 2,
        "aggregate": aggregate,
        "per_specimen": per_specimen,
        "per_image": per_image,
        "configuration": {
            "binary_confidence_threshold": 0.25,
            "final_test_threshold_tuning_allowed": False,
        },
        "native_segmentation": {"mask_map_50": 0.1},
        "proposal_90_percent_target": {"status": "NOT ACHIEVED"},
    }


class FinalAcademicValidationTests(unittest.TestCase):
    def test_reconstruction_grouping_excludes_sparse_fallback(self):
        rows = [
            reconstruction_row("QZ-03"),
            reconstruction_row("QZ-30", provenance="SPARSE_FALLBACK", included="no"),
        ]

        groups = bootstrap_uncertainty.reconstruction_groups_from_rows(rows)
        summary = bootstrap_uncertainty.bootstrap_reconstruction(
            rows,
            {"principal_oriented": {}, "proposal_target": {"status": "NOT_ACHIEVED"}},
            iterations=10,
            seed=1,
        )

        self.assertEqual([group["specimen_id"] for group in groups], ["QZ-03"])
        self.assertEqual(len(groups[0]["dimension_abs_errors_mm"]), 3)
        self.assertEqual(summary["resampling_unit"], "physical_specimen")
        self.assertFalse(summary["pixel_or_dimension_independent_bootstrap_used"])
        self.assertIn("QZ-30", summary["sparse_fallback_excluded"])

    def test_current_reconstruction_uncertainty_uses_only_true_dense_specimens(self):
        rows = bootstrap_uncertainty.read_csv_rows(
            bootstrap_uncertainty.RECON_RESULTS_CSV
        )

        groups = bootstrap_uncertainty.reconstruction_groups_from_rows(rows)

        self.assertEqual(
            {group["specimen_id"] for group in groups},
            set(bootstrap_uncertainty.TRUE_DENSE_SPECIMENS),
        )
        self.assertNotIn("QZ-09", {group["specimen_id"] for group in groups})
        self.assertNotIn("QZ-30", {group["specimen_id"] for group in groups})

    def test_bootstrap_seed_is_deterministic(self):
        groups = [
            {"specimen_id": "a", "dimension_abs_errors_mm": [1.0, 2.0, 3.0]},
            {"specimen_id": "b", "dimension_abs_errors_mm": [4.0, 5.0, 6.0]},
        ]

        first = bootstrap_uncertainty.bootstrap_metric_intervals(
            groups,
            bootstrap_uncertainty.reconstruction_metrics,
            seed=123,
            iterations=100,
        )
        second = bootstrap_uncertainty.bootstrap_metric_intervals(
            groups,
            bootstrap_uncertainty.reconstruction_metrics,
            seed=123,
            iterations=100,
        )

        self.assertEqual(first, second)

    def test_defect_confusion_counts_reconstruct_from_selected_groups(self):
        groups = [
            {"specimen_id": "a", "tp": 1, "fp": 2, "fn": 3, "tn": 4},
            {"specimen_id": "b", "tp": 5, "fp": 7, "fn": 11, "tn": 13},
        ]

        counts = bootstrap_uncertainty.aggregate_confusion_counts(groups, [1, 0, 1])
        metrics = bootstrap_uncertainty.compute_defect_metrics(counts)

        self.assertEqual(counts, {"tp": 11, "fp": 16, "fn": 25, "tn": 30})
        self.assertAlmostEqual(metrics["precision"], 11 / 27)
        self.assertAlmostEqual(metrics["recall"], 11 / 36)

    def test_defect_bootstrap_uses_group_level_not_pixels_or_images(self):
        summary = bootstrap_uncertainty.bootstrap_defect(
            defect_result(),
            seed=5,
            iterations=100,
        )

        self.assertEqual(
            summary["resampling_unit"],
            "independent_physical_specimen_group",
        )
        self.assertEqual(summary["specimen_group_count"], 2)
        self.assertEqual(summary["image_count"], 10)
        self.assertTrue(summary["all_groups_have_five_images"])
        self.assertTrue(summary["group_sum_check"]["matches_frozen_aggregate"])
        self.assertFalse(summary["pixel_level_bootstrap_used"])
        self.assertFalse(summary["image_level_bootstrap_used"])
        self.assertFalse(summary["native_ultralytics_inference_run"])
        self.assertFalse(summary["thresholds_or_model_mutated"])
        self.assertFalse(summary["configuration"]["final_test_threshold_tuning_allowed"])

    def test_defect_map50_ci_is_not_fabricated(self):
        summary = bootstrap_uncertainty.bootstrap_defect(
            defect_result(),
            seed=5,
            iterations=10,
        )

        self.assertEqual(
            summary["native_mask_map50_ci"]["status"],
            "CI_NOT_ESTIMATED",
        )


if __name__ == "__main__":
    unittest.main()
