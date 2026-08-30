import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai_runner import run_ai_pipeline  # noqa: E402


class FakeModel:
    def __init__(self, results=None, error=None):
        self.results = results
        self.error = error
        self.names = {0: "Crack", 1: "Inclusion"}

    def predict(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.results


class ClassAwareAiRunnerTests(unittest.TestCase):
    def _job(self, root):
        job = Path(root) / "job"
        images = job / "images"
        images.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            images / "sample.jpg"
        )
        model = Path(root) / "model.pt"
        model.write_bytes(b"synthetic-model")
        return job, model

    def _taxonomy(self, root, status="confirmed_defect", confirmed=True):
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
            writer.writerow(
                {
                    "class_id": 1,
                    "class_name": "Inclusion",
                    "semantic_definition": "Synthetic test class",
                    "taxonomy_status": status,
                    "is_visible_defect": True,
                    "eligible_for_3d_mapping": True,
                    "eligible_for_no_cut_zone": True,
                    "minimum_confidence": "",
                    "expert_confirmed": confirmed,
                    "confirmed_by": "",
                    "confirmed_date": "",
                    "notes": "",
                }
            )
        return path

    def _config(self, root, taxonomy, policy="yolo_only", strict=True):
        path = Path(root) / f"{policy}.json"
        value = {
            "policy": policy,
            "taxonomy_path": str(taxonomy),
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
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def _result(self):
        mask = np.zeros((1, 8, 8), dtype=float)
        mask[0, 2:6, 2:6] = 1.0
        return SimpleNamespace(
            masks=SimpleNamespace(data=mask),
            boxes=SimpleNamespace(
                cls=np.asarray([1.0]),
                conf=np.asarray([0.88]),
            ),
            names={1: "Inclusion"},
            orig_shape=(8, 8),
        )

    def test_yolo_record_preserves_class_confidence_hash_and_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, model_path = self._job(tmp)
            taxonomy = self._taxonomy(tmp)
            config = self._config(tmp, taxonomy)
            fake = FakeModel(results=[self._result()])
            summary = run_ai_pipeline(
                job,
                policy_config_path=config,
                model_path=model_path,
                model_factory=lambda _: fake,
                cv_mask_factory=lambda _: np.ones((8, 8), dtype=np.uint8) * 255,
            )
            raw = json.loads(
                (job / "detections" / "raw_predictions.json").read_text()
            )
            decisions = json.loads(
                (job / "detections" / "policy_decisions.json").read_text()
            )
            raw_record = [
                item for item in raw["records"] if item["source"] == "yolo"
            ][0]
            decision = [
                item for item in decisions["records"] if item["source"] == "yolo"
            ][0]
            raw_mask_exists = (job / raw_record["mask_path"]).exists()

        self.assertEqual(raw_record["class_id"], 1)
        self.assertEqual(raw_record["class_name"], "Inclusion")
        self.assertAlmostEqual(raw_record["confidence"], 0.88)
        self.assertTrue(raw_record["model_sha256"])
        self.assertFalse(raw_record["accepted_for_3d_mapping"])
        self.assertTrue(decision["accepted_for_3d_mapping"])
        self.assertTrue(decision["mapping_mask_path"])
        self.assertEqual(summary["raw_prediction_count"], 2)
        self.assertEqual(summary["mapping_count"], 1)
        self.assertEqual(summary["no_cut_count"], 1)
        self.assertTrue(raw_mask_exists)

    def test_fallback_activates_for_load_and_inference_failures(self):
        for failure in ("load", "inference"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                job, model_path = self._job(tmp)
                taxonomy = self._taxonomy(tmp)
                config = self._config(
                    tmp,
                    taxonomy,
                    policy="fallback",
                    strict=False,
                )
                if failure == "load":
                    factory = lambda _: (_ for _ in ()).throw(RuntimeError("load"))
                else:
                    factory = lambda _: FakeModel(error=RuntimeError("predict"))
                summary = run_ai_pipeline(
                    job,
                    policy_config_path=config,
                    model_path=model_path,
                    model_factory=factory,
                    cv_mask_factory=lambda _: np.ones(
                        (8, 8), dtype=np.uint8
                    ) * 255,
                )
                decisions = json.loads(
                    (job / "detections" / "policy_decisions.json").read_text()
                )
                opencv = [
                    item
                    for item in decisions["records"]
                    if item["source"] == "opencv"
                ][0]
                self.assertTrue(opencv["accepted_for_no_cut_zone"])
                self.assertIsNone(opencv["confidence"])
                self.assertEqual(summary["no_cut_count"], 1)

    def test_zero_detections_does_not_activate_fallback_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, model_path = self._job(tmp)
            taxonomy = self._taxonomy(tmp)
            config = self._config(
                tmp,
                taxonomy,
                policy="fallback",
                strict=False,
            )
            summary = run_ai_pipeline(
                job,
                policy_config_path=config,
                model_path=model_path,
                model_factory=lambda _: FakeModel(results=[]),
                cv_mask_factory=lambda _: np.ones((8, 8), dtype=np.uint8) * 255,
            )
            decisions = json.loads(
                (job / "detections" / "policy_decisions.json").read_text()
            )
            opencv = decisions["records"][0]

        self.assertEqual(decisions["image_events"]["sample.jpg"], "zero_detections")
        self.assertFalse(opencv["accepted_for_no_cut_zone"])
        self.assertEqual(opencv["rejection_reason"], "fallback_not_activated")
        self.assertEqual(summary["no_cut_count"], 0)


if __name__ == "__main__":
    unittest.main()
