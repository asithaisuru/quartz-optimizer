import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research.model_provenance import inspect_model_provenance  # noqa: E402


class ModelProvenanceTests(unittest.TestCase):
    def _files(self, root, leakage=2):
        root = Path(root)
        model = root / "best.pt"
        model.write_bytes(b"model-bytes")
        yaml = root / "data.yaml"
        yaml.write_text("names: ['Defect']\n")
        args = root / "args.yaml"
        args.write_text(f"model: yolov8n-seg.pt\ndata: {yaml}\n")
        source = root / "train.py"
        source.write_text("print('training')\n")
        audit = root / "audit.json"
        audit.write_text(
            json.dumps(
                {
                    "totals": {
                        "augmentation_parent_leakage_groups": leakage,
                        "augmentation_leakage_affected_images": 8,
                    }
                }
            )
        )
        return model, yaml, args, source, audit

    def test_likely_contaminated_model_is_not_final_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, yaml, args, source, audit = self._files(tmp)
            before = model.read_bytes()
            report = inspect_model_provenance(
                model_path=model,
                dataset_yaml=yaml,
                training_args=args,
                training_source=source,
                dataset_audit=audit,
            )
            after = model.read_bytes()
        self.assertEqual(report["training_split_provenance"], "likely_contaminated")
        self.assertFalse(report["final_research_eligible"])
        self.assertTrue(report["exploratory_pipeline_smoke_test"])
        self.assertEqual(before, after)

    def test_unknown_provenance_blocks_final_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "best.pt"
            model.write_bytes(b"unknown")
            report = inspect_model_provenance(model_path=model)
        self.assertEqual(report["training_split_provenance"], "unknown")
        self.assertFalse(report["final_research_eligible"])

    def test_clean_provenance_requires_matching_model_hash_and_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, yaml, args, source, audit = self._files(tmp, leakage=0)
            import hashlib

            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            report = inspect_model_provenance(
                model_path=model,
                dataset_yaml=yaml,
                training_args=args,
                training_source=source,
                dataset_audit=audit,
                clean_provenance={
                    "model_sha256": digest,
                    "grouped_split_frozen": True,
                    "test_data_excluded": True,
                    "threshold_tuned_on_validation_only": True,
                },
            )
        self.assertEqual(report["training_split_provenance"], "clean")
        self.assertTrue(report["final_research_eligible"])


if __name__ == "__main__":
    unittest.main()
