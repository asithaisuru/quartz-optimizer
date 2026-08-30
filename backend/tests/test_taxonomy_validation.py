import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research.taxonomy_validation import validate_taxonomy_approval  # noqa: E402


class TaxonomyValidationTests(unittest.TestCase):
    FIELDS = [
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

    def _files(self, root, rows, approval=True):
        root = Path(root)
        taxonomy = root / "taxonomy.csv"
        with taxonomy.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        yaml = root / "data.yaml"
        yaml.write_text("names: ['Defect']\n", encoding="utf-8")
        approval_path = root / "approval.json"
        if approval:
            approval_path.write_text(
                json.dumps(
                    {
                        "taxonomy_id": "quartz-visible-defects",
                        "taxonomy_version": "1.0",
                        "taxonomy_file": str(taxonomy),
                        "taxonomy_sha256": hashlib.sha256(
                            taxonomy.read_bytes()
                        ).hexdigest(),
                        "approved_by": "Expert A",
                        "approver_role": "Gem-domain expert",
                        "approval_date": "2026-07-29",
                        "approval_scope": "visible defects",
                        "primary_visible_defect_definition": "Visible defect",
                        "primary_claim_metric": "binary_visible_defect_f1",
                        "claim_threshold": 0.9,
                        "precision_floor": None,
                        "notes": "",
                        "example_only": False,
                    }
                ),
                encoding="utf-8",
            )
        return taxonomy, approval_path, yaml

    def _row(self, **overrides):
        value = {
            "class_id": 0,
            "class_name": "Defect",
            "semantic_definition": "Expert-confirmed visible defect",
            "taxonomy_status": "confirmed_defect",
            "is_visible_defect": "true",
            "eligible_for_3d_mapping": "true",
            "eligible_for_no_cut_zone": "true",
            "minimum_confidence": "0.25",
            "expert_confirmed": "true",
            "confirmed_by": "Expert A",
            "confirmed_date": "2026-07-29",
            "notes": "",
        }
        value.update(overrides)
        return value

    def test_valid_approval_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            taxonomy, approval, yaml = self._files(tmp, [self._row()])
            report = validate_taxonomy_approval(taxonomy, approval, yaml)
        self.assertTrue(report["summary"]["taxonomy_approved"])

    def test_unknown_taxonomy_and_missing_approval_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            taxonomy, approval, yaml = self._files(
                tmp,
                [self._row(taxonomy_status="unknown", semantic_definition="")],
                approval=False,
            )
            report = validate_taxonomy_approval(taxonomy, None, yaml)
        self.assertFalse(report["summary"]["taxonomy_approved"])
        self.assertFalse(report["approval"]["supplied"])

    def test_hash_mismatch_and_example_record_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            taxonomy, approval, yaml = self._files(tmp, [self._row()])
            value = json.loads(approval.read_text())
            value["taxonomy_sha256"] = "0" * 64
            value["example_only"] = True
            approval.write_text(json.dumps(value))
            report = validate_taxonomy_approval(taxonomy, approval, yaml)
        joined = " ".join(report["summary"]["blockers"])
        self.assertIn("hash", joined.casefold())
        self.assertIn("example", joined.casefold())

    def test_unknown_model_class_and_duplicate_id_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [self._row(), self._row(class_name="Other")]
            taxonomy, approval, yaml = self._files(tmp, rows)
            report = validate_taxonomy_approval(taxonomy, approval, yaml)
        joined = " ".join(report["summary"]["blockers"])
        self.assertIn("duplicate", joined.casefold())

    def test_no_cut_without_expert_confirmation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            taxonomy, approval, yaml = self._files(
                tmp,
                [self._row(expert_confirmed="false")],
            )
            report = validate_taxonomy_approval(taxonomy, approval, yaml)
        reasons = report["class_validation"][0]["blocking_reasons"]
        self.assertIn("no_cut_without_confirmed_expert_defect", reasons)

    def test_primary_metric_missing_remains_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            taxonomy, approval, yaml = self._files(tmp, [self._row()])
            value = json.loads(approval.read_text())
            value["primary_claim_metric"] = None
            approval.write_text(json.dumps(value))
            report = validate_taxonomy_approval(taxonomy, approval, yaml)
        self.assertEqual(report["claim_status"], "unavailable")
        self.assertFalse(report["summary"]["taxonomy_approved"])


if __name__ == "__main__":
    unittest.main()
