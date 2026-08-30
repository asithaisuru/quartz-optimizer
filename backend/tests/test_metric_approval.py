import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research.metric_approval import validate_metric_approval  # noqa: E402


class MetricApprovalTests(unittest.TestCase):
    def _files(self, root, metric="binary_visible_defect_f1", floor=None):
        root = Path(root)
        taxonomy = root / "taxonomy.csv"
        taxonomy.write_text("class_id,class_name\n0,Defect\n", encoding="utf-8")
        protocol = root / "protocol.md"
        protocol.write_text("# Test protocol\n", encoding="utf-8")
        approval = root / "approval.json"
        approval.write_text(
            json.dumps(
                {
                    "metric_id": "metric-1",
                    "metric_version": "1",
                    "primary_metric": metric,
                    "claim_threshold": 0.90,
                    "precision_floor": floor,
                    "aggregation_level": "specimen_grouped",
                    "approved_by": "Supervisor",
                    "approver_role": "Research supervisor",
                    "approval_date": "2026-07-29",
                    "taxonomy_sha256": hashlib.sha256(
                        taxonomy.read_bytes()
                    ).hexdigest(),
                    "test_protocol_sha256": hashlib.sha256(
                        protocol.read_bytes()
                    ).hexdigest(),
                    "notes": "",
                    "example_only": False,
                }
            ),
            encoding="utf-8",
        )
        return approval, taxonomy, protocol

    def test_missing_approval_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, taxonomy, protocol = self._files(tmp)
            report = validate_metric_approval(None, taxonomy, protocol)
        self.assertFalse(report["summary"]["metric_approved"])

    def test_recall_requires_precision_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            approval, taxonomy, protocol = self._files(
                tmp, "binary_visible_defect_recall", None
            )
            report = validate_metric_approval(approval, taxonomy, protocol)
        self.assertFalse(report["summary"]["metric_approved"])
        self.assertTrue(
            any("precision floor" in item for item in report["summary"]["blockers"])
        )

    def test_f1_with_formal_090_approval_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            approval, taxonomy, protocol = self._files(tmp)
            report = validate_metric_approval(approval, taxonomy, protocol)
        self.assertTrue(report["summary"]["metric_approved"])
        self.assertEqual(report["approval"]["claim_threshold"], 0.90)

    def test_example_record_is_not_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            approval, taxonomy, protocol = self._files(tmp)
            value = json.loads(approval.read_text(encoding="utf-8"))
            value["example_only"] = True
            approval.write_text(json.dumps(value), encoding="utf-8")
            report = validate_metric_approval(approval, taxonomy, protocol)
        self.assertFalse(report["summary"]["metric_approved"])

    def test_hashes_must_match_taxonomy_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            approval, taxonomy, protocol = self._files(tmp)
            taxonomy.write_text("changed", encoding="utf-8")
            protocol.write_text("changed", encoding="utf-8")
            report = validate_metric_approval(approval, taxonomy, protocol)
        blockers = " ".join(report["summary"]["blockers"])
        self.assertIn("taxonomy hash", blockers)
        self.assertIn("test-protocol hash", blockers)


if __name__ == "__main__":
    unittest.main()
