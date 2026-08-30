import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research.candidate_dataset import (  # noqa: E402
    GEM_IDENTITY_RULE,
    validate_gem_identity_approval,
)
from research.gate2 import (  # noqa: E402
    APPROVAL_DATE,
    ROLE_IDENTIFIERS,
    _privacy_issues,
    _validate_external_concept,
    _validate_initialization,
    _validate_route,
)
from research.metric_approval import validate_metric_approval  # noqa: E402
from research.taxonomy_validation import validate_taxonomy_approval  # noqa: E402


class Gate2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.approvals = cls.root / "research" / "approvals"
        cls.protocols = cls.root / "research" / "protocols"

    def test_real_taxonomy_approval_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "data.yaml"
            yaml_path.write_text(
                "names: ['fracture', 'inclusion']\n",
                encoding="utf-8",
            )
            report = validate_taxonomy_approval(
                self.approvals / "gem_fracture_taxonomy.approved.csv",
                self.approvals
                / "gem_fracture_taxonomy_approval.approved.json",
                yaml_path,
            )
        self.assertTrue(report["summary"]["taxonomy_approved"])

    def test_real_gem_identity_approval_passes(self):
        report = validate_gem_identity_approval(
            self.approvals / "gem_identity_rule.approved.json",
            repo_root=self.root,
            source_manifest_sha256=(
                "4cc974801fac659c8d51d0ba2a58687093085bfdb4b13b2bed25f7e2d5c9c767"
            ),
        )
        self.assertEqual(report["rule"], GEM_IDENTITY_RULE)
        self.assertTrue(report["approved"])

    def test_route_approval_passes(self):
        report = _validate_route(
            self.approvals / "dataset_route_decision.approved.json"
        )
        self.assertTrue(report["approved"])

    def test_metric_approval_passes(self):
        report = validate_metric_approval(
            self.approvals / "defect_metric_approval.approved.json",
            self.approvals / "gem_fracture_taxonomy.approved.csv",
            self.protocols / "defect_model_evaluation.md",
        )
        self.assertTrue(report["summary"]["metric_approved"])
        self.assertEqual(
            report["approval"]["primary_metric"],
            "binary_visible_defect_f1",
        )

    def test_initialization_approval_rejects_legacy_model(self):
        report = _validate_initialization(
            self.approvals / "defect_training_approval.approved.json",
            self.protocols / "clean_defect_model_training.md",
        )
        self.assertTrue(report["approved"])
        self.assertFalse(report["legacy_model_initialization_allowed"])

    def test_external_operational_details_remain_unavailable(self):
        report = _validate_external_concept(
            self.approvals / "external_test_concept_approval.approved.json"
        )
        self.assertTrue(report["concept_approved"])
        self.assertFalse(report["operational_ready"])
        self.assertEqual(len(report["unavailable_fields"]), 5)

    def test_approval_records_use_role_only_identifiers(self):
        for path in self.approvals.glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(_privacy_issues(value), [], path.name)
            for key in (
                "approved_by",
                "approver_role",
                "record_prepared_by",
                "required_reviewer_role",
            ):
                if value.get(key):
                    self.assertIn(value[key], ROLE_IDENTIFIERS)
            if value.get("approval_date"):
                self.assertEqual(value["approval_date"], APPROVAL_DATE)

    def test_training_configuration_remains_disabled(self):
        config = json.loads(
            (
                self.root
                / "research"
                / "configs"
                / "gem_fracture_training.disabled.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(config["training_enabled"])
        self.assertEqual(
            config["authorization_status"],
            "blocked_external_test_operational_details",
        )
        self.assertFalse(config["weights_downloaded"])
        self.assertFalse(config["training_occurred"])
        self.assertIsNone(config["model_size"])
        self.assertIsNone(config["epochs"])
        self.assertIsNone(config["confidence_threshold"])


if __name__ == "__main__":
    unittest.main()
