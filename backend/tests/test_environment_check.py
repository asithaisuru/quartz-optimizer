import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research.environment_check import audit_environment  # noqa: E402


class EnvironmentCheckTests(unittest.TestCase):
    def _config(self):
        return {
            "minimum_python_version": "3.10",
            "colmap_executable": "colmap",
            "dependencies": ["numpy", "trimesh"],
        }

    def test_available_and_missing_dependencies_are_explicit(self):
        available = {"numpy"}
        report = audit_environment(
            self._config(),
            spec_finder=lambda name: object() if name in available else None,
            version_reader=lambda names: "1.2.3",
            executable_finder=lambda name: None,
        )
        rows = {row["key"]: row for row in report["dependencies"]}
        self.assertEqual(rows["numpy"]["status"], "available")
        self.assertEqual(rows["trimesh"]["status"], "missing")
        self.assertEqual(rows["colmap"]["status"], "missing")

    def test_environment_variables_are_not_dumped(self):
        os.environ["QUARTZ_TEST_SECRET"] = "must-not-appear"
        report = audit_environment(
            self._config(),
            spec_finder=lambda name: None,
            executable_finder=lambda name: None,
        )
        serialized = json.dumps(report)
        self.assertNotIn("must-not-appear", serialized)
        self.assertNotIn("QUARTZ_TEST_SECRET", serialized)
        self.assertFalse(report["environment_variables_recorded"])

    def test_dependency_output_is_deterministic_for_same_probe(self):
        kwargs = {
            "spec_finder": lambda name: None,
            "executable_finder": lambda name: None,
        }
        first = audit_environment(self._config(), **kwargs)
        second = audit_environment(self._config(), **kwargs)
        first.pop("started_utc")
        first.pop("ended_utc")
        second.pop("started_utc")
        second.pop("ended_utc")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
