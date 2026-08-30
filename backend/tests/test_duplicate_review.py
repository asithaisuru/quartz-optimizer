import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research.duplicate_review import (  # noqa: E402
    build_review_rows,
    generate_duplicate_review,
    grouping_pairs,
)


class DuplicateReviewTests(unittest.TestCase):
    def _images(self, root):
        root = Path(root)
        first = root / "a.jpg"
        second = root / "b.jpg"
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(first)
        Image.fromarray(np.ones((16, 16, 3), dtype=np.uint8) * 10).save(second)
        return first, second

    def test_candidate_is_exported_and_originals_remain_unchanged(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            first, second = self._images(root)
            before = (first.read_bytes(), second.read_bytes())
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "near_duplicate_candidates": [
                            {
                                "left": first.relative_to(Path.cwd()).as_posix(),
                                "right": second.relative_to(Path.cwd()).as_posix(),
                                "hamming_distance": 2,
                            }
                        ]
                    }
                )
            )
            report, files = generate_duplicate_review(
                dataset_audit_path=audit,
                output_directory=root / "output",
            )
            after = (first.read_bytes(), second.read_bytes())
            csv_text = files["review"].read_text()
        self.assertEqual(report["candidate_count"], 1)
        self.assertIn("unreviewed", csv_text)
        self.assertEqual(before, after)

    def test_grouping_decisions_join_only_confirmed_relations(self):
        rows = [
            {"image_a": "a", "image_b": "b", "review_decision": "same_source"},
            {"image_a": "c", "image_b": "d", "review_decision": "not_duplicate"},
        ]
        self.assertEqual(grouping_pairs(rows), [("a", "b")])

    def test_unreviewed_is_a_strict_blocker(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            first, second = self._images(root)
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "near_duplicate_candidates": [
                            {
                                "left": first.relative_to(Path.cwd()).as_posix(),
                                "right": second.relative_to(Path.cwd()).as_posix(),
                                "hamming_distance": 1,
                            }
                        ]
                    }
                )
            )
            report, _ = generate_duplicate_review(
                dataset_audit_path=audit,
                output_directory=root / "out",
                strict=True,
            )
        self.assertTrue(report["strict_blocked"])
        self.assertEqual(report["unresolved_count"], 1)


if __name__ == "__main__":
    unittest.main()
