import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import colmap_runner  # noqa: E402


COLMAP_313_FEATURE_HELP = """
  --FeatureExtraction.use_gpu arg (=1)
  --FeatureExtraction.gpu_index arg (=-1)
  --SiftExtraction.max_image_size arg (=3200)
"""

COLMAP_313_MATCHER_HELP = """
  --FeatureMatching.use_gpu arg (=1)
  --FeatureMatching.gpu_index arg (=-1)
  --SiftMatching.cpu_brute_force_matcher arg (=0)
"""


def fake_help(command):
    if command == "feature_extractor":
        return COLMAP_313_FEATURE_HELP
    if command == "exhaustive_matcher":
        return COLMAP_313_MATCHER_HELP
    return ""


class ColmapRunnerGpuFlagTests(unittest.TestCase):
    def test_cpu_mode_emits_colmap_313_no_gpu_flags_for_sift_stages(self):
        with (
            patch.object(colmap_runner, "COLMAP_BIN", "colmap"),
            patch.object(colmap_runner, "_HAS_CUDA", False),
            patch.object(colmap_runner, "_colmap_command_help", side_effect=fake_help),
        ):
            feature_cmd = colmap_runner._feature_extractor_command("db.db", "images")
            matcher_cmd = colmap_runner._exhaustive_matcher_command("db.db")

        feature_flag = feature_cmd.index("--FeatureExtraction.use_gpu")
        matcher_flag = matcher_cmd.index("--FeatureMatching.use_gpu")
        self.assertEqual(feature_cmd[feature_flag + 1], "0")
        self.assertEqual(matcher_cmd[matcher_flag + 1], "0")

    def test_gpu_mode_preserves_colmap_gpu_flags_for_sift_stages(self):
        with (
            patch.object(colmap_runner, "COLMAP_BIN", "colmap"),
            patch.object(colmap_runner, "_HAS_CUDA", True),
            patch.object(colmap_runner, "_colmap_command_help", side_effect=fake_help),
        ):
            feature_cmd = colmap_runner._feature_extractor_command("db.db", "images")
            matcher_cmd = colmap_runner._exhaustive_matcher_command("db.db")

        feature_flag = feature_cmd.index("--FeatureExtraction.use_gpu")
        matcher_flag = matcher_cmd.index("--FeatureMatching.use_gpu")
        self.assertEqual(feature_cmd[feature_flag + 1], "1")
        self.assertEqual(matcher_cmd[matcher_flag + 1], "1")

    def test_cuda_setup_errors_are_classified_separately(self):
        self.assertTrue(
            colmap_runner._looks_like_cuda_setup_failure(
                "Check failed: cudaSetDevice failed: no CUDA-capable device"
            )
        )
        self.assertFalse(
            colmap_runner._looks_like_cuda_setup_failure(
                "No good initial image pair found"
            )
        )


if __name__ == "__main__":
    unittest.main()
