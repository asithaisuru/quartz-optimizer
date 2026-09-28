import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import colmap_runner  # noqa: E402


COLMAP_411_FEATURE_HELP = """
  --image_list_path arg
  --FeatureExtraction.num_threads arg (=-1)
  --FeatureExtraction.use_gpu arg (=1)
  --FeatureExtraction.gpu_index arg (=-1)
  --FeatureExtraction.max_image_size arg (=-1)
  --SiftExtraction.max_num_features arg (=8192)
  --SiftExtraction.peak_threshold arg (=0.00667)
"""

COLMAP_LEGACY_FEATURE_HELP = """
  --SiftExtraction.num_threads arg (=-1)
  --FeatureExtraction.use_gpu arg (=1)
  --FeatureExtraction.gpu_index arg (=-1)
  --SiftExtraction.max_image_size arg (=3200)
"""

COLMAP_313_MATCHER_HELP = """
  --FeatureMatching.num_threads arg (=-1)
  --FeatureMatching.use_gpu arg (=1)
  --FeatureMatching.gpu_index arg (=-1)
  --SiftMatching.cpu_brute_force_matcher arg (=0)
"""

COLMAP_411_PATCH_MATCH_HELP = """
  --PatchMatchStereo.gpu_index arg (=-1)
  --PatchMatchStereo.geom_consistency arg (=1)
"""


def fake_help(command):
    if command == "feature_extractor":
        return COLMAP_411_FEATURE_HELP
    if command == "exhaustive_matcher":
        return COLMAP_313_MATCHER_HELP
    if command == "patch_match_stereo":
        return COLMAP_411_PATCH_MATCH_HELP
    return ""


class ColmapRunnerGpuFlagTests(unittest.TestCase):
    def setUp(self):
        # Exercise the normal default even if the invoking shell opts out.
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop(colmap_runner.REPRODUCIBLE_MODE_ENV, None)

    def test_reproducible_mode_defaults_on_and_rejects_invalid_values(self):
        self.assertTrue(colmap_runner._reproducible_mode({}))
        for value in ("true", "1", "yes", "on", " TRUE "):
            self.assertTrue(colmap_runner._reproducible_mode(
                {colmap_runner.REPRODUCIBLE_MODE_ENV: value}))
        for value in ("false", "0", "no", "off"):
            self.assertFalse(colmap_runner._reproducible_mode(
                {colmap_runner.REPRODUCIBLE_MODE_ENV: value}))
        with self.assertRaises(ValueError):
            colmap_runner._reproducible_mode(
                {colmap_runner.REPRODUCIBLE_MODE_ENV: "maybe"})

    def test_explicit_modes_control_all_sparse_commands_and_metadata(self):
        for mode, threads in (("true", "1"), ("false", "8")):
            with (
                self.subTest(mode=mode),
                patch.dict(os.environ, {colmap_runner.REPRODUCIBLE_MODE_ENV: mode}),
                patch.object(colmap_runner, "_colmap_command_help", side_effect=fake_help),
            ):
                feature = colmap_runner._feature_extractor_command("db", "images", "order")
                match = colmap_runner._exhaustive_matcher_command("db")
                mapper = colmap_runner._mapper_command("db", "images", "sparse", "order")
                hierarchy = colmap_runner._hierarchical_mapper_command(
                    "db", "images", "sparse", "order")
                for cmd, flag in (
                    (feature, "--FeatureExtraction.num_threads"),
                    (match, "--FeatureMatching.num_threads"),
                    (mapper, "--Mapper.num_threads"),
                    (hierarchy, "--Mapper.num_threads"),
                    (hierarchy, "--num_threads"),
                ):
                    self.assertEqual(cmd[cmd.index(flag) + 1], threads)
                    self.assertEqual(cmd[cmd.index("--default_random_seed") + 1], "0")
                self.assertEqual(match[match.index("--TwoViewGeometry.random_seed") + 1], "0")
                self.assertEqual(mapper[mapper.index("--Mapper.random_seed") + 1], "0")
                self.assertEqual(hierarchy[hierarchy.index("--num_workers") + 1], "1")
                settings = colmap_runner._reconstruction_settings()
                for stage in ("feature_extraction", "feature_matching", "mapper"):
                    self.assertEqual(settings[stage]["num_threads"], int(threads))
                self.assertTrue(settings["patch_match_stereo"]["use_gpu"])
                self.assertFalse(settings["patch_match_stereo"]["full_cuda_determinism"])
                # Check persisted provenance for both default and intentional opt-out.
                with tempfile.TemporaryDirectory() as tmp, patch.object(
                    colmap_runner, "_gpu_runtime_metadata",
                    return_value={"gpu_name": None, "cuda_version": None, "driver_version": None},
                ):
                    path = colmap_runner._write_reconstruction_environment(
                        tmp, "missing-test-executable", "test", ["frame_0000.jpg"])
                    with open(path, encoding="utf-8") as handle:
                        metadata = colmap_runner.json.load(handle)
                sparse = metadata["sparse_execution"]
                self.assertEqual(sparse["reproducible_mode"], mode == "true")
                for field in ("feature_threads", "matching_threads", "mapper_threads"):
                    self.assertEqual(sparse[field], int(threads))
                self.assertFalse(metadata["reproducibility"]["full_pipeline_bitwise_deterministic"])

    def test_colmap_env_removes_disable_all_cuda_mask_without_mutating_parent(self):
        with (
            patch.dict(os.environ, {colmap_runner.CUDA_VISIBLE_DEVICES_ENV: "-1"}),
            patch.object(colmap_runner, "_COLMAP_CUDA_ENV_SANITIZED_LOGGED", False),
            patch("builtins.print") as mocked_print,
        ):
            child = colmap_runner._colmap_subprocess_env()

            self.assertEqual(os.environ[colmap_runner.CUDA_VISIBLE_DEVICES_ENV], "-1")
            self.assertNotIn(colmap_runner.CUDA_VISIBLE_DEVICES_ENV, child)
            mocked_print.assert_called_once()

    def test_colmap_env_preserves_valid_cuda_mask(self):
        parent = {
            colmap_runner.CUDA_VISIBLE_DEVICES_ENV: "0",
            "PATH": r"C:\Windows",
        }
        with patch("builtins.print") as mocked_print:
            child = colmap_runner._colmap_subprocess_env(parent)

        self.assertEqual(parent[colmap_runner.CUDA_VISIBLE_DEVICES_ENV], "0")
        self.assertEqual(child[colmap_runner.CUDA_VISIBLE_DEVICES_ENV], "0")
        mocked_print.assert_not_called()

    def test_colmap_env_preserves_multi_gpu_cuda_mask(self):
        parent = {
            colmap_runner.CUDA_VISIBLE_DEVICES_ENV: "0,1",
            "PATH": r"C:\Windows",
        }
        child = colmap_runner._colmap_subprocess_env(parent)

        self.assertEqual(child[colmap_runner.CUDA_VISIBLE_DEVICES_ENV], "0,1")

    def test_colmap_env_leaves_unset_cuda_mask_unset(self):
        parent = {"PATH": r"C:\Windows"}
        child = colmap_runner._colmap_subprocess_env(parent)

        self.assertNotIn(colmap_runner.CUDA_VISIBLE_DEVICES_ENV, parent)
        self.assertNotIn(colmap_runner.CUDA_VISIBLE_DEVICES_ENV, child)

    def test_colmap_env_sanitization_logs_once(self):
        parent = {colmap_runner.CUDA_VISIBLE_DEVICES_ENV: "-1"}
        with (
            patch.object(colmap_runner, "_COLMAP_CUDA_ENV_SANITIZED_LOGGED", False),
            patch("builtins.print") as mocked_print,
        ):
            colmap_runner._colmap_subprocess_env(parent)
            colmap_runner._colmap_subprocess_env(parent)

        mocked_print.assert_called_once()

    def test_run_command_uses_sanitized_colmap_child_env(self):
        completed = colmap_runner.subprocess.CompletedProcess(
            args=["colmap", "-h"],
            returncode=0,
            stdout="",
            stderr="",
        )
        with (
            patch.dict(os.environ, {colmap_runner.CUDA_VISIBLE_DEVICES_ENV: "-1"}),
            patch.object(colmap_runner, "_COLMAP_CUDA_ENV_SANITIZED_LOGGED", True),
            patch.object(colmap_runner.subprocess, "run", return_value=completed) as run,
            patch("builtins.print"),
        ):
            self.assertTrue(colmap_runner.run_command(["colmap", "-h"]))
            self.assertEqual(os.environ[colmap_runner.CUDA_VISIBLE_DEVICES_ENV], "-1")

        child_env = run.call_args.kwargs["env"]
        self.assertNotIn(colmap_runner.CUDA_VISIBLE_DEVICES_ENV, child_env)

    def test_default_local_colmap_411_path_is_preferred_before_path(self):
        old_path = r"D:\colmap\bin\colmap.exe"
        resolved = colmap_runner._resolve_colmap_bin(
            env={},
            which=lambda name: old_path if name == "colmap" else None,
            isfile=lambda path: path == colmap_runner.DEFAULT_COLMAP_BIN,
        )
        self.assertEqual(resolved, colmap_runner.DEFAULT_COLMAP_BIN)

    def test_configured_colmap_path_overrides_default(self):
        configured = r"E:\tools\colmap\colmap.exe"
        resolved = colmap_runner._resolve_colmap_bin(
            env={colmap_runner.COLMAP_BIN_ENV: configured},
            which=lambda name: None,
            isfile=lambda path: path == configured,
        )
        self.assertEqual(resolved, configured)

    def test_missing_configured_colmap_path_does_not_fall_back_silently(self):
        configured = r"Z:\missing\colmap.exe"
        resolved = colmap_runner._resolve_colmap_bin(
            env={colmap_runner.COLMAP_BIN_ENV: configured},
            which=lambda name: r"D:\colmap-new\bin\colmap.exe",
            isfile=lambda path: path != configured,
        )
        self.assertEqual(resolved, configured)

    def test_missing_colmap_binary_fails_clearly(self):
        with (
            patch.object(colmap_runner, "COLMAP_BIN", r"Z:\missing\colmap.exe"),
            patch.object(colmap_runner.os.path, "isfile", return_value=False),
            patch.object(colmap_runner.shutil, "which", return_value=None),
        ):
            with self.assertRaises(FileNotFoundError) as err:
                colmap_runner._ensure_colmap_available()

        self.assertIn("COLMAP executable not found", str(err.exception))
        self.assertIn(colmap_runner.COLMAP_BIN_ENV, str(err.exception))

    def test_cpu_mode_emits_colmap_411_no_gpu_flags_for_sift_stages(self):
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
        feature_threads = feature_cmd.index("--FeatureExtraction.num_threads")
        matcher_threads = matcher_cmd.index("--FeatureMatching.num_threads")
        self.assertEqual(feature_cmd[feature_threads + 1], "1")
        self.assertEqual(matcher_cmd[matcher_threads + 1], "1")
        feature_seed = feature_cmd.index("--default_random_seed")
        matcher_seed = matcher_cmd.index("--default_random_seed")
        self.assertEqual(feature_cmd[feature_seed + 1], "0")
        self.assertEqual(matcher_cmd[matcher_seed + 1], "0")
        two_view_seed = matcher_cmd.index("--TwoViewGeometry.random_seed")
        self.assertEqual(matcher_cmd[two_view_seed + 1], "0")
        image_size_flag = feature_cmd.index("--FeatureExtraction.max_image_size")
        self.assertEqual(feature_cmd[image_size_flag + 1], "1200")
        self.assertNotIn("--SiftExtraction.max_image_size", feature_cmd)
        self.assertIn("--SiftExtraction.max_num_features", feature_cmd)
        self.assertIn("--SiftExtraction.peak_threshold", feature_cmd)

    def test_gpu_capability_does_not_reenable_sift_gpu_processing(self):
        with (
            patch.object(colmap_runner, "COLMAP_BIN", "colmap"),
            patch.object(colmap_runner, "_HAS_CUDA", True),
            patch.object(colmap_runner, "_colmap_command_help", side_effect=fake_help),
        ):
            feature_cmd = colmap_runner._feature_extractor_command("db.db", "images")
            matcher_cmd = colmap_runner._exhaustive_matcher_command("db.db")

        feature_flag = feature_cmd.index("--FeatureExtraction.use_gpu")
        matcher_flag = matcher_cmd.index("--FeatureMatching.use_gpu")
        self.assertEqual(feature_cmd[feature_flag + 1], "0")
        self.assertEqual(matcher_cmd[matcher_flag + 1], "0")

    def test_feature_image_size_falls_back_to_legacy_sift_option(self):
        def legacy_help(command):
            if command == "feature_extractor":
                return COLMAP_LEGACY_FEATURE_HELP
            return fake_help(command)

        with (
            patch.object(colmap_runner, "COLMAP_BIN", "colmap"),
            patch.object(colmap_runner, "_HAS_CUDA", False),
            patch.object(colmap_runner, "_colmap_command_help", side_effect=legacy_help),
        ):
            feature_cmd = colmap_runner._feature_extractor_command("db.db", "images")

        image_size_flag = feature_cmd.index("--SiftExtraction.max_image_size")
        self.assertEqual(feature_cmd[image_size_flag + 1], "1200")
        self.assertNotIn("--FeatureExtraction.max_image_size", feature_cmd)

    def test_feature_image_size_option_must_be_supported(self):
        help_without_image_size = """
          --FeatureExtraction.use_gpu arg (=1)
          --FeatureExtraction.num_threads arg (=-1)
        """
        with (
            patch.object(colmap_runner, "COLMAP_BIN", "colmap"),
            patch.object(colmap_runner, "_HAS_CUDA", False),
            patch.object(
                colmap_runner,
                "_colmap_command_help",
                return_value=help_without_image_size,
            ),
        ):
            with self.assertRaises(RuntimeError) as err:
                colmap_runner._feature_extractor_command("db.db", "images")

        self.assertIn("feature_extractor", str(err.exception))
        self.assertIn("max_image_size", str(err.exception))

    def test_mappers_have_fixed_seed_reproducible_threads_and_image_order(self):
        cmd = colmap_runner._mapper_command(
            "db.db",
            "images",
            "sparse",
            "image-order.txt",
        )
        hierarchical_cmd = colmap_runner._hierarchical_mapper_command(
            "db.db",
            "images",
            "sparse",
            "image-order.txt",
        )

        self.assertEqual(cmd[cmd.index("--Mapper.random_seed") + 1], "0")
        self.assertEqual(cmd[cmd.index("--Mapper.num_threads") + 1], "1")
        self.assertEqual(cmd[cmd.index("--default_random_seed") + 1], "0")
        self.assertEqual(
            cmd[cmd.index("--Mapper.image_list_path") + 1],
            "image-order.txt",
        )
        self.assertEqual(
            hierarchical_cmd[
                hierarchical_cmd.index("--Mapper.random_seed") + 1
            ],
            "0",
        )
        self.assertEqual(
            hierarchical_cmd[
                hierarchical_cmd.index("--Mapper.num_threads") + 1
            ],
            "1",
        )
        self.assertEqual(
            hierarchical_cmd[hierarchical_cmd.index("--num_threads") + 1],
            "1",
        )
        self.assertEqual(
            hierarchical_cmd[hierarchical_cmd.index("--num_workers") + 1],
            "1",
        )

    def test_feature_extraction_receives_sorted_image_list(self):
        with (
            patch.object(colmap_runner, "COLMAP_BIN", "colmap"),
            patch.object(colmap_runner, "_colmap_command_help", side_effect=fake_help),
        ):
            cmd = colmap_runner._feature_extractor_command(
                "db.db",
                "images",
                "image-order.txt",
            )

        self.assertEqual(
            cmd[cmd.index("--image_list_path") + 1],
            "image-order.txt",
        )

    def test_frame_and_dense_configuration_order_is_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            images_dir = os.path.join(tmp, "images")
            stereo_dir = os.path.join(tmp, "dense", "stereo")
            os.makedirs(images_dir)
            os.makedirs(stereo_dir)
            for name in ("frame_0002.jpg", "frame_0000.jpg", "frame_0001.jpg"):
                with open(os.path.join(images_dir, name), "wb") as handle:
                    handle.write(b"image")

            image_names = colmap_runner._sorted_image_filenames(images_dir)
            image_list_path = colmap_runner._write_colmap_image_order(
                tmp, image_names
            )
            patch_match_path = os.path.join(stereo_dir, "patch-match.cfg")
            fusion_path = os.path.join(stereo_dir, "fusion.cfg")
            with open(patch_match_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "frame_0002.jpg\n__auto__, 20\n"
                    "frame_0000.jpg\n__auto__, 20\n"
                    "frame_0001.jpg\n__auto__, 20\n"
                )
            with open(fusion_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "frame_0002.jpg\nframe_0000.jpg\nframe_0001.jpg\n"
                )

            result = colmap_runner._canonicalize_dense_config_order(
                os.path.join(tmp, "dense")
            )

            with open(image_list_path, encoding="utf-8") as handle:
                self.assertEqual(handle.read().splitlines(), image_names)
            with open(patch_match_path, encoding="utf-8") as handle:
                patch_lines = handle.read().splitlines()
            with open(fusion_path, encoding="utf-8") as handle:
                fusion_lines = handle.read().splitlines()

        self.assertEqual(
            image_names,
            ["frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg"],
        )
        self.assertEqual(
            patch_lines[::2],
            ["frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg"],
        )
        self.assertEqual(fusion_lines, image_names)
        self.assertEqual(result, {"patch_match": True, "fusion": True})

    def test_reconstruction_environment_records_runtime_and_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = os.path.join(tmp, "colmap.exe")
            with open(executable, "wb") as handle:
                handle.write(b"colmap-binary")
            with (
                patch.object(
                    colmap_runner,
                    "_gpu_runtime_metadata",
                    return_value={
                        "gpu_name": "Test GPU",
                        "cuda_version": "12.8",
                        "driver_version": "600.1",
                    },
                ),
                patch.object(
                    colmap_runner,
                    "_installed_package_version",
                    return_value="0.19.0",
                ),
            ):
                output_path = colmap_runner._write_reconstruction_environment(
                    tmp,
                    executable,
                    "COLMAP test",
                    ["frame_0000.jpg"],
                )
            with open(output_path, encoding="utf-8") as handle:
                metadata = colmap_runner.json.load(handle)

        self.assertEqual(metadata["colmap_version"], "COLMAP test")
        self.assertEqual(metadata["open3d_version"], "0.19.0")
        self.assertEqual(metadata["gpu_name"], "Test GPU")
        self.assertEqual(metadata["cuda_version"], "12.8")
        self.assertEqual(metadata["image_order"], ["frame_0000.jpg"])
        self.assertEqual(
            metadata["reconstruction_settings"]["mapper"]["random_seed"],
            0,
        )
        self.assertEqual(
            metadata["reconstruction_settings"]["feature_extraction"][
                "num_threads"
            ],
            1,
        )
        self.assertEqual(
            metadata["sparse_execution"],
            {
                "reproducible_mode": True,
                "feature_threads": 1,
                "matching_threads": 1,
                "mapper_threads": 1,
                "gpu_enabled": False,
                "random_seed": 0,
            },
        )
        self.assertEqual(len(metadata["colmap_executable_sha256"]), 64)

    def test_patch_match_uses_gpu_index_zero_when_supported(self):
        with (
            patch.object(colmap_runner, "COLMAP_BIN", "colmap"),
            patch.object(colmap_runner, "_colmap_command_help", side_effect=fake_help),
        ):
            cmd = colmap_runner._patch_match_stereo_command("dense")

        flag = cmd.index("--PatchMatchStereo.gpu_index")
        self.assertEqual(cmd[flag + 1], "0")
        self.assertIn("--PatchMatchStereo.geom_consistency", cmd)
        self.assertIn("--PatchMatchStereo.window_radius", cmd)
        self.assertIn("--PatchMatchStereo.num_iterations", cmd)

    def test_patch_match_retries_once_for_cuda_setup_failure(self):
        cmd = ["colmap", "patch_match_stereo"]
        with tempfile.TemporaryDirectory() as tmp:
            stereo_dir = os.path.join(tmp, "stereo")
            os.makedirs(stereo_dir)
            config_path = os.path.join(stereo_dir, "patch-match.cfg")
            with open(config_path, "w") as handle:
                handle.write("frame.jpg\n__auto__, 20\n")
            for output_dir in ("depth_maps", "normal_maps", "consistency_graphs"):
                path = os.path.join(stereo_dir, output_dir)
                os.makedirs(path)
                with open(os.path.join(path, "partial.bin"), "w") as handle:
                    handle.write("partial")

            with (
                patch.object(colmap_runner, "_patch_match_stereo_command", return_value=cmd),
                patch.object(
                    colmap_runner,
                    "run_command",
                    side_effect=[
                        (False, "CUDA error: no CUDA-capable device is detected"),
                        (True, ""),
                    ],
                ) as run_command,
                patch.object(colmap_runner.time, "sleep") as sleep,
            ):
                self.assertTrue(colmap_runner._run_patch_match_stereo(tmp))

            self.assertEqual(run_command.call_count, 2)
            sleep.assert_called_once()
            self.assertTrue(os.path.exists(config_path))
            for output_dir in ("depth_maps", "normal_maps", "consistency_graphs"):
                path = os.path.join(stereo_dir, output_dir)
                self.assertTrue(os.path.isdir(path))
                self.assertEqual(os.listdir(path), [])

    def test_patch_match_persistent_cuda_setup_failure_is_clear(self):
        cmd = ["colmap", "patch_match_stereo"]
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "stereo", "depth_maps"))
            with (
                patch.object(colmap_runner, "_patch_match_stereo_command", return_value=cmd),
                patch.object(
                    colmap_runner,
                    "run_command",
                    return_value=(False, "cudaSetDevice failed: no CUDA-capable device"),
                ) as run_command,
                patch.object(colmap_runner.time, "sleep"),
            ):
                with self.assertRaises(RuntimeError) as err:
                    colmap_runner._run_patch_match_stereo(tmp)

            self.assertEqual(run_command.call_count, 2)
            self.assertIn("COLMAP PatchMatch failed during CUDA setup", str(err.exception))
            self.assertIn("not evidence of image quality", str(err.exception))

    def test_patch_match_non_cuda_failure_keeps_fallback_path(self):
        cmd = ["colmap", "patch_match_stereo"]
        with tempfile.TemporaryDirectory() as tmp:
            partial_dir = os.path.join(tmp, "stereo", "depth_maps")
            os.makedirs(partial_dir)
            partial_path = os.path.join(partial_dir, "partial.bin")
            with open(partial_path, "w") as handle:
                handle.write("partial")

            with (
                patch.object(colmap_runner, "_patch_match_stereo_command", return_value=cmd),
                patch.object(
                    colmap_runner,
                    "run_command",
                    return_value=(False, "workspace configuration is invalid"),
                ) as run_command,
                patch.object(colmap_runner.time, "sleep") as sleep,
            ):
                self.assertFalse(colmap_runner._run_patch_match_stereo(tmp))

            self.assertTrue(os.path.exists(partial_path))

        run_command.assert_called_once()
        sleep.assert_not_called()

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
