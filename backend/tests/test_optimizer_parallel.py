"""Ordered process evaluation matches the sequential geometry decisions."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import optimizer
from defect_aware_geometry import ConfirmedGeometry
from optimizer_parallel import configuration


@pytest.mark.parametrize("logical,expected", [(1, 1), (2, 1), (8, 7), (24, 23)])
def test_auto_workers_use_all_but_one_logical_cpu(logical, expected):
    settings = configuration({"OPTIMIZER_PERFORMANCE_MODE": "full", "OPTIMIZER_MAX_WORKERS": "auto"}, logical)
    assert settings["optimizer_worker_count"] == expected
    assert settings["logical_cpu_count"] == logical
    assert configuration({"OPTIMIZER_PERFORMANCE_MODE": "sequential"}, logical)["optimizer_worker_count"] == 1


def test_unavailable_cpu_detection_and_invalid_worker_setting_are_safe():
    with patch("optimizer_parallel.os.cpu_count", return_value=None):
        assert configuration({})["optimizer_worker_count"] == 1
    assert configuration({"OPTIMIZER_MAX_WORKERS": "bad"}, 8)["optimizer_worker_count"] == 1


def context(kind):
    annotations = []
    if kind != "none":
        annotations = [{"id": "test", "type": "fracture" if kind == "tube" else "inclusion",
            "source": "manual_3d", "status": "confirmed", "notes": "", "source_frames": [],
            "geometry_type": "tube_polyline" if kind == "tube" else "ellipsoid",
            "geometry": {"points_mm": [[0, -4, 0], [0, 4, 0]], "radius_mm": .25} if kind == "tube" else
                        {"center_mm": [0, 0, 0], "radii_mm": [1, 1, 1]}}]
    grid = np.full((25, 25, 25), 5.)
    guard = ConfirmedGeometry({"annotations": annotations}, 1.)
    origin = np.full(3, -12.)
    return optimizer.FitContext(grid, origin, 1., grid>0, guard.mask(grid.shape, origin, 1.),
        optimizer._fit_settings(np.full(3, 20.), 1., mm_per_mesh_unit=1.), confirmed_guard=guard)


def candidates(ctx):
    rough = trimesh.creation.box(extents=(20, 20, 20))
    shapes = {"Box": trimesh.creation.box(extents=(2, 1.5, 1))}
    variants = optimizer._make_variants(shapes)[:3]
    with patch.object(optimizer, "_make_variants", return_value=variants):
        return optimizer._generate_candidates(rough, shapes, np.array([[-4., 0, 0], [0., 0, 0], [4., 0, 0]]), ctx)


@pytest.mark.parametrize("kind", ["none", "ellipsoid", "tube"])
def test_parallel_and_sequential_candidate_results_and_defect_decisions_match(kind):
    sequential = context(kind)
    parallel = context(kind)
    parallel.performance = configuration({"OPTIMIZER_MAX_WORKERS": "2"}, 4)
    original_env = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "RECONSTRUCTION_REPRODUCIBLE_MODE")}
    expected, rejected = candidates(sequential)
    actual, parallel_rejected = candidates(parallel)
    assert parallel.performance["parallel_fallback"] is False
    assert parallel.performance["optimizer_worker_count"] == 2
    assert parallel.performance["parallel_evaluation_seconds"] > 0
    assert len(actual) == len(expected) > 0
    assert parallel_rejected == rejected
    assert parallel.confirmed_guard.rejected == sequential.confirmed_guard.rejected
    for a, b in zip(actual, expected):
        assert a.variant.key == b.variant.key
        assert a.scale == pytest.approx(b.scale, abs=1e-12)
        np.testing.assert_array_equal(a.occ_flat, b.occ_flat)
        assert not parallel.confirmed_guard.rejects(a.variant.verts*a.scale+a.pos)
    assert {key: os.environ.get(key) for key in original_env} == original_env


def test_process_initialization_failure_falls_back_to_sequential():
    ctx = context("ellipsoid")
    ctx.performance = configuration({"OPTIMIZER_MAX_WORKERS": "2"}, 4)
    with patch("loky.ProcessPoolExecutor", side_effect=OSError("process launch unavailable")):
        actual, _ = candidates(ctx)
    expected, _ = candidates(context("ellipsoid"))
    assert len(actual) == len(expected) > 0
    assert ctx.performance["parallel_fallback"] is True
    assert ctx.performance["optimizer_worker_count"] == 1


def test_reconstruction_configuration_remains_one_thread():
    from colmap_runner import _sparse_threads
    # The existing reconstruction regression suite checks the generated flags;
    # parallel evaluation must never change that function or parent environment.
    assert configuration({}, 24)["optimizer_worker_count"] == 23
    with patch.dict(os.environ, {"RECONSTRUCTION_REPRODUCIBLE_MODE": "true"}):
        assert _sparse_threads() == 1
