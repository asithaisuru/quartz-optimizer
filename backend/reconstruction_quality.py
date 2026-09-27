"""Pre-optimization validation for reconstructed rough-stone meshes.

This module deliberately does not change optimizer behavior.  It validates
that a reconstruction can supply the closed, manifold volume representation
the existing optimizer expects, and records the measurements used to make
that decision.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import trimesh


MIN_USABLE_SDF_VOXELS = 64
RECONSTRUCTION_INSUFFICIENT_PREFIX = "Reconstruction insufficient"


def _as_mesh(loaded):
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            return None
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    return None


def _failure_reason(code, message, value=None, threshold=None):
    reason = {"code": code, "message": message}
    if value is not None:
        reason["value"] = value
    if threshold is not None:
        reason["threshold"] = threshold
    return reason


def assess_reconstruction_quality(
    mesh_path,
    *,
    sdf_resolution=None,
    minimum_usable_sdf_voxels=MIN_USABLE_SDF_VOXELS,
):
    """Return a JSON-serializable quality-gate report for ``mesh_path``."""

    path = Path(mesh_path)
    report = {
        "schema_version": "1.0",
        "mesh_path": str(path),
        "generated_at_epoch": time.time(),
        "status": "FAIL",
        "passed": False,
        "checks": {},
        "metrics": {},
        "failure_reasons": [],
    }

    if not path.is_file():
        report["failure_reasons"].append(
            _failure_reason("mesh_missing", "Reconstructed mesh file is missing.")
        )
        report["message"] = (
            f"{RECONSTRUCTION_INSUFFICIENT_PREFIX}: reconstructed mesh file is missing."
        )
        return report

    try:
        mesh = _as_mesh(trimesh.load(path, process=False))
    except Exception as exc:
        report["failure_reasons"].append(
            _failure_reason(
                "mesh_unreadable",
                f"Reconstructed mesh could not be read: {str(exc)[:240]}",
            )
        )
        report["message"] = (
            f"{RECONSTRUCTION_INSUFFICIENT_PREFIX}: reconstructed mesh could not be read."
        )
        return report

    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        report["failure_reasons"].append(
            _failure_reason("mesh_empty", "Reconstructed mesh has no usable faces.")
        )
        report["message"] = (
            f"{RECONSTRUCTION_INSUFFICIENT_PREFIX}: reconstructed mesh has no usable faces."
        )
        return report

    extents = np.asarray(mesh.extents, dtype=float)
    try:
        mesh_volume = float(abs(mesh.volume))
        if not math.isfinite(mesh_volume):
            mesh_volume = None
    except Exception:
        mesh_volume = None
    report["metrics"].update(
        vertex_count=int(len(mesh.vertices)),
        face_count=int(len(mesh.faces)),
        mesh_volume=mesh_volume,
        bounding_dimensions=[
            float(value) if math.isfinite(float(value)) else None
            for value in extents
        ],
    )

    watertight = bool(mesh.is_watertight)
    report["checks"]["watertight"] = {
        "passed": watertight,
        "value": watertight,
    }
    if not watertight:
        report["failure_reasons"].append(
            _failure_reason(
                "mesh_not_watertight",
                "Mesh contains open boundaries or does not enclose a closed volume.",
            )
        )

    try:
        edge_use_counts = np.bincount(mesh.edges_unique_inverse)
        boundary_edges = int(np.count_nonzero(edge_use_counts == 1))
        non_manifold_edges = int(np.count_nonzero(edge_use_counts > 2))
    except Exception as exc:
        boundary_edges = None
        non_manifold_edges = None
        report["failure_reasons"].append(
            _failure_reason(
                "topology_check_failed",
                f"Mesh edge topology could not be evaluated: {str(exc)[:240]}",
            )
        )

    topology_passed = non_manifold_edges == 0
    report["checks"]["non_manifold_edges"] = {
        "passed": topology_passed,
        "value": non_manifold_edges,
        "required": 0,
    }
    report["metrics"]["boundary_edge_count"] = boundary_edges
    report["metrics"]["non_manifold_edge_count"] = non_manifold_edges
    if non_manifold_edges not in (None, 0):
        report["failure_reasons"].append(
            _failure_reason(
                "non_manifold_edges_detected",
                "Mesh has edges shared by more than two faces.",
                value=non_manifold_edges,
                threshold=0,
            )
        )

    usable_sdf_voxels = 0
    usable_sdf_volume = 0.0
    sdf_pitch = None
    sdf_error = None
    try:
        if not np.all(np.isfinite(extents)) or float(np.max(extents)) <= 0:
            raise ValueError("mesh bounding dimensions are not finite and positive")

        # Reuse the optimizer's existing SDF representation so the gate checks
        # the exact volume semantics that optimization will consume.
        from optimizer import _build_sdf_grid

        kwargs = {}
        if sdf_resolution is not None:
            kwargs["res"] = int(sdf_resolution)
        grid, _origin, sdf_pitch = _build_sdf_grid(mesh, **kwargs)
        sdf_pitch = float(sdf_pitch)
        if not math.isfinite(sdf_pitch) or sdf_pitch <= 0:
            raise ValueError("SDF pitch is not finite and positive")
        usable_mask = np.isfinite(grid) & (grid > -(sdf_pitch * 0.20))
        usable_sdf_voxels = int(np.count_nonzero(usable_mask))
        usable_sdf_volume = float(usable_sdf_voxels * (sdf_pitch ** 3))
        if not math.isfinite(usable_sdf_volume):
            raise ValueError("usable SDF volume is not finite")
    except Exception as exc:
        sdf_error = str(exc)[:240]

    sdf_passed = (
        sdf_error is None
        and usable_sdf_voxels >= int(minimum_usable_sdf_voxels)
        and usable_sdf_volume > 0
    )
    report["checks"]["usable_sdf_volume"] = {
        "passed": sdf_passed,
        "usable_voxel_count": usable_sdf_voxels,
        "minimum_usable_voxel_count": int(minimum_usable_sdf_voxels),
        "volume_mesh_units_cubed": usable_sdf_volume,
        "sdf_pitch_mesh_units": sdf_pitch,
        "error": sdf_error,
    }
    report["metrics"].update(
        usable_sdf_voxel_count=usable_sdf_voxels,
        usable_sdf_volume=usable_sdf_volume,
        sdf_pitch=sdf_pitch,
    )
    if not sdf_passed:
        report["failure_reasons"].append(
            _failure_reason(
                "usable_sdf_volume_insufficient",
                (
                    f"SDF construction failed: {sdf_error}"
                    if sdf_error
                    else "Mesh has too little usable SDF volume for optimization."
                ),
                value=usable_sdf_voxels,
                threshold=int(minimum_usable_sdf_voxels),
            )
        )

    passed = watertight and topology_passed and sdf_passed
    report["passed"] = passed
    report["status"] = "PASS" if passed else "FAIL"
    if passed:
        report["message"] = "Reconstruction quality gate passed."
    else:
        details = "; ".join(
            reason["message"] for reason in report["failure_reasons"]
        )
        report["message"] = f"{RECONSTRUCTION_INSUFFICIENT_PREFIX}: {details}"
    return report
