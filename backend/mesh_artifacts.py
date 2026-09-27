"""Topology-preserving preparation and selection of optimizer mesh artifacts.

``final_textured_model.ply`` is the canonical reconstruction.  Derived viewer
and optimizer meshes may translate that geometry, but they must not change its
topology or volume.  If validation fails, callers receive the canonical path as
the safe optimizer input.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


CANONICAL_MESH_FILENAME = "final_textured_model.ply"
ALIGNED_MESH_FILENAME = "visual_aligned_stone.ply"
OPTIMIZER_MESH_FILENAME = "final_textured_model_opt_input.ply"
VALIDATION_REPORT_FILENAME = "optimizer_mesh_validation.json"
VOLUME_RELATIVE_TOLERANCE = 1e-6
VOLUME_ABSOLUTE_TOLERANCE = 1e-12


def _as_mesh(loaded):
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("Mesh scene contains no geometry.")
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("Artifact is not a triangle mesh.")
    return loaded


def load_mesh_preserving_topology(path):
    """Load a mesh without Trimesh's automatic repair/vertex merging."""

    return _as_mesh(trimesh.load(Path(path), process=False))


def _face_component_count(mesh):
    face_count = int(len(mesh.faces))
    if face_count == 0:
        return 0
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency) == 0:
        return face_count
    rows = np.concatenate((adjacency[:, 0], adjacency[:, 1]))
    columns = np.concatenate((adjacency[:, 1], adjacency[:, 0]))
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(face_count, face_count),
    ).tocsr()
    return int(
        connected_components(graph, directed=False, return_labels=False)
    )


def mesh_invariants(mesh):
    edge_use_counts = np.bincount(mesh.edges_unique_inverse)
    volume = float(abs(mesh.volume))
    return {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "connected_components": _face_component_count(mesh),
        "boundary_edge_count": int(np.count_nonzero(edge_use_counts == 1)),
        "non_manifold_edge_count": int(np.count_nonzero(edge_use_counts > 2)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "volume_mesh_units_cubed": volume if math.isfinite(volume) else None,
        "bounding_dimensions": [float(value) for value in mesh.extents],
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compare_mesh_invariants(canonical, candidate):
    reasons = []

    for key in ("vertex_count", "face_count", "connected_components"):
        if candidate[key] != canonical[key]:
            reasons.append(
                {
                    "code": f"{key}_changed",
                    "canonical": canonical[key],
                    "optimizer": candidate[key],
                }
            )

    if canonical["watertight"] and not candidate["watertight"]:
        reasons.append(
            {
                "code": "watertight_status_regressed",
                "canonical": True,
                "optimizer": False,
            }
        )
    if (
        canonical["winding_consistent"]
        and not candidate["winding_consistent"]
    ):
        reasons.append(
            {
                "code": "winding_consistency_regressed",
                "canonical": True,
                "optimizer": False,
            }
        )

    for key in ("boundary_edge_count", "non_manifold_edge_count"):
        if candidate[key] > canonical[key]:
            reasons.append(
                {
                    "code": f"{key}_increased",
                    "canonical": canonical[key],
                    "optimizer": candidate[key],
                }
            )

    canonical_volume = canonical["volume_mesh_units_cubed"]
    optimizer_volume = candidate["volume_mesh_units_cubed"]
    if canonical_volume is None or optimizer_volume is None:
        volume_difference = None
        volume_relative_difference = None
        reasons.append(
            {
                "code": "volume_not_finite",
                "canonical": canonical_volume,
                "optimizer": optimizer_volume,
            }
        )
    else:
        volume_difference = abs(optimizer_volume - canonical_volume)
        denominator = max(abs(canonical_volume), VOLUME_ABSOLUTE_TOLERANCE)
        volume_relative_difference = volume_difference / denominator
        tolerance = max(
            VOLUME_ABSOLUTE_TOLERANCE,
            abs(canonical_volume) * VOLUME_RELATIVE_TOLERANCE,
        )
        if volume_difference > tolerance:
            reasons.append(
                {
                    "code": "volume_changed",
                    "canonical": canonical_volume,
                    "optimizer": optimizer_volume,
                    "absolute_difference": volume_difference,
                    "relative_difference": volume_relative_difference,
                    "relative_tolerance": VOLUME_RELATIVE_TOLERANCE,
                }
            )

    return {
        "safe_to_use": not reasons,
        "failure_reasons": reasons,
        "volume_absolute_difference": volume_difference,
        "volume_relative_difference": volume_relative_difference,
        "volume_relative_tolerance": VOLUME_RELATIVE_TOLERANCE,
    }


def validate_mesh_artifact(canonical_path, candidate_path):
    canonical_mesh = load_mesh_preserving_topology(canonical_path)
    candidate_mesh = load_mesh_preserving_topology(candidate_path)
    canonical = mesh_invariants(canonical_mesh)
    candidate = mesh_invariants(candidate_mesh)
    comparison = compare_mesh_invariants(canonical, candidate)
    return {
        "canonical": canonical,
        "candidate": candidate,
        "comparison": comparison,
    }


def _optimizer_payload(mesh):
    """Return an input accepted by ``trimesh.load`` without auto-processing.

    ``optimizer.optimize_cut`` owns its load call, so passing a serialized mesh
    dictionary with ``process=False`` keeps the existing optimizer API and
    algorithms unchanged while making that load topology-preserving.
    """

    return {
        "vertices": np.array(mesh.vertices, dtype=np.float64, copy=True),
        "faces": np.array(mesh.faces, dtype=np.int64, copy=True),
        "process": False,
    }


def _validate_optimizer_payload(canonical_mesh, payload):
    optimizer_mesh = _as_mesh(trimesh.load(payload))
    canonical = mesh_invariants(canonical_mesh)
    candidate = mesh_invariants(optimizer_mesh)
    return {
        "canonical": canonical,
        "candidate": candidate,
        "comparison": compare_mesh_invariants(canonical, candidate),
    }


def _atomic_write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_optimizer_mesh_artifacts(canonical_path, job_folder=None):
    """Create validated derived artifacts and select a safe optimizer mesh.

    The canonical file is read only.  A centered in-memory copy is exported for
    the viewer and optimizer, then reloaded with ``process=False`` and compared
    with the canonical mesh.  Any negative topology or volume change causes the
    canonical path to be selected instead.
    """

    canonical_path = Path(canonical_path).resolve()
    if canonical_path.name != CANONICAL_MESH_FILENAME:
        raise ValueError(
            f"Optimizer preparation requires {CANONICAL_MESH_FILENAME}."
        )
    if not canonical_path.is_file():
        raise FileNotFoundError(str(canonical_path))

    canonical_hash_before = _sha256(canonical_path)
    canonical_mesh = load_mesh_preserving_topology(canonical_path)
    centered_mesh = canonical_mesh.copy()
    centered_mesh.apply_translation(-centered_mesh.bounds.mean(axis=0))

    aligned_path = canonical_path.with_name(ALIGNED_MESH_FILENAME)
    optimizer_candidate_path = canonical_path.with_name(OPTIMIZER_MESH_FILENAME)
    centered_mesh.export(aligned_path)
    centered_mesh.export(optimizer_candidate_path)

    aligned_validation = validate_mesh_artifact(canonical_path, aligned_path)
    optimizer_validation = validate_mesh_artifact(
        canonical_path,
        optimizer_candidate_path,
    )
    canonical_hash_after = _sha256(canonical_path)
    canonical_unchanged = canonical_hash_before == canonical_hash_after
    if not canonical_unchanged:
        optimizer_validation["comparison"]["safe_to_use"] = False
        optimizer_validation["comparison"]["failure_reasons"].append(
            {"code": "canonical_mesh_changed_during_preparation"}
        )

    optimizer_file_safe = bool(
        optimizer_validation["comparison"]["safe_to_use"]
        and canonical_unchanged
    )
    aligned_safe = bool(
        aligned_validation["comparison"]["safe_to_use"]
        and canonical_unchanged
    )
    selected_optimizer_path = (
        optimizer_candidate_path if optimizer_file_safe else canonical_path
    )
    selected_viewer_path = aligned_path if aligned_safe else canonical_path

    selected_optimizer_mesh = load_mesh_preserving_topology(
        selected_optimizer_path
    )
    optimizer_input = _optimizer_payload(selected_optimizer_mesh)
    optimizer_runtime_validation = _validate_optimizer_payload(
        canonical_mesh,
        optimizer_input,
    )

    # The dictionary payload is what optimize_cut actually passes through
    # trimesh.load. If that runtime representation ever regresses despite the
    # file check, reject the derived artifact and rebuild from the canonical.
    runtime_candidate_validation = optimizer_runtime_validation
    if (
        not optimizer_runtime_validation["comparison"]["safe_to_use"]
        and selected_optimizer_path != canonical_path
    ):
        selected_optimizer_path = canonical_path
        optimizer_input = _optimizer_payload(canonical_mesh)
        optimizer_runtime_validation = _validate_optimizer_payload(
            canonical_mesh,
            optimizer_input,
        )
    if not optimizer_runtime_validation["comparison"]["safe_to_use"]:
        raise ValueError(
            "Canonical mesh could not be represented for optimization "
            "without changing topology or volume."
        )

    fallback_to_canonical = selected_optimizer_path == canonical_path

    report = {
        "schema_version": "1.0",
        "generated_at_epoch": time.time(),
        "canonical_mesh": str(canonical_path),
        "canonical_sha256_before": canonical_hash_before,
        "canonical_sha256_after": canonical_hash_after,
        "canonical_unchanged": canonical_unchanged,
        "aligned_mesh": str(aligned_path),
        "optimizer_candidate_mesh": str(optimizer_candidate_path),
        "selected_optimizer_mesh": str(selected_optimizer_path),
        "selected_viewer_mesh": str(selected_viewer_path),
        "fallback_to_canonical": fallback_to_canonical,
        "status": (
            "FALLBACK_CANONICAL" if fallback_to_canonical else "PASS"
        ),
        "aligned_validation": aligned_validation,
        "optimizer_validation": optimizer_validation,
        "optimizer_runtime_candidate_validation": (
            runtime_candidate_validation
        ),
        "optimizer_runtime_validation": optimizer_runtime_validation,
    }
    if job_folder:
        report_root = Path(job_folder)
    elif canonical_path.parent.name.casefold() == "dense":
        report_root = canonical_path.parent.parent
    else:
        report_root = canonical_path.parent
    _atomic_write_json(report_root / VALIDATION_REPORT_FILENAME, report)
    return {
        "mesh": centered_mesh,
        "optimizer_mesh_path": str(selected_optimizer_path),
        "optimizer_input": optimizer_input,
        "viewer_mesh_path": str(selected_viewer_path),
        "report": report,
    }
