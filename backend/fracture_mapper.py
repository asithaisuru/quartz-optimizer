"""Associate policy-approved 2D masks with existing COLMAP sparse points."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from colmap_loader import read_images_binary, read_points3D_binary


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _load_policy_payload(job: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = job / "detections" / "policy_decisions.json"
    if not path.exists():
        return None, (
            "No structured policy decisions were found. Historical masks are "
            "legacy/unverified and were not mapped."
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        return None, f"Policy decisions could not be read: {exc}"
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        return None, "Policy decisions do not contain a records list."
    return payload, None


def _resolve_mask_path(job: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = job / path
    try:
        resolved = path.resolve()
        resolved.relative_to(job.resolve())
    except (OSError, ValueError):
        return None
    return resolved if resolved.exists() else None


def _approved_records(
    job: Path,
    payload: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    records_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in payload.get("records", []):
        if not isinstance(value, dict):
            continue
        mapping_accepted = bool(value.get("accepted_for_3d_mapping"))
        no_cut_accepted = bool(value.get("accepted_for_no_cut_zone"))
        if not mapping_accepted and not no_cut_accepted:
            continue
        mapping_path = _resolve_mask_path(job, value.get("mapping_mask_path"))
        no_cut_path = _resolve_mask_path(job, value.get("no_cut_mask_path"))
        if mapping_accepted and mapping_path is None:
            continue
        record = dict(value)
        record["_mapping_path"] = mapping_path
        record["_no_cut_path"] = no_cut_path
        records_by_image[str(value.get("image_id", ""))].append(record)
    return records_by_image


def _mask_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def _find_sparse_path(job: Path) -> Path | None:
    preferred = job / "sparse" / "0"
    if preferred.exists():
        return preferred
    sparse_root = job / "sparse"
    if not sparse_root.exists():
        return None
    subfolders = sorted(
        path for path in sparse_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    return subfolders[0] if subfolders else None


def _mesh_offset(job: Path) -> tuple[np.ndarray, str | None]:
    mesh_path = job / "dense" / "final_textured_model.ply"
    if not mesh_path.exists():
        return np.zeros(3, dtype=float), None
    try:
        import trimesh

        mesh = trimesh.load(mesh_path)
        return -np.asarray(mesh.bounds, dtype=float).mean(axis=0), None
    except Exception as exc:
        return np.zeros(3, dtype=float), (
            f"Visual mesh alignment offset was unavailable: {exc}"
        )


def _write_point_cloud(path: Path, vertices: list[np.ndarray], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for vertex in vertices:
            handle.write(
                f"{float(vertex[0]):.9g} {float(vertex[1]):.9g} "
                f"{float(vertex[2]):.9g} {color[0]} {color[1]} {color[2]}\n"
            )


def _metadata_values(records: list[dict[str, Any]]) -> dict[str, Any]:
    class_ids = sorted(
        {
            int(record["class_id"])
            for record in records
            if record.get("class_id") is not None
        }
    )
    confidences = sorted(
        {
            round(float(record["confidence"]), 6)
            for record in records
            if record.get("confidence") is not None
        }
    )
    return {
        "prediction_ids": sorted(
            {str(record.get("prediction_id")) for record in records}
        ),
        "sources": sorted({str(record.get("source")) for record in records}),
        "class_ids": class_ids,
        "class_names": sorted(
            {str(record.get("class_name")) for record in records}
        ),
        "confidences": confidences,
    }


def _unavailable_result(
    job: Path,
    policy: str | None,
    reason: str,
    legacy_present: bool,
) -> None:
    _write_json(
        job / "dense" / "defect_associations.json",
        {
            "schema_version": "1.0",
            "status": "unavailable",
            "method": "mask_filtered_colmap_sparse_feature_point_association",
            "policy": policy,
            "mapping_point_count": 0,
            "no_cut_point_count": 0,
            "legacy_masks_present": legacy_present,
            "legacy_masks_used": False,
            "reason": reason,
            "claim_boundary": (
                "No accepted defect evidence does not prove a defect-free stone."
            ),
            "associations": [],
        },
    )


def map_fractures_to_3d(
    job_folder: str | Path,
    images_loader: Callable[[str], dict[Any, Any]] = read_images_binary,
    points_loader: Callable[[str], dict[Any, Any]] = read_points3D_binary,
) -> str | None:
    """Map only approved masks to already-reconstructed sparse feature points."""

    print("--- Mapping Approved 2D Candidates to COLMAP Sparse Points ---")
    job = Path(job_folder).resolve()
    legacy_present = (job / "fractures").exists()
    payload, error = _load_policy_payload(job)
    if payload is None:
        _unavailable_result(job, None, error or "Policy metadata unavailable.", legacy_present)
        print(f"   No approved policy input: {error}")
        return None

    policy = str(payload.get("policy", "unavailable"))
    records_by_image = _approved_records(job, payload)
    if not records_by_image:
        _unavailable_result(
            job,
            policy,
            "No policy-approved mapping masks were available.",
            legacy_present,
        )
        print("   No policy-approved mapping masks were available.")
        return None

    sparse_path = _find_sparse_path(job)
    if sparse_path is None:
        _unavailable_result(
            job,
            policy,
            "No COLMAP sparse model was found.",
            legacy_present,
        )
        print("   No sparse model found.")
        return None

    try:
        images = images_loader(str(sparse_path / "images.bin"))
        points3d = points_loader(str(sparse_path / "points3D.bin"))
    except Exception as exc:
        _unavailable_result(
            job,
            policy,
            f"COLMAP sparse data could not be read: {exc}",
            legacy_present,
        )
        print(f"   COLMAP sparse data could not be read: {exc}")
        return None

    mapping_hits: dict[int, list[dict[str, Any]]] = defaultdict(list)
    no_cut_hits: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for image_data in images.values():
        records = records_by_image.get(str(image_data.name), [])
        if not records:
            continue
        loaded_masks = []
        for record in records:
            mapping_path = record.get("_mapping_path")
            no_cut_path = record.get("_no_cut_path")
            loaded_masks.append(
                (
                    record,
                    _mask_array(mapping_path) if mapping_path else None,
                    _mask_array(no_cut_path) if no_cut_path else None,
                )
            )

        for index, point_id in enumerate(image_data.point3D_ids):
            if point_id == -1:
                continue
            x, y = image_data.xys[index]
            if not (math.isfinite(float(x)) and math.isfinite(float(y))):
                continue
            u, v = int(round(float(x))), int(round(float(y)))
            for record, mapping_mask, no_cut_mask in loaded_masks:
                if (
                    mapping_mask is not None
                    and 0 <= v < mapping_mask.shape[0]
                    and 0 <= u < mapping_mask.shape[1]
                    and mapping_mask[v, u]
                ):
                    mapping_hits[int(point_id)].append(record)
                if (
                    no_cut_mask is not None
                    and 0 <= v < no_cut_mask.shape[0]
                    and 0 <= u < no_cut_mask.shape[1]
                    and no_cut_mask[v, u]
                ):
                    no_cut_hits[int(point_id)].append(record)

    offset, offset_warning = _mesh_offset(job)
    mapping_vertices = []
    no_cut_vertices = []
    associations = []
    for point_id in sorted(mapping_hits):
        point = points3d.get(point_id)
        if point is None:
            continue
        xyz = np.asarray(point.xyz, dtype=float) + offset
        mapping_vertices.append(xyz)
        no_cut_records = no_cut_hits.get(point_id, [])
        associations.append(
            {
                "point3D_id": point_id,
                "xyz_aligned": [round(float(value), 9) for value in xyz],
                "mapping": _metadata_values(mapping_hits[point_id]),
                "accepted_for_no_cut_zone": bool(no_cut_records),
                "no_cut": _metadata_values(no_cut_records) if no_cut_records else None,
            }
        )
    for point_id in sorted(no_cut_hits):
        point = points3d.get(point_id)
        if point is not None:
            no_cut_vertices.append(np.asarray(point.xyz, dtype=float) + offset)

    warnings = [offset_warning] if offset_warning else []
    metadata = {
        "schema_version": "1.0",
        "status": "complete" if mapping_vertices else "unavailable",
        "method": "mask_filtered_colmap_sparse_feature_point_association",
        "policy": policy,
        "strict_research_mode": bool(payload.get("strict_research_mode", True)),
        "mapping_point_count": len(mapping_vertices),
        "no_cut_point_count": len(no_cut_vertices),
        "legacy_masks_present": legacy_present,
        "legacy_masks_used": False,
        "warnings": warnings,
        "claim_boundary": (
            "Points are existing COLMAP sparse feature points selected by approved "
            "2D masks; they are not a validated internal defect volume."
        ),
        "associations": associations,
    }
    if not mapping_vertices:
        metadata["reason"] = (
            "Approved masks did not overlap any reconstructed COLMAP sparse points. "
            "This does not prove a defect-free stone."
        )
    _write_json(job / "dense" / "defect_associations.json", metadata)

    if not mapping_vertices:
        print("   Approved masks did not overlap reconstructed sparse points.")
        return None

    visualization_path = job / "dense" / "defects.ply"
    _write_point_cloud(visualization_path, mapping_vertices, (255, 64, 32))
    if no_cut_vertices:
        _write_point_cloud(
            job / "dense" / "no_cut_defects.ply",
            no_cut_vertices,
            (255, 0, 0),
        )
    print(
        f"   Saved {len(mapping_vertices)} mapped visualization points and "
        f"{len(no_cut_vertices)} optimizer no-cut points."
    )
    return str(visualization_path)
