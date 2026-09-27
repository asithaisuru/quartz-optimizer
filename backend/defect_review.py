"""Human-reviewed safety-zone approximations; detector artifacts remain immutable."""
from __future__ import annotations

import errno
import logging
import threading
import time
import weakref
import copy
import hashlib
import json
import math
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote

TYPES = {"fracture", "inclusion", "cloud", "cavity", "other"}
SOURCES = {"ai_yolo", "opencv", "manual_3d", "manual_2d", "expert"}
STATUSES = {"provisional", "confirmed", "rejected"}
CLAIM = "expert/manual safety-zone approximations."


def now():
    return datetime.now(timezone.utc).isoformat()


_JSON_LOCKS = weakref.WeakValueDictionary()
_JSON_LOCKS_GUARD = threading.Lock()
_RETRY_DELAYS = (.005, .01, .02, .04, .05, .05, .05)


def json_file_lock(path):
    key = os.path.normcase(str(Path(path).resolve()))
    with _JSON_LOCKS_GUARD:
        lock = _JSON_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _JSON_LOCKS[key] = lock
        return lock


def _transient_file_operation(operation):
    # Windows readers/AV can temporarily deny replacement (5/32/33).
    # Retry only sharing/access errors; never loop on permanent IO errors.
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            return operation()
        except OSError as exc:
            transient = (isinstance(exc, PermissionError) or
                         exc.errno in {errno.EACCES, errno.EPERM, errno.EBUSY} or
                         getattr(exc, "winerror", None) in {5, 32, 33})
            if not transient or attempt == len(_RETRY_DELAYS):
                raise
            time.sleep(_RETRY_DELAYS[attempt])


def read_json(path, default=None):
    path = Path(path)
    with json_file_lock(path):
        def read():
            try:
                with path.open(encoding="utf-8") as handle:
                    return json.load(handle)
            except FileNotFoundError:
                return copy.deepcopy(default)
        return _transient_file_operation(read)


def atomic_json(path, value):
    path = Path(path)
    with json_file_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            _transient_file_operation(lambda: os.replace(temporary, path))
        finally:
            try:
                _transient_file_operation(lambda: temporary.unlink(missing_ok=True))
            except OSError:
                logging.getLogger(__name__).warning("Could not remove temporary JSON file", exc_info=True)


def finite(value, label, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(label + " must be a finite number.")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum) or (
        maximum is not None and value > maximum
    ):
        raise ValueError(label + " is outside its allowed range.")
    return value


def vector(value, label, positive=False):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(label + " must have three coordinates.")
    values = [finite(x, label) for x in value]
    if positive and min(values) <= 0:
        raise ValueError(label + " must be positive.")
    return values


def validate_annotation(record):
    if (not isinstance(record.get("type"), str) or record["type"] not in TYPES
            or not isinstance(record.get("source"), str) or record["source"] not in SOURCES):
        raise ValueError("Unsupported defect type or source.")
    if not isinstance(record.get("status"), str) or record["status"] not in STATUSES:
        raise ValueError("Unsupported defect status.")
    if record.get("confidence") is not None:
        finite(record["confidence"], "confidence", 0, 1)
    if not isinstance(record.get("notes"), str) or len(record["notes"]) > 10000:
        raise ValueError("notes must be text, at most 10000 characters.")
    if not isinstance(record.get("source_frames"), list):
        raise ValueError("source_frames must be a list.")
    geometry = record.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("geometry must be an object.")
    if "ellipsoid" in geometry or "tube_polyline" in geometry:
        raise ValueError("Use geometry_type plus a flat geometry object; nested geometry is not supported.")
    kind = record.get("geometry_type")
    if kind == "ellipsoid":
        if record["type"] == "fracture":
            raise ValueError("Fractures require tube_polyline geometry.")
        vector(geometry.get("center_mm"), "center_mm")
        vector(geometry.get("radii_mm"), "radii_mm", positive=True)
    elif kind == "tube_polyline":
        if record["type"] != "fracture":
            raise ValueError("tube_polyline is reserved for fractures.")
        points = geometry.get("points_mm")
        if not isinstance(points, list) or not 2 <= len(points) <= 2000:
            raise ValueError("points_mm requires 2 to 2000 points.")
        for point in points:
            vector(point, "points_mm")
        if finite(geometry.get("radius_mm"), "radius_mm") <= 0:
            raise ValueError("radius_mm must be positive.")
    elif kind == "sparse_candidate":
        if set(geometry) - {"points_mesh_units", "coordinate_frame"}:
            raise ValueError("Safety-zone geometry requires an explicit geometry_type.")
        if record["status"] == "confirmed":
            raise ValueError("Confirmation requires an ellipsoid or tube_polyline safety zone.")
    else:
        raise ValueError("Unsupported geometry_type.")


def source_frames(job, values):
    """Canonical frame objects. URLs only identify existing images inside this job."""
    root = Path(job).resolve()
    prefix = "/files/" + quote(root.name, safe="") + "/"
    output = {}
    for value in values:
        if isinstance(value, str):
            label, location = value, value
        elif isinstance(value, dict) and isinstance(value.get("frame"), str):
            label, location = value["frame"], value.get("url") or value["frame"]
        else:
            raise ValueError("source_frames items must be {frame: string, url: string|null}.")
        if not isinstance(location, str):
            raise ValueError("source_frames url must be text or null.")
        name = label.replace("\\", "/").rsplit("/", 1)[-1]
        location = location.replace("\\", "/")
        url = None
        choices = []
        if location.startswith(prefix):
            choices = [root / unquote(location[len(prefix):])]
        elif "://" not in location and not location.startswith("//"):
            candidate = Path(location)
            choices = [candidate if candidate.is_absolute() else root / candidate]
            if "/" not in location and ":" not in location:
                choices.append(root / "images" / location)
        for choice in choices:
            resolved = choice.resolve()
            if (resolved.is_relative_to(root) and resolved.is_file()
                    and resolved.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}):
                url = prefix + quote(resolved.relative_to(root).as_posix(), safe="/")
                break
        if name not in output or url is not None:
            output[name] = {"frame": name, "url": url}
    return list(output.values())


def annotation_response(job, item):
    value = copy.deepcopy(item)
    value["source_frames"] = source_frames(job, value.get("source_frames", []))
    return value


def candidates(job):
    job = Path(job)
    payload = read_json(job / "detections" / "policy_decisions.json", {})
    association = read_json(job / "dense" / "defect_associations.json", {})
    point_map = {}
    # A mapper alignment warning means these coordinates are not reliable.
    if association.get("status") == "complete" and not association.get("warnings"):
        for point in association.get("associations", []):
            xyz = point.get("xyz_aligned")
            try:
                xyz = vector(xyz, "xyz_aligned")
            except ValueError:
                continue
            for prediction in point.get("mapping", {}).get("prediction_ids", []):
                point_map.setdefault(str(prediction), []).append(xyz)
    output = []
    for index, item in enumerate(payload.get("records", [])):
        source = {"yolo": "ai_yolo", "ai_yolo": "ai_yolo", "opencv": "opencv"}.get(item.get("source"))
        if source is None:
            continue
        prediction = str(item.get("prediction_id", index))
        identity = source + ":" + prediction
        defect_type = str(item.get("class_name", "other")).lower()
        output.append({
            "id": "DEF-AI-" + hashlib.sha256(identity.encode()).hexdigest()[:20],
            "type": defect_type if defect_type in TYPES else "other",
            "source": source, "status": "provisional",
            "confidence": item.get("confidence"), "geometry_type": "sparse_candidate",
            "geometry": {"points_mesh_units": point_map.get(prediction, []),
                         "coordinate_frame": "centered_rough_mesh"},
            "source_frames": [item[k] for k in ("image_id", "source_image_path") if item.get(k)],
            "notes": "Provisional detector evidence; human review and safety-zone geometry required.",
            "created_at": item.get("created_at_utc", ""), "updated_at": item.get("created_at_utc", ""),
            "provenance": {"prediction_id": prediction, "original_detector_source": item.get("source"),
                           "class_name": item.get("class_name"), "mask_path": item.get("mask_path"),
                           "artifact": "detections/policy_decisions.json"},
        })
    return [annotation_response(job, item) for item in output]


def review(job):
    saved = read_json(Path(job) / "defect_review.json", {
        "schema_version": "1.0", "revision": 0, "annotations": [], "history": [],
    })
    annotations = [annotation_response(job, item) for item in saved["annotations"]]
    overridden = {x["id"] for x in annotations} | set(saved.get("deleted_candidate_ids", []))
    visible = [x for x in candidates(job) if x["id"] not in overridden]
    counts = Counter(x["status"] for x in visible + annotations)
    return {
        "policy": "confirmed_only", "candidates": visible, "annotations": annotations,
        "summary": {status: counts[status] for status in ("provisional", "confirmed", "rejected")},
        "schema_version": saved["schema_version"], "revision": saved["revision"],
        "geometry_claim": CLAIM,
        "coordinate_frame": {"name": "centered_rough_mesh", "annotation_units": "mm",
                             "candidate_units": "mesh_units",
                             "origin": "canonical_axis_aligned_bounding_box_center",
                             "axes": "unchanged_from_canonical_mesh"},
    }


def mutate(job, operation, payload=None, annotation_id=None):
    path = Path(job) / "defect_review.json"
    saved = read_json(path, {"schema_version": "1.0", "revision": 0, "annotations": [],
                             "history": [], "deleted_candidate_ids": []})
    payload = payload or {}
    records = saved["annotations"]
    old = next((x for x in records if x["id"] == annotation_id), None)
    if old is None and operation != "create":
        old = next((x for x in review(job)["candidates"] if x["id"] == annotation_id), None)
    if operation != "create" and old is None:
        raise KeyError(annotation_id)
    stamp = now()
    if operation == "delete":
        records[:] = [x for x in records if x["id"] != annotation_id]
        if annotation_id.startswith("DEF-AI-"):
            saved.setdefault("deleted_candidate_ids", []).append(annotation_id)
        updated = None
    else:
        allowed = {"type", "geometry_type", "geometry", "notes", "status", "confidence"}
        if operation == "create":
            allowed |= {"source", "source_frames"}
        if set(payload) - allowed:
            raise ValueError("Unsupported fields: " + ", ".join(sorted(set(payload) - allowed)))
        if operation == "create":
            if payload.get("source", "manual_3d") not in ("manual_3d", "manual_2d", "expert"):
                raise ValueError("POST creates manual/expert annotations only.")
            updated = {
                "id": "DEF-" + uuid.uuid4().hex, "type": "other", "source": "manual_3d",
                "status": "provisional", "confidence": None, "geometry_type": "sparse_candidate",
                "geometry": {}, "source_frames": [], "notes": "", "created_at": stamp,
            }
        else:
            updated = copy.deepcopy(old)
        updated.update(payload)
        updated["updated_at"] = stamp
        validate_annotation(updated)
        updated = annotation_response(job, updated)
        if updated["status"] == "confirmed" and (old is None or old.get("status") != "confirmed"):
            updated["confirmation"] = {"method": "manual_review", "confirmed_at": stamp}
        records[:] = [x for x in records if x["id"] != updated["id"]]
        records.append(updated)
    saved["revision"] += 1
    saved["updated_at"] = stamp
    saved["history"].append({"revision": saved["revision"], "operation": operation,
                             "at": stamp, "before": old, "after": copy.deepcopy(updated)})
    atomic_json(path, saved)
    return updated


def confirmed_annotations(snapshot):
    result = []
    for item in snapshot.get("annotations", []):
        if item.get("status") != "confirmed":
            continue
        validate_annotation(item)
        if item["source"] in {"expert", "manual_2d", "manual_3d"} or item.get("confirmation"):
            result.append(item)
    return result
