"""Read-only integrity audit for YOLO segmentation datasets."""

from __future__ import annotations

import ast
import csv
import math
import os
import platform
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

from . import TOOL_VERSION
from .common import (
    STATUS_PASS,
    STATUS_UNAVAILABLE,
    atomic_write_json,
    deterministic_random,
    git_evidence,
    normalize_path,
    read_json,
    sha256_file,
    utc_timestamp,
)

DEFAULT_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)
SPLIT_ALIASES = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
ISSUE_COLUMNS = (
    "severity",
    "category",
    "split",
    "file",
    "line_number",
    "error_code",
    "message",
)
MANIFEST_FIELDS = (
    "sample_id",
    "file_path",
    "label_path",
    "specimen_id",
    "capture_session_id",
    "recording_id",
    "source_frame_id",
    "augmentation_parent_id",
    "exact_duplicate_group_id",
    "near_duplicate_group_id",
    "split",
    "original_split",
    "source_dataset",
    "source_dataset_version",
    "license",
    "annotator_id",
    "annotation_status",
    "expert_review_status",
    "expert_reviewer_id",
    "expert_review_date",
    "notes",
)
REQUIRED_MANIFEST_FIELDS = (
    "sample_id",
    "file_path",
    "label_path",
    "specimen_id",
    "capture_session_id",
    "recording_id",
    "source_frame_id",
    "augmentation_parent_id",
    "split",
    "source_dataset",
    "license",
    "annotator_id",
    "annotation_status",
)


@dataclass
class ImageRecord:
    """Internal immutable-source record for one dataset image."""

    split: str
    path: Path
    relative_path: str
    stem: str
    size_bytes: int
    sha256: str = ""
    width: int | None = None
    height: int | None = None
    image_format: str = ""
    difference_hash: int | None = None
    label_path: Path | None = None
    label_sha256: str = ""
    class_ids: set[int] = field(default_factory=set)

    @property
    def augmentation_parent_id(self) -> str:
        return augmentation_parent_id(self.stem)


def augmentation_parent_id(stem: str) -> str:
    """Return the deterministic Roboflow source identifier for an image stem."""

    return stem.split(".rf.", 1)[0]


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return {}
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("\"'")


def _limited_yaml_load(path: Path) -> dict[str, Any]:
    """Parse the simple mapping/list subset used by Ultralytics data YAML files."""

    result: dict[str, Any] = {}
    parent_key: str | None = None
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            indentation = len(raw_line) - len(raw_line.lstrip(" "))
            line = raw_line.strip()
            if ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            key = key.strip()
            if indentation == 0:
                value = _parse_scalar(raw_value)
                result[key] = value
                parent_key = key if isinstance(value, dict) else None
            elif parent_key is not None:
                parent = result.setdefault(parent_key, {})
                if isinstance(parent, dict):
                    parent[key] = _parse_scalar(raw_value)
    return result


def load_dataset_yaml(path: str | Path) -> dict[str, Any]:
    """Load dataset YAML with PyYAML when available and a safe local fallback."""

    yaml_path = Path(path)
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return _limited_yaml_load(yaml_path)

    with yaml_path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {yaml_path}")
    return value


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd().resolve()


def load_audit_config(
    path: str | Path,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    """Load a dataset audit configuration and resolve repository-relative paths."""

    config_path = Path(path).resolve()
    config = read_json(config_path)
    repo_root = _find_repo_root(config_path.parent)
    config["_config_path"] = str(config_path)
    config["_repo_root"] = str(repo_root)
    if output_override is not None:
        config["output_directory"] = str(output_override)
    return config


def _resolve_from_root(value: str | Path | None, repo_root: Path) -> Path | None:
    if value in (None, ""):
        return None
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _resolve_yaml_path(value: str | Path, yaml_path: Path, yaml_data: dict[str, Any]) -> Path:
    candidate = Path(str(value).replace("\\", os.sep))
    if candidate.is_absolute():
        return candidate.resolve()
    configured_root = yaml_data.get("path")
    base = yaml_path.parent
    if configured_root:
        root = Path(str(configured_root).replace("\\", os.sep))
        base = root if root.is_absolute() else (yaml_path.parent / root)
    return (base / candidate).resolve()


def _split_image_directories(
    yaml_path: Path,
    yaml_data: dict[str, Any],
) -> dict[str, list[Path]]:
    directories: dict[str, list[Path]] = {}
    for yaml_key, split in SPLIT_ALIASES.items():
        if yaml_key not in yaml_data or split in directories:
            continue
        raw_locations = yaml_data[yaml_key]
        locations = raw_locations if isinstance(raw_locations, list) else [raw_locations]
        directories[split] = [
            _resolve_yaml_path(location, yaml_path, yaml_data)
            for location in locations
        ]
    return directories


def _class_names(yaml_data: dict[str, Any]) -> list[str]:
    names = yaml_data.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(value) for value in names]
    return []


def _label_directory(image_directory: Path) -> Path:
    if image_directory.name.casefold() == "images":
        return image_directory.parent / "labels"
    return image_directory.parent / "labels"


def _issue(
    issues: list[dict[str, Any]],
    *,
    severity: str,
    category: str,
    split: str = "",
    file: str = "",
    line_number: int | str = "",
    error_code: str,
    message: str,
) -> None:
    issues.append(
        {
            "severity": severity,
            "category": category,
            "split": split,
            "file": file,
            "line_number": line_number,
            "error_code": error_code,
            "message": message,
        }
    )


def _difference_hash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = grayscale.tobytes()
    value = 0
    bit = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            if pixels[offset + column] > pixels[offset + column + 1]:
                value |= 1 << bit
            bit += 1
    return value


def _inspect_image(record: ImageRecord, issues: list[dict[str, Any]]) -> None:
    if record.size_bytes == 0:
        _issue(
            issues,
            severity="error",
            category="image",
            split=record.split,
            file=record.relative_path,
            error_code="zero_byte_image",
            message="Image file is empty.",
        )
        return

    try:
        record.sha256 = sha256_file(record.path)
        with Image.open(record.path) as image:
            record.width, record.height = image.size
            record.image_format = str(image.format or "")
            image.verify()
        with Image.open(record.path) as image:
            record.difference_hash = _difference_hash(image)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        _issue(
            issues,
            severity="error",
            category="image",
            split=record.split,
            file=record.relative_path,
            error_code="unreadable_image",
            message=f"Image could not be decoded: {exc}",
        )
        return

    if min(record.width or 0, record.height or 0) < 8:
        _issue(
            issues,
            severity="warning",
            category="image",
            split=record.split,
            file=record.relative_path,
            error_code="unusually_small_image",
            message=f"Image dimensions are {record.width}x{record.height}.",
        )


def _polygon_area(coordinates: list[float]) -> float:
    points = list(zip(coordinates[0::2], coordinates[1::2]))
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _validate_label(
    record: ImageRecord,
    class_names: list[str],
    issues: list[dict[str, Any]],
    class_objects: dict[str, Counter[int]],
    class_images: dict[str, dict[int, set[str]]],
    repo_root: Path,
) -> tuple[int, bool]:
    label_path = record.label_path
    if label_path is None:
        return 0, False

    try:
        record.label_sha256 = sha256_file(label_path)
        text = label_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        _issue(
            issues,
            severity="error",
            category="label",
            split=record.split,
            file=normalize_path(label_path, repo_root),
            error_code="inaccessible_label",
            message=f"Label could not be read: {exc}",
        )
        return 0, False

    lines = [(number, line.strip()) for number, line in enumerate(text.splitlines(), 1)]
    populated = [(number, line) for number, line in lines if line]
    if not populated:
        _issue(
            issues,
            severity="warning",
            category="label",
            split=record.split,
            file=normalize_path(label_path, repo_root),
            error_code="empty_label",
            message="Label file contains no annotations.",
        )
        return 0, True

    invalid_rows: set[int] = set()
    object_count = 0
    for line_number, line in populated:
        tokens = line.split()
        if len(tokens) < 7 or (len(tokens) - 1) % 2 != 0:
            invalid_rows.add(line_number)
            _issue(
                issues,
                severity="error",
                category="annotation",
                split=record.split,
                file=normalize_path(label_path, repo_root),
                line_number=line_number,
                error_code="invalid_coordinate_count",
                message="A segmentation row requires a class and at least three x/y points.",
            )
            continue

        try:
            class_id = int(tokens[0])
        except ValueError:
            invalid_rows.add(line_number)
            _issue(
                issues,
                severity="error",
                category="annotation",
                split=record.split,
                file=normalize_path(label_path, repo_root),
                line_number=line_number,
                error_code="invalid_class_id",
                message="Class ID must be an integer.",
            )
            continue

        class_valid = 0 <= class_id < len(class_names)
        if not class_valid:
            invalid_rows.add(line_number)
            _issue(
                issues,
                severity="error",
                category="annotation",
                split=record.split,
                file=normalize_path(label_path, repo_root),
                line_number=line_number,
                error_code="class_id_out_of_range",
                message=f"Class ID {class_id} is not configured.",
            )

        try:
            coordinates = [float(value) for value in tokens[1:]]
        except ValueError:
            invalid_rows.add(line_number)
            _issue(
                issues,
                severity="error",
                category="annotation",
                split=record.split,
                file=normalize_path(label_path, repo_root),
                line_number=line_number,
                error_code="non_numeric_coordinate",
                message="Polygon coordinates must be numeric.",
            )
            continue

        if any(not math.isfinite(value) for value in coordinates):
            invalid_rows.add(line_number)
            _issue(
                issues,
                severity="error",
                category="annotation",
                split=record.split,
                file=normalize_path(label_path, repo_root),
                line_number=line_number,
                error_code="non_finite_coordinate",
                message="Polygon contains NaN or infinite coordinates.",
            )
            continue

        if any(value < 0.0 or value > 1.0 for value in coordinates):
            invalid_rows.add(line_number)
            _issue(
                issues,
                severity="error",
                category="annotation",
                split=record.split,
                file=normalize_path(label_path, repo_root),
                line_number=line_number,
                error_code="coordinate_out_of_range",
                message="Normalized polygon coordinates must be within [0, 1].",
            )

        if _polygon_area(coordinates) <= 1e-12:
            invalid_rows.add(line_number)
            _issue(
                issues,
                severity="error",
                category="annotation",
                split=record.split,
                file=normalize_path(label_path, repo_root),
                line_number=line_number,
                error_code="zero_area_polygon",
                message="Polygon area is zero.",
            )

        if class_valid:
            object_count += 1
            record.class_ids.add(class_id)
            class_objects[record.split][class_id] += 1
            class_images[record.split][class_id].add(record.relative_path)

    return len(invalid_rows), False


def _paths_for_split(
    split: str,
    directories: list[Path],
    extensions: set[str],
    repo_root: Path,
    issues: list[dict[str, Any]],
) -> tuple[list[ImageRecord], list[Path]]:
    images: list[Path] = []
    labels: list[Path] = []
    for image_directory in directories:
        if not image_directory.exists():
            _issue(
                issues,
                severity="error",
                category="dataset",
                split=split,
                file=normalize_path(image_directory, repo_root),
                error_code="missing_image_directory",
                message="Configured image directory does not exist.",
            )
            continue
        if image_directory.is_file():
            _issue(
                issues,
                severity="error",
                category="dataset",
                split=split,
                file=normalize_path(image_directory, repo_root),
                error_code="unsupported_image_list",
                message="Text-file image lists are not supported by this audit version.",
            )
            continue
        images.extend(
            path
            for path in image_directory.iterdir()
            if path.is_file() and path.suffix.casefold() in extensions
        )
        label_directory = _label_directory(image_directory)
        if label_directory.exists():
            labels.extend(
                path
                for path in label_directory.iterdir()
                if path.is_file() and path.suffix.casefold() == ".txt"
            )
        else:
            _issue(
                issues,
                severity="error",
                category="dataset",
                split=split,
                file=normalize_path(label_directory, repo_root),
                error_code="missing_label_directory",
                message="Expected label directory does not exist.",
            )

    records = [
        ImageRecord(
            split=split,
            path=path.resolve(),
            relative_path=normalize_path(path, repo_root),
            stem=path.stem,
            size_bytes=path.stat().st_size,
        )
        for path in sorted(set(images), key=lambda item: normalize_path(item).casefold())
    ]
    return records, sorted(set(labels), key=lambda item: normalize_path(item).casefold())


def _pair_records(
    split: str,
    records: list[ImageRecord],
    labels: list[Path],
    repo_root: Path,
    issues: list[dict[str, Any]],
) -> None:
    images_by_stem: dict[str, list[ImageRecord]] = defaultdict(list)
    labels_by_stem: dict[str, list[Path]] = defaultdict(list)
    for record in records:
        images_by_stem[record.stem.casefold()].append(record)
    for label in labels:
        labels_by_stem[label.stem.casefold()].append(label)

    for stem, matches in images_by_stem.items():
        if len(matches) > 1:
            _issue(
                issues,
                severity="error",
                category="pairing",
                split=split,
                file=";".join(record.relative_path for record in matches),
                error_code="duplicate_image_stem",
                message=f"Multiple images use the stem {stem}.",
            )
        label_matches = labels_by_stem.get(stem, [])
        if not label_matches:
            for record in matches:
                _issue(
                    issues,
                    severity="error",
                    category="pairing",
                    split=split,
                    file=record.relative_path,
                    error_code="image_without_label",
                    message="No label file has the same stem.",
                )
        elif len(label_matches) > 1:
            _issue(
                issues,
                severity="error",
                category="pairing",
                split=split,
                file=";".join(normalize_path(path, repo_root) for path in label_matches),
                error_code="duplicate_label_stem",
                message=f"Multiple labels use the stem {stem}.",
            )
        for record in matches:
            if label_matches:
                record.label_path = label_matches[0].resolve()

    for stem, label_matches in labels_by_stem.items():
        if stem not in images_by_stem:
            for label in label_matches:
                _issue(
                    issues,
                    severity="error",
                    category="pairing",
                    split=split,
                    file=normalize_path(label, repo_root),
                    error_code="label_without_image",
                    message="No supported image has the same stem.",
                )


def _duplicate_groups(
    records: list[ImageRecord],
    labels_by_split: dict[str, list[Path]],
    repo_root: Path,
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    image_hashes: dict[str, list[ImageRecord]] = defaultdict(list)
    label_hashes: dict[str, list[tuple[str, Path, ImageRecord | None]]] = defaultdict(list)
    records_by_key = {
        (record.split, record.stem.casefold()): record for record in records
    }
    for record in records:
        if record.sha256:
            image_hashes[record.sha256].append(record)
    for split, label_paths in labels_by_split.items():
        for label_path in label_paths:
            record = records_by_key.get((split, label_path.stem.casefold()))
            try:
                digest = (
                    record.label_sha256
                    if record is not None and record.label_sha256
                    else sha256_file(label_path)
                )
            except OSError as exc:
                _issue(
                    issues,
                    severity="error",
                    category="label",
                    split=split,
                    file=normalize_path(label_path, repo_root),
                    error_code="inaccessible_label",
                    message=f"Label could not be hashed: {exc}",
                )
                continue
            label_hashes[digest].append((split, label_path, record))

    for digest, members in sorted(image_hashes.items()):
        if len(members) < 2:
            continue
        splits = sorted({member.split for member in members})
        label_digests = {member.label_sha256 for member in members if member.label_sha256}
        groups.append(
            {
                "duplicate_type": "image_sha256",
                "sha256": digest,
                "count": len(members),
                "splits": splits,
                "paths": [member.relative_path for member in members],
                "cross_split": len(splits) > 1,
                "conflicting_labels": len(label_digests) > 1,
                "different_images": False,
            }
        )

    for digest, members in sorted(label_hashes.items()):
        if len(members) < 2:
            continue
        splits = sorted({split for split, _, _ in members})
        image_digests = {
            record.sha256
            for _, _, record in members
            if record is not None and record.sha256
        }
        groups.append(
            {
                "duplicate_type": "label_sha256",
                "sha256": digest,
                "count": len(members),
                "splits": splits,
                "paths": [
                    normalize_path(label_path, repo_root)
                    for _, label_path, _ in members
                ],
                "cross_split": len(splits) > 1,
                "conflicting_labels": False,
                "different_images": len(image_digests) > 1,
            }
        )
    return groups


def _augmentation_leakage(records: list[ImageRecord]) -> list[dict[str, Any]]:
    parents: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        parents[record.augmentation_parent_id].append(record)
    leakage: list[dict[str, Any]] = []
    for parent, members in sorted(parents.items()):
        splits = sorted({member.split for member in members})
        if len(splits) < 2:
            continue
        leakage.append(
            {
                "level": "augmentation_parent",
                "group_id": parent,
                "splits": splits,
                "affected_images": len(members),
                "representative_paths": [
                    member.relative_path for member in sorted(
                        members,
                        key=lambda item: item.relative_path,
                    )[:5]
                ],
            }
        )
    return leakage


def _near_duplicate_candidates(
    records: list[ImageRecord],
    threshold: int,
    maximum: int,
) -> tuple[list[dict[str, Any]], bool]:
    valid = [
        record
        for record in records
        if record.difference_hash is not None and record.sha256
    ]
    candidates: list[dict[str, Any]] = []
    truncated = False
    for left_index, left in enumerate(valid):
        for right in valid[left_index + 1:]:
            if left.sha256 == right.sha256:
                continue
            distance = (left.difference_hash ^ right.difference_hash).bit_count()
            if distance > threshold:
                continue
            if len(candidates) >= maximum:
                truncated = True
                return candidates, truncated
            candidates.append(
                {
                    "left": left.relative_path,
                    "right": right.relative_path,
                    "left_split": left.split,
                    "right_split": right.split,
                    "hamming_distance": distance,
                    "cross_split": left.split != right.split,
                    "status": "candidate",
                }
            )
    return candidates, truncated


def _manifest_path(
    raw_path: str,
    manifest_path: Path,
    repo_root: Path,
) -> Path:
    candidate = Path(raw_path.replace("\\", os.sep))
    if candidate.is_absolute():
        return candidate.resolve()
    repo_candidate = (repo_root / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (manifest_path.parent / candidate).resolve()


def _overlap_groups(
    rows: list[dict[str, str]],
    field_name: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = row.get(field_name, "").strip()
        if value:
            groups[value].append(row)
    leakage: list[dict[str, Any]] = []
    for group_id, members in sorted(groups.items()):
        splits = sorted({member.get("split", "").strip() for member in members if member.get("split", "").strip()})
        if len(splits) < 2:
            continue
        leakage.append(
            {
                "level": field_name.removesuffix("_id"),
                "group_id": group_id,
                "splits": splits,
                "affected_images": len(members),
                "representative_paths": [
                    member.get("file_path", "") for member in members[:5]
                ],
            }
        )
    return leakage


def _read_metadata_manifest(
    manifest_path: Path | None,
    records: list[ImageRecord],
    repo_root: Path,
    issues: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any], list[dict[str, Any]]]:
    if manifest_path is None:
        return (
            [],
            {
                "status": STATUS_UNAVAILABLE,
                "reason": "Specimen identity metadata was not supplied.",
                "specimen_check": {"status": STATUS_UNAVAILABLE},
                "capture_session_check": {"status": STATUS_UNAVAILABLE},
                "recording_check": {"status": STATUS_UNAVAILABLE},
                "source_frame_check": {"status": STATUS_UNAVAILABLE},
                "specimen_id_coverage": 0.0,
                "capture_session_id_coverage": 0.0,
                "recording_id_coverage": 0.0,
                "source_frame_id_coverage": 0.0,
                "expert_review_coverage": 0.0,
            },
            [],
        )

    if not manifest_path.exists():
        _issue(
            issues,
            severity="error",
            category="manifest",
            file=normalize_path(manifest_path, repo_root),
            error_code="missing_metadata_manifest",
            message="Configured metadata manifest does not exist.",
        )
        return [], {
            "status": STATUS_UNAVAILABLE,
            "reason": "Configured metadata manifest was not found.",
        }, []

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        missing_headers = [
            field for field in REQUIRED_MANIFEST_FIELDS
            if field not in headers
        ]
        rows = [
            {key: str(value or "") for key, value in row.items()}
            for row in reader
        ]

    if missing_headers:
        _issue(
            issues,
            severity="error",
            category="manifest",
            file=normalize_path(manifest_path, repo_root),
            error_code="missing_manifest_columns",
            message=f"Missing manifest columns: {', '.join(missing_headers)}",
        )

    sample_ids = Counter(row.get("sample_id", "").strip() for row in rows)
    for sample_id, count in sorted(sample_ids.items()):
        if sample_id and count > 1:
            _issue(
                issues,
                severity="error",
                category="manifest",
                file=normalize_path(manifest_path, repo_root),
                error_code="duplicate_sample_id",
                message=f"Sample ID {sample_id} occurs {count} times.",
            )

    records_by_path = {record.path.resolve(): record for record in records}
    for row in rows:
        raw_image = row.get("file_path", "").strip()
        if not raw_image:
            _issue(
                issues,
                severity="error",
                category="manifest",
                error_code="missing_manifest_image_path",
                message="Manifest row does not provide file_path.",
            )
            continue
        image_path = _manifest_path(raw_image, manifest_path, repo_root)
        if not image_path.exists():
            _issue(
                issues,
                severity="error",
                category="manifest",
                file=raw_image,
                error_code="manifest_image_missing",
                message="Manifest image path does not exist.",
            )
            continue
        record = records_by_path.get(image_path)
        if record is not None and row.get("split", "").strip() != record.split:
            _issue(
                issues,
                severity="error",
                category="manifest",
                split=record.split,
                file=raw_image,
                error_code="manifest_split_mismatch",
                message=f"Manifest split is {row.get('split', '')}; physical split is {record.split}.",
            )

        raw_label = row.get("label_path", "").strip()
        if raw_label and not _manifest_path(raw_label, manifest_path, repo_root).exists():
            _issue(
                issues,
                severity="error",
                category="manifest",
                file=raw_label,
                error_code="manifest_label_missing",
                message="Manifest label path does not exist.",
            )

    leakage: list[dict[str, Any]] = []
    check_fields = (
        "specimen_id",
        "capture_session_id",
        "recording_id",
        "source_frame_id",
        "augmentation_parent_id",
    )
    row_count = len(rows)

    def coverage(field_name: str, accepted: set[str] | None = None) -> float:
        if not row_count:
            return 0.0
        populated = 0
        for row in rows:
            value = row.get(field_name, "").strip()
            if accepted is None:
                populated += bool(value)
            else:
                populated += value.casefold() in accepted
        return round(100.0 * populated / row_count, 2)

    checks: dict[str, Any] = {
        "status": STATUS_PASS,
        "row_count": row_count,
        "specimen_id_coverage": coverage("specimen_id"),
        "capture_session_id_coverage": coverage("capture_session_id"),
        "recording_id_coverage": coverage("recording_id"),
        "source_frame_id_coverage": coverage("source_frame_id"),
        "expert_review_coverage": coverage(
            "expert_review_status",
            {"approved", "corrected", "rejected", "needs_second_review"},
        ),
    }
    for field_name in check_fields:
        available = any(row.get(field_name, "").strip() for row in rows)
        field_leakage = _overlap_groups(rows, field_name) if available else []
        checks[field_name.removesuffix("_id") + "_check"] = (
            {
                "status": "fail" if field_leakage else STATUS_PASS,
                "leakage_group_count": len(field_leakage),
            }
            if available
            else {
                "status": STATUS_UNAVAILABLE,
                "reason": f"{field_name} metadata was not supplied.",
            }
        )
        leakage.extend(field_leakage)

    if checks["specimen_check"]["status"] == STATUS_UNAVAILABLE:
        checks["status"] = STATUS_UNAVAILABLE
        checks["reason"] = "Specimen identity metadata was not supplied."
    elif leakage:
        checks["status"] = "fail"
    return rows, checks, leakage


def _assign_proposed_splits(
    records: list[ImageRecord],
    metadata_rows: list[dict[str, str]],
    repo_root: Path,
    ratios: dict[str, float],
    seed: int,
) -> tuple[list[dict[str, Any]], str, str]:
    metadata_by_path: dict[Path, dict[str, str]] = {}
    for row in metadata_rows:
        raw_path = row.get("file_path", "").strip()
        if raw_path:
            candidate = Path(raw_path.replace("\\", os.sep))
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            metadata_by_path[candidate.resolve()] = row

    def complete(field_name: str) -> bool:
        return bool(records) and all(
            metadata_by_path.get(record.path, {}).get(field_name, "").strip()
            for record in records
        )

    if complete("specimen_id"):
        grouping_level = "specimen"
        key_for = lambda item: metadata_by_path[item.path]["specimen_id"].strip()
        note = "Grouped by supplied physical specimen identity."
    elif complete("recording_id"):
        grouping_level = "recording"
        key_for = lambda item: metadata_by_path[item.path]["recording_id"].strip()
        note = "Specimen identity was incomplete; grouped by recording identity."
    else:
        grouping_level = "augmentation_parent"
        key_for = lambda item: item.augmentation_parent_id
        note = (
            "Grouped by augmentation parent only. This proposal does not prove "
            "that physical specimens are independent across splits."
        )

    normalized_ratios = {
        split: max(0.0, float(ratios.get(split, 0.0)))
        for split in ("train", "valid", "test")
    }
    ratio_total = sum(normalized_ratios.values())
    if ratio_total <= 0:
        normalized_ratios = {"train": 0.8, "valid": 0.1, "test": 0.1}
        ratio_total = 1.0
    normalized_ratios = {
        split: value / ratio_total for split, value in normalized_ratios.items()
    }

    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[str(key_for(record))].append(record)
    group_items = sorted(groups.items(), key=lambda item: item[0])
    deterministic_random(seed).shuffle(group_items)

    targets = {
        split: normalized_ratios[split] * len(records)
        for split in normalized_ratios
    }
    assigned_counts = {split: 0 for split in normalized_ratios}
    assignment: dict[str, str] = {}
    for group_id, members in group_items:
        destination = max(
            ("train", "valid", "test"),
            key=lambda split: (
                targets[split] - assigned_counts[split],
                normalized_ratios[split],
                split,
            ),
        )
        assignment[group_id] = destination
        assigned_counts[destination] += len(members)

    proposed = []
    for record in sorted(records, key=lambda item: item.relative_path):
        group_id = str(key_for(record))
        proposed.append(
            {
                "sample_id": metadata_by_path.get(record.path, {}).get(
                    "sample_id",
                    record.stem,
                ),
                "file_path": record.relative_path,
                "current_split": record.split,
                "proposed_split": assignment[group_id],
                "grouping_level": grouping_level,
                "group_id": group_id,
            }
        )
    return proposed, grouping_level, note


def _write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    columns: Iterable[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    column_list = list(columns)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=column_list, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: ";".join(str(item) for item in value)
                        if isinstance(value, list)
                        else value
                        for key, value in row.items()
                    }
                )
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _summary_markdown(report: dict[str, Any]) -> str:
    readiness = report["claim_readiness"]
    lines = [
        "# Dataset Integrity Audit",
        "",
        f"- Audit status: {report['audit_status']}",
        f"- Claim ready: {str(readiness['claim_ready']).lower()}",
        f"- Images: {report['totals']['images']}",
        f"- Labels: {report['totals']['labels']}",
        f"- Exact duplicate image groups: {report['totals']['exact_duplicate_image_groups']}",
        f"- Cross-split augmentation parents: {report['totals']['augmentation_parent_leakage_groups']}",
        f"- Images affected by augmentation leakage: {report['totals']['augmentation_leakage_affected_images']}",
        "",
        "## Claim Boundary",
        "",
        "This audit evaluates dataset integrity. It does not establish model accuracy, "
        "expert ground truth, or physical specimen independence when specimen metadata "
        "is unavailable.",
        "",
        "## Blocking Reasons",
        "",
    ]
    if readiness["blocking_reasons"]:
        lines.extend(f"- {reason}" for reason in readiness["blocking_reasons"])
    else:
        lines.append("- None")
    lines.extend(["", "## Unavailable Checks", ""])
    if readiness["unavailable_checks"]:
        lines.extend(f"- {reason}" for reason in readiness["unavailable_checks"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Class Semantics",
            "",
            "Configured class names are reported exactly as supplied. The meaning of "
            "colour-named classes requires research-team or expert confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def _research_manifest(
    *,
    repo_root: Path,
    yaml_path: Path,
    metadata_manifest: Path | None,
    config: dict[str, Any],
    command: str,
    started_at: str,
    ended_at: str,
    output_files: list[Path],
) -> dict[str, Any]:
    tool_sources = [
        Path(__file__).resolve(),
        Path(__file__).with_name("common.py").resolve(),
        (repo_root / "backend" / "research_benchmark.py").resolve(),
    ]
    config_path = _resolve_from_root(config.get("_config_path"), repo_root)
    return {
        "tool": "quartz_dataset_integrity_audit",
        "tool_version": TOOL_VERSION,
        "status": STATUS_PASS,
        "started_utc": started_at,
        "ended_utc": ended_at,
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "command": command,
        "deterministic_seed": int(config.get("deterministic_seed", 42)),
        "git": git_evidence(repo_root),
        "tool_sources": [
            {
                "path": normalize_path(source, repo_root),
                "sha256": sha256_file(source),
            }
            for source in tool_sources
            if source.exists()
        ],
        "inputs": {
            "configuration": (
                {
                    "path": normalize_path(config_path, repo_root),
                    "sha256": sha256_file(config_path),
                }
                if config_path is not None and config_path.exists()
                else {
                    "status": STATUS_UNAVAILABLE,
                    "reason": "Audit configuration path was not recorded.",
                }
            ),
            "dataset_yaml": {
                "path": normalize_path(yaml_path, repo_root),
                "sha256": sha256_file(yaml_path),
            },
            "dataset_manifest": (
                {
                    "path": normalize_path(metadata_manifest, repo_root),
                    "sha256": sha256_file(metadata_manifest),
                }
                if metadata_manifest is not None and metadata_manifest.exists()
                else {
                    "status": STATUS_UNAVAILABLE,
                    "reason": "Dataset metadata manifest was not supplied.",
                }
            ),
        },
        "outputs": [
            {
                "path": output.name,
                "sha256": sha256_file(output),
                "size_bytes": output.stat().st_size,
            }
            for output in sorted(output_files, key=lambda item: item.name)
            if output.exists() and output.name != "manifest.json"
        ],
    }


def audit_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Audit a dataset in memory without writing or changing source files."""

    started_at = utc_timestamp()
    repo_root = Path(config.get("_repo_root") or Path.cwd()).resolve()
    yaml_path = _resolve_from_root(config.get("dataset_yaml"), repo_root)
    if yaml_path is None or not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    yaml_data = load_dataset_yaml(yaml_path)
    class_names = _class_names(yaml_data)
    extensions = {
        str(value).casefold()
        for value in config.get("supported_image_extensions", DEFAULT_EXTENSIONS)
    }
    extensions = {
        value if value.startswith(".") else f".{value}" for value in extensions
    }

    issues: list[dict[str, Any]] = []
    split_directories = _split_image_directories(yaml_path, yaml_data)
    records_by_split: dict[str, list[ImageRecord]] = {}
    labels_by_split: dict[str, list[Path]] = {}
    for split in ("train", "valid", "test"):
        directories = split_directories.get(split, [])
        if not directories:
            _issue(
                issues,
                severity="error",
                category="dataset",
                split=split,
                error_code="missing_split_configuration",
                message=f"No image path is configured for {split}.",
            )
        records, labels = _paths_for_split(
            split,
            directories,
            extensions,
            repo_root,
            issues,
        )
        _pair_records(split, records, labels, repo_root, issues)
        records_by_split[split] = records
        labels_by_split[split] = labels

    all_records = [
        record
        for split in ("train", "valid", "test")
        for record in records_by_split[split]
    ]
    class_objects = {
        split: Counter() for split in ("train", "valid", "test")
    }
    class_images = {
        split: defaultdict(set) for split in ("train", "valid", "test")
    }
    invalid_annotation_rows = 0
    empty_labels = 0
    for record in all_records:
        _inspect_image(record, issues)
        invalid_rows, is_empty = _validate_label(
            record,
            class_names,
            issues,
            class_objects,
            class_images,
            repo_root,
        )
        invalid_annotation_rows += invalid_rows
        empty_labels += int(is_empty)

    duplicate_groups = _duplicate_groups(
        all_records,
        labels_by_split,
        repo_root,
        issues,
    )
    augmentation_leakage = _augmentation_leakage(all_records)

    near_config = config.get("near_duplicate_detection", {})
    near_enabled = bool(near_config.get("enabled", True))
    near_candidates: list[dict[str, Any]] = []
    near_truncated = False
    if near_enabled:
        near_candidates, near_truncated = _near_duplicate_candidates(
            all_records,
            threshold=max(0, int(near_config.get("hamming_distance_threshold", 5))),
            maximum=max(1, int(near_config.get("max_candidates", 5000))),
        )

    manifest_path = _resolve_from_root(config.get("metadata_manifest"), repo_root)
    metadata_rows, metadata_status, metadata_leakage = _read_metadata_manifest(
        manifest_path,
        all_records,
        repo_root,
        issues,
    )
    leakage_groups = augmentation_leakage + metadata_leakage

    proposed, grouping_level, grouping_note = _assign_proposed_splits(
        all_records,
        metadata_rows,
        repo_root,
        config.get(
            "proposed_split_ratios",
            {"train": 0.8, "valid": 0.1, "test": 0.1},
        ),
        int(config.get("deterministic_seed", 42)),
    )

    summary_rows = []
    for split in ("train", "valid", "test"):
        split_issues = [issue for issue in issues if issue["split"] == split]
        summary_rows.append(
            {
                "split": split,
                "image_count": len(records_by_split[split]),
                "label_count": len(labels_by_split[split]),
                "error_count": sum(issue["severity"] == "error" for issue in split_issues),
                "warning_count": sum(issue["severity"] == "warning" for issue in split_issues),
                "empty_label_count": sum(
                    issue["error_code"] == "empty_label" for issue in split_issues
                ),
            }
        )

    class_distribution = []
    for split in ("train", "valid", "test"):
        for class_id, class_name in enumerate(class_names):
            class_distribution.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "object_count": class_objects[split][class_id],
                    "image_count": len(class_images[split][class_id]),
                }
            )

    exact_image_groups = [
        group for group in duplicate_groups
        if group["duplicate_type"] == "image_sha256"
    ]
    cross_split_exact = [group for group in exact_image_groups if group["cross_split"]]
    corrupt_or_unreadable = [
        issue for issue in issues
        if issue["error_code"] in {"zero_byte_image", "unreadable_image"}
    ]
    pairing_errors = [
        issue for issue in issues
        if issue["category"] == "pairing" and issue["severity"] == "error"
    ]
    manifest_errors = [
        issue for issue in issues
        if issue["category"] == "manifest" and issue["severity"] == "error"
    ]

    blockers: list[str] = []
    warnings: list[str] = []
    unavailable_checks: list[str] = []
    if augmentation_leakage:
        blockers.append(
            f"{len(augmentation_leakage)} augmentation-parent IDs cross dataset splits."
        )
    for level in ("specimen", "capture_session", "recording", "source_frame"):
        matches = [group for group in metadata_leakage if group["level"] == level]
        if matches:
            blockers.append(f"{len(matches)} {level} IDs cross dataset splits.")
    if cross_split_exact:
        blockers.append(
            f"{len(cross_split_exact)} exact duplicate image groups cross dataset splits."
        )
    if corrupt_or_unreadable:
        blockers.append(f"{len(corrupt_or_unreadable)} images are corrupt or unreadable.")
    if invalid_annotation_rows:
        blockers.append(f"{invalid_annotation_rows} annotation rows are invalid.")
    if pairing_errors:
        blockers.append(f"{len(pairing_errors)} image/label pairing errors were found.")
    if manifest_errors:
        blockers.append(f"{len(manifest_errors)} metadata manifest errors were found.")
    if metadata_status.get("specimen_check", {}).get("status") == STATUS_UNAVAILABLE:
        unavailable_checks.append("Specimen identity metadata was not supplied.")
        blockers.append(
            "Specimen-level independence cannot be established without specimen metadata."
        )
    if exact_image_groups and not cross_split_exact:
        warnings.append(
            f"{len(exact_image_groups)} exact duplicate image groups exist within splits."
        )
    if near_candidates:
        warnings.append(
            f"{len(near_candidates)} near-duplicate candidates require review."
        )
    if near_truncated:
        warnings.append("Near-duplicate candidate output reached the configured limit.")
    if empty_labels:
        warnings.append(f"{empty_labels} empty label files were found.")

    roboflow = yaml_data.get("roboflow", {})
    source_metadata = dict(config.get("source_metadata", {}))
    license_metadata = dict(config.get("license_metadata", {}))
    if isinstance(roboflow, dict):
        source_metadata = {**roboflow, **source_metadata}
        if roboflow.get("license") and not license_metadata.get("name"):
            license_metadata["name"] = roboflow["license"]
    if not source_metadata:
        blockers.append("Critical dataset source provenance is missing.")
    if not license_metadata.get("name"):
        blockers.append("Dataset license metadata is missing.")

    issue_error_count = sum(issue["severity"] == "error" for issue in issues)
    claim_readiness = {
        "claim_ready": not blockers,
        "blocking_reasons": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "unavailable_checks": list(dict.fromkeys(unavailable_checks)),
    }
    return {
        "tool": "quartz_dataset_integrity_audit",
        "tool_version": TOOL_VERSION,
        "started_utc": started_at,
        "ended_utc": utc_timestamp(),
        "audit_status": "fail" if issue_error_count else (
            "warning" if issues or blockers or warnings else STATUS_PASS
        ),
        "dataset": {
            "yaml_path": normalize_path(yaml_path, repo_root),
            "yaml_sha256": sha256_file(yaml_path),
            "class_names": class_names,
            "source_metadata": source_metadata,
            "license_metadata": license_metadata,
            "class_semantics_note": (
                "Class names are reported as configured. The meaning of colour-named "
                "classes requires research-team or expert confirmation."
            ),
        },
        "splits": summary_rows,
        "totals": {
            "images": len(all_records),
            "labels": sum(len(value) for value in labels_by_split.values()),
            "issues": len(issues),
            "errors": issue_error_count,
            "warnings": sum(issue["severity"] == "warning" for issue in issues),
            "empty_labels": empty_labels,
            "invalid_annotation_rows": invalid_annotation_rows,
            "exact_duplicate_image_groups": len(exact_image_groups),
            "exact_duplicate_label_groups": sum(
                group["duplicate_type"] == "label_sha256"
                for group in duplicate_groups
            ),
            "cross_split_exact_duplicate_groups": len(cross_split_exact),
            "augmentation_parent_leakage_groups": len(augmentation_leakage),
            "augmentation_leakage_affected_images": sum(
                group["affected_images"] for group in augmentation_leakage
            ),
            "near_duplicate_candidates": len(near_candidates),
        },
        "issues": issues,
        "duplicate_groups": duplicate_groups,
        "near_duplicate_candidates": near_candidates,
        "near_duplicate_candidates_truncated": near_truncated,
        "leakage_groups": leakage_groups,
        "class_distribution": class_distribution,
        "metadata_validation": metadata_status,
        "claim_readiness": claim_readiness,
        "proposed_split": {
            "grouping_level": grouping_level,
            "note": grouping_note,
            "assignments": proposed,
        },
    }


def write_audit_outputs(
    report: dict[str, Any],
    config: dict[str, Any],
    output_directory: str | Path,
    command: str,
) -> dict[str, Path]:
    """Write the compact, provenance-rich audit evidence bundle."""

    repo_root = Path(config.get("_repo_root") or Path.cwd()).resolve()
    output_path = Path(output_directory)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    files = {
        "audit": output_path / "dataset_audit.json",
        "summary": output_path / "dataset_summary.csv",
        "issues": output_path / "dataset_issues.csv",
        "duplicates": output_path / "duplicate_groups.csv",
        "leakage": output_path / "leakage_groups.csv",
        "classes": output_path / "class_distribution.csv",
        "proposed_split": output_path / "proposed_split.csv",
        "markdown": output_path / "audit_summary.md",
        "manifest": output_path / "manifest.json",
    }

    atomic_write_json(files["audit"], report)
    _write_csv(
        files["summary"],
        report["splits"],
        (
            "split",
            "image_count",
            "label_count",
            "error_count",
            "warning_count",
            "empty_label_count",
        ),
    )
    _write_csv(files["issues"], report["issues"], ISSUE_COLUMNS)
    _write_csv(
        files["duplicates"],
        report["duplicate_groups"],
        (
            "duplicate_type",
            "sha256",
            "count",
            "splits",
            "paths",
            "cross_split",
            "conflicting_labels",
            "different_images",
        ),
    )
    _write_csv(
        files["leakage"],
        report["leakage_groups"],
        ("level", "group_id", "splits", "affected_images", "representative_paths"),
    )
    _write_csv(
        files["classes"],
        report["class_distribution"],
        ("split", "class_id", "class_name", "object_count", "image_count"),
    )
    _write_csv(
        files["proposed_split"],
        report["proposed_split"]["assignments"],
        (
            "sample_id",
            "file_path",
            "current_split",
            "proposed_split",
            "grouping_level",
            "group_id",
        ),
    )
    files["markdown"].write_text(
        _summary_markdown(report),
        encoding="utf-8",
        newline="\n",
    )

    yaml_path = _resolve_from_root(config.get("dataset_yaml"), repo_root)
    if yaml_path is None:
        raise ValueError("dataset_yaml is required")
    metadata_manifest = _resolve_from_root(config.get("metadata_manifest"), repo_root)
    manifest = _research_manifest(
        repo_root=repo_root,
        yaml_path=yaml_path,
        metadata_manifest=metadata_manifest,
        config=config,
        command=command,
        started_at=report["started_utc"],
        ended_at=utc_timestamp(),
        output_files=list(files.values()),
    )
    atomic_write_json(files["manifest"], manifest)
    return files


def run_dataset_audit(
    config: dict[str, Any],
    output_override: str | Path | None = None,
    command: str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Run the read-only audit and write a generated evidence bundle."""

    report = audit_dataset(config)
    output = output_override or config.get("output_directory")
    if not output:
        raise ValueError("An output directory is required.")
    files = write_audit_outputs(
        report,
        config,
        output,
        command or " ".join(sys.argv),
    )
    return report, files
