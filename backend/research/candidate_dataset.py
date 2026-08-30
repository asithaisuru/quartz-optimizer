"""Gate 1 candidate registration and gem-grouped development planning."""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .common import atomic_write_json, git_evidence, sha256_file, utc_timestamp
from .dataset_audit import load_dataset_yaml
from .readiness_common import (
    atomic_write_csv,
    atomic_write_text,
    load_json_config,
    resolve_path,
    stable_json_sha256,
    write_manifest,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
SOURCE_SPLITS = ("train", "valid", "test")
DEVELOPMENT_SPLITS = ("train", "valid")
DEFAULT_LINEAGE_PATTERN = (
    r"^Gem_(?P<gem_id>\d+)_Vid_(?P<video_id>\d+)_"
    r"Fr_(?P<frame_id>\d+)_jpg\.rf\.(?P<export_id>[^.]+)$"
)
GEM_IDENTITY_RULE = (
    "All images sharing the same Gem_<number> filename prefix represent "
    "the same physical gemstone."
)
MODEL_STATUS = "legacy_or_exploratory_only"

FILE_MANIFEST_COLUMNS = (
    "relative_path",
    "category",
    "size_bytes",
    "sha256",
)
LINEAGE_COLUMNS = (
    "sample_id",
    "source_image_path",
    "source_label_path",
    "relative_image_path",
    "relative_label_path",
    "image_sha256",
    "label_sha256",
    "gem_id",
    "specimen_id",
    "video_id",
    "frame_id",
    "export_id",
    "original_split",
    "class_ids",
    "object_count",
    "difference_hash",
)
DEVELOPMENT_MANIFEST_COLUMNS = (
    *LINEAGE_COLUMNS[:-1],
    "assigned_split",
    "atomic_group_id",
    "final_test_eligible",
)
CLASS_DISTRIBUTION_COLUMNS = (
    "split",
    "class_id",
    "class_name",
    "image_count",
    "object_count",
)
GEM_DISTRIBUTION_COLUMNS = (
    "gem_id",
    "specimen_id",
    "atomic_group_id",
    "assigned_split",
    "image_count",
    "video_count",
    "object_count",
    "class_object_counts",
)
PERCEPTUAL_COLUMNS = (
    "candidate_id",
    "left_sample_id",
    "right_sample_id",
    "left_gem_id",
    "right_gem_id",
    "left_original_split",
    "right_original_split",
    "hamming_distance",
    "same_gem",
)
CANDIDATE_TAXONOMY_COLUMNS = (
    "class_id",
    "class_name",
    "semantic_definition",
    "taxonomy_status",
    "is_visible_defect",
    "eligible_for_visualization",
    "eligible_for_3d_mapping",
    "eligible_for_no_cut_zone",
    "expert_confirmed",
    "confirmed_by",
    "confirmed_date",
    "notes",
)


def development_manifest_sha256(rows: Iterable[dict[str, Any]]) -> str:
    """Hash the exact CSV bytes used for the grouped development manifest."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(DEVELOPMENT_MANIFEST_COLUMNS),
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
        )
    return hashlib.sha256(output.getvalue().encode("utf-8")).hexdigest()
EXTERNAL_TEST_PLAN_COLUMNS = (
    "specimen_id",
    "specimen_source",
    "legacy_candidate_exclusion_confirmed",
    "exclusion_confirmed_by",
    "planned_capture_date",
    "planned_image_count",
    "mask_creator",
    "expert_reviewer",
    "status",
    "notes",
)


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        first = self.find(left)
        second = self.find(right)
        if first != second:
            self.parent[max(first, second)] = min(first, second)


def load_candidate_config(
    path: str | Path,
    *,
    source_override: str | Path | None = None,
    zip_override: str | Path | None = None,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    config = load_json_config(path, output_override=output_override)
    if source_override is not None:
        config["_source_override"] = str(source_override)
    if zip_override is not None:
        config["_zip_override"] = str(zip_override)
    return config


def _configured_external_path(
    config: dict[str, Any],
    value_key: str,
    environment_key: str,
    override_key: str,
) -> Path | None:
    value = config.get(override_key)
    if value in (None, ""):
        environment_name = config.get(environment_key)
        if environment_name:
            value = os.environ.get(str(environment_name))
    if value in (None, ""):
        value = config.get(value_key)
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(config["_repo_root"]) / path
    return path.resolve()


def candidate_source_root(config: dict[str, Any]) -> Path | None:
    return _configured_external_path(
        config,
        "source_root",
        "source_root_env",
        "_source_override",
    )


def candidate_zip_path(config: dict[str, Any]) -> Path | None:
    return _configured_external_path(
        config,
        "candidate_zip",
        "candidate_zip_env",
        "_zip_override",
    )


def parse_lineage(
    filename: str | Path,
    pattern: str = DEFAULT_LINEAGE_PATTERN,
) -> dict[str, str] | None:
    match = re.fullmatch(pattern, Path(filename).stem, flags=re.IGNORECASE)
    if match is None:
        return None
    values = match.groupdict()
    required = {"gem_id", "video_id", "frame_id", "export_id"}
    if not required.issubset(values) or any(values[key] in (None, "") for key in required):
        return None
    return {
        "gem_id": str(int(values["gem_id"])),
        "video_id": str(int(values["video_id"])),
        "frame_id": str(int(values["frame_id"])),
        "export_id": values["export_id"],
    }


def _class_names(yaml_data: dict[str, Any]) -> list[str]:
    names = yaml_data.get("names", [])
    if isinstance(names, dict):
        return [
            str(names[key])
            for key in sorted(names, key=lambda item: int(item))
        ]
    if isinstance(names, list):
        return [str(value) for value in names]
    return []


def _file_manifest(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows = []
    digest = hashlib.sha256()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        file_digest = sha256_file(path)
        size = path.stat().st_size
        category = "metadata"
        if path.parent.name.casefold() == "images":
            category = "image"
        elif path.parent.name.casefold() == "labels":
            category = "label"
        rows.append(
            {
                "relative_path": relative,
                "category": category,
                "size_bytes": size,
                "sha256": file_digest,
            }
        )
        digest.update(f"{relative}\0{size}\0{file_digest}\n".encode("utf-8"))
    return rows, digest.hexdigest()


def _record_set_hash(rows: list[dict[str, Any]], category: str) -> str:
    digest = hashlib.sha256()
    for row in rows:
        if row["category"] != category:
            continue
        digest.update(
            (
                f"{row['relative_path']}\0{row['size_bytes']}\0"
                f"{row['sha256']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _difference_hash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    flattened = getattr(grayscale, "get_flattened_data", None)
    pixels = list(flattened() if flattened is not None else grayscale.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return value


def _inspect_image(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            return {
                "width": int(image.width),
                "height": int(image.height),
                "mode": image.mode,
                "difference_hash": f"{_difference_hash(image):016x}",
            }, None
    except Exception as exc:  # Pillow reports several format-specific errors.
        return {}, f"{type(exc).__name__}: {exc}"


def _inspect_label(
    path: Path,
    class_count: int,
) -> tuple[Counter[int], list[dict[str, Any]], bool]:
    counts: Counter[int] = Counter()
    issues = []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        return counts, [{"line": 0, "reason": f"unreadable label: {exc}"}], False
    empty = not text.strip()
    for line_number, raw in enumerate(text.splitlines(), 1):
        tokens = raw.split()
        if not tokens:
            continue
        if len(tokens) < 7 or (len(tokens) - 1) % 2:
            issues.append(
                {
                    "line": line_number,
                    "reason": (
                        "expected an integer class followed by at least "
                        "three x/y polygon points"
                    ),
                }
            )
            continue
        try:
            class_value = float(tokens[0])
            coordinates = [float(value) for value in tokens[1:]]
        except ValueError:
            issues.append({"line": line_number, "reason": "non-numeric token"})
            continue
        if not class_value.is_integer():
            issues.append(
                {"line": line_number, "reason": "class ID is not an integer"}
            )
            continue
        class_id = int(class_value)
        if not 0 <= class_id < class_count:
            issues.append(
                {
                    "line": line_number,
                    "reason": f"class ID {class_id} is outside [0, {class_count})",
                }
            )
        invalid = [
            value
            for value in coordinates
            if not math.isfinite(value) or not 0.0 <= value <= 1.0
        ]
        if invalid:
            issues.append(
                {
                    "line": line_number,
                    "reason": "polygon coordinate is outside [0, 1]",
                    "invalid_values": invalid,
                }
            )
        counts[class_id] += 1
    return counts, issues, empty


def _yaml_declared_paths(
    source_root: Path,
    yaml_data: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    configured_base = source_root
    configured_root = yaml_data.get("path")
    if configured_root:
        root = Path(str(configured_root))
        configured_base = root if root.is_absolute() else source_root / root
    for key in ("train", "val", "test"):
        if key not in yaml_data:
            continue
        values = yaml_data[key]
        if not isinstance(values, list):
            values = [values]
        for value in values:
            path = Path(str(value).replace("\\", os.sep))
            resolved = (
                path.resolve()
                if path.is_absolute()
                else (configured_base / path).resolve()
            )
            rows.append(
                {
                    "yaml_key": key,
                    "declared": str(value),
                    "resolved": str(resolved),
                    "exists": resolved.is_dir(),
                }
            )
    return rows


def _readme_metadata(source_root: Path) -> dict[str, Any]:
    readme = source_root / "README.roboflow.txt"
    text = readme.read_text(encoding="utf-8-sig") if readme.is_file() else ""
    normalized = " ".join(text.casefold().split())
    combined_none = "no pre-processing or augmentation was applied" in normalized
    no_preprocessing = combined_none or "no preprocessing was applied" in normalized
    no_augmentation = combined_none or "no augmentation was applied" in normalized
    return {
        "path": str(readme),
        "available": readme.is_file(),
        "sha256": sha256_file(readme) if readme.is_file() else None,
        "states_no_preprocessing": no_preprocessing,
        "states_no_augmentation": no_augmentation,
        "rf_suffix_interpretation": (
            "roboflow_export_identifier"
            if no_augmentation
            else "augmentation_status_unresolved"
        ),
    }


def _perceptual_candidates(
    records: list[dict[str, Any]],
    threshold: int,
) -> list[dict[str, Any]]:
    candidates = []
    for left, right in itertools.combinations(records, 2):
        if left["image_sha256"] == right["image_sha256"]:
            continue
        distance = (
            int(left["difference_hash"], 16)
            ^ int(right["difference_hash"], 16)
        ).bit_count()
        if distance > threshold:
            continue
        pair = sorted((left["sample_id"], right["sample_id"]))
        candidates.append(
            {
                "candidate_id": "perceptual-" + stable_json_sha256(pair)[:12],
                "left_sample_id": left["sample_id"],
                "right_sample_id": right["sample_id"],
                "left_gem_id": left["gem_id"],
                "right_gem_id": right["gem_id"],
                "left_original_split": left["original_split"],
                "right_original_split": right["original_split"],
                "hamming_distance": distance,
                "same_gem": left["gem_id"] == right["gem_id"],
            }
        )
    return sorted(
        candidates,
        key=lambda row: (
            row["hamming_distance"],
            row["left_sample_id"],
            row["right_sample_id"],
        ),
    )


def _source_key(path: str | Path) -> str:
    return re.sub(
        r"_(?:jpg|jpeg|png)\.rf\.[^.]+$",
        "",
        Path(path).stem,
        flags=re.IGNORECASE,
    )


def _project_comparison(
    candidate_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    project_root = resolve_path(config.get("project_dataset_root"), config["_repo_root"])
    if project_root is None or not project_root.is_dir():
        return {
            "status": "unavailable",
            "exact_image_overlap_count": None,
            "normalized_source_key_overlap_count": None,
        }
    project_hashes = set()
    project_keys = set()
    image_count = 0
    for path in project_root.rglob("*"):
        if (
            path.is_file()
            and path.parent.name.casefold() == "images"
            and path.suffix.casefold() in IMAGE_EXTENSIONS
        ):
            image_count += 1
            project_hashes.add(sha256_file(path))
            project_keys.add(_source_key(path))
    candidate_hashes = {row["image_sha256"] for row in candidate_records}
    candidate_keys = {row["source_key"] for row in candidate_records}
    return {
        "status": "complete",
        "project_image_count": image_count,
        "exact_image_overlap_count": len(candidate_hashes & project_hashes),
        "normalized_source_key_overlap_count": len(candidate_keys & project_keys),
        "taxonomy_automatically_compatible": False,
    }


def audit_candidate_dataset(config: dict[str, Any]) -> dict[str, Any]:
    source_root = candidate_source_root(config)
    if source_root is None or not source_root.is_dir():
        raise ValueError(
            "Candidate source root is unavailable. Supply --source or the "
            "configured environment variable."
        )
    yaml_path = source_root / str(config.get("dataset_yaml", "data.yaml"))
    if not yaml_path.is_file():
        raise ValueError(f"Candidate data.yaml is missing: {yaml_path}")
    yaml_data = load_dataset_yaml(yaml_path)
    class_names = _class_names(yaml_data)
    declared_class_count = int(yaml_data.get("nc", len(class_names)))
    lineage_pattern = str(
        config.get("lineage_pattern", DEFAULT_LINEAGE_PATTERN)
    )
    file_rows, directory_manifest_hash = _file_manifest(source_root)
    image_set_hash = _record_set_hash(file_rows, "image")
    label_set_hash = _record_set_hash(file_rows, "label")

    records = []
    issues = []
    missing_images = []
    missing_labels = []
    empty_labels = []
    corrupt_images = []
    lineage_failures = []
    class_objects: Counter[int] = Counter()
    class_images: Counter[int] = Counter()
    split_counts = {}

    for split in SOURCE_SPLITS:
        image_directory = source_root / split / "images"
        label_directory = source_root / split / "labels"
        if not image_directory.is_dir():
            issues.append(
                {"category": "structure", "path": str(image_directory), "reason": "missing"}
            )
            images = []
        else:
            images = sorted(
                path
                for path in image_directory.iterdir()
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
            )
        labels = (
            sorted(
                path
                for path in label_directory.iterdir()
                if path.is_file() and path.suffix.casefold() == ".txt"
            )
            if label_directory.is_dir()
            else []
        )
        image_by_stem = {path.stem: path for path in images}
        label_by_stem = {path.stem: path for path in labels}
        missing_labels.extend(
            f"{split}/images/{stem}{image_by_stem[stem].suffix}"
            for stem in sorted(set(image_by_stem) - set(label_by_stem))
        )
        missing_images.extend(
            f"{split}/labels/{stem}.txt"
            for stem in sorted(set(label_by_stem) - set(image_by_stem))
        )
        for image_path in images:
            label_path = label_by_stem.get(image_path.stem)
            lineage = parse_lineage(image_path.name, lineage_pattern)
            if lineage is None:
                lineage_failures.append(
                    image_path.relative_to(source_root).as_posix()
                )
                lineage = {
                    "gem_id": "",
                    "video_id": "",
                    "frame_id": "",
                    "export_id": "",
                }
            image_details, image_error = _inspect_image(image_path)
            if image_error:
                corrupt_images.append(
                    {
                        "path": image_path.relative_to(source_root).as_posix(),
                        "reason": image_error,
                    }
                )
            label_counts: Counter[int] = Counter()
            if label_path is not None:
                label_counts, label_issues, empty = _inspect_label(
                    label_path,
                    declared_class_count,
                )
                if empty:
                    empty_labels.append(
                        label_path.relative_to(source_root).as_posix()
                    )
                for issue in label_issues:
                    issues.append(
                        {
                            "category": "label",
                            "path": label_path.relative_to(source_root).as_posix(),
                            **issue,
                        }
                    )
            image_classes = set(label_counts)
            class_objects.update(label_counts)
            class_images.update(image_classes)
            relative_image = image_path.relative_to(source_root).as_posix()
            relative_label = (
                label_path.relative_to(source_root).as_posix()
                if label_path is not None
                else ""
            )
            sample_id = (
                "sample-"
                + hashlib.sha256(relative_image.encode("utf-8")).hexdigest()[:16]
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "source_image_path": str(image_path.resolve()),
                    "source_label_path": (
                        str(label_path.resolve()) if label_path is not None else ""
                    ),
                    "relative_image_path": relative_image,
                    "relative_label_path": relative_label,
                    "image_sha256": sha256_file(image_path),
                    "label_sha256": (
                        sha256_file(label_path) if label_path is not None else ""
                    ),
                    "gem_id": lineage["gem_id"],
                    "specimen_id": (
                        f"Gem_{lineage['gem_id']}" if lineage["gem_id"] else ""
                    ),
                    "video_id": lineage["video_id"],
                    "frame_id": lineage["frame_id"],
                    "export_id": lineage["export_id"],
                    "original_split": split,
                    "class_counts": dict(sorted(label_counts.items())),
                    "class_ids": sorted(image_classes),
                    "object_count": sum(label_counts.values()),
                    "difference_hash": image_details.get("difference_hash", ""),
                    "width": image_details.get("width"),
                    "height": image_details.get("height"),
                    "mode": image_details.get("mode"),
                    "source_key": _source_key(image_path),
                }
            )
        split_counts[split] = {
            "images": len(images),
            "labels": len(labels),
            "matched_pairs": len(set(image_by_stem) & set(label_by_stem)),
        }

    exact_hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        exact_hashes[record["image_sha256"]].append(record)
    exact_groups = []
    for digest, members in sorted(exact_hashes.items()):
        if len(members) < 2:
            continue
        exact_groups.append(
            {
                "group_id": "exact-" + digest[:12],
                "sha256": digest,
                "member_count": len(members),
                "sample_ids": [member["sample_id"] for member in members],
                "gem_ids": sorted({member["gem_id"] for member in members}),
                "splits": sorted({member["original_split"] for member in members}),
            }
        )

    threshold = int(
        (config.get("development_split") or {}).get(
            "perceptual_hamming_threshold",
            5,
        )
    )
    perceptual_candidates = _perceptual_candidates(
        [record for record in records if record["difference_hash"]],
        threshold,
    )

    gem_splits: dict[str, Counter[str]] = defaultdict(Counter)
    video_splits: dict[str, Counter[str]] = defaultdict(Counter)
    frame_splits: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        if not record["gem_id"]:
            continue
        gem_splits[record["gem_id"]][record["original_split"]] += 1
        video_key = f"Gem_{record['gem_id']}_Vid_{record['video_id']}"
        frame_key = f"{video_key}_Fr_{record['frame_id']}"
        video_splits[video_key][record["original_split"]] += 1
        frame_splits[frame_key][record["original_split"]] += 1
    gem_crossings = {
        key: dict(value)
        for key, value in sorted(gem_splits.items(), key=lambda item: int(item[0]))
        if len(value) > 1
    }
    video_crossings = {
        key: dict(value)
        for key, value in sorted(video_splits.items())
        if len(value) > 1
    }
    frame_crossings = {
        key: dict(value)
        for key, value in sorted(frame_splits.items())
        if len(value) > 1
    }
    development_gems = {
        record["gem_id"]
        for record in records
        if record["original_split"] in {"train", "valid"}
    }
    test_gems = {
        record["gem_id"]
        for record in records
        if record["original_split"] == "test"
    }
    test_overlap = sorted(test_gems & development_gems, key=int)
    test_overlap_images = sum(
        record["original_split"] == "test" and record["gem_id"] in test_overlap
        for record in records
    )

    readme = _readme_metadata(source_root)
    yaml_paths = _yaml_declared_paths(source_root, yaml_data)
    candidate_zip = candidate_zip_path(config)
    zip_hash = (
        sha256_file(candidate_zip)
        if candidate_zip is not None and candidate_zip.is_file()
        else None
    )
    expected = config.get("expected") or {}
    integrity_mismatches = []

    def expect(name: str, actual: Any, expected_value: Any) -> None:
        if expected_value not in (None, "") and actual != expected_value:
            integrity_mismatches.append(
                {
                    "field": name,
                    "expected": expected_value,
                    "actual": actual,
                }
            )

    expect(
        "directory_manifest_sha256",
        directory_manifest_hash,
        expected.get("directory_manifest_sha256"),
    )
    expect("zip_sha256", zip_hash, expected.get("zip_sha256"))
    expect("image_count", len(records), expected.get("image_count"))
    expect(
        "label_count",
        sum(value["labels"] for value in split_counts.values()),
        expected.get("label_count"),
    )
    expect(
        "polygon_count",
        sum(class_objects.values()),
        expected.get("polygon_count"),
    )
    expect("class_names", class_names, expected.get("class_names"))

    blockers = []
    if declared_class_count != len(class_names):
        blockers.append("Declared nc does not match the class-name count.")
    if missing_images or missing_labels:
        blockers.append("Missing image/label pairs were found.")
    if issues:
        blockers.append("Invalid YOLO segmentation labels were found.")
    if empty_labels:
        blockers.append("Empty label files were found.")
    if corrupt_images:
        blockers.append("Corrupt image files were found.")
    if lineage_failures:
        blockers.append("Filename lineage could not be parsed for every image.")
    if integrity_mismatches:
        blockers.append("Candidate source integrity differs from the expected record.")

    warnings = []
    if any(not row["exists"] for row in yaml_paths):
        warnings.append(
            "The source data.yaml paths do not resolve to the physical split "
            "directories and must be corrected only in derived output."
        )
    if gem_crossings:
        warnings.append(
            "The current Roboflow split is not gem-independent and cannot "
            "be used for final metrics."
        )
    if not readme["states_no_augmentation"]:
        warnings.append("README evidence does not establish a no-augmentation export.")

    comparison = _project_comparison(records, config)
    model_path = resolve_path(config.get("legacy_model"), config["_repo_root"])
    model_hash = (
        sha256_file(model_path)
        if model_path is not None and model_path.is_file()
        else None
    )
    format_valid = not blockers
    return {
        "schema_version": "1.0",
        "audit_type": "candidate_dataset_registration",
        "created_utc": utc_timestamp(),
        "candidate_id": config.get("candidate_id", "gem-fracture-yolo-candidate"),
        "source": {
            "root": str(source_root),
            "dataset_yaml": str(yaml_path),
            "data_yaml_sha256": sha256_file(yaml_path),
            "zip_path": str(candidate_zip) if candidate_zip else None,
            "zip_sha256": zip_hash,
            "directory_manifest_sha256": directory_manifest_hash,
            "image_set_sha256": image_set_hash,
            "label_set_sha256": label_set_hash,
            "file_count": len(file_rows),
            "source_modified": False,
        },
        "dataset": {
            "format": "yolov8_segmentation",
            "declared_class_count": declared_class_count,
            "class_names": class_names,
            "split_counts": split_counts,
            "image_count": len(records),
            "label_count": sum(
                value["labels"] for value in split_counts.values()
            ),
            "polygon_count": sum(class_objects.values()),
            "class_object_counts": dict(sorted(class_objects.items())),
            "class_image_counts": dict(sorted(class_images.items())),
            "yaml_declared_paths": yaml_paths,
            "readme": readme,
        },
        "records": records,
        "file_manifest": file_rows,
        "integrity": {
            "format_valid": format_valid,
            "source_integrity_pass": not integrity_mismatches,
            "integrity_mismatches": integrity_mismatches,
            "missing_images": missing_images,
            "missing_labels": missing_labels,
            "empty_labels": empty_labels,
            "corrupt_images": corrupt_images,
            "label_issues": issues,
            "lineage_failures": lineage_failures,
            "exact_duplicate_groups": exact_groups,
            "perceptual_duplicate_candidate_count": len(perceptual_candidates),
        },
        "lineage": {
            "pattern": lineage_pattern,
            "parsed_count": len(records) - len(lineage_failures),
            "coverage": (
                (len(records) - len(lineage_failures)) / len(records)
                if records
                else 0.0
            ),
            "unique_gem_count": len(gem_splits),
            "unique_video_count": len(video_splits),
            "unique_source_frame_count": len(frame_splits),
            "gem_split_counts": {
                key: dict(value)
                for key, value in sorted(
                    gem_splits.items(),
                    key=lambda item: int(item[0]),
                )
            },
            "gem_ids_crossing_splits": gem_crossings,
            "video_ids_crossing_splits": video_crossings,
            "source_frames_crossing_splits": frame_crossings,
        },
        "current_split_assessment": {
            "trusted": False,
            "gemstone_independent": not test_overlap,
            "test_gem_ids": sorted(test_gems, key=int),
            "test_gem_overlap_with_development": test_overlap,
            "test_images_with_development_gem": test_overlap_images,
            "current_test_image_count": split_counts.get("test", {}).get(
                "images",
                0,
            ),
            "usable_for_final_metrics": False,
        },
        "perceptual_duplicate_candidates": perceptual_candidates,
        "project_dataset_comparison": comparison,
        "model_provenance": {
            "model_path": str(model_path) if model_path else None,
            "model_sha256": model_hash,
            "model_status": MODEL_STATUS,
            "candidate_dataset_attributed_to_model": False,
            "final_initialization_eligible": False,
        },
        "summary": {
            "status": (
                "pass_with_warnings"
                if format_valid and warnings
                else "pass"
                if format_valid
                else "blocked"
            ),
            "format_valid": format_valid,
            "source_integrity_pass": not integrity_mismatches,
            "registration_ready": bool(format_valid and not integrity_mismatches),
            "current_split_usable": False,
            "blocker_count": len(blockers),
            "blockers": blockers,
            "warnings": warnings,
            "training_occurred": False,
            "final_metrics_calculated": False,
            "claim_90_percent_available": False,
        },
    }


def _audit_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    dataset = report["dataset"]
    lineage = report["lineage"]
    current = report["current_split_assessment"]
    return "\n".join(
        [
            "# Candidate YOLO Segmentation Dataset Registration",
            "",
            f"- Status: {summary['status']}",
            f"- Images: {dataset['image_count']}",
            f"- Labels: {dataset['label_count']}",
            f"- Polygons: {dataset['polygon_count']}",
            f"- Classes: {', '.join(dataset['class_names'])}",
            f"- Gem IDs: {lineage['unique_gem_count']}",
            f"- Lineage coverage: {lineage['coverage']:.3f}",
            f"- Current split trusted: {str(current['trusted']).lower()}",
            (
                "- Current test gem-independent: "
                f"{str(current['gemstone_independent']).lower()}"
            ),
            "- Training occurred: false",
            "- Final metrics calculated: false",
            "",
            *[f"- Warning: {warning}" for warning in summary["warnings"]],
            *[f"- Blocker: {blocker}" for blocker in summary["blockers"]],
            "",
        ]
    )


def write_candidate_audit_outputs(
    report: dict[str, Any],
    config: dict[str, Any],
    output_directory: str | Path,
    command: str,
) -> dict[str, Path]:
    output = resolve_path(output_directory, config["_repo_root"])
    if output is None:
        raise ValueError("An output directory is required.")
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "report": output / "candidate_dataset_audit.json",
        "file_manifest": output / "candidate_file_manifest.csv",
        "lineage_manifest": output / "candidate_lineage_manifest.csv",
        "class_distribution": output / "candidate_class_distribution.csv",
        "exact_duplicates": output / "exact_duplicate_groups.csv",
        "perceptual_candidates": output / "perceptual_duplicate_candidates.csv",
        "markdown": output / "audit_summary.md",
        "manifest": output / "manifest.json",
    }
    serializable = {
        key: value
        for key, value in report.items()
        if key not in {"records", "file_manifest", "perceptual_duplicate_candidates"}
    }
    atomic_write_json(files["report"], serializable)
    atomic_write_csv(
        files["file_manifest"],
        report["file_manifest"],
        FILE_MANIFEST_COLUMNS,
    )
    atomic_write_csv(
        files["lineage_manifest"],
        report["records"],
        LINEAGE_COLUMNS,
    )
    class_rows = []
    for split, counts in report["dataset"]["split_counts"].items():
        split_records = [
            row for row in report["records"] if row["original_split"] == split
        ]
        for class_id, class_name in enumerate(report["dataset"]["class_names"]):
            class_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "image_count": sum(
                        class_id in row["class_ids"] for row in split_records
                    ),
                    "object_count": sum(
                        int(row["class_counts"].get(class_id, 0))
                        for row in split_records
                    ),
                }
            )
    atomic_write_csv(
        files["class_distribution"],
        class_rows,
        CLASS_DISTRIBUTION_COLUMNS,
    )
    atomic_write_csv(
        files["exact_duplicates"],
        report["integrity"]["exact_duplicate_groups"],
        ("group_id", "sha256", "member_count", "sample_ids", "gem_ids", "splits"),
    )
    atomic_write_csv(
        files["perceptual_candidates"],
        report["perceptual_duplicate_candidates"],
        PERCEPTUAL_COLUMNS,
    )
    atomic_write_text(files["markdown"], _audit_markdown(report))
    source_root = Path(report["source"]["root"])
    write_manifest(
        path=files["manifest"],
        repo_root=config["_repo_root"],
        tool_name="quartz_candidate_dataset_registration",
        command=command,
        tool_sources=(
            "backend/research/candidate_dataset.py",
            "backend/research/readiness_common.py",
            "backend/research_benchmark.py",
        ),
        inputs=(
            ("configuration", config.get("_config_path")),
            ("candidate_data_yaml", source_root / "data.yaml"),
            ("candidate_readme", source_root / "README.roboflow.txt"),
            ("candidate_zip", report["source"].get("zip_path")),
        ),
        outputs=tuple(
            (name, path)
            for name, path in files.items()
            if name != "manifest"
        ),
        extra={
            "candidate_directory_manifest_sha256": report["source"][
                "directory_manifest_sha256"
            ],
            "source_dataset_modified": False,
            "current_roboflow_split_trusted": False,
            "training_occurred": False,
            "final_metrics_calculated": False,
        },
    )
    return files


def run_candidate_audit(
    config: dict[str, Any],
    *,
    output_override: str | Path | None = None,
    command: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    report = audit_candidate_dataset(config)
    output = output_override or config.get("output_directory")
    if not output:
        raise ValueError("An output directory is required.")
    files = write_candidate_audit_outputs(report, config, output, command)
    return report, files


def validate_gem_identity_approval(
    approval_path: str | Path | None,
    *,
    repo_root: str | Path,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    path = resolve_path(approval_path, repo_root)
    blockers = []
    value: dict[str, Any] = {}
    if path is None or not path.is_file():
        blockers.append("Gem identity approval was not supplied.")
    else:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(f"Gem identity approval is unreadable: {exc}")
    if value:
        if value.get("example_only", False):
            blockers.append("Example gem identity records are not approvals.")
        if value.get("rule") != GEM_IDENTITY_RULE:
            blockers.append("Gem identity approval rule does not match the required rule.")
        if value.get("approved") is not True:
            blockers.append("Gem identity rule has not been approved.")
        for field in ("approved_by", "approver_role", "approval_date", "evidence"):
            if not value.get(field):
                blockers.append(f"Gem identity approval field is missing: {field}")
        supplied_hash = value.get("candidate_directory_manifest_sha256")
        if supplied_hash != source_manifest_sha256:
            blockers.append(
                "Gem identity approval is not linked to this candidate manifest."
            )
    return {
        "path": str(path) if path else None,
        "supplied": bool(value),
        "approved": bool(value and not blockers),
        "rule": value.get("rule", GEM_IDENTITY_RULE),
        "exceptions": value.get("exceptions", []),
        "blockers": blockers,
    }


def _human_approval(
    value_or_path: Any,
    *,
    repo_root: str | Path,
    expected: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    path = None
    if isinstance(value_or_path, dict):
        value = value_or_path
    else:
        path = resolve_path(value_or_path, repo_root)
        if path is not None and path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = {}
    blockers = []
    if not value:
        blockers.append(f"{label} approval was not supplied.")
    else:
        if value.get("example_only", False):
            blockers.append(f"Example {label} records are not approvals.")
        if value.get("approved") is not True:
            blockers.append(f"{label} has not been approved.")
        for field, expected_value in expected.items():
            if value.get(field) != expected_value:
                blockers.append(
                    f"{label} field {field} must equal {expected_value!r}."
                )
        for field in ("approved_by", "approver_role", "approval_date"):
            if not value.get(field):
                blockers.append(f"{label} approval field is missing: {field}")
    return {
        "path": str(path) if path else None,
        "approved": bool(value and not blockers),
        "value": value,
        "blockers": blockers,
    }


def _candidate_taxonomy_status(
    config: dict[str, Any],
    class_names: list[str],
) -> dict[str, Any]:
    taxonomy_path = resolve_path(config.get("candidate_taxonomy"), config["_repo_root"])
    approval_path = resolve_path(
        config.get("candidate_taxonomy_approval"),
        config["_repo_root"],
    )
    blockers = []
    rows = []
    if taxonomy_path is None or not taxonomy_path.is_file():
        blockers.append("Candidate taxonomy file is missing.")
    else:
        with taxonomy_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if [row.get("class_name") for row in rows] != class_names:
            blockers.append("Candidate taxonomy classes do not match data.yaml.")
        for row in rows:
            for field in (
                "semantic_definition",
                "confirmed_by",
                "confirmed_date",
            ):
                if not str(row.get(field, "")).strip():
                    blockers.append(
                        f"Class {row.get('class_name', '')} taxonomy field "
                        f"is missing: {field}"
                    )
            if str(row.get("expert_confirmed", "")).casefold() != "true":
                blockers.append(
                    f"Class {row.get('class_name', '')} is not expert-confirmed."
                )
    approval = {}
    if approval_path is None or not approval_path.is_file():
        blockers.append("Candidate taxonomy approval was not supplied.")
    else:
        try:
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append("Candidate taxonomy approval is unreadable.")
    if approval:
        if approval.get("example_only", False):
            blockers.append("Example taxonomy approval is not a real approval.")
        if not approval.get("approved_by") or not approval.get("approval_date"):
            blockers.append("Candidate taxonomy approval identity/date is incomplete.")
        if (
            taxonomy_path is None
            or not taxonomy_path.is_file()
            or approval.get("taxonomy_sha256") != sha256_file(taxonomy_path)
        ):
            blockers.append("Candidate taxonomy approval hash does not match.")
    return {
        "path": str(taxonomy_path) if taxonomy_path else None,
        "approval_path": str(approval_path) if approval_path else None,
        "approved": bool(rows and approval and not blockers),
        "blockers": list(dict.fromkeys(blockers)),
    }


def _atomic_gem_groups(
    records: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    gem_ids = sorted({record["gem_id"] for record in records}, key=int)
    union = UnionFind(gem_ids)
    exact_hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        exact_hashes[record["image_sha256"]].append(record)
    exact_groups = 0
    for members in exact_hashes.values():
        if len(members) < 2:
            continue
        exact_groups += 1
        for member in members[1:]:
            union.union(members[0]["gem_id"], member["gem_id"])
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_root[union.find(record["gem_id"])].append(record)
    groups = {}
    for members in by_root.values():
        member_gems = sorted({member["gem_id"] for member in members}, key=int)
        group_id = "gem-group-" + stable_json_sha256(member_gems)[:12]
        groups[group_id] = sorted(
            members,
            key=lambda row: (
                int(row["gem_id"]),
                int(row["video_id"]),
                int(row["frame_id"]),
                row["sample_id"],
            ),
        )
    return dict(sorted(groups.items())), {
        "gem_count": len(gem_ids),
        "atomic_group_count": len(groups),
        "exact_duplicate_group_count": exact_groups,
    }


def _assignment_score(
    valid_images: int,
    valid_objects: tuple[int, ...],
    valid_group_count: int,
    *,
    total_images: int,
    total_objects: tuple[int, ...],
    total_groups: int,
    valid_ratio: float,
) -> float:
    target_images = total_images * valid_ratio
    score = ((valid_images - target_images) / max(target_images, 1.0)) ** 2
    for class_id, total in enumerate(total_objects):
        target = total * valid_ratio
        score += 0.55 * (
            (valid_objects[class_id] - target) / max(target, 1.0)
        ) ** 2
        train_value = total - valid_objects[class_id]
        if total and (valid_objects[class_id] == 0 or train_value == 0):
            score += 25.0
    target_groups = total_groups * valid_ratio
    score += 0.05 * (
        (valid_group_count - target_groups) / max(target_groups, 1.0)
    ) ** 2
    if valid_group_count == 0 or valid_group_count == total_groups:
        score += 100.0
    return score


def assign_gem_groups(
    groups: dict[str, list[dict[str, Any]]],
    *,
    class_count: int,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
    beam_width: int = 4096,
) -> dict[str, str]:
    ratio_total = train_ratio + valid_ratio
    if ratio_total <= 0:
        raise ValueError("Train and validation ratios must have a positive sum.")
    valid_ratio = valid_ratio / ratio_total
    stats = {}
    total_objects = Counter()
    for group_id, members in groups.items():
        objects = Counter()
        for member in members:
            objects.update(
                {
                    int(key): int(value)
                    for key, value in member["class_counts"].items()
                }
            )
        stats[group_id] = {
            "images": len(members),
            "objects": tuple(objects[index] for index in range(class_count)),
        }
        total_objects.update(objects)
    order = sorted(
        groups,
        key=lambda group_id: (
            -stats[group_id]["images"],
            hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest(),
        ),
    )
    total_images = sum(item["images"] for item in stats.values())
    totals = tuple(total_objects[index] for index in range(class_count))
    states = [(tuple(), 0, tuple(0 for _ in range(class_count)), 0)]
    remaining_images = total_images
    remaining_objects = list(totals)
    for group_id in order:
        group = stats[group_id]
        remaining_images -= group["images"]
        remaining_objects = [
            remaining_objects[index] - group["objects"][index]
            for index in range(class_count)
        ]
        expanded = []
        for selected, images, objects, count in states:
            expanded.append((selected, images, objects, count))
            expanded.append(
                (
                    (*selected, group_id),
                    images + group["images"],
                    tuple(
                        objects[index] + group["objects"][index]
                        for index in range(class_count)
                    ),
                    count + 1,
                )
            )

        target_images = total_images * valid_ratio
        target_objects = [total * valid_ratio for total in totals]

        def lower_bound(state: tuple[Any, ...]) -> tuple[Any, ...]:
            selected, images, objects, _ = state
            image_error = 0.0
            if images > target_images:
                image_error = images - target_images
            elif images + remaining_images < target_images:
                image_error = target_images - images - remaining_images
            object_error = 0.0
            for index, value in enumerate(objects):
                if value > target_objects[index]:
                    object_error += value - target_objects[index]
                elif value + remaining_objects[index] < target_objects[index]:
                    object_error += (
                        target_objects[index]
                        - value
                        - remaining_objects[index]
                    )
            tie = stable_json_sha256([seed, list(selected)])
            return (round(image_error + 0.35 * object_error, 12), tie)

        states = sorted(expanded, key=lower_bound)[: max(2, int(beam_width))]

    best = min(
        states,
        key=lambda state: (
            round(
                _assignment_score(
                    state[1],
                    state[2],
                    state[3],
                    total_images=total_images,
                    total_objects=totals,
                    total_groups=len(groups),
                    valid_ratio=valid_ratio,
                ),
                12,
            ),
            stable_json_sha256([seed, list(state[0])]),
        ),
    )
    valid_groups = set(best[0])
    return {
        group_id: ("valid" if group_id in valid_groups else "train")
        for group_id in groups
    }


def plan_development_split(config: dict[str, Any]) -> dict[str, Any]:
    audit = audit_candidate_dataset(config)
    if not audit["summary"]["registration_ready"]:
        raise ValueError("Candidate registration is not ready for split planning.")
    records = audit["records"]
    groups, grouping = _atomic_gem_groups(records)
    split_config = config.get("development_split") or {}
    ratios = split_config.get("ratios") or {"train": 0.8, "valid": 0.2}
    train_ratio = float(ratios.get("train", 0.8))
    valid_ratio = float(ratios.get("valid", 0.2))
    seed = int(split_config.get("seed", 42))
    assignments = assign_gem_groups(
        groups,
        class_count=len(audit["dataset"]["class_names"]),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
        beam_width=int(split_config.get("beam_width", 4096)),
    )

    group_by_sample = {}
    rows = []
    gem_rows = []
    for group_id, members in groups.items():
        assigned = assignments[group_id]
        by_gem: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in members:
            group_by_sample[record["sample_id"]] = group_id
            by_gem[record["gem_id"]].append(record)
            row = {
                key: record.get(key, "")
                for key in DEVELOPMENT_MANIFEST_COLUMNS
            }
            row.update(
                {
                    "assigned_split": assigned,
                    "atomic_group_id": group_id,
                    "final_test_eligible": False,
                }
            )
            rows.append(row)
        for gem_id, gem_members in sorted(by_gem.items(), key=lambda item: int(item[0])):
            object_counts = Counter()
            for member in gem_members:
                object_counts.update(
                    {
                        int(key): int(value)
                        for key, value in member["class_counts"].items()
                    }
                )
            gem_rows.append(
                {
                    "gem_id": gem_id,
                    "specimen_id": f"Gem_{gem_id}",
                    "atomic_group_id": group_id,
                    "assigned_split": assigned,
                    "image_count": len(gem_members),
                    "video_count": len(
                        {member["video_id"] for member in gem_members}
                    ),
                    "object_count": sum(object_counts.values()),
                    "class_object_counts": dict(sorted(object_counts.items())),
                }
            )
    rows.sort(
        key=lambda row: (
            row["assigned_split"],
            int(row["gem_id"]),
            int(row["video_id"]),
            int(row["frame_id"]),
        )
    )
    gem_rows.sort(key=lambda row: int(row["gem_id"]))

    class_distribution = []
    for split in DEVELOPMENT_SPLITS:
        split_records = [
            record
            for record in records
            if assignments[group_by_sample[record["sample_id"]]] == split
        ]
        for class_id, class_name in enumerate(audit["dataset"]["class_names"]):
            class_distribution.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "image_count": sum(
                        class_id in record["class_ids"]
                        for record in split_records
                    ),
                    "object_count": sum(
                        int(record["class_counts"].get(class_id, 0))
                        for record in split_records
                    ),
                }
            )

    split_gems = {
        split: {
            row["gem_id"]
            for row in gem_rows
            if row["assigned_split"] == split
        }
        for split in DEVELOPMENT_SPLITS
    }
    gem_overlap = sorted(split_gems["train"] & split_gems["valid"], key=int)
    exact_cross = []
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        hash_splits[row["image_sha256"]].add(row["assigned_split"])
    for digest, splits in hash_splits.items():
        if len(splits) > 1:
            exact_cross.append(digest)

    sample_split = {
        row["sample_id"]: row["assigned_split"]
        for row in rows
    }
    perceptual_cross = []
    for candidate in audit["perceptual_duplicate_candidates"]:
        left = sample_split[candidate["left_sample_id"]]
        right = sample_split[candidate["right_sample_id"]]
        if left != right:
            perceptual_cross.append(
                {
                    **candidate,
                    "left_assigned_split": left,
                    "right_assigned_split": right,
                }
            )

    identity = validate_gem_identity_approval(
        config.get("gem_identity_approval"),
        repo_root=config["_repo_root"],
        source_manifest_sha256=audit["source"]["directory_manifest_sha256"],
    )
    route = _human_approval(
        config.get("dataset_route_approval"),
        repo_root=config["_repo_root"],
        expected={"dataset_route": "external_specimen_holdout"},
        label="dataset route",
    )
    taxonomy = _candidate_taxonomy_status(
        config,
        audit["dataset"]["class_names"],
    )
    leakage = {
        "gem_overlap_count": len(gem_overlap),
        "gem_overlap_ids": gem_overlap,
        "exact_duplicate_overlap_count": len(exact_cross),
        "exact_duplicate_overlap_hashes": exact_cross,
        "source_frame_overlap_count": 0,
        "perceptual_candidate_count": len(
            audit["perceptual_duplicate_candidates"]
        ),
        "perceptual_candidates_crossing_split_count": len(perceptual_cross),
        "perceptual_candidates_crossing_split": perceptual_cross,
        "current_roboflow_gem_crossing_count": len(
            audit["lineage"]["gem_ids_crossing_splits"]
        ),
        "current_roboflow_test_trusted": False,
        "core_grouping_passes": not gem_overlap and not exact_cross,
        "passes": not gem_overlap and not exact_cross and not perceptual_cross,
    }

    class_group_counts = Counter()
    for members in groups.values():
        present = {
            class_id
            for member in members
            for class_id in member["class_ids"]
        }
        class_group_counts.update(present)
    rare_warnings = []
    for class_id, class_name in enumerate(audit["dataset"]["class_names"]):
        if class_group_counts[class_id] < len(DEVELOPMENT_SPLITS):
            rare_warnings.append(
                f"Class {class_name} appears in only "
                f"{class_group_counts[class_id]} gem group(s)."
            )
    if perceptual_cross:
        rare_warnings.append(
            f"{len(perceptual_cross)} perceptual duplicate candidate pair(s) "
            "cross the planned split and require review before approval."
        )

    split_counts = Counter(row["assigned_split"] for row in rows)
    gem_counts = Counter(row["assigned_split"] for row in gem_rows)
    split_core = {
        "candidate_directory_manifest_sha256": audit["source"][
            "directory_manifest_sha256"
        ],
        "assignments": [
            (row["sample_id"], row["assigned_split"]) for row in rows
        ],
        "seed": seed,
        "ratios": {"train": train_ratio, "valid": valid_ratio},
    }
    split_id = "development-" + stable_json_sha256(split_core)[:16]
    planned_manifest_hash = development_manifest_sha256(rows)
    required_overlap_counts = {
        "gem": len(gem_overlap),
        "exact_duplicate": len(exact_cross),
        "source_frame": leakage["source_frame_overlap_count"],
        "flagged_perceptual_pair": len(perceptual_cross),
    }
    split_approval = _human_approval(
        config.get("development_split_approval") or {},
        repo_root=config["_repo_root"],
        expected={
            "split_id": split_id,
            "candidate_directory_manifest_sha256": audit["source"][
                "directory_manifest_sha256"
            ],
            "grouped_development_manifest_sha256": planned_manifest_hash,
            "seed": seed,
            "expected_image_counts": {
                "train": split_counts["train"],
                "valid": split_counts["valid"],
            },
            "expected_gem_counts": {
                "train": gem_counts["train"],
                "valid": gem_counts["valid"],
            },
            "required_overlap_counts": required_overlap_counts,
        },
        label="development split",
    )
    preapproval_ready = bool(
        identity["approved"]
        and route["approved"]
        and leakage["passes"]
        and audit["summary"]["registration_ready"]
    )
    approved = bool(preapproval_ready and split_approval["approved"])
    frozen = bool(approved and split_config.get("request_freeze", False))
    split_approval_path = split_approval.get("path")
    split_approval_hash = (
        sha256_file(split_approval_path)
        if split_approval_path
        else stable_json_sha256(split_approval.get("value") or {})
    )
    lock = {
        "schema_version": "1.0",
        "lock_type": "gem_grouped_development_split",
        "split_id": split_id,
        "candidate_directory_manifest_sha256": audit["source"][
            "directory_manifest_sha256"
        ],
        "gem_identity_rule_approved": identity["approved"],
        "dataset_route": "external_specimen_holdout",
        "dataset_route_approved": route["approved"],
        "split_approved": approved,
        "frozen": frozen,
        "approval_status": "approved" if approved else "unavailable",
        "grouped_development_manifest_sha256": planned_manifest_hash,
        "split_approval_sha256": split_approval_hash,
        "training_authorized": False,
    }
    lock["lock_hash"] = stable_json_sha256(lock)
    blockers = []
    blockers.extend(identity["blockers"])
    blockers.extend(route["blockers"])
    blockers.extend(split_approval["blockers"])
    if not leakage["passes"]:
        blockers.append("Development split leakage review is incomplete.")
    blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": "1.0",
        "audit_type": "gem_grouped_development_split_plan",
        "created_utc": utc_timestamp(),
        "rows": rows,
        "gem_distribution": gem_rows,
        "class_distribution": class_distribution,
        "leakage_audit": leakage,
        "perceptual_duplicate_candidates": audit[
            "perceptual_duplicate_candidates"
        ],
        "external_test_identity_policy": {
            "route": "external_specimen_holdout",
            "development_gem_ids": [
                f"Gem_{gem_id}"
                for gem_id in sorted(
                    {record["gem_id"] for record in records},
                    key=int,
                )
            ],
            "reserved_external_prefix": "EXT-GEM-",
            "first_reserved_identity": "EXT-GEM-001",
            "must_be_new_physical_specimen": True,
            "reject_exact_development_duplicates": True,
            "reject_perceptual_duplicate_candidates": True,
            "prohibited_uses": [
                "training",
                "validation",
                "threshold_tuning",
                "early_stopping",
                "architecture_selection",
                "augmentation_tuning",
                "model_selection",
            ],
            "external_test_rows_created": 0,
        },
        "identity_approval": identity,
        "dataset_route_approval": route,
        "development_split_approval": split_approval,
        "taxonomy": taxonomy,
        "summary": {
            "status": "blocked" if not approved else "approved",
            "structural_plan_generated": True,
            "source_image_count": len(rows),
            "source_gem_count": len(gem_rows),
            "atomic_group_count": grouping["atomic_group_count"],
            "split_image_counts": dict(split_counts),
            "split_gem_counts": dict(gem_counts),
            "ratios": {"train": train_ratio, "valid": valid_ratio},
            "seed": seed,
            "all_current_images_are_development": True,
            "current_images_used_as_final_test": 0,
            "taxonomy_approved": taxonomy["approved"],
            "taxonomy_blocks_structural_plan": False,
            "gem_identity_rule_approved": identity["approved"],
            "dataset_route_approved": route["approved"],
            "preapproval_ready": preapproval_ready,
            "split_approved": approved,
            "frozen": frozen,
            "materialization_allowed": bool(approved and frozen and leakage["passes"]),
            "blocker_count": len(blockers),
            "blockers": blockers,
            "warnings": rare_warnings,
            "training_authorized": False,
            "training_occurred": False,
            "final_metrics_calculated": False,
            "claim_90_percent_available": False,
        },
        "development_split_lock": lock,
        "class_names": audit["dataset"]["class_names"],
        "candidate_audit_summary": {
            "source": audit["source"],
            "current_split_assessment": audit["current_split_assessment"],
            "model_provenance": audit["model_provenance"],
        },
    }


def _development_yaml(plan: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Future materialized development dataset; source export is unchanged.",
            "# Current candidate images are never the final research test set.",
            "path: .",
            "train: train/images",
            "val: valid/images",
            f"nc: {len(plan['class_names'])}",
            "names: " + json.dumps(plan["class_names"], ensure_ascii=True),
            "",
        ]
    )


def _development_markdown(plan: dict[str, Any]) -> str:
    summary = plan["summary"]
    return "\n".join(
        [
            "# Gem-Grouped Development Split Plan",
            "",
            f"- Status: {summary['status']}",
            f"- Images: {summary['source_image_count']}",
            f"- Gem IDs: {summary['source_gem_count']}",
            f"- Atomic groups: {summary['atomic_group_count']}",
            (
                "- Split image counts: "
                + json.dumps(summary["split_image_counts"], sort_keys=True)
            ),
            (
                "- Split gem counts: "
                + json.dumps(summary["split_gem_counts"], sort_keys=True)
            ),
            "- Current images used as final test: 0",
            (
                "- Gem identity approved: "
                f"{str(summary['gem_identity_rule_approved']).lower()}"
            ),
            (
                "- Dataset route approved: "
                f"{str(summary['dataset_route_approved']).lower()}"
            ),
            f"- Split approved: {str(summary['split_approved']).lower()}",
            f"- Frozen: {str(summary['frozen']).lower()}",
            (
                "- Materialization allowed: "
                f"{str(summary['materialization_allowed']).lower()}"
            ),
            "- Training authorized: false",
            "- Training occurred: false",
            "- Final metrics calculated: false",
            "",
            *[f"- Warning: {item}" for item in summary["warnings"]],
            *[f"- Blocker: {item}" for item in summary["blockers"]],
            "",
        ]
    )


def _write_human_input_pack(
    plan: dict[str, Any],
    config: dict[str, Any],
    output: Path,
) -> dict[str, Path]:
    pack = output / "human_input_pack"
    pack.mkdir(parents=True, exist_ok=True)
    files = {
        "taxonomy": pack / "candidate_taxonomy.pending.csv",
        "identity": pack / "gem_identity_rule.pending.json",
        "route": pack / "dataset_route_decision.pending.json",
        "metric": pack / "defect_metric_approval.pending.json",
        "training": pack / "defect_training_approval.pending.json",
        "external": pack / "external_test_plan.pending.csv",
        "checklist": pack / "gate_closure_checklist.md",
    }
    taxonomy_source = resolve_path(
        config.get("candidate_taxonomy"),
        config["_repo_root"],
    )
    taxonomy_rows = []
    if taxonomy_source is not None and taxonomy_source.is_file():
        with taxonomy_source.open("r", encoding="utf-8-sig", newline="") as handle:
            taxonomy_rows = list(csv.DictReader(handle))
    atomic_write_csv(
        files["taxonomy"],
        taxonomy_rows,
        CANDIDATE_TAXONOMY_COLUMNS,
    )
    atomic_write_json(
        files["identity"],
        {
            "schema_version": "1.0",
            "rule": GEM_IDENTITY_RULE,
            "candidate_directory_manifest_sha256": plan[
                "development_split_lock"
            ]["candidate_directory_manifest_sha256"],
            "approved": False,
            "approved_by": None,
            "approver_role": None,
            "approval_date": None,
            "exceptions": [],
            "evidence": None,
            "notes": "Pending human confirmation; this is not an approval.",
            "example_only": True,
        },
    )
    atomic_write_json(
        files["route"],
        {
            "schema_version": "1.0",
            "dataset_route": "external_specimen_holdout",
            "duplicate_policy": "conservative_union",
            "approved": False,
            "approved_by": None,
            "approver_role": None,
            "approval_date": None,
            "limitations_acknowledged": [
                "current_roboflow_split_not_gem_independent"
            ],
            "notes": "Recommended route only; human approval is required.",
            "example_only": True,
        },
    )
    metric_source = resolve_path(
        config.get("metric_approval_template"),
        config["_repo_root"],
    )
    metric_value = {
        "primary_metric": "binary_visible_defect_f1",
        "claim_threshold": 0.9,
        "approved_by": None,
        "example_only": True,
    }
    if metric_source is not None and metric_source.is_file():
        metric_value = json.loads(metric_source.read_text(encoding="utf-8"))
    atomic_write_json(files["metric"], metric_value)
    training_source = resolve_path(
        config.get("training_approval_template"),
        config["_repo_root"],
    )
    training_value = {
        "initialization_policy": "official_generic_pretrained_initialization",
        "approved_by": None,
        "example_only": True,
    }
    if training_source is not None and training_source.is_file():
        training_value = json.loads(training_source.read_text(encoding="utf-8"))
    atomic_write_json(files["training"], training_value)
    atomic_write_csv(files["external"], [], EXTERNAL_TEST_PLAN_COLUMNS)
    atomic_write_text(
        files["checklist"],
        "\n".join(
            [
                "# Gate 1 Human Input Checklist",
                "",
                "| File | Completed by | Evidence | Blocks |",
                "|---|---|---|---|",
                (
                    "| candidate_taxonomy.pending.csv | Gem expert or dataset "
                    "creator | Definitions and class eligibility | Training taxonomy |"
                ),
                (
                    "| gem_identity_rule.pending.json | Dataset creator or "
                    "custodian | Confirmation that Gem_<n> is one stone | Split freeze |"
                ),
                (
                    "| dataset_route_decision.pending.json | Supervisor | Signed "
                    "external-holdout route decision | Dataset route |"
                ),
                (
                    "| defect_metric_approval.pending.json | Supervisor | Metric "
                    "and threshold approval | 90 percent claim definition |"
                ),
                (
                    "| defect_training_approval.pending.json | Supervisor | Clean "
                    "initialization decision | Future training |"
                ),
                (
                    "| external_test_plan.pending.csv | Research team | New specimen "
                    "and expert-review plan | Final held-out test |"
                ),
                "",
                "After completion, reference the reviewed files in the Gate 1 config,",
                "rerun candidate-dataset-audit and development-split-plan, and review",
                "the new lock hashes. Do not set example_only to false unless a real",
                "named approver supplied the decision.",
                "",
            ]
        ),
    )
    return files


def write_development_split_outputs(
    plan: dict[str, Any],
    config: dict[str, Any],
    output_directory: str | Path,
    command: str,
) -> dict[str, Path]:
    output = resolve_path(output_directory, config["_repo_root"])
    if output is None:
        raise ValueError("An output directory is required.")
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "manifest_csv": output / "grouped_development_manifest.csv",
        "summary": output / "development_split_summary.json",
        "class_distribution": output / "development_class_distribution.csv",
        "gem_distribution": output / "development_gem_distribution.csv",
        "leakage": output / "development_leakage_audit.json",
        "lock": output / "development_split_lock.json",
        "yaml": output / "data_grouped.yaml",
        "perceptual_candidates": output / "perceptual_duplicate_candidates.csv",
        "external_policy": output / "external_test_identity_policy.json",
        "markdown": output / "audit_summary.md",
        "manifest": output / "manifest.json",
    }
    atomic_write_csv(
        files["manifest_csv"],
        plan["rows"],
        DEVELOPMENT_MANIFEST_COLUMNS,
    )
    manifest_hash = sha256_file(files["manifest_csv"])
    expected_manifest_hash = plan["development_split_lock"].get(
        "grouped_development_manifest_sha256"
    )
    if manifest_hash != expected_manifest_hash:
        raise ValueError(
            "Written development manifest does not match the approved plan hash."
        )
    lock = dict(plan["development_split_lock"])
    lock["manifest_sha256"] = manifest_hash
    lock["generated_utc"] = plan["created_utc"]
    lock["lock_hash"] = stable_json_sha256(
        {key: value for key, value in lock.items() if key != "generated_utc"}
    )
    summary = {
        **plan["summary"],
        "split_id": lock["split_id"],
        "manifest_sha256": manifest_hash,
        "identity_approval": plan["identity_approval"],
        "dataset_route_approval": plan["dataset_route_approval"],
        "development_split_approval": plan["development_split_approval"],
        "taxonomy": plan["taxonomy"],
        "candidate_source": plan["candidate_audit_summary"]["source"],
    }
    atomic_write_json(files["summary"], summary)
    atomic_write_csv(
        files["class_distribution"],
        plan["class_distribution"],
        CLASS_DISTRIBUTION_COLUMNS,
    )
    atomic_write_csv(
        files["gem_distribution"],
        plan["gem_distribution"],
        GEM_DISTRIBUTION_COLUMNS,
    )
    atomic_write_json(files["leakage"], plan["leakage_audit"])
    atomic_write_json(files["lock"], lock)
    atomic_write_text(files["yaml"], _development_yaml(plan))
    atomic_write_csv(
        files["perceptual_candidates"],
        plan["perceptual_duplicate_candidates"],
        PERCEPTUAL_COLUMNS,
    )
    atomic_write_json(
        files["external_policy"],
        plan["external_test_identity_policy"],
    )
    atomic_write_text(files["markdown"], _development_markdown(plan))
    human_files = _write_human_input_pack(plan, config, output)
    write_manifest(
        path=files["manifest"],
        repo_root=config["_repo_root"],
        tool_name="quartz_gem_grouped_development_split",
        command=command,
        tool_sources=(
            "backend/research/candidate_dataset.py",
            "backend/research/readiness_common.py",
            "backend/research_benchmark.py",
        ),
        inputs=(
            ("configuration", config.get("_config_path")),
            ("gem_identity_approval", config.get("gem_identity_approval")),
            ("candidate_taxonomy", config.get("candidate_taxonomy")),
            (
                "candidate_taxonomy_approval",
                config.get("candidate_taxonomy_approval"),
            ),
            ("dataset_route_approval", config.get("dataset_route_approval")),
            (
                "development_split_approval",
                config.get("development_split_approval"),
            ),
        ),
        outputs=tuple(
            [
                (name, path)
                for name, path in files.items()
                if name != "manifest"
            ]
            + [
                (f"human_input_{name}", path)
                for name, path in human_files.items()
            ]
        ),
        extra={
            "candidate_directory_manifest_sha256": lock[
                "candidate_directory_manifest_sha256"
            ],
            "source_dataset_modified": False,
            "all_current_images_are_development": True,
            "current_images_used_as_final_test": 0,
            "materialized": False,
            "training_occurred": False,
            "final_metrics_calculated": False,
        },
    )
    return {**files, **{f"human_{key}": value for key, value in human_files.items()}}


def run_development_split_plan(
    config: dict[str, Any],
    *,
    output_override: str | Path | None = None,
    command: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    plan = plan_development_split(config)
    output = (
        output_override
        or config.get("development_output_directory")
        or config.get("output_directory")
    )
    if not output:
        raise ValueError("A development split output directory is required.")
    files = write_development_split_outputs(plan, config, output, command)
    return plan, files


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def materialize_development_split(
    *,
    plan_directory: str | Path,
    output_directory: str | Path,
    repo_root: str | Path,
    apply: bool = False,
    confirm: bool = False,
    mode: str = "copy",
    provenance_inputs: Iterable[str | Path] = (),
    taxonomy_path: str | Path | None = None,
) -> dict[str, Any]:
    plan_dir = Path(plan_directory).resolve()
    output = Path(output_directory).resolve()
    repository = Path(repo_root).resolve()
    required = {
        "manifest": plan_dir / "grouped_development_manifest.csv",
        "summary": plan_dir / "development_split_summary.json",
        "lock": plan_dir / "development_split_lock.json",
        "yaml": plan_dir / "data_grouped.yaml",
    }
    if not all(path.is_file() for path in required.values()):
        return {"status": "blocked", "reason": "Development split plan files are missing."}
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    lock = json.loads(required["lock"].read_text(encoding="utf-8"))
    if not apply:
        return {
            "status": "preview",
            "apply": False,
            "confirm": bool(confirm),
            "output_created": False,
            "materialization_allowed": bool(summary.get("materialization_allowed")),
            "training_occurred": False,
        }
    if not confirm:
        return {
            "status": "blocked",
            "reason": "A separate explicit --confirm flag is required.",
            "output_created": False,
        }
    if mode not in {"copy", "hardlink"}:
        return {"status": "blocked", "reason": f"Unsupported mode: {mode}"}
    if not (
        summary.get("materialization_allowed")
        and lock.get("split_approved")
        and lock.get("frozen")
        and lock.get("gem_identity_rule_approved")
        and lock.get("dataset_route_approved")
    ):
        return {
            "status": "blocked",
            "reason": (
                "Approved identity, route, split lock, and freeze are required."
            ),
            "output_created": False,
        }
    if output.exists():
        return {"status": "blocked", "reason": "Output directory already exists."}
    if _is_within(output, repository):
        return {
            "status": "blocked",
            "reason": "Derived output must be outside the repository.",
        }
    with required["manifest"].open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(row.get("assigned_split") not in DEVELOPMENT_SPLITS for row in rows):
        return {
            "status": "blocked",
            "reason": "Development materialization may contain only train and valid.",
        }
    sources = []
    destinations = set()
    source_roots = set()
    for row in rows:
        image = Path(row["source_image_path"]).resolve()
        label = Path(row["source_label_path"]).resolve()
        if not image.is_file() or not label.is_file():
            return {
                "status": "blocked",
                "reason": f"Source pair is missing for {row.get('sample_id', '')}.",
            }
        source_roots.add(image.parents[2])
        image_destination = (
            output / row["assigned_split"] / "images" / image.name
        )
        label_destination = (
            output / row["assigned_split"] / "labels" / label.name
        )
        keys = {
            str(image_destination).casefold(),
            str(label_destination).casefold(),
        }
        if destinations & keys:
            return {
                "status": "blocked",
                "reason": f"Output collision for {row.get('sample_id', '')}.",
            }
        destinations.update(keys)
        if sha256_file(image) != row["image_sha256"]:
            return {
                "status": "blocked",
                "reason": f"Source image hash changed for {row.get('sample_id', '')}.",
            }
        if sha256_file(label) != row["label_sha256"]:
            return {
                "status": "blocked",
                "reason": f"Source label hash changed for {row.get('sample_id', '')}.",
            }
        sources.append((row, image, label, image_destination, label_destination))
    if any(_is_within(output, root) for root in source_roots):
        return {
            "status": "blocked",
            "reason": "Derived output must not be inside the source dataset.",
        }

    output.mkdir(parents=True)
    manifests_directory = output / "manifests"
    provenance_directory = output / "provenance"
    approval_directory = provenance_directory / "approvals"
    manifests_directory.mkdir()
    approval_directory.mkdir(parents=True)
    materialized = []
    for row, image, label, image_destination, label_destination in sources:
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == "hardlink":
            os.link(image, image_destination)
            os.link(label, label_destination)
        else:
            shutil.copy2(image, image_destination)
            shutil.copy2(label, label_destination)
        image_hash = sha256_file(image_destination)
        label_hash = sha256_file(label_destination)
        if image_hash != row["image_sha256"] or label_hash != row["label_sha256"]:
            raise ValueError(f"Materialized hash mismatch for {row['sample_id']}")
        materialized.append(
            {
                "sample_id": row["sample_id"],
                "split": row["assigned_split"],
                "source_image_path": str(image),
                "source_label_path": str(label),
                "image_path": str(image_destination),
                "label_path": str(label_destination),
                "derived_image_path": image_destination.relative_to(output).as_posix(),
                "derived_label_path": label_destination.relative_to(output).as_posix(),
                "image_sha256": image_hash,
                "label_sha256": label_hash,
            }
        )
    shutil.copy2(required["yaml"], output / "data.yaml")

    plan_artifacts = (
        "grouped_development_manifest.csv",
        "development_split_summary.json",
        "development_class_distribution.csv",
        "development_gem_distribution.csv",
        "development_leakage_audit.json",
        "development_split_lock.json",
        "perceptual_duplicate_candidates.csv",
        "external_test_identity_policy.json",
    )
    copied_plan_artifacts = []
    for name in plan_artifacts:
        source = plan_dir / name
        if not source.is_file():
            continue
        destination = manifests_directory / name
        shutil.copy2(source, destination)
        copied_plan_artifacts.append(
            {
                "name": name,
                "sha256": sha256_file(destination),
            }
        )

    copied_approvals = []
    for raw_path in provenance_inputs:
        source = Path(raw_path)
        if not source.is_absolute():
            source = repository / source
        source = source.resolve()
        if not source.is_file():
            return {
                "status": "blocked",
                "reason": f"Provenance input is missing: {source}",
                "output_created": True,
            }
        destination = approval_directory / source.name
        if destination.exists():
            return {
                "status": "blocked",
                "reason": f"Provenance filename collision: {source.name}",
                "output_created": True,
            }
        shutil.copy2(source, destination)
        copied_approvals.append(
            {
                "file": source.name,
                "source_path": str(source),
                "sha256": sha256_file(destination),
            }
        )

    taxonomy = None
    if taxonomy_path:
        taxonomy = Path(taxonomy_path)
        if not taxonomy.is_absolute():
            taxonomy = repository / taxonomy
        taxonomy = taxonomy.resolve()
    taxonomy_hash = (
        sha256_file(taxonomy) if taxonomy is not None and taxonomy.is_file() else None
    )
    counts = Counter(row["split"] for row in materialized)
    materialization_payload = {
        "schema_version": "1.0",
        "created_utc": utc_timestamp(),
        "split_id": lock["split_id"],
        "split_lock_sha256": sha256_file(required["lock"]),
        "grouped_development_manifest_sha256": sha256_file(required["manifest"]),
        "candidate_directory_manifest_sha256": lock[
            "candidate_directory_manifest_sha256"
        ],
        "taxonomy_sha256": taxonomy_hash,
        "mode": mode,
        "file_pair_count": len(materialized),
        "split_counts": {
            "train": counts["train"],
            "valid": counts["valid"],
            "test": 0,
        },
        "source_files_moved": False,
        "labels_preserved_byte_for_byte": True,
        "final_test_directory_created": False,
        "approval_files": copied_approvals,
        "records": materialized,
    }
    materialization_path = manifests_directory / "materialization_manifest.json"
    atomic_write_json(materialization_path, materialization_payload)
    shutil.copy2(materialization_path, output / "materialization_manifest.json")

    atomic_write_csv(
        manifests_directory / "source_to_derived_mapping.csv",
        materialized,
        (
            "sample_id",
            "split",
            "source_image_path",
            "source_label_path",
            "derived_image_path",
            "derived_label_path",
            "image_sha256",
            "label_sha256",
        ),
    )
    provenance_payload = {
        "schema_version": "1.0",
        "created_utc": utc_timestamp(),
        "tool": "quartz_gate2_development_materializer",
        "tool_version": "1.0",
        "git": git_evidence(repository),
        "source": {
            "candidate_directory_manifest_sha256": lock[
                "candidate_directory_manifest_sha256"
            ],
            "source_files_moved": False,
        },
        "development_split": {
            "split_id": lock["split_id"],
            "split_lock_sha256": sha256_file(required["lock"]),
            "grouped_development_manifest_sha256": sha256_file(
                required["manifest"]
            ),
            "frozen": lock["frozen"],
        },
        "taxonomy_sha256": taxonomy_hash,
        "approval_files": copied_approvals,
        "copied_plan_artifacts": copied_plan_artifacts,
        "training_occurred": False,
        "weights_downloaded": False,
        "final_evaluation_occurred": False,
    }
    provenance_path = provenance_directory / "provenance.json"
    atomic_write_json(provenance_path, provenance_payload)

    derived_manifest_path = manifests_directory / "derived_file_manifest.csv"
    derived_rows = []
    for path in sorted(
        (
            item
            for item in output.rglob("*")
            if item.is_file() and item != derived_manifest_path
        ),
        key=lambda item: item.relative_to(output).as_posix().casefold(),
    ):
        derived_rows.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    atomic_write_csv(
        derived_manifest_path,
        derived_rows,
        ("relative_path", "size_bytes", "sha256"),
    )
    return {
        "status": "complete",
        "output": str(output),
        "file_pair_count": len(materialized),
        "split_counts": materialization_payload["split_counts"],
        "materialization_manifest": str(materialization_path),
        "provenance": str(provenance_path),
        "derived_file_manifest": str(derived_manifest_path),
        "source_files_moved": False,
        "final_test_directory_created": False,
        "weights_downloaded": False,
        "training_occurred": False,
    }


def audit_materialized_development_dataset(
    dataset_root: str | Path,
    *,
    plan_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Audit a materialized train/validation dataset without changing it."""

    root = Path(dataset_root).resolve()
    plan_dir = Path(plan_directory).resolve() if plan_directory else None
    data_yaml = root / "data.yaml"
    materialization_path = root / "manifests" / "materialization_manifest.json"
    blockers = []
    issues = []
    if not root.is_dir():
        blockers.append("Derived dataset root does not exist.")
    if not data_yaml.is_file():
        blockers.append("Derived data.yaml is missing.")
        yaml_data = {}
    else:
        try:
            yaml_data = load_dataset_yaml(data_yaml)
        except Exception as exc:
            yaml_data = {}
            blockers.append(f"Derived data.yaml is invalid: {exc}")
    class_names = _class_names(yaml_data)
    if class_names != ["fracture", "inclusion"]:
        blockers.append("Derived class names must be fracture and inclusion.")

    if not materialization_path.is_file():
        blockers.append("Materialization manifest is missing.")
        materialization = {}
    else:
        try:
            materialization = json.loads(
                materialization_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            materialization = {}
            blockers.append(f"Materialization manifest is invalid: {exc}")
    manifest_records = {
        Path(row.get("derived_image_path", "")).stem: row
        for row in materialization.get("records", [])
        if row.get("sample_id") and row.get("derived_image_path")
    }

    records = []
    split_counts = {}
    class_objects = Counter()
    class_images = Counter()
    for split in DEVELOPMENT_SPLITS:
        declared = yaml_data.get("train" if split == "train" else "val")
        expected_declared = f"{split}/images"
        if declared != expected_declared:
            blockers.append(
                f"Derived YAML path for {split} must be {expected_declared}."
            )
        images_directory = root / split / "images"
        labels_directory = root / split / "labels"
        images = {
            path.stem: path
            for path in images_directory.glob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        }
        labels = {
            path.stem: path
            for path in labels_directory.glob("*.txt")
            if path.is_file()
        }
        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        for stem in missing_labels:
            issues.append(
                {"split": split, "sample_id": stem, "reason": "missing_label"}
            )
        for stem in missing_images:
            issues.append(
                {"split": split, "sample_id": stem, "reason": "missing_image"}
            )
        split_counts[split] = len(images)
        for stem in sorted(set(images) & set(labels)):
            image = images[stem]
            label = labels[stem]
            image_info, image_error = _inspect_image(image)
            object_counts, label_issues, empty = _inspect_label(
                label,
                len(class_names),
            )
            if image_error:
                issues.append(
                    {
                        "split": split,
                        "sample_id": stem,
                        "reason": f"corrupt_image: {image_error}",
                    }
                )
            if empty:
                issues.append(
                    {"split": split, "sample_id": stem, "reason": "empty_label"}
                )
            for issue in label_issues:
                issues.append(
                    {
                        "split": split,
                        "sample_id": stem,
                        "reason": (
                            f"invalid_label_line_{issue['line']}: "
                            f"{issue['reason']}"
                        ),
                    }
                )
            lineage = parse_lineage(image.name)
            if lineage is None:
                issues.append(
                    {
                        "split": split,
                        "sample_id": stem,
                        "reason": "unparseable_lineage",
                    }
                )
                lineage = {
                    "gem_id": "",
                    "video_id": "",
                    "frame_id": "",
                    "export_id": "",
                }
            image_hash = sha256_file(image)
            label_hash = sha256_file(label)
            source_record = manifest_records.get(stem)
            stable_sample_id = (
                source_record.get("sample_id") if source_record else stem
            )
            if source_record is None:
                issues.append(
                    {
                        "split": split,
                        "sample_id": stem,
                        "reason": "missing_materialization_record",
                    }
                )
            else:
                if source_record.get("split") != split:
                    issues.append(
                        {
                            "split": split,
                            "sample_id": stem,
                            "reason": "manifest_split_mismatch",
                        }
                    )
                if source_record.get("image_sha256") != image_hash:
                    issues.append(
                        {
                            "split": split,
                            "sample_id": stem,
                            "reason": "derived_image_hash_mismatch",
                        }
                    )
                if source_record.get("label_sha256") != label_hash:
                    issues.append(
                        {
                            "split": split,
                            "sample_id": stem,
                            "reason": "derived_label_hash_mismatch",
                        }
                    )
                source_image = Path(source_record.get("source_image_path", ""))
                source_label = Path(source_record.get("source_label_path", ""))
                if (
                    not source_image.is_file()
                    or sha256_file(source_image) != image_hash
                ):
                    issues.append(
                        {
                            "split": split,
                            "sample_id": stem,
                            "reason": "source_image_hash_mismatch",
                        }
                    )
                if (
                    not source_label.is_file()
                    or sha256_file(source_label) != label_hash
                ):
                    issues.append(
                        {
                            "split": split,
                            "sample_id": stem,
                            "reason": "source_label_hash_mismatch",
                        }
                    )
            for class_id, count in object_counts.items():
                class_objects[(split, class_id)] += count
                class_images[(split, class_id)] += 1
            records.append(
                {
                    "sample_id": stable_sample_id,
                    "split": split,
                    "image_sha256": image_hash,
                    "label_sha256": label_hash,
                    **lineage,
                    **image_info,
                }
            )

    test_directory_exists = (root / "test").exists()
    if test_directory_exists:
        blockers.append("A final test directory was created from development data.")
    if len(records) != len(manifest_records):
        blockers.append(
            "Materialized record count does not match the materialization manifest."
        )

    gem_splits: dict[str, set[str]] = defaultdict(set)
    frame_splits: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    sample_splits = {}
    for record in records:
        gem_splits[record["gem_id"]].add(record["split"])
        frame_splits[
            (record["gem_id"], record["video_id"], record["frame_id"])
        ].add(record["split"])
        hash_splits[record["image_sha256"]].add(record["split"])
        sample_splits[record["sample_id"]] = record["split"]
    gem_overlap = sorted(key for key, value in gem_splits.items() if len(value) > 1)
    frame_overlap = sorted(
        key for key, value in frame_splits.items() if len(value) > 1
    )
    exact_overlap = sorted(
        key for key, value in hash_splits.items() if len(value) > 1
    )

    perceptual_path = root / "manifests" / "perceptual_duplicate_candidates.csv"
    if not perceptual_path.is_file() and plan_dir:
        perceptual_path = plan_dir / "perceptual_duplicate_candidates.csv"
    perceptual_cross = []
    if perceptual_path.is_file():
        with perceptual_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            for row in csv.DictReader(handle):
                left = sample_splits.get(row.get("left_sample_id"))
                right = sample_splits.get(row.get("right_sample_id"))
                if left and right and left != right:
                    perceptual_cross.append(row.get("candidate_id"))

    expected_counts = materialization.get("split_counts") or {}
    if split_counts.get("train") != expected_counts.get("train"):
        blockers.append("Train image count does not match the approved manifest.")
    if split_counts.get("valid") != expected_counts.get("valid"):
        blockers.append("Validation image count does not match the approved manifest.")
    if expected_counts.get("test") not in (None, 0):
        blockers.append("Materialization manifest contains current test images.")
    if gem_overlap:
        blockers.append("Gem identities cross train and validation.")
    if frame_overlap:
        blockers.append("Source frames cross train and validation.")
    if exact_overlap:
        blockers.append("Exact duplicate images cross train and validation.")
    if perceptual_cross:
        blockers.append("Flagged perceptual pairs cross train and validation.")
    if issues:
        blockers.append("One or more derived files failed integrity validation.")
    blockers = list(dict.fromkeys(blockers))

    class_distribution = []
    for split in DEVELOPMENT_SPLITS:
        for class_id, class_name in enumerate(class_names):
            class_distribution.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "image_count": class_images[(split, class_id)],
                    "object_count": class_objects[(split, class_id)],
                }
            )
    return {
        "schema_version": "1.0",
        "audit_type": "materialized_development_dataset",
        "created_utc": utc_timestamp(),
        "dataset_root": str(root),
        "data_yaml": str(data_yaml),
        "class_names": class_names,
        "split_counts": {
            "train": split_counts.get("train", 0),
            "valid": split_counts.get("valid", 0),
            "test": 0,
        },
        "pair_count": len(records),
        "manifest_record_count": len(manifest_records),
        "class_distribution": class_distribution,
        "issues": issues,
        "leakage": {
            "gem_overlap_count": len(gem_overlap),
            "exact_duplicate_overlap_count": len(exact_overlap),
            "source_frame_overlap_count": len(frame_overlap),
            "flagged_perceptual_pair_overlap_count": len(perceptual_cross),
        },
        "test_directory_exists": test_directory_exists,
        "labels_preserved_byte_for_byte": not any(
            "label_hash_mismatch" in row["reason"] for row in issues
        ),
        "summary": {
            "status": "pass" if not blockers else "blocked",
            "integrity_verified": not blockers,
            "blocker_count": len(blockers),
            "blockers": blockers,
            "training_occurred": False,
            "weights_downloaded": False,
            "final_metrics_calculated": False,
        },
    }


def run_materialized_dataset_audit(
    *,
    dataset_root: str | Path,
    plan_directory: str | Path | None,
    output_directory: str | Path,
    repo_root: str | Path,
    command: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    report = audit_materialized_development_dataset(
        dataset_root,
        plan_directory=plan_directory,
    )
    output = Path(output_directory)
    if not output.is_absolute():
        output = Path(repo_root) / output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "report": output / "materialized_dataset_audit.json",
        "class_distribution": output / "materialized_class_distribution.csv",
        "issues": output / "materialized_dataset_issues.csv",
        "markdown": output / "audit_summary.md",
        "manifest": output / "manifest.json",
    }
    atomic_write_json(files["report"], report)
    atomic_write_csv(
        files["class_distribution"],
        report["class_distribution"],
        CLASS_DISTRIBUTION_COLUMNS,
    )
    atomic_write_csv(
        files["issues"],
        report["issues"],
        ("split", "sample_id", "reason"),
    )
    atomic_write_text(
        files["markdown"],
        "\n".join(
            [
                "# Materialized Development Dataset Audit",
                "",
                f"- Status: {report['summary']['status']}",
                f"- Integrity verified: {str(report['summary']['integrity_verified']).lower()}",
                f"- Train images: {report['split_counts']['train']}",
                f"- Validation images: {report['split_counts']['valid']}",
                "- Final test images: 0",
                f"- Issues: {len(report['issues'])}",
                f"- Gem overlap: {report['leakage']['gem_overlap_count']}",
                f"- Exact duplicate overlap: {report['leakage']['exact_duplicate_overlap_count']}",
                f"- Source-frame overlap: {report['leakage']['source_frame_overlap_count']}",
                f"- Flagged perceptual overlap: {report['leakage']['flagged_perceptual_pair_overlap_count']}",
                "- Training occurred: false",
                "- Final metrics calculated: false",
                "",
            ]
        ),
    )
    write_manifest(
        path=files["manifest"],
        repo_root=repo_root,
        tool_name="quartz_materialized_development_dataset_audit",
        command=command,
        tool_sources=(
            "backend/research/candidate_dataset.py",
            "backend/research_benchmark.py",
        ),
        inputs=(
            ("data_yaml", Path(dataset_root) / "data.yaml"),
            (
                "materialization_manifest",
                Path(dataset_root)
                / "manifests"
                / "materialization_manifest.json",
            ),
        ),
        outputs=tuple(
            (name, path)
            for name, path in files.items()
            if name != "manifest"
        ),
        extra={
            "integrity_verified": report["summary"]["integrity_verified"],
            "training_occurred": False,
            "weights_downloaded": False,
            "final_metrics_calculated": False,
        },
    )
    return report, files
