"""Deterministic grouped split planning without modifying source datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common import atomic_write_json, normalize_path, sha256_file, utc_timestamp
from .dataset_audit import augmentation_parent_id, load_dataset_yaml
from .duplicate_review import GROUPING_DECISIONS
from .readiness_common import (
    aggregate_file_hash,
    atomic_write_csv,
    atomic_write_text,
    find_repo_root,
    load_json_config,
    resolve_path,
    stable_json_sha256,
    write_manifest,
)
from .taxonomy_validation import validate_taxonomy_approval

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MANIFEST_COLUMNS = (
    "sample_id",
    "file_path",
    "label_path",
    "image_sha256",
    "label_sha256",
    "specimen_id",
    "capture_session_id",
    "recording_id",
    "source_frame_id",
    "augmentation_parent_id",
    "exact_duplicate_group_id",
    "near_duplicate_group_id",
    "original_split",
    "assigned_split",
    "atomic_group_id",
    "grouping_level",
    "class_ids",
    "object_count",
)


class UnionFind:
    def __init__(self, values: Iterable[str]):
        items = list(values)
        self.parent = {value: value for value in items}
        self.rank = {value: 0 for value in items}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        left = self.find(first)
        right = self.find(second)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def load_grouped_split_config(
    path: str | Path,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    return load_json_config(path, output_override=output_override)


def _resolve_yaml_location(
    value: str | Path,
    yaml_path: Path,
    yaml_data: dict[str, Any],
) -> Path:
    path = Path(str(value).replace("\\", os.sep))
    if path.is_absolute():
        return path.resolve()
    base = yaml_path.parent
    configured_root = yaml_data.get("path")
    if configured_root:
        root = Path(str(configured_root).replace("\\", os.sep))
        base = root if root.is_absolute() else yaml_path.parent / root
    return (base / path).resolve()


def _yaml_class_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(value) for value in names]
    return []


def _label_for_image(path: Path) -> Path:
    if path.parent.name.casefold() == "images":
        return path.parent.parent / "labels" / f"{path.stem}.txt"
    return path.parent / "labels" / f"{path.stem}.txt"


def _label_counts(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            tokens = line.split()
            if not tokens:
                continue
            try:
                counts[int(tokens[0])] += 1
            except ValueError:
                continue
    return counts


def _metadata_rows(
    manifest_path: Path | None,
    repo_root: Path,
) -> tuple[dict[Path, dict[str, str]], list[str], dict[str, float]]:
    if manifest_path is None:
        return {}, [], {
            "specimen_id_coverage": 0.0,
            "capture_session_id_coverage": 0.0,
            "recording_id_coverage": 0.0,
            "source_frame_id_coverage": 0.0,
            "expert_review_coverage": 0.0,
        }
    if not manifest_path.exists():
        return {}, ["Configured metadata manifest was not found."], {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {key: str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    blockers = []
    ids = Counter(row.get("sample_id", "") for row in rows)
    duplicate_ids = [key for key, count in ids.items() if key and count > 1]
    if duplicate_ids:
        blockers.append(f"Duplicate manifest sample IDs: {sorted(duplicate_ids)}")
    mapping = {}
    for row in rows:
        file_path = resolve_path(row.get("file_path"), repo_root)
        if file_path is None or not file_path.exists():
            blockers.append(
                f"Manifest image is missing: {row.get('file_path', '')}"
            )
            continue
        mapping[file_path] = row

    total = max(len(rows), 1)
    def coverage(field: str, reviewed: bool = False) -> float:
        count = 0
        for row in rows:
            value = row.get(field, "")
            if reviewed:
                count += value.casefold() in {
                    "approved",
                    "corrected",
                    "rejected",
                    "needs_second_review",
                }
            else:
                count += bool(value)
        return round(100.0 * count / total, 2) if rows else 0.0

    coverage_values = {
        "specimen_id_coverage": coverage("specimen_id"),
        "capture_session_id_coverage": coverage("capture_session_id"),
        "recording_id_coverage": coverage("recording_id"),
        "source_frame_id_coverage": coverage("source_frame_id"),
        "expert_review_coverage": coverage("expert_review_status", True),
    }
    return mapping, blockers, coverage_values


def collect_dataset_records(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repo_root = Path(config["_repo_root"]).resolve()
    yaml_path = resolve_path(config.get("dataset_yaml"), repo_root)
    if yaml_path is None or not yaml_path.exists():
        raise ValueError("Configured dataset_yaml does not exist.")
    yaml_data = load_dataset_yaml(yaml_path)
    class_names = _yaml_class_names(yaml_data)
    manifest_path = resolve_path(config.get("metadata_manifest"), repo_root)
    metadata, blockers, coverage = _metadata_rows(manifest_path, repo_root)
    records = []
    seen_paths = set()
    for yaml_key, split in (("train", "train"), ("val", "valid"), ("valid", "valid"), ("test", "test")):
        if yaml_key not in yaml_data or (split == "valid" and any(
            record["original_split"] == "valid" for record in records
        )):
            continue
        locations = yaml_data[yaml_key]
        if not isinstance(locations, list):
            locations = [locations]
        for location in locations:
            image_dir = _resolve_yaml_location(location, yaml_path, yaml_data)
            if not image_dir.exists():
                blockers.append(f"Image directory is missing: {image_dir}")
                continue
            for image_path in sorted(
                path for path in image_dir.rglob("*")
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
            ):
                image_path = image_path.resolve()
                if image_path in seen_paths:
                    continue
                seen_paths.add(image_path)
                label_path = _label_for_image(image_path).resolve()
                if not label_path.exists():
                    blockers.append(
                        f"Label is missing for {normalize_path(image_path, repo_root)}"
                    )
                relative = normalize_path(image_path, repo_root)
                metadata_row = metadata.get(image_path, {})
                sample_id = metadata_row.get("sample_id") or (
                    "sample-" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
                )
                counts = _label_counts(label_path)
                record = {
                    "sample_id": sample_id,
                    "path": image_path,
                    "label_path_object": label_path,
                    "file_path": relative,
                    "label_path": normalize_path(label_path, repo_root),
                    "image_sha256": sha256_file(image_path),
                    "label_sha256": (
                        sha256_file(label_path) if label_path.exists() else ""
                    ),
                    "specimen_id": metadata_row.get("specimen_id", ""),
                    "capture_session_id": metadata_row.get("capture_session_id", ""),
                    "recording_id": metadata_row.get("recording_id", ""),
                    "source_frame_id": metadata_row.get("source_frame_id", ""),
                    "augmentation_parent_id": (
                        metadata_row.get("augmentation_parent_id")
                        or augmentation_parent_id(image_path.stem)
                    ),
                    "exact_duplicate_group_id": "",
                    "near_duplicate_group_id": metadata_row.get(
                        "near_duplicate_group_id", ""
                    ),
                    "original_split": split,
                    "class_counts": counts,
                    "class_ids": sorted(counts),
                    "object_count": sum(counts.values()),
                }
                records.append(record)
    sample_counts = Counter(record["sample_id"] for record in records)
    duplicates = sorted(
        sample_id for sample_id, count in sample_counts.items() if count > 1
    )
    if duplicates:
        blockers.append(f"Duplicate sample IDs in dataset records: {duplicates}")
    return records, {
        "repo_root": repo_root,
        "dataset_yaml": yaml_path,
        "class_names": class_names,
        "metadata_manifest": manifest_path,
        "metadata_coverage": coverage,
        "blockers": blockers,
    }


def _join_field(
    union: UnionFind,
    records: list[dict[str, Any]],
    field: str,
) -> int:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        value = str(record.get(field, "")).strip()
        if value:
            grouped[value].append(record["sample_id"])
    joins = 0
    for members in grouped.values():
        for member in members[1:]:
            union.union(members[0], member)
            joins += 1
    return joins


def _load_near_review(
    path: Path | None,
) -> tuple[list[dict[str, str]], int, list[str]]:
    if path is None or not path.exists():
        return [], 0, []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {key: str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    unresolved = sum(
        row.get("review_decision") in {"", "unreviewed", "uncertain"}
        for row in rows
    )
    invalid = [
        row.get("candidate_group_id", "")
        for row in rows
        if row.get("review_decision") not in {
            "same_source",
            "same_specimen",
            "different_specimen",
            "augmentation_related",
            "not_duplicate",
            "uncertain",
            "unreviewed",
        }
    ]
    return rows, unresolved, invalid


def _grouping_level(
    records: list[dict[str, Any]],
) -> tuple[str, str]:
    def complete(field: str) -> bool:
        return bool(records) and all(str(record.get(field, "")).strip() for record in records)

    if complete("specimen_id"):
        return "specimen", "All samples have supplied specimen identities."
    if complete("recording_id"):
        return "recording", (
            "Specimen identity is incomplete; all samples have recording lineage."
        )
    if complete("source_frame_id") or complete("augmentation_parent_id"):
        return "provisional_parent_grouped", (
            "Only source-frame/augmentation-parent grouping is complete. "
            "This is not specimen-independent."
        )
    return "unavailable", "No complete approved grouping lineage is available."


def build_atomic_groups(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    union = UnionFind(record["sample_id"] for record in records)
    relations = {}
    fields = [
        "specimen_id",
        "capture_session_id",
        "recording_id",
        "source_frame_id",
        "augmentation_parent_id",
    ]
    fields.extend(
        field for field in config.get("additional_grouping_fields", [])
        if field not in fields
    )
    for field in fields:
        relations[field] = _join_field(union, records, field)

    hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        hashes[record["image_sha256"]].append(record)
    exact_group_count = 0
    for digest, members in hashes.items():
        if len(members) < 2:
            continue
        group_id = "exact-" + digest[:12]
        exact_group_count += 1
        for record in members:
            record["exact_duplicate_group_id"] = group_id
        for member in members[1:]:
            union.union(members[0]["sample_id"], member["sample_id"])

    by_path = {
        record["file_path"]: record for record in records
    }
    review_path = resolve_path(
        config.get("near_duplicate_review"),
        config["_repo_root"],
    )
    review_required_missing = bool(
        config.get("near_duplicate_review")
        and (review_path is None or not review_path.exists())
    )
    review_rows, unresolved, invalid_decisions = _load_near_review(review_path)
    near_joins = 0
    for row in review_rows:
        if row.get("review_decision") not in GROUPING_DECISIONS:
            continue
        left = by_path.get(row.get("image_a", ""))
        right = by_path.get(row.get("image_b", ""))
        if left and right:
            union.union(left["sample_id"], right["sample_id"])
            group_id = row.get("candidate_group_id", "")
            left["near_duplicate_group_id"] = group_id
            right["near_duplicate_group_id"] = group_id
            near_joins += 1

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[union.find(record["sample_id"])].append(record)
    canonical = {}
    for members in groups.values():
        sample_ids = sorted(record["sample_id"] for record in members)
        group_id = "group-" + stable_json_sha256(sample_ids)[:12]
        canonical[group_id] = sorted(members, key=lambda item: item["sample_id"])
    return dict(sorted(canonical.items())), {
        "field_joins": relations,
        "exact_duplicate_group_count": exact_group_count,
        "near_duplicate_join_count": near_joins,
        "near_duplicate_unresolved_count": unresolved,
        "invalid_review_decisions": invalid_decisions,
        "review_required_missing": review_required_missing,
        "review_path": review_path,
    }


def _normalized_ratios(config: dict[str, Any]) -> dict[str, float]:
    raw = config.get("ratios", {"train": 0.8, "valid": 0.1, "test": 0.1})
    values = {split: max(0.0, float(raw.get(split, 0.0))) for split in SPLITS}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Split ratios must have a positive sum.")
    return {split: values[split] / total for split in SPLITS}


def assign_groups(
    groups: dict[str, list[dict[str, Any]]],
    class_count: int,
    ratios: dict[str, float],
    seed: int,
) -> tuple[dict[str, str], list[str]]:
    total_images = sum(len(members) for members in groups.values())
    total_objects: Counter[int] = Counter()
    groups_per_class: Counter[int] = Counter()
    stats = {}
    for group_id, members in groups.items():
        objects: Counter[int] = Counter()
        for record in members:
            objects.update(record["class_counts"])
        stats[group_id] = {"images": len(members), "objects": objects}
        for class_id in objects:
            groups_per_class[class_id] += 1
        total_objects.update(objects)
    rare_warnings = [
        (
            f"Class {class_id} appears in only {groups_per_class[class_id]} "
            "atomic group(s); all three splits cannot contain it."
        )
        for class_id in range(class_count)
        if total_objects[class_id] and groups_per_class[class_id] < len(SPLITS)
    ]
    targets_images = {
        split: total_images * ratios[split] for split in SPLITS
    }
    targets_objects = {
        split: {
            class_id: total_objects[class_id] * ratios[split]
            for class_id in range(class_count)
        }
        for split in SPLITS
    }
    current_images = Counter()
    current_objects: dict[str, Counter[int]] = {
        split: Counter() for split in SPLITS
    }

    def order_key(group_id: str) -> tuple[Any, ...]:
        objects = stats[group_id]["objects"]
        rarity = sum(
            count / max(groups_per_class[class_id], 1)
            for class_id, count in objects.items()
        )
        tie = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
        return (-rarity, -stats[group_id]["images"], tie)

    split_order = sorted(
        SPLITS,
        key=lambda split: hashlib.sha256(
            f"{seed}:{split}".encode("utf-8")
        ).hexdigest(),
    )
    assignments = {}
    for group_id in sorted(groups, key=order_key):
        group = stats[group_id]
        costs = []
        for split in split_order:
            image_values = {
                name: current_images[name] + (
                    group["images"] if name == split else 0
                )
                for name in SPLITS
            }
            image_cost = sum(
                (
                    (image_values[name] - targets_images[name])
                    / max(targets_images[name], 1.0)
                ) ** 2
                for name in SPLITS
            )
            object_cost = 0.0
            for name in SPLITS:
                for class_id in range(class_count):
                    value = current_objects[name][class_id]
                    if name == split:
                        value += group["objects"][class_id]
                    target = targets_objects[name][class_id]
                    object_cost += ((value - target) / max(target, 1.0)) ** 2
            costs.append((image_cost + 0.35 * object_cost, split))
        _, selected = min(costs, key=lambda item: (round(item[0], 12), split_order.index(item[1])))
        assignments[group_id] = selected
        current_images[selected] += group["images"]
        current_objects[selected].update(group["objects"])
    return assignments, rare_warnings


def _taxonomy_status(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    taxonomy = resolve_path(config.get("taxonomy_path"), repo_root)
    approval = resolve_path(config.get("taxonomy_approval"), repo_root)
    yaml_path = resolve_path(config.get("dataset_yaml"), repo_root)
    if taxonomy is None or yaml_path is None:
        return {"approved": False, "blockers": ["Taxonomy inputs are missing."]}
    report = validate_taxonomy_approval(taxonomy, approval, yaml_path)
    return {
        "approved": bool(report["summary"]["taxonomy_approved"]),
        "taxonomy_sha256": report["taxonomy"]["sha256"],
        "blockers": report["summary"]["blockers"],
    }


def plan_grouped_split(config: dict[str, Any]) -> dict[str, Any]:
    records, dataset = collect_dataset_records(config)
    groups, grouping = build_atomic_groups(records, config)
    ratios = _normalized_ratios(config)
    seed = int(config.get("deterministic_seed", 42))
    assignments, rare_warnings = assign_groups(
        groups,
        len(dataset["class_names"]),
        ratios,
        seed,
    )
    grouping_level, grouping_note = _grouping_level(records)
    rows = []
    group_distribution = []
    for group_id, members in groups.items():
        assigned = assignments[group_id]
        group_distribution.append(
            {
                "atomic_group_id": group_id,
                "assigned_split": assigned,
                "image_count": len(members),
                "object_count": sum(record["object_count"] for record in members),
                "class_ids": sorted(
                    {
                        class_id
                        for record in members
                        for class_id in record["class_ids"]
                    }
                ),
            }
        )
        for record in members:
            row = {
                key: record.get(key, "")
                for key in MANIFEST_COLUMNS
            }
            row.update(
                {
                    "assigned_split": assigned,
                    "atomic_group_id": group_id,
                    "grouping_level": grouping_level,
                    "class_ids": record["class_ids"],
                }
            )
            rows.append(row)
    rows.sort(key=lambda item: item["sample_id"])

    leakage = []
    for group_id, members in groups.items():
        assigned = {assignments[group_id]}
        if len(assigned) > 1:
            leakage.append(group_id)
    class_distribution = []
    for split in SPLITS:
        split_rows = [row for row in rows if row["assigned_split"] == split]
        for class_id, class_name in enumerate(dataset["class_names"]):
            class_distribution.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "image_count": sum(
                        class_id in row["class_ids"] for row in split_rows
                    ),
                    "object_count": sum(
                        record["class_counts"][class_id]
                        for record in records
                        if record["sample_id"] in {
                            row["sample_id"] for row in split_rows
                        }
                    ),
                }
            )
    split_counts = Counter(row["assigned_split"] for row in rows)
    taxonomy = _taxonomy_status(config, dataset["repo_root"])
    blockers = list(dataset["blockers"])
    blockers.extend(taxonomy["blockers"])
    if grouping_level == "provisional_parent_grouped":
        blockers.append(
            "Specimen/recording independence is unavailable; grouping is parent-only."
        )
    elif grouping_level == "unavailable":
        blockers.append("No sufficient grouping metadata is available.")
    if grouping["near_duplicate_unresolved_count"]:
        blockers.append(
            f"{grouping['near_duplicate_unresolved_count']} near-duplicate "
            "candidates remain unresolved."
        )
    if grouping["review_required_missing"]:
        blockers.append("Configured near-duplicate review evidence is missing.")
    if grouping["invalid_review_decisions"]:
        blockers.append("Near-duplicate review contains invalid decisions.")
    if leakage:
        blockers.append("Atomic group leakage was detected.")
    preapproval_ready = bool(
        not blockers
        and grouping_level in {"specimen", "recording"}
        and taxonomy["approved"]
    )
    split_approval = config.get("split_approval") or {}
    approval_supplied = bool(
        split_approval.get("approved_by")
        and split_approval.get("approval_date")
    )
    split_approved = bool(preapproval_ready and approval_supplied)
    frozen = bool(split_approved and config.get("request_freeze", False))
    source_files = [
        record["path"] for record in records
    ] + [
        record["label_path_object"]
        for record in records
        if record["label_path_object"].exists()
    ]
    source_hash = aggregate_file_hash(source_files, dataset["repo_root"])
    test_rows = [
        {
            "sample_id": row["sample_id"],
            "image_sha256": row["image_sha256"],
            "label_sha256": row["label_sha256"],
        }
        for row in rows
        if row["assigned_split"] == "test"
    ]
    lock_core = {
        "split_id": "split-" + stable_json_sha256(
            {
                "samples": [
                    (row["sample_id"], row["assigned_split"]) for row in rows
                ],
                "seed": seed,
                "ratios": ratios,
            }
        )[:16],
        "source_dataset_hash": source_hash,
        "taxonomy_hash": taxonomy.get("taxonomy_sha256"),
        "grouping_level": grouping_level,
        "seed": seed,
        "ratios": ratios,
        "test_set_hash": stable_json_sha256(test_rows),
        "approval_status": "approved" if split_approved else "unavailable",
        "frozen": frozen,
    }
    lock_core["lock_hash"] = stable_json_sha256(lock_core)
    return {
        "schema_version": "1.0",
        "audit_type": "grouped_split_plan",
        "created_utc": utc_timestamp(),
        "rows": rows,
        "groups": group_distribution,
        "class_distribution": class_distribution,
        "leakage_audit": {
            "atomic_group_overlap_count": len(leakage),
            "overlapping_groups": leakage,
            "augmentation_parent_overlap_count": 0,
            "exact_duplicate_overlap_count": 0,
            "confirmed_near_duplicate_overlap_count": 0,
            "passes": not leakage,
        },
        "metadata_coverage": dataset["metadata_coverage"],
        "grouping": {
            "level": grouping_level,
            "note": grouping_note,
            **{
                key: value
                for key, value in grouping.items()
                if key != "review_path"
            },
        },
        "summary": {
            "image_count": len(rows),
            "atomic_group_count": len(groups),
            "split_image_counts": dict(split_counts),
            "ratios": ratios,
            "grouping_level": grouping_level,
            "provisional": grouping_level == "provisional_parent_grouped",
            "taxonomy_approved": taxonomy["approved"],
            "near_duplicates_reviewed": (
                grouping["near_duplicate_unresolved_count"] == 0
                and bool(grouping.get("review_path"))
            ),
            "preapproval_ready": preapproval_ready,
            "split_approved": split_approved,
            "frozen": frozen,
            "materialization_allowed": bool(
                frozen and split_approved and not leakage
            ),
            "blocker_count": len(list(dict.fromkeys(blockers))),
            "blockers": list(dict.fromkeys(blockers)),
            "warnings": rare_warnings,
        },
        "split_lock": lock_core,
        "class_names": dataset["class_names"],
        "dataset_yaml": dataset["dataset_yaml"],
        "metadata_manifest": dataset["metadata_manifest"],
        "near_duplicate_review": grouping.get("review_path"),
    }


def _summary_markdown(plan: dict[str, Any]) -> str:
    summary = plan["summary"]
    return "\n".join(
        [
            "# Grouped Dataset Split Plan",
            "",
            f"- Images: {summary['image_count']}",
            f"- Atomic groups: {summary['atomic_group_count']}",
            f"- Grouping level: {summary['grouping_level']}",
            f"- Provisional: {str(summary['provisional']).lower()}",
            f"- Taxonomy approved: {str(summary['taxonomy_approved']).lower()}",
            f"- Split approved: {str(summary['split_approved']).lower()}",
            f"- Test set frozen: {str(summary['frozen']).lower()}",
            f"- Materialization allowed: {str(summary['materialization_allowed']).lower()}",
            f"- Blockers: {summary['blocker_count']}",
            "",
            *[f"- {blocker}" for blocker in summary["blockers"]],
            "",
            plan["grouping"]["note"],
            "",
        ]
    )


def _grouped_yaml(plan: dict[str, Any]) -> str:
    names = json.dumps(plan["class_names"], ensure_ascii=True)
    return "\n".join(
        [
            "# Manifest-only grouped split plan; source dataset is unchanged.",
            "format: grouped_manifest",
            "manifest: grouped_split_manifest.csv",
            f"split_id: {plan['split_lock']['split_id']}",
            f"grouping_level: {plan['grouping']['level']}",
            f"approved: {str(plan['summary']['split_approved']).lower()}",
            f"frozen: {str(plan['summary']['frozen']).lower()}",
            f"nc: {len(plan['class_names'])}",
            f"names: {names}",
            "",
        ]
    )


def write_grouped_split_outputs(
    plan: dict[str, Any],
    config: dict[str, Any],
    output_directory: str | Path,
    command: str,
) -> dict[str, Path]:
    repo_root = Path(config["_repo_root"])
    output = resolve_path(output_directory, repo_root)
    if output is None:
        raise ValueError("An output directory is required.")
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "manifest_csv": output / "grouped_split_manifest.csv",
        "summary": output / "split_summary.json",
        "class_distribution": output / "split_class_distribution.csv",
        "group_distribution": output / "split_group_distribution.csv",
        "leakage": output / "split_leakage_audit.json",
        "lock": output / "split_lock.json",
        "yaml": output / "data_grouped.yaml",
        "markdown": output / "audit_summary.md",
        "evidence_manifest": output / "manifest.json",
    }
    atomic_write_csv(files["manifest_csv"], plan["rows"], MANIFEST_COLUMNS)
    manifest_hash = sha256_file(files["manifest_csv"])
    lock = dict(plan["split_lock"])
    lock["manifest_hash"] = manifest_hash
    lock["generated_utc"] = plan["created_utc"]
    lock["lock_hash"] = stable_json_sha256(
        {key: value for key, value in lock.items() if key != "generated_utc"}
    )
    summary = {
        **plan["summary"],
        "metadata_coverage": plan["metadata_coverage"],
        "grouping": plan["grouping"],
        "split_id": lock["split_id"],
        "manifest_hash": manifest_hash,
    }
    atomic_write_json(files["summary"], summary)
    atomic_write_csv(
        files["class_distribution"],
        plan["class_distribution"],
        ("split", "class_id", "class_name", "image_count", "object_count"),
    )
    atomic_write_csv(
        files["group_distribution"],
        plan["groups"],
        ("atomic_group_id", "assigned_split", "image_count", "object_count", "class_ids"),
    )
    atomic_write_json(files["leakage"], plan["leakage_audit"])
    atomic_write_json(files["lock"], lock)
    atomic_write_text(files["yaml"], _grouped_yaml(plan))
    atomic_write_text(files["markdown"], _summary_markdown(plan))
    write_manifest(
        path=files["evidence_manifest"],
        repo_root=repo_root,
        tool_name="quartz_grouped_split_plan",
        command=command,
        tool_sources=(
            "backend/research/grouped_split.py",
            "backend/research/readiness_common.py",
            "backend/research/duplicate_review.py",
            "backend/research/taxonomy_validation.py",
            "backend/research_benchmark.py",
        ),
        inputs=(
            ("configuration", config.get("_config_path")),
            ("dataset_yaml", config.get("dataset_yaml")),
            ("metadata_manifest", config.get("metadata_manifest")),
            ("near_duplicate_review", config.get("near_duplicate_review")),
            ("taxonomy", config.get("taxonomy_path")),
            ("taxonomy_approval", config.get("taxonomy_approval")),
        ),
        outputs=tuple(
            (name, path)
            for name, path in files.items()
            if name != "evidence_manifest"
        ),
        extra={
            "source_dataset_hash": lock["source_dataset_hash"],
            "original_dataset_modified": False,
        },
    )
    return files


def run_grouped_split_plan(
    config: dict[str, Any],
    output_override: str | Path | None = None,
    command: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    plan = plan_grouped_split(config)
    output = output_override or config.get("output_directory")
    if not output:
        raise ValueError("An output directory is required.")
    files = write_grouped_split_outputs(plan, config, output, command)
    return plan, files


def materialize_grouped_split(
    *,
    plan_directory: str | Path,
    output_directory: str | Path,
    apply: bool = False,
    confirm: bool = False,
    mode: str = "copy",
) -> dict[str, Any]:
    plan_dir = Path(plan_directory).resolve()
    output = Path(output_directory).resolve()
    lock_path = plan_dir / "split_lock.json"
    summary_path = plan_dir / "split_summary.json"
    manifest_path = plan_dir / "grouped_split_manifest.csv"
    if not all(path.exists() for path in (lock_path, summary_path, manifest_path)):
        return {"status": "blocked", "reason": "Grouped split plan files are missing."}
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not apply:
        return {
            "status": "preview",
            "apply": False,
            "output_created": False,
            "materialization_allowed": bool(summary.get("materialization_allowed")),
        }
    if not confirm:
        return {
            "status": "blocked",
            "reason": "A separate explicit confirmation flag is required.",
        }
    if not summary.get("materialization_allowed") or not lock.get("frozen"):
        return {
            "status": "blocked",
            "reason": "Scientific readiness gates do not permit materialization.",
        }
    if mode not in {"copy", "hardlink"}:
        return {"status": "blocked", "reason": f"Unsupported materialization mode: {mode}"}
    if output.exists():
        return {"status": "blocked", "reason": "Output path already exists."}

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    source_root = find_repo_root(plan_dir)
    resolved_sources = []
    destinations = set()
    for row in rows:
        image_source = resolve_path(row.get("file_path"), source_root)
        label_source = resolve_path(row.get("label_path"), source_root)
        if (
            image_source is None
            or label_source is None
            or not image_source.is_file()
            or not label_source.is_file()
        ):
            return {
                "status": "blocked",
                "reason": f"Source pair is missing for {row.get('sample_id', '')}.",
            }
        split = row["assigned_split"]
        pair = (
            output / split / "images" / image_source.name,
            output / split / "labels" / label_source.name,
        )
        if any(destination in destinations for destination in pair):
            return {
                "status": "blocked",
                "reason": f"Output collision for {row.get('sample_id', '')}.",
            }
        destinations.update(pair)
        resolved_sources.append((row, image_source, label_source, pair))
    output.mkdir(parents=True)
    materialized = []
    for row, image_source, label_source, pair in resolved_sources:
        split = row["assigned_split"]
        image_destination, label_destination = pair
        for destination in (image_destination, label_destination):
            if destination.exists():
                raise FileExistsError(f"Output collision: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == "hardlink":
            os.link(image_source, image_destination)
            os.link(label_source, label_destination)
        else:
            shutil.copy2(image_source, image_destination)
            shutil.copy2(label_source, label_destination)
        image_hash = sha256_file(image_destination)
        label_hash = sha256_file(label_destination)
        if image_hash != row["image_sha256"] or label_hash != row["label_sha256"]:
            raise ValueError(f"Materialized hash mismatch for {row['sample_id']}")
        materialized.append(
            {
                "sample_id": row["sample_id"],
                "split": split,
                "image": str(image_destination),
                "label": str(label_destination),
                "image_sha256": image_hash,
                "label_sha256": label_hash,
            }
        )
    atomic_write_json(
        output / "materialization_manifest.json",
        {
            "schema_version": "1.0",
            "split_id": lock["split_id"],
            "mode": mode,
            "file_pair_count": len(materialized),
            "source_files_moved": False,
            "records": materialized,
        },
    )
    return {
        "status": "complete",
        "file_pair_count": len(materialized),
        "source_files_moved": False,
        "output": str(output),
    }
