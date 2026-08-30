"""Class-aware detector policy and traceability primitives."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

POLICIES = {
    "yolo_only",
    "opencv_only",
    "union",
    "gated_union",
    "fallback",
    "legacy_union_all",
}
TAXONOMY_STATUSES = {
    "confirmed_defect",
    "confirmed_non_defect",
    "provisional_defect",
    "unknown",
}
MODEL_FAILURE_EVENTS = {"model_unavailable", "model_load_error", "inference_error"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _as_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


@dataclass(frozen=True)
class TaxonomyEntry:
    class_id: int
    class_name: str
    semantic_definition: str = ""
    taxonomy_status: str = "unknown"
    is_visible_defect: bool = False
    eligible_for_3d_mapping: bool = False
    eligible_for_no_cut_zone: bool = False
    minimum_confidence: float | None = None
    expert_confirmed: bool = False
    confirmed_by: str = ""
    confirmed_date: str = ""
    notes: str = ""


@dataclass
class DetectionRecord:
    prediction_id: str
    image_id: str
    source_image_path: str
    source: str
    class_id: int | None
    class_name: str
    confidence: float | None
    mask_path: str
    mask_width: int
    mask_height: int
    mask_area_pixels: int
    taxonomy_status: str = "unknown"
    eligible_for_visualization: bool = False
    eligible_for_3d_mapping: bool = False
    eligible_for_no_cut_zone: bool = False
    accepted_for_visualization: bool = False
    accepted_for_3d_mapping: bool = False
    accepted_for_no_cut_zone: bool = False
    rejection_reason: str | None = None
    policy_name: str = ""
    model_path: str | None = None
    model_sha256: str | None = None
    model_version: str | None = None
    threshold: float | None = None
    created_at_utc: str = field(default_factory=utc_now)
    visualization_mask_path: str | None = None
    mapping_mask_path: str | None = None
    no_cut_mask_path: str | None = None
    policy_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DetectionRecord":
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})


def load_taxonomy(path: str | Path) -> dict[int, TaxonomyEntry]:
    taxonomy: dict[int, TaxonomyEntry] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            status = str(row.get("taxonomy_status", "unknown")).strip()
            if status not in TAXONOMY_STATUSES:
                raise ValueError(f"Unsupported taxonomy_status: {status}")
            entry = TaxonomyEntry(
                class_id=int(row["class_id"]),
                class_name=str(row["class_name"]).strip(),
                semantic_definition=str(row.get("semantic_definition", "")).strip(),
                taxonomy_status=status,
                is_visible_defect=_as_bool(row.get("is_visible_defect", False)),
                eligible_for_3d_mapping=_as_bool(
                    row.get("eligible_for_3d_mapping", False)
                ),
                eligible_for_no_cut_zone=_as_bool(
                    row.get("eligible_for_no_cut_zone", False)
                ),
                minimum_confidence=_as_optional_float(
                    row.get("minimum_confidence")
                ),
                expert_confirmed=_as_bool(row.get("expert_confirmed", False)),
                confirmed_by=str(row.get("confirmed_by", "")).strip(),
                confirmed_date=str(row.get("confirmed_date", "")).strip(),
                notes=str(row.get("notes", "")).strip(),
            )
            if entry.class_id in taxonomy:
                raise ValueError(f"Duplicate taxonomy class_id: {entry.class_id}")
            taxonomy[entry.class_id] = entry
    return taxonomy


def load_policy_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Detector policy configuration must be a JSON object.")
    validate_policy_config(config)
    config["_config_path"] = str(config_path)
    return config


def validate_policy_config(config: dict[str, Any]) -> None:
    policy = config.get("policy")
    if policy not in POLICIES:
        raise ValueError(f"Unsupported detector policy: {policy}")
    if config.get("unknown_class_action", "exclude") != "exclude":
        raise ValueError("Only safe unknown_class_action='exclude' is supported.")
    default = float(config.get("minimum_confidence_default", 0.25))
    if not 0.0 <= default <= 1.0:
        raise ValueError("minimum_confidence_default must be within [0, 1].")
    for value in config.get("minimum_confidence_by_class", {}).values():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("Per-class confidence values must be within [0, 1].")
    if (
        policy == "legacy_union_all"
        and config.get("strict_research_mode", True)
    ):
        raise ValueError(
            "legacy_union_all is rejected while strict_research_mode is enabled."
        )
    if (
        policy == "legacy_union_all"
        and not config.get("legacy", {}).get("allow_legacy_union_all", False)
    ):
        raise ValueError("legacy_union_all requires explicit legacy permission.")


def resolve_taxonomy_path(config: dict[str, Any], repo_root: str | Path) -> Path:
    candidate = Path(str(config["taxonomy_path"]))
    if not candidate.is_absolute():
        candidate = Path(repo_root) / candidate
    return candidate.resolve()


def approved_no_cut_cloud_path(
    job_path: str | Path,
) -> tuple[Path | None, str]:
    """Return a no-cut cloud only when structured policy provenance permits it."""

    job = Path(job_path)
    summary_path = job / "detections" / "policy_summary.json"
    if not summary_path.exists():
        return None, "missing_policy_summary"
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, ValueError):
        return None, "invalid_policy_summary"
    if int(summary.get("no_cut_count", 0)) <= 0:
        return None, "no_approved_no_cut_masks"
    if summary.get("legacy_input_used"):
        return None, "legacy_input_excluded"
    if (
        summary.get("strict_research_mode", True)
        and summary.get("provisional_input_used")
    ):
        return None, "provisional_input_excluded_in_strict_mode"
    path = job / "dense" / "no_cut_defects.ply"
    if not path.exists():
        return None, "approved_no_cut_cloud_missing"
    return path, "approved"


def threshold_for(
    record: DetectionRecord,
    entry: TaxonomyEntry | None,
    config: dict[str, Any],
) -> float | None:
    if record.source != "yolo":
        return None
    by_class = config.get("minimum_confidence_by_class", {})
    for key in (str(record.class_id), record.class_name):
        if key in by_class:
            return float(by_class[key])
    if entry is not None and entry.minimum_confidence is not None:
        return float(entry.minimum_confidence)
    return float(config.get("minimum_confidence_default", 0.25))


def _base_yolo_decision(
    record: DetectionRecord,
    taxonomy: dict[int, TaxonomyEntry],
    config: dict[str, Any],
) -> None:
    entry = taxonomy.get(record.class_id) if record.class_id is not None else None
    record.threshold = threshold_for(record, entry, config)
    if entry is None:
        record.taxonomy_status = "unknown"
        record.rejection_reason = "class_missing_from_taxonomy"
        return

    record.class_name = entry.class_name
    record.taxonomy_status = entry.taxonomy_status
    record.eligible_for_visualization = entry.is_visible_defect
    record.eligible_for_3d_mapping = entry.eligible_for_3d_mapping
    record.eligible_for_no_cut_zone = entry.eligible_for_no_cut_zone

    if (
        record.confidence is None
        or record.threshold is None
        or record.confidence < record.threshold
    ):
        record.rejection_reason = "below_confidence_threshold"
        return
    if entry.taxonomy_status == "confirmed_non_defect":
        record.rejection_reason = "confirmed_non_defect"
        return
    if entry.taxonomy_status == "unknown":
        record.rejection_reason = "unknown_taxonomy"
        return

    if entry.taxonomy_status == "provisional_defect":
        record.accepted_for_visualization = bool(
            config.get("allow_provisional_for_visualization", True)
            and record.eligible_for_visualization
        )
        record.accepted_for_3d_mapping = bool(
            not config.get("strict_research_mode", True)
            and record.eligible_for_3d_mapping
        )
        record.accepted_for_no_cut_zone = bool(
            not config.get("strict_research_mode", True)
            and config.get("allow_provisional_for_no_cut", False)
            and record.eligible_for_no_cut_zone
        )
        record.rejection_reason = (
            None
            if record.accepted_for_visualization
            else "provisional_not_allowed"
        )
        return

    if entry.taxonomy_status != "confirmed_defect" or not entry.expert_confirmed:
        record.rejection_reason = "taxonomy_not_expert_confirmed"
        return

    record.accepted_for_visualization = record.eligible_for_visualization
    record.accepted_for_3d_mapping = record.eligible_for_3d_mapping
    record.accepted_for_no_cut_zone = (
        record.eligible_for_no_cut_zone
        and record.accepted_for_3d_mapping
    )
    if not any(
        (
            record.accepted_for_visualization,
            record.accepted_for_3d_mapping,
            record.accepted_for_no_cut_zone,
        )
    ):
        record.rejection_reason = "class_not_eligible_for_output"


def _base_opencv_decision(
    record: DetectionRecord,
    config: dict[str, Any],
) -> None:
    settings = config.get("opencv", {})
    record.taxonomy_status = "provisional_defect"
    record.eligible_for_visualization = bool(
        settings.get("eligible_for_visualization", False)
    )
    record.eligible_for_3d_mapping = bool(
        settings.get("eligible_for_3d_mapping", False)
    )
    record.eligible_for_no_cut_zone = bool(
        settings.get("eligible_for_no_cut", False)
    )
    if not settings.get("enabled", False):
        record.rejection_reason = "opencv_disabled"
        return
    if record.mask_area_pixels < int(settings.get("minimum_area_pixels", 0)):
        record.rejection_reason = "below_opencv_minimum_area"
        return
    strict = bool(config.get("strict_research_mode", True))
    record.accepted_for_visualization = record.eligible_for_visualization
    record.accepted_for_3d_mapping = (
        record.eligible_for_3d_mapping and not strict
    )
    record.accepted_for_no_cut_zone = (
        record.eligible_for_no_cut_zone
        and record.accepted_for_3d_mapping
        and (
            not strict
            or config.get("allow_provisional_for_no_cut", False)
        )
    )
    if not any(
        (
            record.accepted_for_visualization,
            record.accepted_for_3d_mapping,
            record.accepted_for_no_cut_zone,
        )
    ):
        record.rejection_reason = "opencv_not_eligible_for_output"


def mask_iou(first_path: str | Path, second_path: str | Path) -> float:
    first = np.asarray(Image.open(first_path).convert("L")) > 127
    second_image = Image.open(second_path).convert("L")
    if second_image.size != (first.shape[1], first.shape[0]):
        second_image = second_image.resize(
            (first.shape[1], first.shape[0]),
            Image.Resampling.NEAREST,
        )
    second = np.asarray(second_image) > 127
    union = np.count_nonzero(first | second)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(first & second) / union)


def _fallback_active(event: str, config: dict[str, Any]) -> bool:
    settings = config.get("opencv", {})
    if event in MODEL_FAILURE_EVENTS:
        key = {
            "model_unavailable": "fallback_on_model_unavailable",
            "model_load_error": "fallback_on_model_unavailable",
            "inference_error": "fallback_on_inference_error",
        }[event]
        return bool(settings.get(key, False))
    if event == "zero_detections":
        return bool(settings.get("fallback_on_zero_detections", False))
    return False


def apply_policy(
    records: list[DetectionRecord],
    taxonomy: dict[int, TaxonomyEntry],
    config: dict[str, Any],
    image_events: dict[str, str] | None = None,
) -> list[DetectionRecord]:
    """Apply deterministic channel decisions to raw prediction records."""

    validate_policy_config(config)
    policy = str(config["policy"])
    events = image_events or {}

    for record in records:
        record.policy_name = policy
        record.accepted_for_visualization = False
        record.accepted_for_3d_mapping = False
        record.accepted_for_no_cut_zone = False
        record.rejection_reason = None
        record.policy_metadata = dict(record.policy_metadata)
        if record.source == "yolo":
            _base_yolo_decision(record, taxonomy, config)
        elif record.source == "opencv":
            _base_opencv_decision(record, config)
        else:
            record.rejection_reason = "unsupported_source"

    yolo_by_image: dict[str, list[DetectionRecord]] = defaultdict(list)
    for record in records:
        if record.source == "yolo":
            yolo_by_image[record.image_id].append(record)

    for record in records:
        if policy == "yolo_only" and record.source != "yolo":
            _reject_channels(record, "excluded_by_yolo_only")
        elif policy == "opencv_only" and record.source != "opencv":
            _reject_channels(record, "excluded_by_opencv_only")
        elif policy == "gated_union" and record.source == "opencv":
            candidates = [
                item
                for item in yolo_by_image.get(record.image_id, [])
                if item.accepted_for_3d_mapping
                or item.accepted_for_no_cut_zone
            ]
            overlaps = [
                mask_iou(record.mask_path, item.mask_path) for item in candidates
            ]
            best = max(overlaps, default=0.0)
            minimum = float(
                config.get("opencv", {}).get("gated_union_min_iou", 0.1)
            )
            record.policy_metadata.update(
                {
                    "overlap_metric": "mask_iou",
                    "overlap_value": round(best, 6),
                    "overlap_threshold": minimum,
                }
            )
            if best < minimum:
                _reject_channels(record, "gated_union_overlap_below_threshold")
        elif policy == "fallback":
            event = events.get(record.image_id, "success")
            if record.source == "opencv":
                record.policy_metadata["fallback_event"] = event
                if not _fallback_active(event, config):
                    _reject_channels(record, "fallback_not_activated")
            elif record.source == "yolo" and event in MODEL_FAILURE_EVENTS:
                _reject_channels(record, "yolo_unavailable_for_image")
        elif policy == "legacy_union_all":
            record.accepted_for_visualization = True
            record.accepted_for_3d_mapping = True
            record.accepted_for_no_cut_zone = True
            record.rejection_reason = None
            record.policy_metadata["warning"] = "unsafe_legacy_policy"

    return records


def _reject_channels(record: DetectionRecord, reason: str) -> None:
    record.accepted_for_visualization = False
    record.accepted_for_3d_mapping = False
    record.accepted_for_no_cut_zone = False
    record.rejection_reason = reason


def _copy_channel_mask(
    record: DetectionRecord,
    detections_dir: Path,
    channel: str,
) -> str:
    source = Path(record.mask_path)
    destination = detections_dir / channel / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination.relative_to(detections_dir.parent).as_posix()


def materialize_policy_outputs(
    records: list[DetectionRecord],
    job_path: str | Path,
) -> None:
    detections_dir = Path(job_path) / "detections"
    for channel in (
        "raw_masks",
        "visualization_masks",
        "mapping_masks",
        "no_cut_masks",
    ):
        (detections_dir / channel).mkdir(parents=True, exist_ok=True)
    for record in records:
        if record.accepted_for_visualization:
            record.visualization_mask_path = _copy_channel_mask(
                record, detections_dir, "visualization_masks"
            )
        if record.accepted_for_3d_mapping:
            record.mapping_mask_path = _copy_channel_mask(
                record, detections_dir, "mapping_masks"
            )
        if record.accepted_for_no_cut_zone:
            record.no_cut_mask_path = _copy_channel_mask(
                record, detections_dir, "no_cut_masks"
            )


def policy_summary(
    records: list[DetectionRecord],
    config: dict[str, Any],
    taxonomy: dict[int, TaxonomyEntry] | None = None,
    taxonomy_path: str | Path | None = None,
    model_sha256: str | None = None,
    runtime_warnings: list[str] | None = None,
) -> dict[str, Any]:
    counts_by_class = Counter(
        record.class_name for record in records if record.source == "yolo"
    )
    counts_by_source = Counter(record.source for record in records)
    confirmed_no_cut = [
        record
        for record in records
        if record.accepted_for_no_cut_zone
        and record.taxonomy_status == "confirmed_defect"
    ]
    warnings = list(runtime_warnings or [])
    if config.get("policy") == "legacy_union_all":
        warnings.append("unsafe_legacy_policy")
    provisional_used = any(
        record.taxonomy_status == "provisional_defect"
        and (
            record.accepted_for_3d_mapping
            or record.accepted_for_no_cut_zone
        )
        for record in records
    )
    if provisional_used:
        warnings.append("provisional_input_used")
    taxonomy_version = None
    if taxonomy_path and Path(taxonomy_path).exists():
        from hashlib import sha256

        taxonomy_version = sha256(Path(taxonomy_path).read_bytes()).hexdigest()

    has_confirmed_taxonomy = bool(
        confirmed_no_cut
        or any(
            entry.taxonomy_status == "confirmed_defect"
            and entry.expert_confirmed
            for entry in (taxonomy or {}).values()
        )
    )
    return {
        "policy": config.get("policy"),
        "strict_research_mode": bool(
            config.get("strict_research_mode", True)
        ),
        "taxonomy_version": taxonomy_version,
        "model_sha256": model_sha256,
        "raw_prediction_count": len(records),
        "visualization_count": sum(
            record.accepted_for_visualization for record in records
        ),
        "mapping_count": sum(
            record.accepted_for_3d_mapping for record in records
        ),
        "no_cut_count": sum(
            record.accepted_for_no_cut_zone for record in records
        ),
        "rejected_count": sum(
            not any(
                (
                    record.accepted_for_visualization,
                    record.accepted_for_3d_mapping,
                    record.accepted_for_no_cut_zone,
                )
            )
            for record in records
        ),
        "counts_by_class": dict(sorted(counts_by_class.items())),
        "counts_by_source": dict(sorted(counts_by_source.items())),
        "legacy_input_used": config.get("policy") == "legacy_union_all",
        "provisional_input_used": provisional_used,
        "claim_status": "unavailable",
        "reason": (
            (
                "Detector performance is not evaluated in Phase 2; zero accepted "
                "evidence does not prove a defect-free stone."
                if not confirmed_no_cut
                else "Detector performance is not evaluated in Phase 2."
            )
            if has_confirmed_taxonomy
            else "No confirmed defect taxonomy was supplied."
        ),
        "warnings": sorted(set(warnings)),
    }
