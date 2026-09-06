"""Pre-COLMAP capture-quality screen for reconstruction batch videos.

This script is intentionally limited to video/file quality signals. It does
not run COLMAP, reconstruction, optimization, masking, defect detection, or use
physical dimensions. It dry-runs the current production frame-selection logic
from backend/video_utils.py and writes consolidated capture-screen evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from video_utils import extract_best_frames, get_sharpness  # noqa: E402


OUT_DIR = REPO_ROOT / "final_research_evidence" / "reconstruction_multi"
INPUT_ROOT = OUT_DIR / "batch_inputs"
CSV_OUT = OUT_DIR / "capture_screen_results.csv"
JSON_OUT = OUT_DIR / "capture_screen_summary.json"
REPORT_OUT = OUT_DIR / "capture_screen_report.md"

EXPECTED_VIDEO_STEMS = ("01", "02", "03", "04")
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi"}
REFERENCE_CAPTURE_IDS = ("QZ-05", "QZ-14", "QZ-08")

MAX_OVERLAP_FRAMES = 8
ORB_FEATURES = 1000
ORB_RATIO_TEST = 0.75

DUPLICATE_DIFF_THRESHOLD = 0.012
LOW_FOREGROUND_PROXY_THRESHOLD = 0.015
VERY_LOW_FOREGROUND_PROXY_THRESHOLD = 0.003
CROPPED_OCCUPANCY_THRESHOLD = 0.72
EDGE_TOUCH_THRESHOLD = 0.20


@dataclass(frozen=True)
class ProductionSettings:
    target_frames: int
    sharpness_threshold: float
    fallback_min_saved: int
    fallback_total_frames: int
    source_sha256: str


@dataclass
class FrameSample:
    frame_index: int
    sharpness: float
    small_gray: np.ndarray
    orb_gray: np.ndarray
    occupancy: float | None
    edge_touch: float | None
    mean_gray: float
    gray_std: float
    dark_percent: float
    bright_percent: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path | str | None) -> str:
    if path is None:
        return ""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def percent(numerator: float, denominator: float) -> float | None:
    if not denominator:
        return None
    return 100.0 * float(numerator) / float(denominator)


def median(values: list[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return float(np.median(np.asarray(clean, dtype=float)))


def percentile(values: list[float | int | None], pct: float) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return float(np.percentile(np.asarray(clean, dtype=float), pct))


def rounded(value: Any, digits: int = 3) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return ""
        return round(float(value), digits)
    return value


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return rel(value)
    return value


def production_settings() -> ProductionSettings:
    source = inspect.getsource(extract_best_frames)
    signature = inspect.signature(extract_best_frames)
    target_default = signature.parameters["target_frames"].default
    if target_default is inspect.Parameter.empty:
        raise RuntimeError("Production extract_best_frames has no target_frames default.")

    sharp_match = re.search(r"sharpness\s*>\s*([0-9]+(?:\.[0-9]+)?)", source)
    fallback_match = re.search(r"saved_count\s*<\s*([0-9]+)", source)
    total_fallback_match = re.search(r"total_frames\s*=\s*([0-9]+)\s*(?:#.*)?\n", source)
    if not sharp_match or not fallback_match:
        raise RuntimeError("Could not derive production sharpness/fallback settings.")

    return ProductionSettings(
        target_frames=int(target_default),
        sharpness_threshold=float(sharp_match.group(1)),
        fallback_min_saved=int(fallback_match.group(1)),
        fallback_total_frames=int(total_fallback_match.group(1)) if total_fallback_match else 300,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def resize_max(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    side = max(height, width)
    if side <= max_side:
        return image
    scale = max_side / float(side)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def frame_sample(frame: np.ndarray, frame_index: int) -> FrameSample:
    sharpness = float(get_sharpness(frame))
    small = resize_max(frame, 180)
    small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    orb_image = resize_max(frame, 720)
    orb_gray = cv2.cvtColor(orb_image, cv2.COLOR_BGR2GRAY)
    orb_gray = cv2.equalizeHist(orb_gray)

    gray = cv2.cvtColor(resize_max(frame, 360), cv2.COLOR_BGR2GRAY)
    mean_gray = float(np.mean(gray))
    gray_std = float(np.std(gray))
    dark_percent = float(np.mean(gray <= 5) * 100.0)
    bright_percent = float(np.mean(gray >= 250) * 100.0)
    occupancy, edge_touch = approximate_foreground_occupancy(resize_max(frame, 360))

    return FrameSample(
        frame_index=frame_index,
        sharpness=sharpness,
        small_gray=small_gray,
        orb_gray=orb_gray,
        occupancy=occupancy,
        edge_touch=edge_touch,
        mean_gray=mean_gray,
        gray_std=gray_std,
        dark_percent=dark_percent,
        bright_percent=bright_percent,
    )


def approximate_foreground_occupancy(frame: np.ndarray) -> tuple[float | None, float | None]:
    if frame is None or frame.size == 0:
        return None, None
    height, width = frame.shape[:2]
    if height < 8 or width < 8:
        return None, None

    border = max(3, int(min(height, width) * 0.04))
    top = frame[:border, :, :]
    bottom = frame[-border:, :, :]
    left = frame[:, :border, :]
    right = frame[:, -border:, :]
    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3),
    ], axis=0).astype(np.float32)
    background = np.median(border_pixels, axis=0)
    border_dist = np.linalg.norm(border_pixels - background, axis=1)
    frame_float = frame.astype(np.float32)
    dist = np.linalg.norm(frame_float - background, axis=2)

    if float(np.std(dist)) < 3.0:
        return None, None

    threshold = max(18.0, float(np.percentile(border_dist, 95)) + 10.0)
    mask = (dist > threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    occupancy = float(np.count_nonzero(mask) / mask.size)
    edge_pixels = np.concatenate([
        mask[0, :],
        mask[-1, :],
        mask[:, 0],
        mask[:, -1],
    ])
    edge_touch = float(np.count_nonzero(edge_pixels) / edge_pixels.size)
    return occupancy, edge_touch


def candidate_indices(effective_total_frames: int, target_frames: int) -> tuple[list[int], int]:
    step = max(1, int(effective_total_frames) // int(target_frames))
    return list(range(0, max(0, int(effective_total_frames)), step)), step


def read_candidates_by_seek(
    video_path: Path,
    indices: list[int],
) -> tuple[list[FrameSample], int]:
    cap = cv2.VideoCapture(str(video_path))
    samples: list[FrameSample] = []
    failures = 0
    try:
        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                failures += 1
                continue
            samples.append(frame_sample(frame, frame_index))
    finally:
        cap.release()
    return samples, failures


def read_candidates_sequential(
    video_path: Path,
    step: int,
) -> tuple[list[FrameSample], int, int]:
    cap = cv2.VideoCapture(str(video_path))
    samples: list[FrameSample] = []
    decoded = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                break
            if decoded % step == 0:
                samples.append(frame_sample(frame, decoded))
            decoded += 1
    finally:
        cap.release()
    return samples, 0, decoded


def duplicate_rate(samples: list[FrameSample]) -> tuple[float | None, float | None, int, int]:
    if len(samples) < 2:
        return None, None, 0, 0
    near = 0
    diffs: list[float] = []
    for left, right in zip(samples, samples[1:]):
        a = left.small_gray.astype(np.float32)
        b = right.small_gray.astype(np.float32)
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        diff = float(np.mean(np.abs(a - b)) / 255.0)
        diffs.append(diff)
        if diff < DUPLICATE_DIFF_THRESHOLD:
            near += 1
    return near / len(diffs), median(diffs), near, len(diffs)


def choose_evenly(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return items
    if limit <= 1:
        return [items[0]]
    indices = sorted({round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)})
    return [items[i] for i in indices]


def orb_features(samples: list[FrameSample]) -> list[dict[str, Any]]:
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    features: list[dict[str, Any]] = []
    for sample in choose_evenly(samples, MAX_OVERLAP_FRAMES):
        keypoints, descriptors = orb.detectAndCompute(sample.orb_gray, None)
        features.append({
            "frame_index": sample.frame_index,
            "keypoints": 0 if keypoints is None else len(keypoints),
            "descriptors": descriptors,
        })
    return features


def good_match_count(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_desc = left.get("descriptors")
    right_desc = right.get("descriptors")
    if left_desc is None or right_desc is None:
        return 0
    if len(left_desc) < 2 or len(right_desc) < 2:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        matches = matcher.knnMatch(left_desc, right_desc, k=2)
    except cv2.error:
        return 0
    good = 0
    for pair in matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ORB_RATIO_TEST * second.distance:
            good += 1
    return good


def overlap_metrics(left_features: list[dict[str, Any]], right_features: list[dict[str, Any]]) -> dict[str, Any]:
    counts: list[int] = []
    for left in left_features:
        for right in right_features:
            counts.append(good_match_count(left, right))
    if not counts:
        return {"max": 0, "median": 0.0, "pairs": 0}
    return {
        "max": int(max(counts)),
        "median": float(np.median(np.asarray(counts, dtype=float))),
        "pairs": len(counts),
    }


def video_metadata(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        opened = bool(cap.isOpened())
        if not opened:
            return {
                "file_readable": False,
                "width": None,
                "height": None,
                "fps": None,
                "declared_frame_count": None,
                "duration_seconds": None,
            }
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (frame_count / fps) if fps > 0 and frame_count > 0 else None
        return {
            "file_readable": True,
            "width": width,
            "height": height,
            "fps": fps,
            "declared_frame_count": frame_count,
            "duration_seconds": duration,
        }
    finally:
        cap.release()


def exposure_flag(video: dict[str, Any]) -> str:
    mean_gray = video.get("median_mean_gray")
    gray_std = video.get("median_gray_std")
    dark = video.get("median_dark_percent")
    bright = video.get("median_bright_percent")
    if mean_gray is None:
        return "unavailable"
    if mean_gray < 25 or (dark is not None and dark > 60):
        return "gross_underexposure"
    if mean_gray > 245 and (gray_std is not None and gray_std < 18):
        return "gross_overexposure"
    if bright is not None and bright > 85 and (gray_std is not None and gray_std < 30):
        return "possible_overexposure"
    return "ok"


def occupancy_flag(video: dict[str, Any]) -> str:
    occ = video.get("median_occupancy")
    edge = video.get("median_edge_touch")
    if occ is None:
        return "unavailable"
    if occ < VERY_LOW_FOREGROUND_PROXY_THRESHOLD:
        return "possible_tiny_stone_or_low_contrast"
    if occ < LOW_FOREGROUND_PROXY_THRESHOLD:
        return "low_foreground_proxy"
    if occ > CROPPED_OCCUPANCY_THRESHOLD:
        return "possible_cropping_or_full_frame_foreground"
    if edge is not None and edge > EDGE_TOUCH_THRESHOLD:
        return "foreground_touches_frame_edge"
    return "ok"


def analyze_video(video_path: Path, label: str, settings: ProductionSettings, force_sequential: bool) -> dict[str, Any]:
    metadata = video_metadata(video_path)
    result: dict[str, Any] = {
        "label": label,
        "path": rel(video_path),
        "filename": video_path.name,
        **metadata,
    }
    if not metadata["file_readable"]:
        result.update({
            "decode_mode": "unreadable",
            "sampled_decode_failures": 0,
            "production_candidate_count": 0,
            "readable_candidate_count": 0,
            "normal_sharp_gate_pass_count": 0,
            "normal_sharp_gate_pass_percent": None,
            "fallback_triggered": False,
            "expected_extracted_frames": 0,
            "median_sharpness": None,
            "p25_sharpness": None,
            "p75_sharpness": None,
            "soft_or_blurred": True,
            "duplicate_rate": None,
            "median_adjacent_diff": None,
            "exposure_flag": "unavailable",
            "occupancy_flag": "unavailable",
            "_samples": [],
            "_orb_features": [],
        })
        return result

    declared_total = int(metadata.get("declared_frame_count") or 0)
    effective_total = declared_total if declared_total > 0 else settings.fallback_total_frames
    indices, step = candidate_indices(effective_total, settings.target_frames)
    decode_mode = "sequential" if force_sequential else "candidate_seek"
    sequential_decoded = None

    if force_sequential:
        samples, failures, sequential_decoded = read_candidates_sequential(video_path, step)
    else:
        samples, failures = read_candidates_by_seek(video_path, indices)
        failure_limit = max(3, int(len(indices) * 0.25))
        if failures > failure_limit:
            samples, failures, sequential_decoded = read_candidates_sequential(video_path, step)
            decode_mode = "sequential_seek_fallback"

    sharpness_values = [sample.sharpness for sample in samples]
    if effective_total < settings.target_frames:
        passed = len(samples)
    else:
        passed = sum(1 for value in sharpness_values if value > settings.sharpness_threshold)
    fallback_triggered = passed < settings.fallback_min_saved
    expected_extracted = passed + (len(samples) if fallback_triggered else 0)
    dup_rate, med_diff, duplicate_pairs, adjacent_pairs = duplicate_rate(samples)

    result.update({
        "decode_mode": decode_mode,
        "effective_total_frames": effective_total,
        "production_step": step,
        "production_candidate_count": len(indices),
        "readable_candidate_count": len(samples),
        "sampled_decode_failures": failures,
        "sequential_decoded_frames": sequential_decoded,
        "normal_sharp_gate_pass_count": passed,
        "normal_sharp_gate_pass_percent": percent(passed, len(samples)),
        "fallback_triggered": fallback_triggered,
        "expected_extracted_frames": expected_extracted,
        "median_sharpness": median(sharpness_values),
        "p25_sharpness": percentile(sharpness_values, 25),
        "p75_sharpness": percentile(sharpness_values, 75),
        "soft_or_blurred": (
            len(samples) == 0
            or median(sharpness_values) is None
            or median(sharpness_values) < settings.sharpness_threshold
            or (percent(passed, len(samples)) is not None and percent(passed, len(samples)) < 50.0)
        ),
        "duplicate_rate": dup_rate,
        "median_adjacent_diff": med_diff,
        "duplicate_pairs": duplicate_pairs,
        "adjacent_pairs": adjacent_pairs,
        "median_occupancy": median([sample.occupancy for sample in samples]),
        "median_edge_touch": median([sample.edge_touch for sample in samples]),
        "median_mean_gray": median([sample.mean_gray for sample in samples]),
        "median_gray_std": median([sample.gray_std for sample in samples]),
        "median_dark_percent": median([sample.dark_percent for sample in samples]),
        "median_bright_percent": median([sample.bright_percent for sample in samples]),
        "_samples": samples,
        "_orb_features": orb_features(samples),
    })
    result["exposure_flag"] = exposure_flag(result)
    result["occupancy_flag"] = occupancy_flag(result)
    return result


def discover_specimen(folder: Path) -> dict[str, Any]:
    video_dir = folder / "videos"
    typo_dirs = sorted([
        child.name
        for child in folder.iterdir()
        if child.is_dir() and child.name.lower() in {"vidoes", "video"}
    ]) if folder.exists() else []
    extra_files: list[Path] = []
    selected: dict[str, Path] = {}
    duplicate_expected: dict[str, list[str]] = {}
    missing: list[str] = []

    if video_dir.exists() and video_dir.is_dir():
        by_stem: dict[str, list[Path]] = {stem: [] for stem in EXPECTED_VIDEO_STEMS}
        for item in sorted(video_dir.iterdir()):
            if not item.is_file():
                continue
            if item.suffix.lower() not in VIDEO_EXTENSIONS:
                extra_files.append(item)
                continue
            stem = item.stem.lower()
            if stem in by_stem:
                by_stem[stem].append(item)
            else:
                extra_files.append(item)
        for stem in EXPECTED_VIDEO_STEMS:
            matches = by_stem[stem]
            if matches:
                selected[stem] = sorted(matches, key=lambda p: p.name.lower())[0]
                if len(matches) > 1:
                    duplicate_expected[stem] = [rel(p) for p in matches]
            else:
                missing.append(f"{stem}.mov")
    else:
        missing = [f"{stem}.mov" for stem in EXPECTED_VIDEO_STEMS]

    return {
        "specimen_id": folder.name,
        "folder": rel(folder),
        "videos_folder": rel(video_dir) if video_dir.exists() else "",
        "has_videos_folder": video_dir.exists() and video_dir.is_dir(),
        "typo_folders": typo_dirs,
        "missing_expected_files": missing,
        "extra_files": [rel(path) for path in extra_files],
        "duplicate_expected_files": duplicate_expected,
        "selected_videos": selected,
        "inspected_files": [rel(selected[stem]) for stem in EXPECTED_VIDEO_STEMS if stem in selected],
    }


def classification_and_score(specimen: dict[str, Any]) -> tuple[str, str, float]:
    fatal: list[str] = []
    borderline: list[str] = []

    videos = specimen.get("videos", {})
    readable_videos = [video for video in videos.values() if video.get("file_readable")]
    missing = specimen.get("missing_expected_files") or []
    typo = specimen.get("typo_folders") or []
    fallback_count = int(specimen.get("fallback_count") or 0)
    expected_total = int(specimen.get("expected_frame_total") or 0)
    sharp_pass = specimen.get("sharp_gate_pass_percent")
    overall_sharp = specimen.get("overall_median_sharpness")
    duplicate = specimen.get("duplicate_rate")
    overlap_values = specimen.get("overlap", {})

    if not specimen.get("has_videos_folder"):
        fatal.append("missing production videos folder")
    if missing:
        fatal.append("missing expected video files: " + ", ".join(missing))
    if len(readable_videos) != 4:
        fatal.append(f"only {len(readable_videos)} of 4 expected videos readable")
    if expected_total < 10:
        fatal.append(f"production extraction would yield only {expected_total} total frames")

    if fallback_count >= 3 and (sharp_pass is None or sharp_pass < 15.0):
        fatal.append(f"{fallback_count} videos trigger production auto-sharpness fallback")
    elif fallback_count > 0:
        borderline.append(f"{fallback_count} videos trigger production auto-sharpness fallback")

    if overall_sharp is not None:
        if overall_sharp < 10.0:
            fatal.append(f"overall median sharpness is very low ({overall_sharp:.1f})")
        elif overall_sharp < 25.0:
            borderline.append(f"overall median sharpness is low ({overall_sharp:.1f})")
    elif len(readable_videos) == 4:
        fatal.append("sharpness unavailable for readable videos")

    if sharp_pass is not None:
        if sharp_pass < 10.0:
            fatal.append(f"normal sharpness gate pass rate is very low ({sharp_pass:.1f}%)")
        elif sharp_pass < 50.0:
            borderline.append(f"normal sharpness gate pass rate is low ({sharp_pass:.1f}%)")

    if duplicate is not None:
        if duplicate > 0.85:
            fatal.append(f"adjacent near-duplicate rate is excessive ({duplicate:.2f})")
        elif duplicate > 0.60:
            borderline.append(f"adjacent near-duplicate rate is high ({duplicate:.2f})")

    severe_gaps = 0
    weak_gaps = 0
    for pair_name, metrics in overlap_values.items():
        max_matches = metrics.get("max", 0) or 0
        med_matches = metrics.get("median", 0) or 0
        if max_matches < 5:
            severe_gaps += 1
        elif max_matches < 10 or med_matches < 1.5:
            weak_gaps += 1
    if severe_gaps >= 2:
        fatal.append(f"{severe_gaps} cross-video continuity gaps by ORB overlap")
    elif severe_gaps or weak_gaps:
        borderline.append(f"{severe_gaps + weak_gaps} weak cross-video overlap transition(s)")

    exposure_issues = [
        video["label"]
        for video in readable_videos
        if str(video.get("exposure_flag", "")).startswith("gross_")
    ]
    possible_exposure = [
        video["label"]
        for video in readable_videos
        if video.get("exposure_flag") == "possible_overexposure"
    ]
    if len(exposure_issues) >= 3:
        fatal.append("gross exposure issue in most videos: " + ", ".join(exposure_issues))
    elif exposure_issues:
        borderline.append("gross exposure issue in " + ", ".join(exposure_issues))
    elif possible_exposure:
        borderline.append("possible overexposure in " + ", ".join(possible_exposure))

    occupancy_issues = [
        video["label"]
        for video in readable_videos
        if video.get("occupancy_flag") in {
            "possible_cropping_or_full_frame_foreground",
            "foreground_touches_frame_edge",
        }
    ]
    if len(occupancy_issues) >= 3:
        fatal.append("stone occupancy/cropping issue in most videos: " + ", ".join(occupancy_issues))
    elif occupancy_issues:
        borderline.append("stone occupancy/cropping issue in " + ", ".join(occupancy_issues))

    if typo:
        borderline.append("typo folder present but ignored: " + ", ".join(typo))

    overlap_component_values = [
        min(1.0, float(metrics.get("max", 0) or 0) / 80.0)
        for metrics in overlap_values.values()
    ]
    overlap_component = 20.0 * (sum(overlap_component_values) / len(overlap_component_values)) if overlap_component_values else 0.0
    sharp_component = 0.0 if overall_sharp is None else min(30.0, math.log10(max(0.0, overall_sharp) + 1.0) / math.log10(151.0) * 30.0)
    pass_component = 0.0 if sharp_pass is None else min(25.0, max(0.0, sharp_pass) / 100.0 * 25.0)
    motion_component = 0.0 if duplicate is None else max(0.0, (1.0 - duplicate) * 15.0)
    extraction_component = min(10.0, expected_total / 160.0 * 10.0)
    penalty = fallback_count * 4.0 + len(fatal) * 18.0 + len(borderline) * 4.0
    score = max(0.0, sharp_component + pass_component + motion_component + overlap_component + extraction_component - penalty)

    if fatal:
        return "FAIL", "; ".join(fatal[:4]), score
    if borderline:
        return "BORDERLINE", "; ".join(borderline[:4]), score
    return "PASS", "complete readable capture set with usable sharpness, motion, exposure, and overlap metrics", score


def folder_status(discovery: dict[str, Any], readable: bool) -> str:
    if not discovery.get("has_videos_folder"):
        return "missing_videos_folder"
    if discovery.get("missing_expected_files"):
        return "incomplete_videos_folder"
    if not readable:
        return "unreadable_video"
    if discovery.get("typo_folders"):
        return "complete_with_ignored_typo_folder"
    if discovery.get("extra_files") or discovery.get("duplicate_expected_files"):
        return "complete_with_extra_files"
    return "complete"


def analyze_specimen(folder: Path, settings: ProductionSettings, force_sequential: bool) -> dict[str, Any]:
    discovery = discover_specimen(folder)
    videos: dict[str, dict[str, Any]] = {}
    for stem in EXPECTED_VIDEO_STEMS:
        path = discovery["selected_videos"].get(stem)
        if path is None:
            continue
        videos[stem] = analyze_video(path, stem, settings, force_sequential)

    overlaps: dict[str, dict[str, Any]] = {}
    for left, right in zip(EXPECTED_VIDEO_STEMS, EXPECTED_VIDEO_STEMS[1:]):
        pair_name = f"{left}_{right}"
        if left in videos and right in videos:
            overlaps[pair_name] = overlap_metrics(
                videos[left].get("_orb_features", []),
                videos[right].get("_orb_features", []),
            )
        else:
            overlaps[pair_name] = {"max": 0, "median": 0.0, "pairs": 0}

    expected_total = sum(int(video.get("expected_extracted_frames") or 0) for video in videos.values())
    candidate_total = sum(int(video.get("readable_candidate_count") or 0) for video in videos.values())
    pass_total = sum(int(video.get("normal_sharp_gate_pass_count") or 0) for video in videos.values())
    duplicate_pairs = sum(int(video.get("duplicate_pairs") or 0) for video in videos.values())
    adjacent_pairs = sum(int(video.get("adjacent_pairs") or 0) for video in videos.values())
    all_sharpness = []
    for video in videos.values():
        for sample in video.get("_samples", []):
            all_sharpness.append(sample.sharpness)

    readable = len(videos) == 4 and all(video.get("file_readable") for video in videos.values())
    specimen: dict[str, Any] = {
        **{key: value for key, value in discovery.items() if key != "selected_videos"},
        "video_count": len(videos),
        "readable": readable,
        "folder_status": "",
        "videos": videos,
        "expected_frame_total": expected_total,
        "fallback_count": sum(1 for video in videos.values() if video.get("fallback_triggered")),
        "sharp_gate_pass_percent": percent(pass_total, candidate_total),
        "overall_median_sharpness": median(all_sharpness),
        "overall_p25_sharpness": percentile(all_sharpness, 25),
        "overall_p75_sharpness": percentile(all_sharpness, 75),
        "duplicate_rate": (duplicate_pairs / adjacent_pairs) if adjacent_pairs else None,
        "overlap": overlaps,
    }
    specimen["folder_status"] = folder_status(discovery, readable)
    screen, reason, score = classification_and_score(specimen)
    specimen["capture_screen"] = screen
    specimen["reason"] = reason
    specimen["quality_score"] = score
    return specimen


def csv_row(specimen: dict[str, Any]) -> dict[str, Any]:
    videos = specimen.get("videos", {})
    overlap = specimen.get("overlap", {})
    row = {
        "specimen_id": specimen["specimen_id"],
        "folder_status": specimen["folder_status"],
        "video_count": specimen["video_count"],
        "readable": "YES" if specimen.get("readable") else "NO",
        "expected_frame_total": specimen.get("expected_frame_total", 0),
        "median_sharpness_01": rounded(videos.get("01", {}).get("median_sharpness")),
        "median_sharpness_02": rounded(videos.get("02", {}).get("median_sharpness")),
        "median_sharpness_03": rounded(videos.get("03", {}).get("median_sharpness")),
        "median_sharpness_04": rounded(videos.get("04", {}).get("median_sharpness")),
        "overall_median_sharpness": rounded(specimen.get("overall_median_sharpness")),
        "fallback_count": specimen.get("fallback_count", 0),
        "sharp_gate_pass_percent": rounded(specimen.get("sharp_gate_pass_percent")),
        "duplicate_rate": rounded(specimen.get("duplicate_rate")),
        "overlap_01_02_max": overlap.get("01_02", {}).get("max", ""),
        "overlap_01_02_median": rounded(overlap.get("01_02", {}).get("median")),
        "overlap_02_03_max": overlap.get("02_03", {}).get("max", ""),
        "overlap_02_03_median": rounded(overlap.get("02_03", {}).get("median")),
        "overlap_03_04_max": overlap.get("03_04", {}).get("max", ""),
        "overlap_03_04_median": rounded(overlap.get("03_04", {}).get("median")),
        "capture_screen": specimen.get("capture_screen"),
        "reason": specimen.get("reason"),
        "quality_score": rounded(specimen.get("quality_score")),
        "inspected_files": "; ".join(specimen.get("inspected_files", [])),
        "missing_expected_files": "; ".join(specimen.get("missing_expected_files", [])),
        "typo_folders": "; ".join(specimen.get("typo_folders", [])),
    }
    return row


def strip_internal(specimen: dict[str, Any]) -> dict[str, Any]:
    clean = dict(specimen)
    videos = {}
    for label, video in specimen.get("videos", {}).items():
        video_clean = {
            key: value
            for key, value in video.items()
            if key not in {"_samples", "_orb_features"}
        }
        videos[label] = video_clean
    clean["videos"] = videos
    return clean


def ranked(specimens: list[dict[str, Any]], screen: str | None = None) -> list[dict[str, Any]]:
    selected = [item for item in specimens if screen is None or item.get("capture_screen") == screen]
    return sorted(selected, key=lambda item: (-float(item.get("quality_score") or 0.0), item["specimen_id"]))


def reference_ranges(specimens: list[dict[str, Any]]) -> dict[str, Any]:
    refs = [item for item in specimens if item["specimen_id"] in REFERENCE_CAPTURE_IDS]
    values = {
        "overall_median_sharpness": [item.get("overall_median_sharpness") for item in refs],
        "sharp_gate_pass_percent": [item.get("sharp_gate_pass_percent") for item in refs],
        "duplicate_rate": [item.get("duplicate_rate") for item in refs],
        "expected_frame_total": [item.get("expected_frame_total") for item in refs],
    }
    overlap_maxes: list[float] = []
    for item in refs:
        for metrics in item.get("overlap", {}).values():
            overlap_maxes.append(float(metrics.get("max", 0) or 0))
    return {
        "reference_specimens": list(REFERENCE_CAPTURE_IDS),
        "note": (
            "Computed after objective per-specimen metrics. These ranges are descriptive only "
            "and are not used as reconstruction-outcome labels for other specimens."
        ),
        "overall_median_sharpness_min": percentile(values["overall_median_sharpness"], 0),
        "overall_median_sharpness_median": median(values["overall_median_sharpness"]),
        "overall_median_sharpness_max": percentile(values["overall_median_sharpness"], 100),
        "sharp_gate_pass_percent_min": percentile(values["sharp_gate_pass_percent"], 0),
        "sharp_gate_pass_percent_median": median(values["sharp_gate_pass_percent"]),
        "sharp_gate_pass_percent_max": percentile(values["sharp_gate_pass_percent"], 100),
        "duplicate_rate_min": percentile(values["duplicate_rate"], 0),
        "duplicate_rate_median": median(values["duplicate_rate"]),
        "duplicate_rate_max": percentile(values["duplicate_rate"], 100),
        "expected_frame_total_min": percentile(values["expected_frame_total"], 0),
        "expected_frame_total_median": median(values["expected_frame_total"]),
        "expected_frame_total_max": percentile(values["expected_frame_total"], 100),
        "cross_video_overlap_max_min": percentile(overlap_maxes, 0),
        "cross_video_overlap_max_median": median(overlap_maxes),
        "cross_video_overlap_max_max": percentile(overlap_maxes, 100),
    }


def compact_rank_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "specimen_id": item["specimen_id"],
        "quality_score": rounded(item.get("quality_score")),
        "capture_screen": item.get("capture_screen"),
        "expected_frame_total": item.get("expected_frame_total"),
        "overall_median_sharpness": rounded(item.get("overall_median_sharpness")),
        "sharp_gate_pass_percent": rounded(item.get("sharp_gate_pass_percent")),
        "duplicate_rate": rounded(item.get("duplicate_rate")),
        "reason": item.get("reason"),
    }


def build_summary(specimens: list[dict[str, Any]], settings: ProductionSettings) -> dict[str, Any]:
    pass_ranked = ranked(specimens, "PASS")
    borderline_ranked = ranked(specimens, "BORDERLINE")
    fail_ranked = ranked(specimens, "FAIL")
    missing_incomplete = [
        item for item in specimens
        if item.get("folder_status") != "complete"
        or item.get("missing_expected_files")
        or item.get("typo_folders")
    ]
    fully_screened = [
        item for item in specimens
        if item.get("has_videos_folder")
        and not item.get("missing_expected_files")
        and len(item.get("videos", {})) == 4
        and all(video.get("file_readable") for video in item.get("videos", {}).values())
    ]
    return {
        "generated_utc": utc_now(),
        "input_root": rel(INPUT_ROOT),
        "outputs": {
            "csv": rel(CSV_OUT),
            "summary_json": rel(JSON_OUT),
            "report_md": rel(REPORT_OUT),
        },
        "production_logic": {
            "source": rel(BACKEND_DIR / "video_utils.py"),
            "function": "extract_best_frames",
            "sharpness_function": "get_sharpness",
            "source_sha256": settings.source_sha256,
            "target_frames": settings.target_frames,
            "sharpness_threshold": settings.sharpness_threshold,
            "fallback_min_saved": settings.fallback_min_saved,
            "fallback_total_frames": settings.fallback_total_frames,
        },
        "method_boundaries": [
            "COLMAP not run",
            "reconstruction not run",
            "optimizer not run",
            "defect detection not run",
            "source videos not modified",
            "physical length/width/height not used",
            "previous reconstruction success/failure not inspected for classification",
            "typo folders detected but not used as production input",
        ],
        "classification_thresholds": {
            "fail_expected_frame_total_lt": 10,
            "fail_overall_median_sharpness_lt": 10,
            "borderline_overall_median_sharpness_lt": 25,
            "fail_sharp_gate_pass_percent_lt": 10,
            "borderline_sharp_gate_pass_percent_lt": 50,
            "fail_duplicate_rate_gt": 0.85,
            "borderline_duplicate_rate_gt": 0.60,
            "fail_overlap_transition_max_lt": 5,
            "borderline_overlap_transition_max_lt": 10,
        },
        "specimen_folders_found": len(specimens),
        "fully_screened": len(fully_screened),
        "counts": {
            "PASS": len(pass_ranked),
            "BORDERLINE": len(borderline_ranked),
            "FAIL": len(fail_ranked),
            "missing_incomplete_or_typo": len(missing_incomplete),
        },
        "pass_specimens_ranked": [compact_rank_item(item) for item in pass_ranked],
        "borderline_specimens_ranked": [compact_rank_item(item) for item in borderline_ranked],
        "fail_specimens": [compact_rank_item(item) for item in fail_ranked],
        "missing_incomplete_typo_specimens": [
            {
                "specimen_id": item["specimen_id"],
                "folder_status": item.get("folder_status"),
                "missing_expected_files": item.get("missing_expected_files"),
                "typo_folders": item.get("typo_folders"),
                "inspected_files": item.get("inspected_files"),
                "capture_screen": item.get("capture_screen"),
                "reason": item.get("reason"),
            }
            for item in missing_incomplete
        ],
        "top_5_best_capture_sets_overall": [compact_rank_item(item) for item in ranked(specimens)[:5]],
        "eligible_for_reconstruction_capture_only": [item["specimen_id"] for item in pass_ranked],
        "recommended_next_for_reconstruction_capture_only": [item["specimen_id"] for item in pass_ranked[:3]],
        "reference_capture_ranges": reference_ranges(specimens),
        "specimens": {
            item["specimen_id"]: strip_internal(item)
            for item in sorted(specimens, key=lambda row: row["specimen_id"])
        },
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "specimen_id",
        "folder_status",
        "video_count",
        "readable",
        "expected_frame_total",
        "median_sharpness_01",
        "median_sharpness_02",
        "median_sharpness_03",
        "median_sharpness_04",
        "overall_median_sharpness",
        "fallback_count",
        "sharp_gate_pass_percent",
        "duplicate_rate",
        "overlap_01_02_max",
        "overlap_01_02_median",
        "overlap_02_03_max",
        "overlap_02_03_median",
        "overlap_03_04_max",
        "overlap_03_04_median",
        "capture_screen",
        "reason",
        "quality_score",
        "inspected_files",
        "missing_expected_files",
        "typo_folders",
    ]
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def rank_table(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["None."]
    lines = [
        "| rank | specimen | score | frames | median sharpness | sharp pass % | dup rate | reason |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, item in enumerate(items, start=1):
        lines.append(
            f"| {index} | {item['specimen_id']} | {rounded(item.get('quality_score'))} | "
            f"{item.get('expected_frame_total', '')} | {rounded(item.get('overall_median_sharpness'))} | "
            f"{rounded(item.get('sharp_gate_pass_percent'))} | {rounded(item.get('duplicate_rate'))} | "
            f"{item.get('reason', '')} |"
        )
    return lines


def write_report(summary: dict[str, Any]) -> None:
    specimens = [summary["specimens"][key] for key in sorted(summary["specimens"])]
    refs = summary["reference_capture_ranges"]
    lines = [
        "# Capture Screen Report",
        "",
        f"Generated UTC: {summary['generated_utc']}",
        "",
        "## Scope",
        "",
        f"Input root: `{summary['input_root']}`",
        f"Specimen folders found: {summary['specimen_folders_found']}",
        f"Fully screened from exact production `videos` folders: {summary['fully_screened']}",
        "",
        "This is a pre-COLMAP video-only screen. It did not run COLMAP, reconstruction, optimizer, defect detection, or use physical L/W/H. Typo folders such as `vidoes` are reported but not used as production input.",
        "",
        "## Production Extraction Logic",
        "",
        f"Source: `{summary['production_logic']['source']}`",
        f"Function: `{summary['production_logic']['function']}`",
        f"Sharpness function: `{summary['production_logic']['sharpness_function']}`",
        f"target_frames: {summary['production_logic']['target_frames']}",
        f"sharpness_threshold: {summary['production_logic']['sharpness_threshold']}",
        f"fallback_min_saved: {summary['production_logic']['fallback_min_saved']}",
        f"source_sha256: `{summary['production_logic']['source_sha256']}`",
        "",
        "## PASS Specimens Ranked",
        "",
        *rank_table(summary["pass_specimens_ranked"]),
        "",
        "## BORDERLINE Specimens Ranked",
        "",
        *rank_table(summary["borderline_specimens_ranked"]),
        "",
        "## FAIL Specimens",
        "",
        *rank_table(summary["fail_specimens"]),
        "",
        "## Missing, Incomplete, Or Typo-Folder Findings",
        "",
        "| specimen | folder status | missing expected files | typo folders | inspected files | screen |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in summary["missing_incomplete_typo_specimens"]:
        inspected = "; ".join(item.get("inspected_files") or []) or "none"
        missing = "; ".join(item.get("missing_expected_files") or []) or ""
        typo = "; ".join(item.get("typo_folders") or []) or ""
        lines.append(
            f"| {item['specimen_id']} | {item.get('folder_status', '')} | {missing} | "
            f"{typo} | {inspected} | {item.get('capture_screen', '')} |"
        )
    lines.extend([
        "",
        "## Top 5 Best Capture Sets Overall",
        "",
        *rank_table(summary["top_5_best_capture_sets_overall"]),
        "",
        "## Reconstruction Eligibility Based Only On Capture Screen",
        "",
        "Eligible now: " + (", ".join(summary["eligible_for_reconstruction_capture_only"]) or "None"),
        "Recommended next: " + (", ".join(summary["recommended_next_for_reconstruction_capture_only"]) or "None"),
        "",
        "## Descriptive Reference Capture Ranges",
        "",
        refs["note"],
        "",
        "| metric | min | median | max |",
        "| --- | ---: | ---: | ---: |",
        f"| overall median sharpness | {rounded(refs['overall_median_sharpness_min'])} | {rounded(refs['overall_median_sharpness_median'])} | {rounded(refs['overall_median_sharpness_max'])} |",
        f"| sharp gate pass percent | {rounded(refs['sharp_gate_pass_percent_min'])} | {rounded(refs['sharp_gate_pass_percent_median'])} | {rounded(refs['sharp_gate_pass_percent_max'])} |",
        f"| duplicate rate | {rounded(refs['duplicate_rate_min'])} | {rounded(refs['duplicate_rate_median'])} | {rounded(refs['duplicate_rate_max'])} |",
        f"| expected frame total | {rounded(refs['expected_frame_total_min'])} | {rounded(refs['expected_frame_total_median'])} | {rounded(refs['expected_frame_total_max'])} |",
        f"| cross-video overlap max | {rounded(refs['cross_video_overlap_max_min'])} | {rounded(refs['cross_video_overlap_max_median'])} | {rounded(refs['cross_video_overlap_max_max'])} |",
        "",
        "## Exact Files Inspected",
        "",
        "| specimen | inspected production videos | ignored typo folders | missing expected files |",
        "| --- | --- | --- | --- |",
    ])
    for item in specimens:
        inspected = "; ".join(item.get("inspected_files") or []) or "none"
        typo = "; ".join(item.get("typo_folders") or []) or ""
        missing = "; ".join(item.get("missing_expected_files") or []) or ""
        lines.append(f"| {item['specimen_id']} | {inspected} | {typo} | {missing} |")

    lines.extend([
        "",
        "## Per-Specimen Detail",
        "",
        "| specimen | screen | status | videos | frames | fallback | median sharpness | sharp pass % | duplicate | 01-02 max/med | 02-03 max/med | 03-04 max/med |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in specimens:
        overlap = item.get("overlap", {})
        def pair(name: str) -> str:
            metrics = overlap.get(name, {})
            return f"{metrics.get('max', 0)}/{rounded(metrics.get('median'))}"
        lines.append(
            f"| {item['specimen_id']} | {item.get('capture_screen', '')} | {item.get('folder_status', '')} | "
            f"{item.get('video_count', 0)} | {item.get('expected_frame_total', 0)} | "
            f"{item.get('fallback_count', 0)} | {rounded(item.get('overall_median_sharpness'))} | "
            f"{rounded(item.get('sharp_gate_pass_percent'))} | {rounded(item.get('duplicate_rate'))} | "
            f"{pair('01_02')} | {pair('02_03')} | {pair('03_04')} |"
        )

    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(INPUT_ROOT))
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Decode videos sequentially like production instead of seeking production candidate frames.",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated specimen IDs for quick diagnostics. Omit for all folders.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global INPUT_ROOT
    INPUT_ROOT = Path(args.input_root)

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input root does not exist: {INPUT_ROOT}")

    settings = production_settings()
    specimen_folders = sorted([path for path in INPUT_ROOT.iterdir() if path.is_dir()], key=lambda p: p.name)
    if args.only:
        requested = {item.strip() for item in args.only.split(",") if item.strip()}
        specimen_folders = [path for path in specimen_folders if path.name in requested]

    specimens: list[dict[str, Any]] = []
    for folder in specimen_folders:
        print(f"Screening {folder.name}...", flush=True)
        specimens.append(analyze_specimen(folder, settings, args.sequential))

    rows = [csv_row(item) for item in sorted(specimens, key=lambda row: row["specimen_id"])]
    write_csv(rows)
    summary = build_summary(specimens, settings)
    JSON_OUT.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(json_ready(summary))

    print(json.dumps({
        "specimen_folders_found": summary["specimen_folders_found"],
        "fully_screened": summary["fully_screened"],
        "counts": summary["counts"],
        "outputs": summary["outputs"],
        "recommended_next_for_reconstruction_capture_only": summary["recommended_next_for_reconstruction_capture_only"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
