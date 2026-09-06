"""Production-safe capture-quality screening helpers.

The checks here are intentionally limited to uploaded video evidence. They do
not run COLMAP, reconstruction, masking, defect detection, optimization, or use
specimen identity/reconstruction outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import math
from pathlib import Path
import re
from typing import Any, Mapping

import cv2
import numpy as np

from video_utils import extract_best_frames, get_sharpness


EXPECTED_VIDEO_STEMS = ("01", "02", "03", "04")
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi"}

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


def percent(numerator: float, denominator: float) -> float | None:
    if not denominator:
        return None
    return 100.0 * float(numerator) / float(denominator)


def median(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values
             if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(np.median(np.asarray(clean, dtype=float)))


def percentile(values: list[float | int | None], pct: float) -> float | None:
    clean = [float(value) for value in values
             if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(np.percentile(np.asarray(clean, dtype=float), pct))


def rounded(value: Any, digits: int = 3) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not math.isfinite(value):
            return None
        return round(value, digits)
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
        return str(value)
    return value


def production_settings() -> ProductionSettings:
    source = inspect.getsource(extract_best_frames)
    signature = inspect.signature(extract_best_frames)
    target_default = signature.parameters["target_frames"].default
    if target_default is inspect.Parameter.empty:
        raise RuntimeError(
            "Production extract_best_frames has no target_frames default."
        )

    sharp_match = re.search(r"sharpness\s*>\s*([0-9]+(?:\.[0-9]+)?)", source)
    fallback_match = re.search(r"saved_count\s*<\s*([0-9]+)", source)
    total_fallback_match = re.search(
        r"total_frames\s*=\s*([0-9]+)\s*(?:#.*)?\n",
        source,
    )
    if not sharp_match or not fallback_match:
        raise RuntimeError(
            "Could not derive production sharpness/fallback settings."
        )

    return ProductionSettings(
        target_frames=int(target_default),
        sharpness_threshold=float(sharp_match.group(1)),
        fallback_min_saved=int(fallback_match.group(1)),
        fallback_total_frames=(
            int(total_fallback_match.group(1)) if total_fallback_match else 300
        ),
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


def approximate_foreground_occupancy(
    frame: np.ndarray,
) -> tuple[float | None, float | None]:
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
    occupancy, edge_touch = approximate_foreground_occupancy(
        resize_max(frame, 360)
    )

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


def candidate_indices(
    effective_total_frames: int,
    target_frames: int,
) -> tuple[list[int], int]:
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


def duplicate_rate(
    samples: list[FrameSample],
) -> tuple[float | None, float | None, int, int]:
    if len(samples) < 2:
        return None, None, 0, 0
    near = 0
    diffs: list[float] = []
    for left, right in zip(samples, samples[1:]):
        a = left.small_gray.astype(np.float32)
        b = right.small_gray.astype(np.float32)
        if a.shape != b.shape:
            b = cv2.resize(
                b,
                (a.shape[1], a.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
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
    indices = sorted({
        round(i * (len(items) - 1) / (limit - 1))
        for i in range(limit)
    })
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


def overlap_metrics(
    left_features: list[dict[str, Any]],
    right_features: list[dict[str, Any]],
) -> dict[str, Any]:
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
        duration = (
            frame_count / fps if fps > 0 and frame_count > 0 else None
        )
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
    if bright is not None and bright > 85 and gray_std is not None and gray_std < 30:
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


def analyze_video(
    video_path: Path,
    label: str,
    settings: ProductionSettings,
    force_sequential: bool = False,
) -> dict[str, Any]:
    metadata = video_metadata(video_path)
    result: dict[str, Any] = {
        "label": label,
        "path": str(video_path),
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
    effective_total = (
        declared_total if declared_total > 0 else settings.fallback_total_frames
    )
    indices, step = candidate_indices(effective_total, settings.target_frames)
    decode_mode = "sequential" if force_sequential else "candidate_seek"
    sequential_decoded = None

    if force_sequential:
        samples, failures, sequential_decoded = read_candidates_sequential(
            video_path,
            step,
        )
    else:
        samples, failures = read_candidates_by_seek(video_path, indices)
        failure_limit = max(3, int(len(indices) * 0.25))
        if failures > failure_limit:
            samples, failures, sequential_decoded = read_candidates_sequential(
                video_path,
                step,
            )
            decode_mode = "sequential_seek_fallback"

    sharpness_values = [sample.sharpness for sample in samples]
    if effective_total < settings.target_frames:
        passed = len(samples)
    else:
        passed = sum(
            1 for value in sharpness_values
            if value > settings.sharpness_threshold
        )
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
            or (
                percent(passed, len(samples)) is not None
                and percent(passed, len(samples)) < 50.0
            )
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
        "median_bright_percent": median(
            [sample.bright_percent for sample in samples]
        ),
        "_samples": samples,
        "_orb_features": orb_features(samples),
    })
    result["exposure_flag"] = exposure_flag(result)
    result["occupancy_flag"] = occupancy_flag(result)
    return result


def folder_status(discovery: dict[str, Any], readable: bool) -> str:
    if not discovery.get("has_videos_folder", True):
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


def classify_continuity(metrics: dict[str, Any]) -> str:
    max_matches = metrics.get("max", 0) or 0
    med_matches = metrics.get("median", 0) or 0
    if max_matches < 5:
        return "poor"
    if max_matches < 10 or med_matches < 1.5:
        return "weak"
    return "good"


def classification_and_score(specimen: dict[str, Any]) -> tuple[str, str, float]:
    fatal: list[str] = []
    borderline: list[str] = []

    videos = specimen.get("videos", {})
    readable_videos = [
        video for video in videos.values() if video.get("file_readable")
    ]
    missing = specimen.get("missing_expected_files") or []
    typo = specimen.get("typo_folders") or []
    fallback_count = int(specimen.get("fallback_count") or 0)
    expected_total = int(specimen.get("expected_frame_total") or 0)
    sharp_pass = specimen.get("sharp_gate_pass_percent")
    overall_sharp = specimen.get("overall_median_sharpness")
    duplicate = specimen.get("duplicate_rate")
    overlap_values = specimen.get("overlap", {})

    if not specimen.get("has_videos_folder", True):
        fatal.append("missing production videos folder")
    if missing:
        fatal.append("missing expected video files: " + ", ".join(missing))
    if len(readable_videos) != 4:
        fatal.append(f"only {len(readable_videos)} of 4 expected videos readable")
    if expected_total < 10:
        fatal.append(
            f"production extraction would yield only {expected_total} total frames"
        )

    if fallback_count >= 3 and (sharp_pass is None or sharp_pass < 15.0):
        fatal.append(
            f"{fallback_count} videos trigger production auto-sharpness fallback"
        )
    elif fallback_count > 0:
        borderline.append(
            f"{fallback_count} videos trigger production auto-sharpness fallback"
        )

    if overall_sharp is not None:
        if overall_sharp < 10.0:
            fatal.append(
                f"overall median sharpness is very low ({overall_sharp:.1f})"
            )
        elif overall_sharp < 25.0:
            borderline.append(
                f"overall median sharpness is low ({overall_sharp:.1f})"
            )
    elif len(readable_videos) == 4:
        fatal.append("sharpness unavailable for readable videos")

    if sharp_pass is not None:
        if sharp_pass < 10.0:
            fatal.append(
                f"normal sharpness gate pass rate is very low ({sharp_pass:.1f}%)"
            )
        elif sharp_pass < 50.0:
            borderline.append(
                f"normal sharpness gate pass rate is low ({sharp_pass:.1f}%)"
            )

    if duplicate is not None:
        if duplicate > 0.85:
            fatal.append(
                f"adjacent near-duplicate rate is excessive ({duplicate:.2f})"
            )
        elif duplicate > 0.60:
            borderline.append(
                f"adjacent near-duplicate rate is high ({duplicate:.2f})"
            )

    severe_gaps = 0
    weak_gaps = 0
    for metrics in overlap_values.values():
        state = classify_continuity(metrics)
        if state == "poor":
            severe_gaps += 1
        elif state == "weak":
            weak_gaps += 1
    if severe_gaps >= 2:
        fatal.append(f"{severe_gaps} cross-video continuity gaps by ORB overlap")
    elif severe_gaps or weak_gaps:
        borderline.append(
            f"{severe_gaps + weak_gaps} weak cross-video overlap transition(s)"
        )

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
        fatal.append(
            "gross exposure issue in most videos: " + ", ".join(exposure_issues)
        )
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
        fatal.append(
            "stone occupancy/cropping issue in most videos: "
            + ", ".join(occupancy_issues)
        )
    elif occupancy_issues:
        borderline.append(
            "stone occupancy/cropping issue in " + ", ".join(occupancy_issues)
        )

    if typo:
        borderline.append("typo folder present but ignored: " + ", ".join(typo))

    overlap_component_values = [
        min(1.0, float(metrics.get("max", 0) or 0) / 80.0)
        for metrics in overlap_values.values()
    ]
    overlap_component = (
        20.0 * (sum(overlap_component_values) / len(overlap_component_values))
        if overlap_component_values else 0.0
    )
    sharp_component = (
        0.0 if overall_sharp is None else
        min(30.0, math.log10(max(0.0, overall_sharp) + 1.0)
            / math.log10(151.0) * 30.0)
    )
    pass_component = (
        0.0 if sharp_pass is None else
        min(25.0, max(0.0, sharp_pass) / 100.0 * 25.0)
    )
    motion_component = (
        0.0 if duplicate is None else max(0.0, (1.0 - duplicate) * 15.0)
    )
    extraction_component = min(10.0, expected_total / 160.0 * 10.0)
    penalty = fallback_count * 4.0 + len(fatal) * 18.0 + len(borderline) * 4.0
    score = max(
        0.0,
        sharp_component + pass_component + motion_component
        + overlap_component + extraction_component - penalty,
    )

    if fatal:
        return "FAIL", "; ".join(fatal[:4]), score
    if borderline:
        return "BORDERLINE", "; ".join(borderline[:4]), score
    return (
        "PASS",
        "complete readable capture set with usable sharpness, motion, exposure, and overlap metrics",
        score,
    )


def analyze_capture_set(
    video_paths: Mapping[str, Path | str],
    settings: ProductionSettings | None = None,
    force_sequential: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = settings or production_settings()
    discovery = {
        "has_videos_folder": True,
        "missing_expected_files": [],
        "typo_folders": [],
        "extra_files": [],
        "duplicate_expected_files": {},
        **(metadata or {}),
    }

    videos: dict[str, dict[str, Any]] = {}
    for stem in EXPECTED_VIDEO_STEMS:
        raw_path = video_paths.get(stem)
        if raw_path is None:
            continue
        videos[stem] = analyze_video(Path(raw_path), stem, settings, force_sequential)

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

    expected_total = sum(
        int(video.get("expected_extracted_frames") or 0)
        for video in videos.values()
    )
    candidate_total = sum(
        int(video.get("readable_candidate_count") or 0)
        for video in videos.values()
    )
    pass_total = sum(
        int(video.get("normal_sharp_gate_pass_count") or 0)
        for video in videos.values()
    )
    duplicate_pairs = sum(
        int(video.get("duplicate_pairs") or 0)
        for video in videos.values()
    )
    adjacent_pairs = sum(
        int(video.get("adjacent_pairs") or 0)
        for video in videos.values()
    )
    all_sharpness: list[float] = []
    for video in videos.values():
        for sample in video.get("_samples", []):
            all_sharpness.append(sample.sharpness)

    readable = (
        len(videos) == 4
        and all(video.get("file_readable") for video in videos.values())
    )
    specimen: dict[str, Any] = {
        **discovery,
        "video_count": len(videos),
        "readable": readable,
        "videos": videos,
        "expected_frame_total": expected_total,
        "fallback_count": sum(
            1 for video in videos.values()
            if video.get("fallback_triggered")
        ),
        "sharp_gate_pass_percent": percent(pass_total, candidate_total),
        "overall_median_sharpness": median(all_sharpness),
        "overall_p25_sharpness": percentile(all_sharpness, 25),
        "overall_p75_sharpness": percentile(all_sharpness, 75),
        "duplicate_rate": (
            duplicate_pairs / adjacent_pairs if adjacent_pairs else None
        ),
        "overlap": overlaps,
    }
    specimen["folder_status"] = folder_status(discovery, readable)
    screen, reason, score = classification_and_score(specimen)
    specimen["capture_screen"] = screen
    specimen["reason"] = reason
    specimen["quality_score"] = score
    return specimen


def strip_internal(specimen: dict[str, Any]) -> dict[str, Any]:
    clean = dict(specimen)
    videos = {}
    for label, video in specimen.get("videos", {}).items():
        videos[label] = {
            key: value
            for key, value in video.items()
            if key not in {"_samples", "_orb_features"}
        }
    clean["videos"] = videos
    return clean


def _lower_status(status: str | None) -> str | None:
    if status is None:
        return None
    return str(status).lower()


def _video_status(video: dict[str, Any]) -> str | None:
    if not video.get("file_readable"):
        return "fail"
    if int(video.get("expected_extracted_frames") or 0) <= 0:
        return "fail"

    failures = int(video.get("sampled_decode_failures") or 0)
    candidates = int(video.get("production_candidate_count") or 0)
    sharp_pass = video.get("normal_sharp_gate_pass_percent")
    median_sharpness = video.get("median_sharpness")
    duplicate = video.get("duplicate_rate")

    if candidates and failures > max(3, int(candidates * 0.25)):
        return "fail"
    if video.get("fallback_triggered") and (
        sharp_pass is None or sharp_pass < 15.0
    ):
        return "fail"
    if median_sharpness is not None and median_sharpness < 10.0:
        return "fail"
    if sharp_pass is not None and sharp_pass < 10.0:
        return "fail"
    if duplicate is not None and duplicate > 0.85:
        return "fail"

    if video.get("fallback_triggered"):
        return "borderline"
    if median_sharpness is not None and median_sharpness < 25.0:
        return "borderline"
    if sharp_pass is not None and sharp_pass < 50.0:
        return "borderline"
    if duplicate is not None and duplicate > 0.60:
        return "borderline"
    if str(video.get("exposure_flag", "")).startswith("gross_"):
        return "borderline"
    if video.get("exposure_flag") == "possible_overexposure":
        return "borderline"
    if video.get("occupancy_flag") in {
        "possible_cropping_or_full_frame_foreground",
        "foreground_touches_frame_edge",
        "possible_tiny_stone_or_low_contrast",
        "low_foreground_proxy",
    }:
        return "borderline"
    return "pass"


def _video_message(video: dict[str, Any], status: str | None) -> str | None:
    label = str(video.get("label") or "").zfill(2)
    if not video.get("file_readable"):
        return "Could not read this video file."
    if status == "fail":
        if video.get("fallback_triggered"):
            return "Too soft; automatic sharpness fallback would trigger."
        return f"Video {label} is not suitable for reconstruction."
    if status == "borderline":
        if video.get("fallback_triggered"):
            return "Automatic sharpness fallback would trigger."
        if video.get("exposure_flag") not in {None, "ok", "unavailable"}:
            return "Exposure should be reviewed before reconstruction."
        if video.get("occupancy_flag") not in {None, "ok", "unavailable"}:
            return "Stone framing should be reviewed before reconstruction."
        return "Capture has quality warnings."
    if status == "pass":
        return "Capture looks suitable."
    return None


def _summary(status: str, reason: str) -> str:
    if status == "pass":
        return "All four videos passed capture-quality screening."
    if status == "borderline":
        return (
            "Capture is usable with warnings; review and acknowledge before "
            f"continuing: {reason}."
        )
    return f"Capture quality is not suitable for reconstruction: {reason}."


def frontend_report(specimen: dict[str, Any]) -> dict[str, Any]:
    status = _lower_status(specimen.get("capture_screen")) or "fail"
    reason = specimen.get("reason") or "capture-quality checks did not complete"

    videos = []
    for index, label in enumerate(EXPECTED_VIDEO_STEMS, start=1):
        raw = specimen.get("videos", {}).get(label)
        if raw is None:
            videos.append({
                "index": index,
                "readable": False,
                "expected_frame_count": None,
                "median_sharpness": None,
                "sharp_gate_pass_percent": None,
                "fallback_triggered": None,
                "duplicate_rate_percent": None,
                "exposure_flag": "unavailable",
                "occupancy_flag": "unavailable",
                "foreground_occupancy_percent": None,
                "decode_failures": None,
                "status": "fail",
                "message": "Expected video was not provided.",
            })
            continue

        video_status = _video_status(raw)
        occupancy = raw.get("median_occupancy")
        duplicate = raw.get("duplicate_rate")
        videos.append({
            "index": index,
            "readable": bool(raw.get("file_readable")),
            "expected_frame_count": rounded(raw.get("expected_extracted_frames"), 3),
            "median_sharpness": rounded(raw.get("median_sharpness"), 3),
            "sharp_gate_pass_percent": rounded(
                raw.get("normal_sharp_gate_pass_percent"),
                3,
            ),
            "fallback_triggered": (
                bool(raw.get("fallback_triggered"))
                if raw.get("fallback_triggered") is not None else None
            ),
            "duplicate_rate_percent": (
                rounded(float(duplicate) * 100.0, 3)
                if duplicate is not None else None
            ),
            "exposure_flag": raw.get("exposure_flag"),
            "occupancy_flag": raw.get("occupancy_flag"),
            "foreground_occupancy_percent": (
                rounded(float(occupancy) * 100.0, 3)
                if occupancy is not None else None
            ),
            "decode_failures": rounded(raw.get("sampled_decode_failures"), 3),
            "status": video_status,
            "message": _video_message(raw, video_status),
        })

    continuity = []
    overlap = specimen.get("overlap", {})
    for left, right in zip(EXPECTED_VIDEO_STEMS, EXPECTED_VIDEO_STEMS[1:]):
        key = f"{left}_{right}"
        pair = f"{left} \u2192 {right}"
        metrics = overlap.get(key, {"max": 0, "median": 0.0})
        state = classify_continuity(metrics)
        message = None
        if state == "weak":
            message = f"Weak overlap between Videos {pair}."
        elif state == "poor":
            message = f"Poor overlap between Videos {pair}."
        continuity.append({
            "pair": pair,
            "state": state,
            "message": message,
        })

    return json_ready({
        "status": status,
        "summary": _summary(status, reason),
        "override_allowed": False,
        "acknowledgement_required": status == "borderline",
        "videos": videos,
        "continuity": continuity,
    })
