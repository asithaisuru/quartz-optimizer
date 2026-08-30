"""Run detector candidates and apply the configured class-aware policy."""

from __future__ import annotations

import importlib.metadata
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from defect_policy import (
    DetectionRecord,
    apply_policy,
    load_policy_config,
    load_taxonomy,
    materialize_policy_outputs,
    policy_summary,
    resolve_taxonomy_path,
)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
MODEL_PATH = BASE_DIR / "best.pt"
DEFAULT_POLICY_PATH = REPO_ROOT / "research" / "configs" / "defect_policy.example.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _sha256_file(path: str | Path) -> str | None:
    from hashlib import sha256

    candidate = Path(path)
    if not candidate.exists():
        return None
    digest = sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _select_device() -> str:
    """Use CUDA only after a small runtime check; otherwise use CPU."""

    try:
        import torch
    except ImportError:
        print("   PyTorch unavailable - detector model cannot be loaded here.")
        return "cpu"

    if not torch.cuda.is_available():
        print("   No CUDA GPU detected - using CPU.")
        return "cpu"
    import warnings

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sample = torch.zeros(2, device="cuda")
            _ = torch.matmul(sample.unsqueeze(0), sample.unsqueeze(1)).item()
        if any(
            "cuda capability" in str(warning.message).casefold()
            for warning in caught
        ):
            print("   CUDA capability is unsupported by this PyTorch build; using CPU.")
            return "cpu"
        print(f"   GPU: {torch.cuda.get_device_name(0)} - using CUDA.")
        return "cuda"
    except Exception as exc:
        print(f"   CUDA test failed ({exc}) - using CPU.")
        return "cpu"


def _classic_cv_defect_mask(img_path: str | Path) -> np.ndarray | None:
    """Return the existing Canny/Hough candidate mask when OpenCV is available."""

    try:
        import cv2
    except ImportError:
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blur, 55, 145)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=38,
        minLineLength=max(18, min(gray.shape[:2]) // 12),
        maxLineGap=8,
    )

    mask = np.zeros_like(gray)
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            cv2.line(mask, (x1, y1), (x2, y2), 255, 2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    area_limit = gray.shape[0] * gray.shape[1] * 0.08
    for contour in contours:
        area = cv2.contourArea(contour)
        if 12 <= area <= area_limit:
            x, y, width, height = cv2.boundingRect(contour)
            aspect = max(width, height) / max(min(width, height), 1)
            if aspect >= 2.2:
                cv2.drawContours(mask, [contour], -1, 255, 1)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask if int(np.count_nonzero(mask)) >= 25 else None


def _save_binary_mask(mask: Any, path: Path, output_size: tuple[int, int]) -> int:
    array = np.asarray(mask, dtype=float)
    if array.ndim > 2:
        array = np.squeeze(array)
    binary = np.where(array > 0.5, 255, 0).astype(np.uint8)
    image = Image.fromarray(binary, mode="L")
    if image.size != output_size:
        image = image.resize(output_size, Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return int(np.count_nonzero(np.asarray(image) > 127))


def _array_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value).reshape(-1).tolist()


def _mask_values(result: Any) -> list[np.ndarray]:
    masks = getattr(result, "masks", None)
    data = getattr(masks, "data", None)
    if data is None:
        return []
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()
    return [np.asarray(mask) for mask in np.asarray(data)]


def _class_name(names: Any, class_id: int | None) -> str:
    if class_id is None:
        return "unknown"
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), f"class_{class_id}")))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def _model_factory_default(model_path: str | Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(model_path))


def _predict(model: Any, image_path: Path, device: str, confidence: float) -> Any:
    return model.predict(
        str(image_path),
        device=device,
        conf=confidence,
        save=False,
        verbose=False,
    )


def _json_record(record: DetectionRecord, job_path: Path) -> dict[str, Any]:
    value = record.to_dict()
    for key in (
        "mask_path",
        "visualization_mask_path",
        "mapping_mask_path",
        "no_cut_mask_path",
    ):
        path_value = value.get(key)
        if not path_value:
            continue
        candidate = Path(path_value)
        if candidate.is_absolute():
            try:
                value[key] = candidate.relative_to(job_path).as_posix()
            except ValueError:
                value[key] = candidate.as_posix()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _legacy_report(
    image_files: list[Path],
    records: list[DetectionRecord],
    summary: dict[str, Any],
) -> dict[str, Any]:
    accepted_by_image: dict[str, list[DetectionRecord]] = defaultdict(list)
    for record in records:
        if record.accepted_for_3d_mapping:
            accepted_by_image[record.image_id].append(record)
    details = []
    for image in image_files:
        accepted = accepted_by_image.get(image.name, [])
        if accepted:
            details.append(
                {
                    "image": image.name,
                    "fracture_count": len(accepted),
                    "masks": [
                        item.mapping_mask_path for item in accepted
                        if item.mapping_mask_path
                    ],
                    "sources": [item.source for item in accepted],
                    "prediction_ids": [item.prediction_id for item in accepted],
                }
            )
    return {
        "total_images_scanned": len(image_files),
        "total_fractures_detected": summary["mapping_count"],
        "raw_prediction_count": summary["raw_prediction_count"],
        "details": details,
        "source_counts": summary["counts_by_source"],
        "defect_detection": summary,
        "compatibility_note": (
            "total_fractures_detected now counts policy-approved mapping masks, "
            "not all raw detector candidates."
        ),
    }


def run_ai_pipeline(
    job_path: str | Path,
    policy_config_path: str | Path | None = None,
    model_path: str | Path = MODEL_PATH,
    model_factory: Callable[[str | Path], Any] | None = None,
    cv_mask_factory: Callable[[str | Path], np.ndarray | None] | None = None,
) -> dict[str, Any]:
    """Run candidate extraction, policy decisions, and separated output writing."""

    print("=" * 55)
    print("=== Class-Aware Defect Candidate Detection")
    print("=" * 55)

    job = Path(job_path).resolve()
    images_dir = job / "images"
    detections_dir = job / "detections"
    raw_masks_dir = detections_dir / "raw_masks"
    raw_masks_dir.mkdir(parents=True, exist_ok=True)

    configured_path = (
        policy_config_path
        or os.environ.get("QUARTZ_DEFECT_POLICY_CONFIG")
        or DEFAULT_POLICY_PATH
    )
    config = load_policy_config(configured_path)
    taxonomy_path = resolve_taxonomy_path(config, REPO_ROOT)
    taxonomy = load_taxonomy(taxonomy_path)
    confidence_threshold = float(
        config.get("minimum_confidence_default", 0.25)
    )
    configured_thresholds = [confidence_threshold]
    configured_thresholds.extend(
        float(value)
        for value in config.get("minimum_confidence_by_class", {}).values()
    )
    configured_thresholds.extend(
        float(entry.minimum_confidence)
        for entry in taxonomy.values()
        if entry.minimum_confidence is not None
    )
    inference_threshold = min(configured_thresholds)
    requested_model = Path(model_path).resolve()
    model_sha256 = _sha256_file(requested_model)
    model_version = _package_version("ultralytics")

    runtime_events: list[dict[str, Any]] = []
    runtime_warnings: list[str] = []
    model = None
    device = "cpu"
    global_model_event = "success"
    factory = model_factory or _model_factory_default
    if not requested_model.exists():
        global_model_event = "model_unavailable"
        runtime_events.append(
            {
                "event": global_model_event,
                "message": f"Model file not found: {requested_model}",
            }
        )
    else:
        try:
            device = _select_device() if model_factory is None else "cpu"
            model = factory(requested_model)
        except Exception as exc:
            global_model_event = "model_load_error"
            runtime_events.append(
                {
                    "event": global_model_event,
                    "message": str(exc),
                }
            )

    image_files = sorted(
        (
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )
    print(f"   Scanning {len(image_files)} images (device={device})...")

    records: list[DetectionRecord] = []
    image_events: dict[str, str] = {}
    cv_factory = cv_mask_factory or _classic_cv_defect_mask
    opencv_available = True
    if cv_mask_factory is None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            opencv_available = False
            runtime_warnings.append("opencv_runtime_unavailable")

    for index, image_path in enumerate(image_files):
        if index % 10 == 0:
            print(f"   [{index + 1}/{len(image_files)}] Analysing {image_path.name}...")
        image_id = image_path.name
        image_event = global_model_event
        yolo_count = 0

        if model is not None:
            try:
                try:
                    results = _predict(
                        model,
                        image_path,
                        device,
                        inference_threshold,
                    )
                except Exception:
                    if device != "cuda":
                        raise
                    print("   CUDA inference failed - retrying this image on CPU.")
                    device = "cpu"
                    results = _predict(
                        model,
                        image_path,
                        device,
                        inference_threshold,
                    )
                result = results[0] if results else None
                masks = _mask_values(result) if result is not None else []
                boxes = getattr(result, "boxes", None) if result is not None else None
                class_ids = _array_values(getattr(boxes, "cls", None))
                confidences = _array_values(getattr(boxes, "conf", None))
                names = getattr(result, "names", getattr(model, "names", {}))
                orig_shape = getattr(result, "orig_shape", None)
                if orig_shape is not None:
                    output_size = (int(orig_shape[1]), int(orig_shape[0]))
                else:
                    with Image.open(image_path) as source_image:
                        output_size = source_image.size

                for prediction_index, mask in enumerate(masks):
                    class_id = (
                        int(class_ids[prediction_index])
                        if prediction_index < len(class_ids)
                        else None
                    )
                    confidence = (
                        float(confidences[prediction_index])
                        if prediction_index < len(confidences)
                        else None
                    )
                    prediction_id = (
                        f"{image_path.stem}-yolo-{prediction_index:04d}"
                    )
                    mask_path = raw_masks_dir / f"{prediction_id}.png"
                    area = _save_binary_mask(mask, mask_path, output_size)
                    records.append(
                        DetectionRecord(
                            prediction_id=prediction_id,
                            image_id=image_id,
                            source_image_path=f"images/{image_path.name}",
                            source="yolo",
                            class_id=class_id,
                            class_name=_class_name(names, class_id),
                            confidence=confidence,
                            mask_path=str(mask_path),
                            mask_width=output_size[0],
                            mask_height=output_size[1],
                            mask_area_pixels=area,
                            model_path=str(requested_model),
                            model_sha256=model_sha256,
                            model_version=model_version,
                        )
                    )
                yolo_count = len(masks)
                image_event = "success" if yolo_count else "zero_detections"
            except Exception as exc:
                image_event = "inference_error"
                runtime_events.append(
                    {
                        "event": image_event,
                        "image_id": image_id,
                        "message": str(exc),
                    }
                )

        image_events[image_id] = image_event
        if config.get("opencv", {}).get("enabled", False) and opencv_available:
            try:
                cv_mask = cv_factory(image_path)
            except Exception as exc:
                cv_mask = None
                runtime_events.append(
                    {
                        "event": "opencv_error",
                        "image_id": image_id,
                        "message": str(exc),
                    }
                )
            if cv_mask is not None:
                prediction_id = f"{image_path.stem}-opencv-0000"
                mask_path = raw_masks_dir / f"{prediction_id}.png"
                height, width = np.asarray(cv_mask).shape[:2]
                area = _save_binary_mask(cv_mask, mask_path, (width, height))
                records.append(
                    DetectionRecord(
                        prediction_id=prediction_id,
                        image_id=image_id,
                        source_image_path=f"images/{image_path.name}",
                        source="opencv",
                        class_id=None,
                        class_name="opencv_candidate",
                        confidence=None,
                        mask_path=str(mask_path),
                        mask_width=width,
                        mask_height=height,
                        mask_area_pixels=area,
                        model_path=None,
                        model_sha256=None,
                        model_version=None,
                        policy_metadata={
                            "algorithm": "clahe_canny_hough_contours",
                            "canny_thresholds": [55, 145],
                            "hough_threshold": 38,
                        },
                    )
                )

    raw_payload = {
        "schema_version": "1.0",
        "records": [_json_record(record, job) for record in records],
        "image_events": image_events,
        "runtime_events": runtime_events,
    }
    _write_json(detections_dir / "raw_predictions.json", raw_payload)

    apply_policy(records, taxonomy, config, image_events=image_events)
    materialize_policy_outputs(records, job)
    summary = policy_summary(
        records,
        config,
        taxonomy=taxonomy,
        taxonomy_path=taxonomy_path,
        model_sha256=model_sha256,
        runtime_warnings=runtime_warnings,
    )
    decisions_payload = {
        "schema_version": "1.0",
        "policy": config["policy"],
        "strict_research_mode": bool(
            config.get("strict_research_mode", True)
        ),
        "records": [_json_record(record, job) for record in records],
        "image_events": image_events,
        "runtime_events": runtime_events,
    }
    _write_json(detections_dir / "policy_decisions.json", decisions_payload)
    _write_json(detections_dir / "policy_summary.json", summary)
    _write_json(job / "ai_results.json", _legacy_report(image_files, records, summary))

    print(
        "Defect candidate analysis complete: "
        f"{len(records)} raw, {summary['mapping_count']} mapping-approved, "
        f"{summary['no_cut_count']} no-cut-approved."
    )
    return summary
