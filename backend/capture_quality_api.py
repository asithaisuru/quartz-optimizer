"""FastAPI router for production capture-quality pre-flight checks."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import tempfile
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/capture-quality", tags=["capture-quality"])

EXPECTED_CAPTURE_VIDEO_COUNT = 4
CAPTURE_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
SUPPORTED_SCAN_MODES = {"turntable", "handheld"}


def _validate_capture_quality_request(
    files: List[UploadFile],
    scan_mode: str,
) -> None:
    mode = (scan_mode or "").strip().lower()
    if mode not in SUPPORTED_SCAN_MODES:
        raise HTTPException(
            status_code=422,
            detail="scan_mode must be 'turntable' or 'handheld'.",
        )
    if len(files or []) != EXPECTED_CAPTURE_VIDEO_COUNT:
        raise HTTPException(
            status_code=422,
            detail="Capture-quality check requires exactly 4 video files.",
        )

    for index, upload in enumerate(files, start=1):
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in CAPTURE_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"File {index} must be a video with extension "
                    ".mp4, .mov, or .avi."
                ),
            )


def _save_uploads_to_temp(
    files: List[UploadFile],
    temp_dir: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for index, upload in enumerate(files, start=1):
        suffix = Path(upload.filename or "").suffix.lower()
        label = f"{index:02d}"
        destination = temp_dir / f"video_{label}{suffix}"
        with destination.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        paths[label] = destination
    return paths


def _run_capture_quality_analysis(
    video_paths: dict[str, Path],
    scan_mode: str,
) -> dict:
    from capture_quality import (
        analyze_capture_set,
        frontend_report,
        production_settings,
    )

    specimen = analyze_capture_set(
        video_paths,
        settings=production_settings(),
        force_sequential=False,
        metadata={"scan_mode": (scan_mode or "").strip().lower()},
    )
    return frontend_report(specimen)


@router.post("/check")
async def check_capture_quality(
    files: List[UploadFile] = File(...),
    scan_mode: str = Form("turntable"),
):
    _validate_capture_quality_request(files, scan_mode)
    try:
        with tempfile.TemporaryDirectory(prefix="quartz_capture_quality_") as tmp:
            video_paths = _save_uploads_to_temp(files, Path(tmp))
            return _run_capture_quality_analysis(video_paths, scan_mode)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Capture-quality check failed.")
        raise HTTPException(
            status_code=500,
            detail="Capture-quality screening could not complete.",
        ) from exc
