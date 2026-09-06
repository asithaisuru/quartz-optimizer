"""Research-only sequential reconstruction batch runner.

Runs selected specimens through the current production API, records job IDs and
logs, generates contact sheets, and validates completed meshes against the
authoritative physical measurement table. It does not modify reconstruction
code and never uses physical L/W/H for scaling, tuning, or rejection.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "final_research_evidence" / "reconstruction_multi"
INPUT_ROOT = OUT_DIR / "batch_inputs"
JOBS_DIR = REPO_ROOT / "jobs"
PHYSICAL_CSV = REPO_ROOT / "final_research_evidence" / "reconstruction" / "physical_measurements.csv"
SOURCE_ANALYZER = REPO_ROOT / "final_research_evidence" / "reconstruction" / "analyze_reconstruction_accuracy.py"

JOB_MAP_CSV = OUT_DIR / "batch_job_map.csv"
RESULTS_CSV = OUT_DIR / "batch_validation_results.csv"
SUMMARY_JSON = OUT_DIR / "batch_validation_summary.json"
RUN_MANIFEST_JSON = OUT_DIR / "batch_run_manifest.json"
INPUT_README = INPUT_ROOT / "README.md"
CONTACT_DIR = OUT_DIR / "contact_sheets"
LOG_DIR = OUT_DIR / "batch_logs"

METHODOLOGY_METADATA_WORDING = (
    "Because all closer-weight candidates available for QZ-30 failed the predefined "
    "capture-quality screen, QZ-30 was retained as degraded low-weight evidence. "
    "QZ-03 was subsequently included as an additional capture-qualified validation "
    "specimen rather than as a weight-equivalent replacement."
)

SPECIMENS = [
    {
        "specimen_id": "QZ-30",
        "known_weight": "10.52",
        "retained_as": "degraded_low_weight_evidence",
        "reconstruction_provenance_note": "SPARSE_FALLBACK",
        "retention_reason": (
            "No available closer-weight candidate passed the corrected pre-COLMAP "
            "capture-quality screen; preserve QZ-30 as degraded low-weight evidence."
        ),
    },
    {"specimen_id": "QZ-14", "known_weight": "50.11"},
    {
        "specimen_id": "QZ-09",
        "known_weight": "120.50",
        "active_validation": False,
        "replaced_by": "QZ-08",
        "replacement_reason": "Existing capture unsuitable for reconstruction; recapture unavailable.",
    },
    {
        "specimen_id": "QZ-08",
        "known_weight": "142.54",
        "replaces": "QZ-09",
        "replacement_basis": (
            "Closest unused specimen weight from authoritative physical measurements only; "
            "selected before inspecting QZ-08 reconstruction outputs."
        ),
    },
    {"specimen_id": "QZ-05", "known_weight": "244.02"},
    {
        "specimen_id": "QZ-03",
        "known_weight": "90.76",
        "supplemental_validation": True,
        "candidate_role": "additional_capture_qualified_true_dense_candidate",
        "capture_screen": "PASS",
        "weight_equivalent_replacement_for_qz30": False,
        "selection_basis": (
            "QZ-03 passed the corrected pre-COLMAP capture-quality audit and is "
            "included as an additional validation candidate, not as a like-for-like "
            "or low-weight-stratum replacement for QZ-30."
        ),
    },
]
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TERMINAL_STATUSES = {"Completed", "Failed", "Cancelled", "Error reading status"}
DENSITY_QUARTZ = 2.65
CARATS_PER_GRAM = 5.0
PROPOSAL_TARGET_MM = 0.1


def rel(path: Path | str | None) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_validation_helpers():
    spec = importlib.util.spec_from_file_location("qz_reconstruction_validation", SOURCE_ANALYZER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def physical_measurements() -> dict[str, dict]:
    rows = {}
    for row in read_csv(PHYSICAL_CSV):
        rows[row["specimen_id"]] = {
            "specimen_id": row["specimen_id"],
            "id_range": row.get("id_range", ""),
            "physical_weight": float(row["physical_weight"]),
            "physical_length_mm": float(row["physical_length_mm"]),
            "physical_width_mm": float(row["physical_width_mm"]),
            "physical_height_mm": float(row["physical_height_mm"]),
        }
    return rows


def specimen_folder(specimen_id: str) -> Path:
    return INPUT_ROOT / specimen_id


def input_files_for(specimen_id: str) -> tuple[list[Path], list[Path], list[str]]:
    folder = specimen_folder(specimen_id)
    ref_dir = folder / "reference"
    video_dir = folder / "videos"
    references = sorted([
        p for p in ref_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]) if ref_dir.exists() else []
    videos = sorted([
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]) if video_dir.exists() else []
    errors = []
    if len(references) != 1:
        errors.append("expected exactly 1 reference image in reference/")
    if len(videos) != 4:
        errors.append("expected exactly 4 videos in videos/")
    return references, videos, errors


def selected_specimens(args) -> list[dict]:
    if not args.only:
        return active_specimens()
    selected = [item for item in SPECIMENS if item["specimen_id"] == args.only]
    if not selected:
        raise ValueError(f"unknown specimen for this batch: {args.only}")
    return selected


def active_specimens() -> list[dict]:
    return [item for item in SPECIMENS if item.get("active_validation", True)]


def active_specimen_ids() -> set[str]:
    return {item["specimen_id"] for item in active_specimens()}


def replacement_metadata() -> list[dict]:
    metadata = []
    by_id = {item["specimen_id"]: item for item in SPECIMENS}
    for item in SPECIMENS:
        replacement_id = item.get("replaced_by")
        if not replacement_id:
            continue
        replacement = by_id.get(replacement_id, {})
        metadata.append({
            "excluded_specimen_id": item["specimen_id"],
            "excluded_weight_ct": item["known_weight"],
            "replacement_specimen_id": replacement_id,
            "replacement_weight_ct": replacement.get("known_weight", ""),
            "reason": item.get("replacement_reason", ""),
            "selection_basis": replacement.get("replacement_basis", ""),
        })
    return metadata


def retained_degraded_metadata() -> list[dict]:
    return [
        {
            "specimen_id": item["specimen_id"],
            "known_weight_ct": item["known_weight"],
            "retained_as": item.get("retained_as", ""),
            "reconstruction_provenance_note": item.get("reconstruction_provenance_note", ""),
            "retention_reason": item.get("retention_reason", ""),
        }
        for item in SPECIMENS
        if item.get("retained_as")
    ]


def supplemental_validation_metadata() -> list[dict]:
    return [
        {
            "specimen_id": item["specimen_id"],
            "known_weight_ct": item["known_weight"],
            "candidate_role": item.get("candidate_role", ""),
            "capture_screen": item.get("capture_screen", ""),
            "weight_equivalent_replacement_for_qz30": item.get(
                "weight_equivalent_replacement_for_qz30", ""
            ),
            "selection_basis": item.get("selection_basis", ""),
        }
        for item in SPECIMENS
        if item.get("supplemental_validation")
    ]


def write_input_readme() -> None:
    lines = [
        "# Reconstruction Batch Inputs",
        "",
        "Place files exactly as follows. The reference photo is for specimen identity evidence only and is not uploaded for reconstruction.",
        "",
        "```text",
        "final_research_evidence/reconstruction_multi/batch_inputs/",
    ]
    for item in SPECIMENS:
        qz = item["specimen_id"]
        lines.extend([
            f"  {qz}/",
            "    reference/",
            f"      {qz}_reference.jpg",
            "    videos/",
            "      01.mov",
            "      02.mov",
            "      03.mov",
            "      04.mov",
        ])
    lines.extend([
        "```",
        "",
        "Use the recorded weights embedded in the runner; do not add physical dimensions to inputs.",
        "",
    ])
    INPUT_README.parent.mkdir(parents=True, exist_ok=True)
    INPUT_README.write_text("\n".join(lines), encoding="utf-8")


def init_inputs() -> None:
    for item in SPECIMENS:
        (specimen_folder(item["specimen_id"]) / "reference").mkdir(parents=True, exist_ok=True)
        (specimen_folder(item["specimen_id"]) / "videos").mkdir(parents=True, exist_ok=True)
    write_input_readme()


def post_upload(api_base: str, videos: list[Path], known_weight: str) -> dict:
    command = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "-X",
        "POST",
        api_base.rstrip("/") + "/upload",
        "-F",
        "is_video=true",
        "-F",
        "scan_mode=turntable",
        "-F",
        f"known_weight={known_weight}",
        "-F",
        "preferred_shape=Auto",
        "-F",
        "cut_mode=multi",
    ]
    for path in videos:
        command.extend(["-F", f"files=@{path};filename={path.name}"])
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "upload failed")
    return json.loads(result.stdout)


def get_status(api_base: str, job_id: str) -> dict:
    with urllib.request.urlopen(
        api_base.rstrip("/") + f"/jobs/{job_id}/status",
        timeout=60,
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def api_reachable(api_base: str) -> tuple[bool, str]:
    try:
        status = get_status(api_base, "00000000-0000-0000-0000-000000000000")
        if status.get("status") in {"Not Found", "Completed", "Processing", "Failed"}:
            return True, ""
        return True, f"unexpected status probe response: {status.get('status')}"
    except Exception as exc:
        return False, str(exc)


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def poll_until_terminal(api_base: str, job_id: str, specimen_id: str,
                        poll_seconds: int, timeout_hours: float) -> dict:
    log_path = LOG_DIR / f"{specimen_id}_{job_id}_status.jsonl"
    deadline = time.time() + timeout_hours * 3600.0
    last = {}
    while time.time() < deadline:
        try:
            status = get_status(api_base, job_id)
        except (urllib.error.URLError, TimeoutError) as exc:
            status = {"status": "poll_error", "message": str(exc), "job_id": job_id}
        last = {"observed_utc": utc_now(), **status}
        append_jsonl(log_path, last)
        if status.get("status") in TERMINAL_STATUSES:
            return last
        time.sleep(max(5, int(poll_seconds)))
    return {
        "observed_utc": utc_now(),
        "status": "Timed Out",
        "message": f"polling exceeded {timeout_hours} hours",
        "job_id": job_id,
    }


def choose_frames(frames: list[Path], count: int = 12) -> list[Path]:
    if len(frames) <= count:
        return frames
    indices = sorted({round(i * (len(frames) - 1) / (count - 1)) for i in range(count)})
    return [frames[i] for i in indices]


def make_contact_sheet(specimen_id: str, job_id: str) -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except Exception:
        return ""

    frames = sorted([
        p for p in (JOBS_DIR / job_id / "images").glob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    frames = choose_frames(frames)
    if not frames:
        return ""

    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONTACT_DIR / f"{specimen_id}_{job_id.split('-')[0]}_contact_sheet.jpg"
    cols = 4
    rows = math.ceil(len(frames) / cols)
    thumb_w, thumb_h = 300, 220
    label_h = 32
    header_h = 54
    margin = 18
    gap = 12
    width = margin * 2 + cols * thumb_w + (cols - 1) * gap
    height = margin * 2 + header_h + rows * (thumb_h + label_h) + (rows - 1) * gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        small = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    draw.text((margin, margin), f"{specimen_id} job {job_id}", fill=(20, 28, 43), font=font)
    draw.text((margin, margin + 24), "Representative extracted production frames.", fill=(75, 85, 99), font=small)
    y0 = margin + header_h
    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x = margin + col * (thumb_w + gap)
        y = y0 + row * (thumb_h + label_h + gap)
        with Image.open(frame) as image:
            image = ImageOps.contain(image.convert("RGB"), (thumb_w, thumb_h), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (thumb_w, thumb_h), (245, 247, 250))
            tile.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline=(209, 213, 219), width=1)
        draw.text((x, y + thumb_h + 7), frame.name, fill=(31, 41, 55), font=small)
    sheet.save(out_path, quality=92)
    return rel(out_path)


def copy_reference(specimen_id: str, reference: Path) -> str:
    destination = OUT_DIR / "reference_photos" / f"{specimen_id}{reference.suffix.lower()}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reference, destination)
    return rel(destination)


def load_existing_map() -> dict[str, dict]:
    return {row["specimen_id"]: row for row in read_csv(JOB_MAP_CSV)}


def archive_existing_failures() -> str:
    if not JOB_MAP_CSV.exists():
        return ""
    rows = read_csv(JOB_MAP_CSV)
    if not any(row.get("status") in {"Runner Failed", "Failed", "Timed Out"} for row in rows):
        return ""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = LOG_DIR / f"batch_job_map_failure_preserved_{timestamp}.csv"
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(JOB_MAP_CSV, archive)
    return rel(archive)


def job_artifact_paths(job_id: str) -> dict[str, str]:
    job_dir = JOBS_DIR / job_id
    return {
        "job_folder": rel(job_dir),
        "analysis_report_path": rel(job_dir / "analysis_report.json") if (job_dir / "analysis_report.json").exists() else "",
        "mesh_path": rel(job_dir / "dense" / "final_textured_model.ply") if (job_dir / "dense" / "final_textured_model.ply").exists() else "",
        "status_path": rel(job_dir / "status.json") if (job_dir / "status.json").exists() else "",
    }


def update_job_map(rows_by_specimen: dict[str, dict]) -> None:
    fields = [
        "specimen_id",
        "known_weight_ct",
        "physical_length_mm",
        "physical_width_mm",
        "physical_height_mm",
        "status",
        "job_id",
        "status_message",
        "submitted_utc",
        "completed_utc",
        "input_folder",
        "reference_photo_path",
        "video_files",
        "contact_sheet_path",
        "job_folder",
        "status_path",
        "analysis_report_path",
        "mesh_path",
        "failure_preserved",
    ]
    ordered_ids = [item["specimen_id"] for item in active_specimens()]
    ordered_ids.extend(
        item["specimen_id"] for item in SPECIMENS
        if item["specimen_id"] not in ordered_ids
    )
    ordered_ids.extend(
        specimen_id for specimen_id in rows_by_specimen
        if specimen_id not in ordered_ids
    )
    rows = [rows_by_specimen[specimen_id] for specimen_id in ordered_ids if specimen_id in rows_by_specimen]
    write_csv(JOB_MAP_CSV, fields, rows)


def submit_and_poll(args, physical: dict[str, dict]) -> dict[str, dict]:
    rows_by_specimen = load_existing_map()
    run_rows = dict(rows_by_specimen)
    if args.dry_run:
        for item in selected_specimens(args):
            specimen_id = item["specimen_id"]
            reference, videos, errors = input_files_for(specimen_id)
            specimen = physical[specimen_id]
            run_rows[specimen_id] = {
                "specimen_id": specimen_id,
                "known_weight_ct": item["known_weight"],
                "physical_length_mm": specimen["physical_length_mm"],
                "physical_width_mm": specimen["physical_width_mm"],
                "physical_height_mm": specimen["physical_height_mm"],
                "status": "Input Missing" if errors else "Dry Run Ready",
                "status_message": "; ".join(errors) if errors else "input folder complete; no API submission",
                "input_folder": rel(specimen_folder(specimen_id)),
                "reference_photo_path": rel(reference[0]) if reference else "",
                "video_files": "; ".join(rel(path) for path in videos),
                "failure_preserved": "yes",
            }
        return run_rows

    reachable, reason = api_reachable(args.api_base)
    if not reachable:
        for item in selected_specimens(args):
            specimen_id = item["specimen_id"]
            existing = rows_by_specimen.get(specimen_id, {})
            run_rows[specimen_id] = {
                **existing,
                "specimen_id": specimen_id,
                "known_weight_ct": item["known_weight"],
                "status": "API Unreachable",
                "status_message": reason,
                "failure_preserved": "yes",
            }
        return run_rows

    for item in selected_specimens(args):
        specimen_id = item["specimen_id"]
        existing = rows_by_specimen.get(specimen_id, {})
        if existing.get("status") == "Completed" and existing.get("job_id") and not args.force_new:
            continue

        reference, videos, errors = input_files_for(specimen_id)
        specimen = physical[specimen_id]
        base_row = {
            "specimen_id": specimen_id,
            "known_weight_ct": item["known_weight"],
            "physical_length_mm": specimen["physical_length_mm"],
            "physical_width_mm": specimen["physical_width_mm"],
            "physical_height_mm": specimen["physical_height_mm"],
            "input_folder": rel(specimen_folder(specimen_id)),
            "video_files": "; ".join(rel(path) for path in videos),
            "failure_preserved": "yes",
        }
        if errors:
            rows_by_specimen[specimen_id] = {
                **base_row,
                "status": "Input Missing",
                "status_message": "; ".join(errors),
            }
            update_job_map(rows_by_specimen)
            continue

        try:
            copied_reference = copy_reference(specimen_id, reference[0])
            submitted = post_upload(args.api_base, videos, item["known_weight"])
            job_id = submitted["job_id"]
            row = {
                **base_row,
                "status": "Submitted",
                "job_id": job_id,
                "status_message": submitted.get("status_url", ""),
                "submitted_utc": utc_now(),
                "reference_photo_path": copied_reference,
            }
            row.update(job_artifact_paths(job_id))
            rows_by_specimen[specimen_id] = row
            update_job_map(rows_by_specimen)
            if args.stop_after_submit:
                return rows_by_specimen

            final_status = poll_until_terminal(
                args.api_base,
                job_id,
                specimen_id,
                args.poll_seconds,
                args.timeout_hours,
            )
            row["status"] = final_status.get("status", "")
            row["status_message"] = final_status.get("message", final_status.get("step", ""))
            row["completed_utc"] = utc_now()
            row.update(job_artifact_paths(job_id))
            if row["status"] == "Completed":
                row["contact_sheet_path"] = make_contact_sheet(specimen_id, job_id)
            rows_by_specimen[specimen_id] = row
            update_job_map(rows_by_specimen)
        except Exception as exc:
            rows_by_specimen[specimen_id] = {
                **base_row,
                "status": "Runner Failed",
                "status_message": str(exc),
                "completed_utc": utc_now(),
            }
            update_job_map(rows_by_specimen)
    return rows_by_specimen


def mm_per_mesh_unit(mesh_path: Path, known_weight_ct: float, report: dict,
                     helpers) -> tuple[float | None, str]:
    try:
        import trimesh

        mesh = trimesh.load(mesh_path)
        if hasattr(mesh, "geometry"):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        vol_mesh = mesh if mesh.is_watertight else mesh.convex_hull
        target_cm3 = (known_weight_ct / CARATS_PER_GRAM) / DENSITY_QUARTZ
        scale_factor = (target_cm3 / abs(float(vol_mesh.volume))) ** (1.0 / 3.0)
        return float(scale_factor * 10.0), "production_mass_density_scale_formula"
    except Exception:
        points = helpers.read_ply_vertices(mesh_path)
        app_dims = report.get("rough_dimensions_mm") or []
        if len(app_dims) == 3:
            raw_axis = helpers.axis_extents(points)
            ratios = [
                float(app_dims[i]) / raw_axis[i]
                for i in range(3)
                if raw_axis[i] > 0
            ]
            if ratios:
                return float(np.median(ratios)), "inferred_from_production_aabb_report"
    return None, "unavailable"


def metric_values(reconstructed: np.ndarray, physical: np.ndarray) -> dict:
    signed = reconstructed - physical
    absolute = np.abs(signed)
    percentages = absolute / np.maximum(physical, 1e-9) * 100.0
    return {
        "signed": signed,
        "absolute": absolute,
        "percentage": percentages,
        "mae": float(np.mean(absolute)),
        "rmse": float(math.sqrt(np.mean(absolute ** 2))),
        "median": float(np.median(absolute)),
        "max": float(np.max(absolute)),
        "mape": float(np.mean(percentages)),
    }


def threshold_count(abs_errors: np.ndarray, threshold: float) -> int:
    return int(np.count_nonzero(abs_errors <= threshold))


def ply_header(path: Path) -> list[str]:
    lines = []
    try:
        with path.open("rb") as handle:
            for _ in range(256):
                raw = handle.readline()
                if not raw:
                    break
                text = raw.decode("ascii", errors="ignore").strip()
                lines.append(text)
                if text == "end_header":
                    break
    except OSError:
        return []
    return lines


def ply_vertex_count(path: Path) -> int | None:
    for line in ply_header(path):
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            try:
                return int(parts[2])
            except ValueError:
                return None
    return None


def ply_has_normal_properties(path: Path) -> bool:
    properties = set()
    for line in ply_header(path):
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "property":
            properties.add(parts[-1])
    return {"nx", "ny", "nz"}.issubset(properties)


def reconstruction_provenance(map_row: dict) -> tuple[str, str]:
    if map_row.get("status") != "Completed":
        return "FAILED", "job status is not Completed"
    job_dir = REPO_ROOT / map_row.get("job_folder", "")
    fused_path = job_dir / "dense" / "fused.ply"
    if not fused_path.exists() and map_row.get("mesh_path"):
        fused_path = (REPO_ROOT / map_row["mesh_path"]).parent / "fused.ply"
    mesh_path = REPO_ROOT / map_row.get("mesh_path", "")
    if not mesh_path.exists():
        return "FAILED", "final mesh artifact is missing"
    if not fused_path.exists():
        return "FAILED", "dense/fused.ply provenance artifact is missing"

    fused_vertices = ply_vertex_count(fused_path)
    if not fused_vertices:
        return "FAILED", "dense/fused.ply has no vertices"
    if ply_has_normal_properties(fused_path):
        return "TRUE_DENSE", (
            f"dense/fused.ply has {fused_vertices} vertices with nx/ny/nz "
            "normal properties from stereo fusion"
        )
    return "SPARSE_FALLBACK", (
        f"dense/fused.ply has {fused_vertices} vertices but no nx/ny/nz "
        "normal properties, consistent with sparse point-cloud fallback"
    )


def validate_completed_jobs(physical: dict[str, dict]) -> None:
    helpers = load_validation_helpers()
    rows = []
    all_errors = {"principal": [], "aabb": []}
    all_pct = {"principal": [], "aabb": []}
    validation_ids = active_specimen_ids()
    for map_row in read_csv(JOB_MAP_CSV):
        if map_row.get("status") != "Completed":
            continue
        specimen_id = map_row["specimen_id"]
        mesh_path = REPO_ROOT / map_row.get("mesh_path", "")
        report_path = REPO_ROOT / map_row.get("analysis_report_path", "")
        if not mesh_path.exists() or not report_path.exists():
            continue
        provenance, provenance_evidence = reconstruction_provenance(map_row)
        in_active_sample = specimen_id in validation_ids
        included_in_dense = in_active_sample and provenance == "TRUE_DENSE"
        if included_in_dense:
            exclusion_reason = ""
        elif not in_active_sample:
            exclusion_reason = "not in active validation sample"
        else:
            exclusion_reason = f"not TRUE_DENSE ({provenance})"
        report = read_json(report_path)
        specimen = physical[specimen_id]
        mm, scale_source = mm_per_mesh_unit(
            mesh_path,
            float(map_row["known_weight_ct"]),
            report,
            helpers,
        )
        if not mm:
            continue
        points = helpers.read_ply_vertices(mesh_path)
        physical_ranked = np.sort(np.array([
            specimen["physical_length_mm"],
            specimen["physical_width_mm"],
            specimen["physical_height_mm"],
        ], dtype=float))[::-1]
        principal_ranked = helpers.principal_extents(points) * mm
        app_dims = report.get("rough_dimensions_mm") or (helpers.axis_extents(points) * mm).tolist()
        aabb_ranked = np.sort(np.array(app_dims, dtype=float))[::-1]
        principal = metric_values(principal_ranked, physical_ranked)
        aabb = metric_values(aabb_ranked, physical_ranked)
        if included_in_dense:
            all_errors["principal"].extend(principal["absolute"].tolist())
            all_errors["aabb"].extend(aabb["absolute"].tolist())
            all_pct["principal"].extend(principal["percentage"].tolist())
            all_pct["aabb"].extend(aabb["percentage"].tolist())
        rows.append({
            "specimen_id": specimen_id,
            "job_id": map_row["job_id"],
            "known_weight_ct": map_row["known_weight_ct"],
            "active_validation_sample": "yes" if in_active_sample else "no",
            "reconstruction_provenance": provenance,
            "provenance_evidence": provenance_evidence,
            "included_in_dense_aggregate": "yes" if included_in_dense else "no",
            "dense_aggregate_exclusion_reason": exclusion_reason,
            "scale_source": scale_source,
            "physical_ranked_mm": " x ".join(f"{v:.3f}" for v in physical_ranked),
            "production_aabb_ranked_mm": " x ".join(f"{v:.3f}" for v in aabb_ranked),
            "principal_oriented_ranked_mm": " x ".join(f"{v:.3f}" for v in principal_ranked),
            "aabb_mae_mm": f"{aabb['mae']:.6f}",
            "aabb_rmse_mm": f"{aabb['rmse']:.6f}",
            "aabb_median_error_mm": f"{aabb['median']:.6f}",
            "aabb_max_error_mm": f"{aabb['max']:.6f}",
            "aabb_mape_percent": f"{aabb['mape']:.6f}",
            "principal_mae_mm": f"{principal['mae']:.6f}",
            "principal_rmse_mm": f"{principal['rmse']:.6f}",
            "principal_median_error_mm": f"{principal['median']:.6f}",
            "principal_max_error_mm": f"{principal['max']:.6f}",
            "principal_mape_percent": f"{principal['mape']:.6f}",
            "principal_within_0_1mm": threshold_count(principal["absolute"], 0.1),
            "principal_within_0_5mm": threshold_count(principal["absolute"], 0.5),
            "principal_within_1mm": threshold_count(principal["absolute"], 1.0),
            "principal_within_2mm": threshold_count(principal["absolute"], 2.0),
            "principal_within_5mm": threshold_count(principal["absolute"], 5.0),
            "aabb_within_0_1mm": threshold_count(aabb["absolute"], 0.1),
            "aabb_within_0_5mm": threshold_count(aabb["absolute"], 0.5),
            "aabb_within_1mm": threshold_count(aabb["absolute"], 1.0),
            "aabb_within_2mm": threshold_count(aabb["absolute"], 2.0),
            "aabb_within_5mm": threshold_count(aabb["absolute"], 5.0),
        })
    fields = [
        "specimen_id", "job_id", "known_weight_ct", "active_validation_sample",
        "reconstruction_provenance", "provenance_evidence",
        "included_in_dense_aggregate", "dense_aggregate_exclusion_reason",
        "scale_source",
        "physical_ranked_mm", "production_aabb_ranked_mm", "principal_oriented_ranked_mm",
        "aabb_mae_mm", "aabb_rmse_mm", "aabb_median_error_mm", "aabb_max_error_mm", "aabb_mape_percent",
        "principal_mae_mm", "principal_rmse_mm", "principal_median_error_mm", "principal_max_error_mm", "principal_mape_percent",
        "principal_within_0_1mm", "principal_within_0_5mm", "principal_within_1mm", "principal_within_2mm", "principal_within_5mm",
        "aabb_within_0_1mm", "aabb_within_0_5mm", "aabb_within_1mm", "aabb_within_2mm", "aabb_within_5mm",
    ]
    write_csv(RESULTS_CSV, fields, rows)
    write_summary(rows, all_errors, all_pct)


def summarize_method(errors: list[float], percentages: list[float]) -> dict:
    if not errors:
        return {
            "dimension_count": 0,
            "mae_mm": None,
            "rmse_mm": None,
            "median_error_mm": None,
            "max_error_mm": None,
            "mape_percent": None,
        }
    arr = np.asarray(errors, dtype=float)
    pct = np.asarray(percentages, dtype=float)
    result = {
        "dimension_count": int(len(arr)),
        "mae_mm": float(np.mean(arr)),
        "rmse_mm": float(math.sqrt(np.mean(arr ** 2))),
        "median_error_mm": float(np.median(arr)),
        "max_error_mm": float(np.max(arr)),
        "mape_percent": float(np.mean(pct)),
    }
    for threshold in [0.1, 0.5, 1.0, 2.0, 5.0]:
        count = threshold_count(arr, threshold)
        label = str(threshold).replace(".", "_")
        result[f"within_{label}mm_count"] = count
        result[f"within_{label}mm_percent"] = 100.0 * count / max(len(arr), 1)
    return result


def write_summary(rows: list[dict], all_errors: dict[str, list[float]],
                  all_pct: dict[str, list[float]]) -> None:
    principal = summarize_method(all_errors["principal"], all_pct["principal"])
    aabb = summarize_method(all_errors["aabb"], all_pct["aabb"])
    dense_rows = [
        row for row in rows
        if row.get("included_in_dense_aggregate") == "yes"
    ]
    excluded_rows = [
        {
            "specimen_id": row.get("specimen_id", ""),
            "job_id": row.get("job_id", ""),
            "reconstruction_provenance": row.get("reconstruction_provenance", ""),
            "exclusion_reason": row.get("dense_aggregate_exclusion_reason", ""),
        }
        for row in rows
        if row.get("included_in_dense_aggregate") != "yes"
    ]
    summary = {
        "generated_utc": utc_now(),
        "specimens_requested": [item["specimen_id"] for item in active_specimens()],
        "preserved_excluded_specimens": [
            item["specimen_id"] for item in SPECIMENS
            if not item.get("active_validation", True)
        ],
        "replacement_metadata": replacement_metadata(),
        "retained_degraded_metadata": retained_degraded_metadata(),
        "supplemental_validation_metadata": supplemental_validation_metadata(),
        "methodology_metadata": METHODOLOGY_METADATA_WORDING,
        "completed_rows_evaluated": len(rows),
        "dense_aggregate_specimen_count": len(dense_rows),
        "dense_aggregate_dimension_count": principal["dimension_count"],
        "dense_specimens_included": [row["specimen_id"] for row in dense_rows],
        "degraded_or_excluded_specimens": excluded_rows,
        "dense_aggregate_inclusion_rule": (
            "Only active validation rows classified TRUE_DENSE from objective "
            "job artifacts are included in final dense accuracy statistics."
        ),
        "validation_results_csv": rel(RESULTS_CSV),
        "job_map_csv": rel(JOB_MAP_CSV),
        "principal_oriented": principal,
        "production_ranked_aabb": aabb,
        "proposal_target": {
            "metric": "principal-oriented absolute dimensional error",
            "target_mm": PROPOSAL_TARGET_MM,
            "achieved": (
                principal["dimension_count"] > 0
                and principal.get("within_0_1mm_count", 0) == principal["dimension_count"]
            ),
            "status": (
                "ACHIEVED"
                if principal["dimension_count"] > 0
                and principal.get("within_0_1mm_count", 0) == principal["dimension_count"]
                else "NOT_ACHIEVED"
            ),
        },
        "aabb_to_principal_mae_reduction_mm": (
            None if principal["mae_mm"] is None or aabb["mae_mm"] is None
            else aabb["mae_mm"] - principal["mae_mm"]
        ),
        "aabb_to_principal_rmse_reduction_mm": (
            None if principal["rmse_mm"] is None or aabb["rmse_mm"] is None
            else aabb["rmse_mm"] - principal["rmse_mm"]
        ),
        "claim_boundary": (
            "Physical length/width/height were used only for validation metrics. "
            "Weight was used only through the current production mass/density scale."
        ),
    }
    write_json(SUMMARY_JSON, summary)


def write_run_manifest(rows_by_specimen: dict[str, dict]) -> None:
    write_json(RUN_MANIFEST_JSON, {
        "generated_utc": utc_now(),
        "runner": rel(Path(__file__)),
        "production_execution_path": "POST /upload, then poll /jobs/{job_id}/status",
        "specimens_requested": [item["specimen_id"] for item in active_specimens()],
        "preserved_excluded_specimens": [
            item["specimen_id"] for item in SPECIMENS
            if not item.get("active_validation", True)
        ],
        "replacement_metadata": replacement_metadata(),
        "retained_degraded_metadata": retained_degraded_metadata(),
        "supplemental_validation_metadata": supplemental_validation_metadata(),
        "methodology_metadata": METHODOLOGY_METADATA_WORDING,
        "settings": {
            "scan_mode": "turntable",
            "cut_mode": "multi",
            "preferred_shape": "Auto",
            "known_weights_ct": {item["specimen_id"]: item["known_weight"] for item in active_specimens()},
            "preserved_known_weights_ct": {
                item["specimen_id"]: item["known_weight"] for item in SPECIMENS
                if not item.get("active_validation", True)
            },
        },
        "input_root": rel(INPUT_ROOT),
        "job_map_csv": rel(JOB_MAP_CSV),
        "validation_summary_json": rel(SUMMARY_JSON),
        "statuses": {
            specimen_id: row.get("status", "")
            for specimen_id, row in rows_by_specimen.items()
        },
        "failure_policy": "Failures and poor reconstructions remain in batch_job_map.csv and are not discarded.",
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--input-root", default=str(INPUT_ROOT))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-hours", type=float, default=6.0)
    parser.add_argument("--init-inputs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Regenerate validation CSV/summary from the existing job map without submitting jobs.",
    )
    parser.add_argument("--force-new", action="store_true")
    parser.add_argument("--only", choices=[item["specimen_id"] for item in SPECIMENS])
    parser.add_argument("--stop-after-submit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global INPUT_ROOT
    INPUT_ROOT = Path(args.input_root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.init_inputs:
        init_inputs()
    physical = physical_measurements()
    archive = "" if args.dry_run or args.validate_existing else archive_existing_failures()
    if args.validate_existing:
        rows_by_specimen = load_existing_map()
        validate_completed_jobs(physical)
    else:
        rows_by_specimen = submit_and_poll(args, physical)
    if not args.dry_run and not args.stop_after_submit and not args.validate_existing:
        validate_completed_jobs(physical)
    write_run_manifest(rows_by_specimen)
    print(json.dumps({
        "failure_archive": archive,
        "job_map_csv": rel(JOB_MAP_CSV),
        "run_manifest_json": rel(RUN_MANIFEST_JSON),
        "validation_summary_json": rel(SUMMARY_JSON) if SUMMARY_JSON.exists() else "",
        "statuses": {
            specimen_id: rows_by_specimen.get(specimen_id, {}).get("status", "")
            for specimen_id in [item["specimen_id"] for item in SPECIMENS]
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
