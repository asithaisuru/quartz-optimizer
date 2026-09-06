"""Prepare multi-stone reconstruction validation evidence.

This research-only helper does not change the reconstruction pipeline and does
not tune meshes from physical dimensions. It audits completed jobs, records
confident specimen mappings, and prepares a fixed stratified capture plan when
too few independent mapped reconstructions exist.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_RECON_DIR = REPO_ROOT / "final_research_evidence" / "reconstruction"
OUT_DIR = REPO_ROOT / "final_research_evidence" / "reconstruction_multi"
JOBS_DIR = REPO_ROOT / "jobs"

PHYSICAL_CSV = SOURCE_RECON_DIR / "physical_measurements.csv"
SOURCE_MAP_CSV = SOURCE_RECON_DIR / "specimen_reconstruction_map.csv"

INVENTORY_CSV = OUT_DIR / "existing_reconstruction_inventory.csv"
CONFIDENT_MAP_CSV = OUT_DIR / "confident_existing_mappings.csv"
SAMPLE_PLAN_CSV = OUT_DIR / "validation_sample_plan.csv"
CAPTURE_MD = OUT_DIR / "CAPTURE_AND_RECONSTRUCTION_WORKFLOW.md"
SUMMARY_JSON = OUT_DIR / "validation_readiness_manifest.json"
REPORT_MD = OUT_DIR / "RECONSTRUCTION_MULTI_READINESS.md"

MINIMUM_CONFIDENT_EXISTING = 5
SELECTED_NEW_CAPTURE_IDS = [
    "QZ-30",
    "QZ-24",
    "QZ-19",
    "QZ-15",
    "QZ-14",
    "QZ-03",
    "QZ-09",
    "QZ-06",
    "QZ-05",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


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


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def physical_rows() -> dict[str, dict]:
    rows = {}
    for row in read_csv(PHYSICAL_CSV):
        row["physical_weight"] = float(row["physical_weight"])
        row["physical_length_mm"] = float(row["physical_length_mm"])
        row["physical_width_mm"] = float(row["physical_width_mm"])
        row["physical_height_mm"] = float(row["physical_height_mm"])
        rows[row["specimen_id"]] = row
    return rows


def weight_matches(weight: float | None, physical: dict[str, dict]) -> list[str]:
    if weight is None:
        return []
    return [
        specimen_id
        for specimen_id, row in physical.items()
        if abs(float(row["physical_weight"]) - float(weight)) < 0.005
    ]


def completed_job_inventory(physical: dict[str, dict]) -> list[dict]:
    rows = []
    for report_path in sorted(JOBS_DIR.glob("*/analysis_report.json")):
        job_dir = report_path.parent
        mesh_path = job_dir / "dense" / "final_textured_model.ply"
        if not mesh_path.exists():
            continue
        report = read_json(report_path)
        raw_weight = report.get("raw_carats")
        try:
            raw_weight = float(raw_weight)
        except (TypeError, ValueError):
            raw_weight = None
        image_dir = job_dir / "images"
        video_files = sorted([
            p.name
            for p in job_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".mov", ".mp4", ".avi"}
        ])
        rows.append({
            "job_id": job_dir.name,
            "analysis_report_path": rel(report_path),
            "mesh_path": rel(mesh_path),
            "raw_carats": raw_weight if raw_weight is not None else "",
            "exact_weight_matches": ";".join(weight_matches(raw_weight, physical)),
            "reported_dimensions_mm": " x ".join(
                str(v) for v in report.get("rough_dimensions_mm", [])
            ),
            "image_frame_count": len(list(image_dir.glob("*"))) if image_dir.exists() else 0,
            "stored_video_files": "; ".join(video_files),
        })
    return rows


def confident_existing_mappings(physical: dict[str, dict]) -> list[dict]:
    rows = []
    for row in read_csv(SOURCE_MAP_CSV):
        specimen_id = row.get("specimen_id", "").strip()
        if not specimen_id or specimen_id not in physical:
            continue
        match_status = row.get("match_status", "")
        confidence = row.get("mapping_confidence", "")
        role = row.get("validation_role", "")
        mesh_path = REPO_ROOT / row.get("mesh_path", "")
        if "confirmed" not in match_status:
            continue
        if confidence != "high":
            continue
        if "secondary" in role:
            continue
        if not mesh_path.exists():
            continue
        try:
            input_weight = float(row.get("input_weight", ""))
        except ValueError:
            input_weight = None
        exact_weight = (
            input_weight is not None
            and abs(input_weight - float(physical[specimen_id]["physical_weight"])) < 0.005
        )
        rows.append({
            "specimen_id": specimen_id,
            "job_id": row.get("job_id", ""),
            "mesh_path": row.get("mesh_path", ""),
            "input_weight_ct": row.get("input_weight", ""),
            "physical_weight_ct": physical[specimen_id]["physical_weight"],
            "exact_weight_match": "yes" if exact_weight else "no",
            "mapping_basis": row.get("candidate_basis", ""),
            "visual_confirmation_status": row.get("visual_confirmation_status", ""),
            "validation_role": role,
        })
    return rows


def stratum_for_weight(weight: float) -> str:
    if weight <= 25.0:
        return "small"
    if weight <= 100.0:
        return "medium"
    return "large"


def sample_plan(physical: dict[str, dict], confident: list[dict]) -> list[dict]:
    confident_ids = {row["specimen_id"] for row in confident}
    rows = []
    for row in confident:
        specimen = physical[row["specimen_id"]]
        rows.append({
            "sample_role": "existing_confirmed_anchor",
            "capture_required": "no",
            "specimen_id": row["specimen_id"],
            "id_range": specimen.get("id_range", ""),
            "weight_stratum": stratum_for_weight(float(specimen["physical_weight"])),
            "physical_weight_ct": specimen["physical_weight"],
            "physical_length_mm": specimen["physical_length_mm"],
            "physical_width_mm": specimen["physical_width_mm"],
            "physical_height_mm": specimen["physical_height_mm"],
            "job_id": row["job_id"],
            "notes": "Existing high-confidence mapping retained; do not recapture unless repeatability is desired.",
        })
    for specimen_id in SELECTED_NEW_CAPTURE_IDS:
        if specimen_id in confident_ids:
            continue
        specimen = physical[specimen_id]
        rows.append({
            "sample_role": "new_capture",
            "capture_required": "yes",
            "specimen_id": specimen_id,
            "id_range": specimen.get("id_range", ""),
            "weight_stratum": stratum_for_weight(float(specimen["physical_weight"])),
            "physical_weight_ct": specimen["physical_weight"],
            "physical_length_mm": specimen["physical_length_mm"],
            "physical_width_mm": specimen["physical_width_mm"],
            "physical_height_mm": specimen["physical_height_mm"],
            "job_id": "",
            "notes": "Fixed stratified sample selected before reconstruction; retain even if reconstruction is poor.",
        })
    return rows


def write_capture_workflow(rows: list[dict]) -> None:
    new_rows = [row for row in rows if row["capture_required"] == "yes"]
    lines = [
        "# Multi-Stone Capture And Reconstruction Workflow",
        "",
        "Use the current production upload/reconstruction pipeline without code changes.",
        "Do not use physical length/width/height to correct, scale, reject, or tune a mesh.",
        "Use each stone's recorded carat weight only as the existing system's mass/density scale input.",
        "",
        "## Fixed New-Capture Stones",
        "",
        "| order | specimen | stratum | weight ct | recorded L x W x H mm |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for idx, row in enumerate(new_rows, start=1):
        dims = (
            f"{row['physical_length_mm']} x "
            f"{row['physical_width_mm']} x "
            f"{row['physical_height_mm']}"
        )
        lines.append(
            f"| {idx} | {row['specimen_id']} | {row['weight_stratum']} | "
            f"{row['physical_weight_ct']} | {dims} |"
        )
    lines.extend([
        "",
        "## Procedure",
        "",
        "1. Process stones one at a time in the listed order.",
        "2. Before filming, place a handwritten label with the specimen ID next to the stone and take one reference photo; keep this photo with the capture files.",
        "3. Capture four overlapping videos per stone, matching the current app expectation that videos upload as `video_0` through `video_3`.",
        "4. Use the same turntable-style acquisition already used by the current pipeline: full 360-degree coverage, steady focus/exposure, visible stone boundary, and no crop that hides the outline.",
        "5. Upload the four videos together through the current backend with `scan_mode=turntable`, `cut_mode=multi`, `preferred_shape=Auto`, and `known_weight` equal to the recorded `physical_weight_ct` for that specimen.",
        "6. Confirm the resulting contact sheet/source media against the labelled reference photo before adding the job to the validation map.",
        "7. If a reconstruction is poor, keep it in the mapped sample and record the failure; do not discard or rerun selectively unless the reason is capture/file corruption.",
        "8. After at least five independent new jobs complete, add only visually confirmed mappings to the reconstruction-multi map and run the existing rotation-independent dimensional analyzer.",
        "",
        "Expected frame extraction is the current production behavior: approximately 40 sharp frames per uploaded video, with failure if fewer than 10 frames are extracted overall.",
        "",
    ])
    CAPTURE_MD.write_text("\n".join(lines), encoding="utf-8")


def write_report(manifest: dict, mappings: list[dict], rows: list[dict]) -> None:
    new_rows = [row for row in rows if row["capture_required"] == "yes"]
    lines = [
        "# Multi-Stone Reconstruction Validation Readiness",
        "",
        f"Confidently mapped existing independent reconstructions: {len(mappings)}.",
        f"Immediate multi-stone validation threshold met: {manifest['immediate_validation_ready']}.",
        "",
        "The current repository has too few confidently mapped independent completed reconstructions, so no new multi-stone accuracy claim was calculated.",
        "",
        "## Existing Confident Mappings",
        "",
        "| specimen | job | weight match | basis |",
        "| --- | --- | --- | --- |",
    ]
    for row in mappings:
        lines.append(
            f"| {row['specimen_id']} | {row['job_id']} | "
            f"{row['exact_weight_match']} | {row['mapping_basis']} |"
        )
    lines.extend([
        "",
        "## New Capture Selection",
        "",
        "| specimen | stratum | weight ct | L x W x H mm |",
        "| --- | --- | ---: | --- |",
    ])
    for row in new_rows:
        dims = (
            f"{row['physical_length_mm']} x "
            f"{row['physical_width_mm']} x "
            f"{row['physical_height_mm']}"
        )
        lines.append(
            f"| {row['specimen_id']} | {row['weight_stratum']} | "
            f"{row['physical_weight_ct']} | {dims} |"
        )
    lines.extend([
        "",
        "## Files",
        "",
        f"- `{rel(INVENTORY_CSV)}`",
        f"- `{rel(CONFIDENT_MAP_CSV)}`",
        f"- `{rel(SAMPLE_PLAN_CSV)}`",
        f"- `{rel(CAPTURE_MD)}`",
        f"- `{rel(SUMMARY_JSON)}`",
        "",
    ])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    physical = physical_rows()
    inventory = completed_job_inventory(physical)
    mappings = confident_existing_mappings(physical)
    sample_rows = sample_plan(physical, mappings)
    new_count = sum(1 for row in sample_rows if row["capture_required"] == "yes")

    write_csv(INVENTORY_CSV, [
        "job_id",
        "analysis_report_path",
        "mesh_path",
        "raw_carats",
        "exact_weight_matches",
        "reported_dimensions_mm",
        "image_frame_count",
        "stored_video_files",
    ], inventory)
    write_csv(CONFIDENT_MAP_CSV, [
        "specimen_id",
        "job_id",
        "mesh_path",
        "input_weight_ct",
        "physical_weight_ct",
        "exact_weight_match",
        "mapping_basis",
        "visual_confirmation_status",
        "validation_role",
    ], mappings)
    write_csv(SAMPLE_PLAN_CSV, [
        "sample_role",
        "capture_required",
        "specimen_id",
        "id_range",
        "weight_stratum",
        "physical_weight_ct",
        "physical_length_mm",
        "physical_width_mm",
        "physical_height_mm",
        "job_id",
        "notes",
    ], sample_rows)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_physical_measurements_csv": rel(PHYSICAL_CSV),
        "source_existing_map_csv": rel(SOURCE_MAP_CSV),
        "output_directory": rel(OUT_DIR),
        "physical_table_rows": len(physical),
        "completed_reconstruction_jobs_with_mesh": len(inventory),
        "confident_existing_independent_mappings": len(mappings),
        "minimum_confident_existing_required_for_immediate_validation": MINIMUM_CONFIDENT_EXISTING,
        "immediate_validation_ready": len(mappings) >= MINIMUM_CONFIDENT_EXISTING,
        "validation_run": "not_run_too_few_confident_existing_mappings",
        "new_capture_count": new_count,
        "total_planned_independent_sample_with_existing_anchor": len(sample_rows),
        "selected_new_capture_specimens": [
            row["specimen_id"]
            for row in sample_rows
            if row["capture_required"] == "yes"
        ],
        "method_constraints": [
            "production reconstruction code unchanged",
            "physical length/width/height are held out for validation only",
            "weight may be used only as the existing mass/density scale input",
            "poor reconstructions must remain in the validation set",
        ],
        "outputs": {
            "inventory_csv": rel(INVENTORY_CSV),
            "confident_mappings_csv": rel(CONFIDENT_MAP_CSV),
            "sample_plan_csv": rel(SAMPLE_PLAN_CSV),
            "capture_workflow_md": rel(CAPTURE_MD),
            "report_md": rel(REPORT_MD),
        },
    }
    SUMMARY_JSON.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_capture_workflow(sample_rows)
    write_report(manifest, mappings, sample_rows)

    print(json.dumps({
        "confident_existing_independent_mappings": len(mappings),
        "immediate_validation_ready": manifest["immediate_validation_ready"],
        "selected_new_capture_specimens": manifest["selected_new_capture_specimens"],
        "outputs": manifest["outputs"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
