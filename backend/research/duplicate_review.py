"""Generate human-review material for near-duplicate image candidates."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .common import atomic_write_json, normalize_path, sha256_file, utc_timestamp
from .readiness_common import (
    atomic_write_csv,
    atomic_write_text,
    find_repo_root,
    resolve_path,
    write_manifest,
)

ALLOWED_DECISIONS = {
    "same_source",
    "same_specimen",
    "different_specimen",
    "augmentation_related",
    "not_duplicate",
    "uncertain",
    "unreviewed",
}
GROUPING_DECISIONS = {
    "same_source",
    "same_specimen",
    "augmentation_related",
}
REVIEW_COLUMNS = (
    "candidate_group_id",
    "image_a",
    "image_b",
    "hash_a",
    "hash_b",
    "perceptual_distance",
    "review_decision",
    "reviewer_id",
    "review_date",
    "reason",
    "notes",
)


def _candidate_id(left: str, right: str) -> str:
    pair = "\0".join(sorted((left, right))).encode("utf-8")
    return "near-" + hashlib.sha256(pair).hexdigest()[:12]


def load_review_rows(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    review_path = Path(path)
    if not review_path.exists():
        return {}
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            candidate_id = str(row.get("candidate_group_id", "")).strip()
            if candidate_id:
                rows[candidate_id] = {
                    key: str(value or "").strip() for key, value in row.items()
                }
        return rows


def grouping_pairs(review_rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return sorted(
        (
            str(row["image_a"]),
            str(row["image_b"]),
        )
        for row in review_rows
        if row.get("review_decision") in GROUPING_DECISIONS
    )


def build_review_rows(
    candidates: list[dict[str, Any]],
    repo_root: str | Path,
    existing_review: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    root = Path(repo_root).resolve()
    existing = load_review_rows(existing_review)
    rows = []
    blockers = []
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            min(str(item.get("left", "")), str(item.get("right", ""))),
            max(str(item.get("left", "")), str(item.get("right", ""))),
            int(item.get("hamming_distance", 0)),
        ),
    )
    for candidate in sorted_candidates:
        left = str(candidate.get("left", ""))
        right = str(candidate.get("right", ""))
        candidate_id = _candidate_id(left, right)
        prior = existing.get(candidate_id, {})
        decision = prior.get("review_decision", "unreviewed") or "unreviewed"
        if decision not in ALLOWED_DECISIONS:
            blockers.append(
                f"{candidate_id} has unsupported review decision: {decision}"
            )
            decision = "unreviewed"
        left_path = resolve_path(left, root)
        right_path = resolve_path(right, root)
        if left_path is None or not left_path.exists():
            blockers.append(f"Candidate image is missing: {left}")
        if right_path is None or not right_path.exists():
            blockers.append(f"Candidate image is missing: {right}")
        rows.append(
            {
                "candidate_group_id": candidate_id,
                "image_a": left,
                "image_b": right,
                "hash_a": (
                    sha256_file(left_path)
                    if left_path is not None and left_path.exists()
                    else ""
                ),
                "hash_b": (
                    sha256_file(right_path)
                    if right_path is not None and right_path.exists()
                    else ""
                ),
                "perceptual_distance": int(
                    candidate.get("hamming_distance", 0)
                ),
                "review_decision": decision,
                "reviewer_id": prior.get("reviewer_id", ""),
                "review_date": prior.get("review_date", ""),
                "reason": prior.get("reason", ""),
                "notes": prior.get("notes", ""),
            }
        )
    return rows, blockers


def _thumbnail(source: Path, destination: Path, maximum: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
        image.convert("RGB").save(destination, format="JPEG", quality=82)


def _contact_sheet(
    rows: list[dict[str, Any]],
    root: Path,
    output: Path,
    thumbnail_directory: Path,
    thumbnail_size: int,
) -> tuple[str, list[str]]:
    warnings = []
    cards = []
    for index, row in enumerate(rows):
        thumbnails = []
        for side, field in (("a", "image_a"), ("b", "image_b")):
            source = resolve_path(row[field], root)
            filename = f"{row['candidate_group_id']}-{side}.jpg"
            destination = thumbnail_directory / filename
            if source is None or not source.exists():
                thumbnails.append("<div class=\"missing\">missing</div>")
                continue
            try:
                _thumbnail(source, destination, thumbnail_size)
                relative = destination.relative_to(output).as_posix()
                thumbnails.append(
                    f'<img src="{html.escape(relative)}" '
                    f'alt="{html.escape(row[field])}">'
                )
            except (OSError, UnidentifiedImageError) as exc:
                warnings.append(f"Thumbnail failed for {row[field]}: {exc}")
                thumbnails.append("<div class=\"missing\">unavailable</div>")
        cards.append(
            "<article>"
            f"<h2>{html.escape(row['candidate_group_id'])}</h2>"
            f"<div class=\"images\">{''.join(thumbnails)}</div>"
            f"<p>A: {html.escape(row['image_a'])}</p>"
            f"<p>B: {html.escape(row['image_b'])}</p>"
            f"<p>Perceptual distance: {row['perceptual_distance']}</p>"
            f"<p>Decision: {html.escape(row['review_decision'])}</p>"
            "</article>"
        )
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Near-Duplicate Candidate Review</title>
<style>
body{font-family:Arial,sans-serif;margin:24px;color:#17202a}
.notice{padding:12px;border-left:4px solid #b9770e;background:#fef9e7}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
article{border:1px solid #ccd1d1;padding:12px}
h2{font-size:16px}.images{display:grid;grid-template-columns:1fr 1fr;gap:8px}
img{width:100%;height:180px;object-fit:contain;background:#111}
p{font-size:12px;overflow-wrap:anywhere}.missing{height:180px;background:#eee;display:grid;place-items:center}
</style></head><body>
<h1>Near-Duplicate Candidate Review</h1>
<p class="notice">These pairs are algorithmic candidates, not confirmed duplicates.
Review decisions must be supplied by a human. Original images are not modified.</p>
<main>""" + "".join(cards) + "</main></body></html>\n"
    return page, warnings


def generate_duplicate_review(
    *,
    dataset_audit_path: str | Path,
    output_directory: str | Path,
    existing_review: str | Path | None = None,
    strict: bool = False,
    thumbnail_size: int = 320,
    command: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    audit_path = Path(dataset_audit_path).resolve()
    repo_root = find_repo_root(audit_path)
    output = resolve_path(output_directory, repo_root)
    if output is None:
        raise ValueError("An output directory is required.")
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    candidates = audit.get("near_duplicate_candidates", [])
    review_path = resolve_path(existing_review, repo_root)
    rows, blockers = build_review_rows(candidates, repo_root, review_path)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "report": output / "duplicate_review.json",
        "review": output / "near_duplicate_review.csv",
        "contact_sheet": output / "near_duplicate_contact_sheet.html",
        "thumbnails": output / "near_duplicate_thumbnails",
        "manifest": output / "manifest.json",
    }
    atomic_write_csv(files["review"], rows, REVIEW_COLUMNS)
    page, thumbnail_warnings = _contact_sheet(
        rows,
        repo_root,
        output,
        files["thumbnails"],
        max(64, int(thumbnail_size)),
    )
    atomic_write_text(files["contact_sheet"], page)
    unresolved = sum(
        row["review_decision"] in {"unreviewed", "uncertain"} for row in rows
    )
    report = {
        "schema_version": "1.0",
        "audit_type": "near_duplicate_human_review_readiness",
        "created_utc": utc_timestamp(),
        "candidate_count": len(rows),
        "reviewed_count": len(rows) - unresolved,
        "unresolved_count": unresolved,
        "grouping_pair_count": len(grouping_pairs(rows)),
        "strict": bool(strict),
        "strict_blocked": bool(strict and (unresolved or blockers)),
        "conservative_behavior": (
            "Unreviewed and uncertain candidates block final split approval. "
            "They are not automatically deleted, merged, or treated as specimens."
        ),
        "blockers": list(dict.fromkeys(blockers + (
            [f"{unresolved} near-duplicate candidates remain unresolved."]
            if unresolved else []
        ))),
        "warnings": thumbnail_warnings,
        "original_images_modified": False,
    }
    atomic_write_json(files["report"], report)
    write_manifest(
        path=files["manifest"],
        repo_root=repo_root,
        tool_name="quartz_near_duplicate_review_package",
        command=command,
        tool_sources=(
            "backend/research/duplicate_review.py",
            "backend/research/readiness_common.py",
            "backend/research_benchmark.py",
        ),
        inputs=(
            ("dataset_audit", audit_path),
            ("existing_review", review_path),
        ),
        outputs=(
            ("report", files["report"]),
            ("review", files["review"]),
            ("contact_sheet", files["contact_sheet"]),
        ),
        extra={
            "thumbnail_count": (
                len(list(files["thumbnails"].glob("*.jpg")))
                if files["thumbnails"].exists()
                else 0
            )
        },
    )
    return report, files
