"""Build a reproducibility record for the frozen Quartz research evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "final_research_evidence" / "final_academic_validation"
REPRODUCIBILITY_RECORD_MD = OUTPUT_DIR / "reproducibility_record.md"
FINAL_SUMMARY_JSON = OUTPUT_DIR / "final_academic_validation_summary.json"

UNCERTAINTY_SUMMARY_JSON = OUTPUT_DIR / "uncertainty_summary.json"
UNCERTAINTY_RESULTS_CSV = OUTPUT_DIR / "uncertainty_results.csv"
FAILURE_ANALYSIS_MD = OUTPUT_DIR / "failure_analysis.md"
THREATS_TO_VALIDITY_MD = OUTPUT_DIR / "threats_to_validity.md"

DEFECT_RESULTS_JSON = (
    ROOT
    / "final_research_evidence"
    / "defect_detection"
    / "final_test"
    / "independent_final_test_results"
    / "independent_final_test_results.json"
)
RECON_SUMMARY_JSON = (
    ROOT
    / "final_research_evidence"
    / "reconstruction_multi"
    / "batch_validation_summary.json"
)
OPTIMIZER_SUMMARY_JSON = (
    ROOT
    / "final_research_evidence"
    / "optimizer_validation"
    / "system_yield_summary.json"
)
FACET_METRICS_JSON = (
    ROOT / "backend" / "research" / "facet_ml" / "facet_orientation_metrics.json"
)
FACET_MANIFEST_JSON = (
    ROOT / "backend" / "research" / "facet_ml" / "facet_orientation_manifest.json"
)

REPOSITORY_URL = "https://github.com/asithaisuru/quartz-optimizer.git"
BRANCH = "dev"
ORIGINAL_THESIS_COMMIT = "2de0f5adc3c3a40923a4b5ac75d765ba8181c628"
ORIGINAL_THESIS_TAG = "thesis-submission-2026-08-30"
FROZEN_MODEL_SHA256 = (
    "5b7a362ed4c6670395e4693ab1a68d415ff1fb2164b2471cef37913338130b98"
)
DEFECT_TEST_MEMBERSHIP_SHA256 = (
    "4633d2e1fd7ce7772d06d8994c084e52f17b8cce1647587693d1558bc050e8cf"
)
FROZEN_LABEL_MANIFEST_SHA256 = (
    "40d3d1cc061924c25192abaddffd6297353dc7afd0bd7e18f8de798436231e07"
)

COMMIT_CATEGORIES = {
    "COLMAP fixes": ("COLMAP", "PatchMatch", "CUDA"),
    "reconstruction validation": ("reconstruction validation",),
    "defect validation": ("independent defect validation",),
    "optimizer/expert validation": (
        "optimizer and expert validation",
        "post-system expert review",
    ),
    "post-validation optimizer": ("manufacturing-complete multi-gem",),
    "capture-quality gate": ("capture quality gate",),
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def current_git_state() -> dict[str, Any]:
    return {
        "head": run_git("rev-parse", "HEAD"),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "tag_commit": run_git("rev-list", "-n", "1", ORIGINAL_THESIS_TAG),
        "original_thesis_tag": ORIGINAL_THESIS_TAG,
        "original_thesis_commit": ORIGINAL_THESIS_COMMIT,
        "original_thesis_tag_matches_expected_commit": (
            run_git("rev-list", "-n", "1", ORIGINAL_THESIS_TAG)
            == ORIGINAL_THESIS_COMMIT
        ),
        "status_porcelain": run_git("status", "--porcelain", "--untracked-files=all"),
    }


def matching_commits(patterns: tuple[str, ...]) -> list[dict[str, str]]:
    raw = run_git(
        "log",
        "--all",
        "--format=%H%x09%s",
        "--regexp-ignore-case",
        *[f"--grep={pattern}" for pattern in patterns],
    )
    commits = []
    seen = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        commit_hash, subject = line.split("\t", 1)
        if commit_hash in seen:
            continue
        seen.add(commit_hash)
        commits.append({"commit": commit_hash, "subject": subject})
    return commits


def relevant_commits() -> dict[str, list[dict[str, str]]]:
    return {
        category: matching_commits(patterns)
        for category, patterns in COMMIT_CATEGORIES.items()
    }


def colmap_default_path() -> str:
    source = ROOT / "backend" / "colmap_runner.py"
    text = source.read_text(encoding="utf-8")
    match = re.search(r'DEFAULT_COLMAP_BIN\s*=\s*r"([^"]+)"', text)
    return match.group(1) if match else "unknown"


def colmap_version(path_text: str) -> dict[str, str]:
    path = Path(path_text)
    if not path.is_file():
        return {"path": path_text, "version": "unknown", "status": "path_not_found"}
    result = subprocess.run(
        [str(path), "-h"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    text = result.stdout + "\n" + result.stderr
    version = next(
        (line.strip() for line in text.splitlines() if line.strip().startswith("COLMAP ")),
        "unknown",
    )
    return {
        "path": path_text,
        "version": version,
        "status": "available" if result.returncode == 0 else "help_failed",
    }


def output_file_status() -> dict[str, dict[str, Any]]:
    paths = {
        "uncertainty_summary": UNCERTAINTY_SUMMARY_JSON,
        "uncertainty_results": UNCERTAINTY_RESULTS_CSV,
        "failure_analysis": FAILURE_ANALYSIS_MD,
        "threats_to_validity": THREATS_TO_VALIDITY_MD,
        "reproducibility_record": REPRODUCIBILITY_RECORD_MD,
        "final_academic_validation_summary": FINAL_SUMMARY_JSON,
    }
    return {
        name: {"path": rel(path), "exists": path.exists()}
        for name, path in paths.items()
    }


def load_core_evidence() -> dict[str, Any]:
    defect = read_json(DEFECT_RESULTS_JSON)
    return {
        "reconstruction": read_json(RECON_SUMMARY_JSON),
        "defect": defect,
        "facet_metrics": read_json(FACET_METRICS_JSON),
        "facet_manifest": read_json(FACET_MANIFEST_JSON),
        "optimizer": read_json(OPTIMIZER_SUMMARY_JSON),
        "defect_configuration": defect.get("configuration", {}),
        "defect_device_selection": defect.get("device_selection", {}),
    }


def command_block() -> dict[str, str]:
    return {
        "reconstruction_validation_only_regeneration": (
            "conda run -n quartz python backend\\research\\reconstruction_validation\\"
            "run_batch_validation.py --validate-existing"
        ),
        "defect_check_only_verification": (
            "conda run -n quartz python backend\\research\\defect_validation\\"
            "evaluate_independent_final_test.py --audit-result"
        ),
        "form_a_validation_after_real_responses": (
            "conda run -n quartz python backend\\research\\expert_validation\\"
            "validate_expert_responses.py final_research_evidence\\expert_validation\\"
            "expert_responses.csv"
        ),
        "objective3_comparison_after_real_form_a": (
            "conda run -n quartz python backend\\research\\expert_validation\\"
            "compare_system_vs_expert.py final_research_evidence\\expert_validation\\"
            "expert_responses.csv"
        ),
        "form_b_linkage_validation_after_post_system_review": (
            "conda run -n quartz python backend\\research\\expert_validation\\"
            "validate_form_b_responses.py final_research_evidence\\expert_validation\\"
            "form_b_responses.csv --form-a-responses final_research_evidence\\"
            "expert_validation\\expert_responses.csv"
        ),
        "academic_uncertainty_regeneration": (
            "conda run -n quartz python backend\\research\\final_academic_validation\\"
            "bootstrap_uncertainty.py"
        ),
        "academic_failure_analysis_regeneration": (
            "conda run -n quartz python backend\\research\\final_academic_validation\\"
            "build_failure_analysis.py"
        ),
        "academic_reproducibility_record_regeneration": (
            "conda run -n quartz python backend\\research\\final_academic_validation\\"
            "build_reproducibility_record.py"
        ),
    }


def reproducibility_markdown(record: dict[str, Any]) -> str:
    evidence = record["core_evidence"]
    defect_config = evidence["defect_configuration"]
    optimizer = evidence["optimizer"]
    facet_manifest = evidence["facet_manifest"]
    lines = [
        "# Reproducibility Record",
        "",
        "This record describes frozen research evidence and later supplemental "
        "academic-validation outputs. It does not move tags, rerun experiments, "
        "or alter frozen evidence.",
        "",
        "## Repository",
        "",
        f"- Repository: {REPOSITORY_URL}",
        f"- Branch: {BRANCH}",
        f"- Current HEAD when record was built: `{record['git']['head']}`",
        f"- Original thesis snapshot commit: `{ORIGINAL_THESIS_COMMIT}`",
        f"- Original thesis snapshot tag: `{ORIGINAL_THESIS_TAG}`",
        "- Tag rule: NEVER MOVE THIS TAG.",
        f"- Tag resolves to expected commit: "
        f"{record['git']['original_thesis_tag_matches_expected_commit']}",
        "",
        "## Relevant Later Commits",
        "",
    ]
    for category, commits in record["relevant_commits"].items():
        lines.append(f"### {category}")
        if not commits:
            lines.append("- No matching commit found by the reproducibility resolver.")
        for item in commits:
            lines.append(f"- `{item['commit']}` {item['subject']}")
        lines.append("")

    lines.extend(
        [
            "## Environment",
            "",
            f"- Python: {record['environment']['python']}",
            f"- Conda environment: {record['environment']['conda_environment']}",
            f"- Platform: {record['environment']['platform']}",
            f"- COLMAP path: `{record['environment']['colmap']['path']}`",
            f"- COLMAP version/status: {record['environment']['colmap']['version']} "
            f"({record['environment']['colmap']['status']})",
            "",
            "## Frozen Defect Evidence",
            "",
            f"- Frozen model SHA256: `{FROZEN_MODEL_SHA256}`",
            f"- Defect test membership SHA256: `{DEFECT_TEST_MEMBERSHIP_SHA256}`",
            f"- Frozen label manifest SHA256: `{FROZEN_LABEL_MANIFEST_SHA256}`",
            f"- Inference image size: {defect_config.get('inference_image_size')}",
            f"- Binary confidence threshold: "
            f"{defect_config.get('binary_confidence_threshold')}",
            f"- Prediction NMS IoU threshold: "
            f"{defect_config.get('prediction_nms_iou_threshold')}",
            f"- Native validation IoU: "
            f"{defect_config.get('native_ultralytics_validation_iou')}",
            f"- Final-test threshold tuning allowed: "
            f"{defect_config.get('final_test_threshold_tuning_allowed')}",
            "",
            "## Frozen Optimizer Verification Settings",
            "",
            f"- Eligible specimens: {', '.join(optimizer.get('eligible_specimens', []))}",
            f"- Result rows: {optimizer.get('result_row_count')}",
            "- Manufacturing-complete rows require "
            "`manufacturing_cut_sequence_complete=YES` and exact cut-sequence "
            "status in the frozen row.",
            "- Diagnostic rows remain diagnostic and are not promoted into the "
            "primary Objective-3 claim.",
            f"- Proposal target status: {optimizer.get('proposal_target', {}).get('status')}",
            "",
            "## Facet ML Evidence",
            "",
            f"- Dataset rows: {facet_manifest.get('dataset', {}).get('row_count')}",
            f"- Scenario groups: "
            f"{facet_manifest.get('dataset', {}).get('scenario_group_count')}",
            "- Target: simulation-derived geometric/ray inclusion-visibility "
            "surrogate.",
            "- Real-world expert/market-quality facet validation: not established.",
            "",
            "## Commands",
            "",
        ]
    )
    for name, command in record["commands"].items():
        lines.append(f"- {name}: `{command}`")
    lines.extend(
        [
            "",
            "## Evidence Paths",
            "",
        ]
    )
    for name, path in record["evidence_paths"].items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            "",
            "## Evidence Categories",
            "",
            "### Frozen Research Evidence",
            "",
            "- Reconstruction TRUE_DENSE metrics and SPARSE_FALLBACK exclusions.",
            "- Independent defect final-test membership, labels, model hash, and "
            "one-shot result.",
            "- Simulation-bound facet ML metrics.",
            "- System-side optimizer verification rows and diagnostic boundaries.",
            "",
            "### Post-Validation System Improvements",
            "",
            "- COLMAP environment fixes.",
            "- Capture-quality gate.",
            "- Manufacturing-complete multi-gem verification tooling.",
            "- These improvements do not revise frozen research metrics.",
            "",
            "### Pending External Expert Evidence",
            "",
            "- Objective-3 independent expert/traditional comparison remains pending.",
            "- Form A is the independent pre-system baseline.",
            "- Form B is supplemental post-system review only.",
            "",
        ]
    )
    return "\n".join(lines)


def build_record(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    global OUTPUT_DIR, REPRODUCIBILITY_RECORD_MD, FINAL_SUMMARY_JSON
    global UNCERTAINTY_SUMMARY_JSON, UNCERTAINTY_RESULTS_CSV
    global FAILURE_ANALYSIS_MD, THREATS_TO_VALIDITY_MD
    OUTPUT_DIR = output_dir
    REPRODUCIBILITY_RECORD_MD = OUTPUT_DIR / "reproducibility_record.md"
    FINAL_SUMMARY_JSON = OUTPUT_DIR / "final_academic_validation_summary.json"
    UNCERTAINTY_SUMMARY_JSON = OUTPUT_DIR / "uncertainty_summary.json"
    UNCERTAINTY_RESULTS_CSV = OUTPUT_DIR / "uncertainty_results.csv"
    FAILURE_ANALYSIS_MD = OUTPUT_DIR / "failure_analysis.md"
    THREATS_TO_VALIDITY_MD = OUTPUT_DIR / "threats_to_validity.md"

    colmap_path = colmap_default_path()
    record = {
        "schema_version": "1.0",
        "report_type": "final_academic_validation_reproducibility_record",
        "frozen_evidence_only": True,
        "repository": REPOSITORY_URL,
        "branch": BRANCH,
        "git": current_git_state(),
        "relevant_commits": relevant_commits(),
        "environment": {
            "python": sys.version.replace("\n", " "),
            "python_version": platform.python_version(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "platform": platform.platform(),
            "colmap": colmap_version(colmap_path),
        },
        "immutable_references": {
            "original_thesis_commit": ORIGINAL_THESIS_COMMIT,
            "original_thesis_tag": ORIGINAL_THESIS_TAG,
            "never_move_original_thesis_tag": True,
            "frozen_model_sha256": FROZEN_MODEL_SHA256,
            "defect_test_membership_sha256": DEFECT_TEST_MEMBERSHIP_SHA256,
            "frozen_label_manifest_sha256": FROZEN_LABEL_MANIFEST_SHA256,
        },
        "core_evidence": load_core_evidence(),
        "commands": command_block(),
        "evidence_paths": {
            "reconstruction_summary": rel(RECON_SUMMARY_JSON),
            "defect_final_test_results": rel(DEFECT_RESULTS_JSON),
            "facet_metrics": rel(FACET_METRICS_JSON),
            "facet_manifest": rel(FACET_MANIFEST_JSON),
            "optimizer_summary": rel(OPTIMIZER_SUMMARY_JSON),
            "uncertainty_summary": rel(UNCERTAINTY_SUMMARY_JSON),
            "uncertainty_results": rel(UNCERTAINTY_RESULTS_CSV),
            "failure_analysis": rel(FAILURE_ANALYSIS_MD),
            "threats_to_validity": rel(THREATS_TO_VALIDITY_MD),
            "reproducibility_record": rel(REPRODUCIBILITY_RECORD_MD),
        },
        "output_file_status": output_file_status(),
        "scientific_boundaries": {
            "frozen_research_evidence_unchanged": True,
            "post_validation_improvements_are_separate": True,
            "objective3_expert_comparison_pending": True,
            "form_a_independent_pre_system_baseline": True,
            "form_b_supplemental_post_system_only": True,
        },
    }
    write_text(REPRODUCIBILITY_RECORD_MD, reproducibility_markdown(record))
    record["output_file_status"] = output_file_status()
    record["output_file_status"]["final_academic_validation_summary"]["exists"] = True
    write_json(FINAL_SUMMARY_JSON, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    record = build_record(args.output_dir)
    print(
        json.dumps(
            {
                "status": "SAFE",
                "head": record["git"]["head"],
                "tag_matches_expected_commit": record["git"][
                    "original_thesis_tag_matches_expected_commit"
                ],
                "outputs": {
                    "reproducibility_record": rel(REPRODUCIBILITY_RECORD_MD),
                    "final_summary": rel(FINAL_SUMMARY_JSON),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
