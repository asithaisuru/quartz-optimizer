import argparse
import json
import os
import sys
from statistics import mean

from research.dataset_audit import load_audit_config, run_dataset_audit
from research.candidate_dataset import (
    load_candidate_config,
    materialize_development_split,
    run_candidate_audit,
    run_development_split_plan,
    run_materialized_dataset_audit,
)
from research.defect_policy_audit import (
    load_defect_audit_config,
    run_defect_policy_audit,
)
from research.defect_evaluation import (
    load_evaluation_config,
    run_evaluation_preflight,
)
from research.duplicate_review import generate_duplicate_review
from research.environment_check import (
    load_environment_config,
    run_environment_check,
)
from research.external_test_readiness import (
    load_external_test_readiness_config,
    run_external_test_readiness,
)
from research.grouped_split import (
    load_grouped_split_config,
    materialize_grouped_split,
    run_grouped_split_plan,
)
from research.gate2 import (
    approval_paths,
    load_gate2_config,
    run_gate2_approval_validation,
)
from research.model_provenance import run_model_provenance
from research.metric_approval import run_metric_approval
from research.phase3b_gate import (
    load_phase3b_gate_config,
    run_phase3b_gate,
)
from research.taxonomy_validation import run_taxonomy_validation


def _read_report(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _metric(report, key, default=0.0):
    value = report
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_reports(job_root):
    reports = []
    for root, _, files in os.walk(job_root):
        if "analysis_report.json" not in files:
            continue
        report = _read_report(os.path.join(root, "analysis_report.json"))
        if report and "error" not in report:
            reports.append(report)

    if not reports:
        return {
            "job_count": 0,
            "message": "No analysis_report.json files found.",
        }

    yields = [_metric(r, "yield_percent") for r in reports]
    utilization = [
        _metric(r, "space_utilization.occupied_percent") for r in reports
    ]
    blade_gaps = [
        _metric(r, "optimizer_diagnostics.blade_clearance.actual_min_gap_mm")
        for r in reports
    ]
    completion = [
        _metric(r, "research_completion.proposal_software_completion_percent")
        for r in reports
    ]
    validation = [
        _metric(r, "research_completion.current_job_validation_percent")
        for r in reports
    ]

    return {
        "job_count": len(reports),
        "average_yield_percent": round(mean(yields), 2),
        "average_space_utilization_percent": round(mean(utilization), 2),
        "minimum_blade_gap_mm": round(min(blade_gaps), 3),
        "average_software_completion_percent": round(mean(completion), 2),
        "average_current_job_validation_percent": round(mean(validation), 2),
        "research_claim_boundary": (
            "This summarizes system outputs. External research validation still "
            "requires ground-truth cut outcomes and expert labels."
        ),
    }


def _legacy_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize Quartz.AI research validation outputs."
    )
    parser.add_argument("job_root", help="Folder containing completed job folders")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path",
    )
    args = parser.parse_args(argv)

    summary = summarize_reports(args.job_root)
    text = json.dumps(summary, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    print(text)
    return 0


def _dataset_audit_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py dataset-audit",
        description="Run a read-only integrity audit of a YOLO segmentation dataset.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the dataset audit JSON configuration.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional evidence output directory override.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 when blocking integrity issues are found.",
    )
    args = parser.parse_args(argv)

    config = load_audit_config(args.config, output_override=args.output)
    if args.strict:
        config["strict"] = True
    command = " ".join(
        ["python", "backend/research_benchmark.py", "dataset-audit", *argv]
    )
    report, files = run_dataset_audit(
        config,
        output_override=args.output,
        command=command,
    )
    summary = {
        "audit_status": report["audit_status"],
        "claim_readiness": report["claim_readiness"],
        "totals": report["totals"],
        "output_directory": str(files["audit"].parent),
    }
    print(json.dumps(summary, indent=2))

    if config.get("strict") and not report["claim_readiness"]["claim_ready"]:
        return 2
    return 0


def _defect_policy_audit_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py defect-policy-audit",
        description=(
            "Audit detector taxonomy, policy safety, and output traceability "
            "without calculating model-performance metrics."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the detector policy JSON configuration.",
    )
    parser.add_argument(
        "--job-dir",
        default=None,
        help="Optional completed job directory for output traceability checks.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional evidence output directory override.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 when policy or traceability blockers exist.",
    )
    args = parser.parse_args(argv)

    config = load_defect_audit_config(
        args.config,
        output_override=args.output,
    )
    command = " ".join(
        ["python", "backend/research_benchmark.py", "defect-policy-audit", *argv]
    )
    report, files = run_defect_policy_audit(
        config,
        job_dir=args.job_dir,
        output_override=args.output,
        command=command,
    )
    summary = {
        "audit_status": report["summary"]["audit_status"],
        "claim_status": report["claim_status"],
        "policy": report["summary"]["policy"],
        "model_class_count": report["summary"]["model_class_count"],
        "unknown_class_count": report["summary"]["unknown_class_count"],
        "blocker_count": report["summary"]["blocker_count"],
        "performance_metrics_calculated": False,
        "output_directory": str(files["audit"].parent),
    }
    print(json.dumps(summary, indent=2))
    if args.strict and report["summary"]["blocker_count"] > 0:
        return 2
    return 0


def _environment_check_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py environment-check",
        description="Run a read-only environment and dependency readiness check.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    config = load_environment_config(args.config, output_override=args.output)
    command = " ".join(
        ["python", "backend/research_benchmark.py", "environment-check", *argv]
    )
    report, files = run_environment_check(
        config,
        output_override=args.output,
        command=command,
    )
    print(json.dumps({
        "status": report["summary"]["status"],
        "available_count": report["summary"]["available_count"],
        "missing_count": report["summary"]["missing_count"],
        "final_model_runtime_ready": report["summary"][
            "final_model_runtime_ready"
        ],
        "output_directory": str(files["report"].parent),
    }, indent=2))
    return 0


def _taxonomy_validate_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py taxonomy-validate",
        description="Validate taxonomy approval evidence without creating approval.",
    )
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--approval", default=None)
    parser.add_argument("--dataset-yaml", default="dataset/data.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    command = " ".join(
        ["python", "backend/research_benchmark.py", "taxonomy-validate", *argv]
    )
    report, files = run_taxonomy_validation(
        taxonomy_path=args.taxonomy,
        approval_path=args.approval,
        dataset_yaml=args.dataset_yaml,
        output_directory=args.output,
        command=command,
    )
    print(json.dumps({
        "status": report["summary"]["status"],
        "taxonomy_approved": report["summary"]["taxonomy_approved"],
        "blocker_count": report["summary"]["blocker_count"],
        "primary_claim_metric": report["approval"]["primary_claim_metric"],
        "output_directory": str(files["report"].parent),
    }, indent=2))
    if args.strict and not report["summary"]["taxonomy_approved"]:
        return 2
    return 0


def _duplicate_review_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py duplicate-review",
        description="Generate a human-review package for near-duplicate candidates.",
    )
    parser.add_argument("--dataset-audit", required=True)
    parser.add_argument("--existing-review", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    command = " ".join(
        ["python", "backend/research_benchmark.py", "duplicate-review", *argv]
    )
    report, files = generate_duplicate_review(
        dataset_audit_path=args.dataset_audit,
        existing_review=args.existing_review,
        output_directory=args.output,
        strict=args.strict,
        command=command,
    )
    print(json.dumps({
        "candidate_count": report["candidate_count"],
        "reviewed_count": report["reviewed_count"],
        "unresolved_count": report["unresolved_count"],
        "strict_blocked": report["strict_blocked"],
        "output_directory": str(files["report"].parent),
    }, indent=2))
    if args.strict and report["strict_blocked"]:
        return 2
    return 0


def _grouped_split_plan_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py grouped-split-plan",
        description="Create a deterministic manifest-only grouped split plan.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    config = load_grouped_split_config(args.config, output_override=args.output)
    command = " ".join(
        ["python", "backend/research_benchmark.py", "grouped-split-plan", *argv]
    )
    plan, files = run_grouped_split_plan(
        config,
        output_override=args.output,
        command=command,
    )
    summary = plan["summary"]
    print(json.dumps({
        "grouping_level": summary["grouping_level"],
        "image_count": summary["image_count"],
        "atomic_group_count": summary["atomic_group_count"],
        "split_approved": summary["split_approved"],
        "frozen": summary["frozen"],
        "blocker_count": summary["blocker_count"],
        "output_directory": str(files["summary"].parent),
    }, indent=2))
    if args.strict and not summary["split_approved"]:
        return 2
    return 0


def _grouped_split_materialize_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py grouped-split-materialize",
        description="Materialize an approved frozen split into a new directory.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args(argv)
    config = load_grouped_split_config(args.config)
    repo_root = config["_repo_root"]
    plan_directory = config.get("output_directory")
    if not os.path.isabs(str(plan_directory)):
        plan_directory = os.path.join(repo_root, str(plan_directory))
    mode = config.get("materialization", {}).get("mode", "copy")
    if mode == "manifest_only":
        mode = "copy"
    result = materialize_grouped_split(
        plan_directory=plan_directory,
        output_directory=args.output,
        apply=args.apply,
        confirm=args.confirm,
        mode=mode,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"preview", "complete"} else 2


def _model_provenance_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py model-provenance",
        description="Inspect legacy model training provenance without loading it.",
    )
    parser.add_argument("--model", default="backend/best.pt")
    parser.add_argument("--dataset-yaml", default="dataset/data.yaml")
    parser.add_argument(
        "--training-args",
        default="training_runs/quartz_fracture_v1/args.yaml",
    )
    parser.add_argument("--training-source", default="backend/train_fractures.py")
    parser.add_argument(
        "--dataset-audit",
        default="research_evidence/runs/dataset-audit-current/dataset_audit.json",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    command = " ".join(
        ["python", "backend/research_benchmark.py", "model-provenance", *argv]
    )
    report, files = run_model_provenance(
        model_path=args.model,
        dataset_yaml=args.dataset_yaml,
        training_args=args.training_args,
        training_source=args.training_source,
        dataset_audit=args.dataset_audit,
        output_directory=args.output,
        command=command,
    )
    print(json.dumps({
        "training_split_provenance": report["training_split_provenance"],
        "final_research_eligible": report["final_research_eligible"],
        "exploratory_pipeline_smoke_test": report[
            "exploratory_pipeline_smoke_test"
        ],
        "output_directory": str(files["report"].parent),
    }, indent=2))
    return 0


def _defect_evaluation_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py defect-evaluation",
        description="Run defect-evaluation preflight or guarded final execution.",
    )
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    config = load_evaluation_config(args.config, output_override=args.output)
    command = " ".join(
        ["python", "backend/research_benchmark.py", "defect-evaluation", *argv]
    )
    report, files = run_evaluation_preflight(
        config,
        output_override=args.output,
        command=command,
    )
    summary = report["summary"]
    result = {
        "status": summary["status"],
        "final_evaluation_ready": summary["final_evaluation_ready"],
        "blocked_gate_count": summary["blocked_gate_count"],
        "final_metrics_calculated": False,
        "claim_90_percent_available": False,
        "output_directory": str(files["report"].parent),
    }
    if args.run:
        if not summary["final_evaluation_ready"]:
            result["execution"] = "blocked_by_scientific_readiness_gates"
            print(json.dumps(result, indent=2))
            return 2
        result["execution"] = (
            "not_executed_phase3a_requires_clean_trained_model_runner"
        )
        print(json.dumps(result, indent=2))
        return 3
    print(json.dumps(result, indent=2))
    return 0


def _metric_approval_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py metric-approval",
        description="Validate the approved operational definition of the 90% target.",
    )
    parser.add_argument("--approval", default=None)
    parser.add_argument(
        "--taxonomy",
        default="research/templates/defect_taxonomy.csv",
    )
    parser.add_argument(
        "--test-protocol",
        default="research/protocols/defect_model_evaluation.md",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    command = " ".join(
        ["python", "backend/research_benchmark.py", "metric-approval", *argv]
    )
    report, files = run_metric_approval(
        approval_path=args.approval,
        taxonomy_path=args.taxonomy,
        test_protocol_path=args.test_protocol,
        output_directory=args.output,
        command=command,
    )
    print(json.dumps({
        "status": report["summary"]["status"],
        "metric_approved": report["summary"]["metric_approved"],
        "primary_metric": report["approval"]["primary_metric"],
        "claim_threshold": report["approval"]["claim_threshold"],
        "blocker_count": report["summary"]["blocker_count"],
        "final_metrics_calculated": False,
        "output_directory": str(files["report"].parent),
    }, indent=2))
    if args.strict and not report["summary"]["metric_approved"]:
        return 2
    return 0


def _phase3b_gate_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py phase3b-gate",
        description=(
            "Run Phase 3B Gate 0 readiness checks without training or "
            "final evaluation."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    config = load_phase3b_gate_config(
        args.config,
        output_override=args.output,
    )
    command = " ".join(
        ["python", "backend/research_benchmark.py", "phase3b-gate", *argv]
    )
    report, files = run_phase3b_gate(
        config,
        output_override=args.output,
        command=command,
    )
    summary = report["summary"]
    print(json.dumps({
        "status": summary["status"],
        "authorized_for_clean_training": summary[
            "authorized_for_clean_training"
        ],
        "passed_gate_count": summary["passed_gate_count"],
        "blocked_gate_count": summary["blocked_gate_count"],
        "training_occurred": False,
        "final_metrics_calculated": False,
        "claim_90_percent_available": False,
        "output_directory": str(files["report"].parent),
    }, indent=2))
    if args.strict and not summary["authorized_for_clean_training"]:
        return 2
    return 0


def _external_test_readiness_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py external-test-readiness",
        description=(
            "Validate external-test operations and collection evidence without "
            "training or evaluation."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    config = load_external_test_readiness_config(
        args.config,
        output_override=args.output,
    )
    command = " ".join(
        [
            "python",
            "backend/research_benchmark.py",
            "external-test-readiness",
            *argv,
        ]
    )
    report, files = run_external_test_readiness(
        config,
        output_override=args.output,
        command=command,
    )
    print(json.dumps({
        "status": report["summary"]["status"],
        "operations_plan_ready": report["summary"]["operations_plan_ready"],
        "specimen_count_planned": len(report["operations"]["specimen_ids"]),
        "specimens_physically_available": report["statuses"][
            "specimens_physically_available"
        ],
        "capture_complete": report["statuses"]["capture_complete"],
        "annotations_complete": report["statuses"]["annotations_complete"],
        "expert_review_complete": report["statuses"][
            "expert_review_complete"
        ],
        "test_set_frozen": report["statuses"]["test_set_frozen"],
        "training_authorized": False,
        "final_evaluation_authorized": report["statuses"][
            "final_evaluation_authorized"
        ],
        "training_occurred": False,
        "evaluation_occurred": False,
        "weights_downloaded": False,
        "performance_metrics_calculated": False,
        "output_directory": str(files["report"].parent),
    }, indent=2))
    if args.strict and not report["summary"]["external_data_ready"]:
        return 2
    return 0


def _candidate_dataset_audit_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py candidate-dataset-audit",
        description=(
            "Register and audit the Gate 1 YOLO segmentation candidate "
            "without modifying source data."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", default=None)
    parser.add_argument("--zip", dest="candidate_zip", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    config = load_candidate_config(
        args.config,
        source_override=args.source,
        zip_override=args.candidate_zip,
        output_override=args.output,
    )
    command = " ".join(
        [
            "python",
            "backend/research_benchmark.py",
            "candidate-dataset-audit",
            *argv,
        ]
    )
    report, files = run_candidate_audit(
        config,
        output_override=args.output,
        command=command,
    )
    summary = report["summary"]
    print(json.dumps({
        "status": summary["status"],
        "registration_ready": summary["registration_ready"],
        "image_count": report["dataset"]["image_count"],
        "label_count": report["dataset"]["label_count"],
        "class_names": report["dataset"]["class_names"],
        "unique_gem_count": report["lineage"]["unique_gem_count"],
        "current_roboflow_split_trusted": False,
        "training_occurred": False,
        "final_metrics_calculated": False,
        "output_directory": str(files["report"].parent),
    }, indent=2))
    if args.strict and not summary["registration_ready"]:
        return 2
    return 0


def _development_split_plan_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py development-split-plan",
        description=(
            "Create a manifest-only gem-grouped train/validation plan."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", default=None)
    parser.add_argument("--zip", dest="candidate_zip", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    config = load_candidate_config(
        args.config,
        source_override=args.source,
        zip_override=args.candidate_zip,
    )
    command = " ".join(
        [
            "python",
            "backend/research_benchmark.py",
            "development-split-plan",
            *argv,
        ]
    )
    plan, files = run_development_split_plan(
        config,
        output_override=args.output,
        command=command,
    )
    summary = plan["summary"]
    print(json.dumps({
        "status": summary["status"],
        "structural_plan_generated": summary["structural_plan_generated"],
        "split_image_counts": summary["split_image_counts"],
        "split_gem_counts": summary["split_gem_counts"],
        "gem_identity_rule_approved": summary[
            "gem_identity_rule_approved"
        ],
        "split_approved": summary["split_approved"],
        "frozen": summary["frozen"],
        "materialization_allowed": summary["materialization_allowed"],
        "current_images_used_as_final_test": 0,
        "training_occurred": False,
        "final_metrics_calculated": False,
        "output_directory": str(files["summary"].parent),
    }, indent=2))
    if args.strict and not summary["split_approved"]:
        return 2
    return 0


def _development_split_materialize_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py development-split-materialize",
        description=(
            "Materialize an approved frozen development split externally."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args(argv)
    config = load_candidate_config(args.config)
    plan_directory = (
        args.plan
        or config.get("development_output_directory")
        or config.get("output_directory")
    )
    if not os.path.isabs(str(plan_directory)):
        plan_directory = os.path.join(
            config["_repo_root"],
            str(plan_directory),
        )
    mode = (config.get("materialization") or {}).get("mode", "copy")
    result = materialize_development_split(
        plan_directory=plan_directory,
        output_directory=args.output,
        repo_root=config["_repo_root"],
        apply=args.apply,
        confirm=args.confirm,
        mode=mode,
        provenance_inputs=approval_paths(config),
        taxonomy_path=config.get("candidate_taxonomy"),
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"preview", "complete"} else 2


def _gate2_approvals_validate_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py gate2-approvals-validate",
        description=(
            "Validate role-only Gate 2 approvals without training or evaluation."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", default=None)
    parser.add_argument("--zip", dest="candidate_zip", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    config = load_gate2_config(
        args.config,
        source_override=args.source,
        zip_override=args.candidate_zip,
        output_override=args.output,
    )
    command = " ".join(
        [
            "python",
            "backend/research_benchmark.py",
            "gate2-approvals-validate",
            *argv,
        ]
    )
    report, files = run_gate2_approval_validation(
        config,
        output_override=args.output,
        command=command,
    )
    summary = report["summary"]
    print(json.dumps({
        "status": summary["status"],
        "development_dataset_approval_ready": summary[
            "development_dataset_approval_ready"
        ],
        "external_test_concept_approved": summary[
            "external_test_concept_approved"
        ],
        "external_test_operational_ready": False,
        "training_authorized": False,
        "training_occurred": False,
        "weights_downloaded": False,
        "final_metrics_calculated": False,
        "output_directory": str(files["report"].parent),
    }, indent=2))
    if args.strict and not summary["development_dataset_approval_ready"]:
        return 2
    return 0


def _materialized_dataset_audit_main(argv):
    parser = argparse.ArgumentParser(
        prog="research_benchmark.py materialized-dataset-audit",
        description=(
            "Audit the external materialized train/validation dataset read-only."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--plan", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    config = load_gate2_config(args.config)
    plan_directory = (
        args.plan
        or config.get("development_output_directory")
        or config.get("output_directory")
    )
    if not os.path.isabs(str(plan_directory)):
        plan_directory = os.path.join(
            config["_repo_root"],
            str(plan_directory),
        )
    output = args.output or config.get("materialized_audit_output_directory")
    command = " ".join(
        [
            "python",
            "backend/research_benchmark.py",
            "materialized-dataset-audit",
            *argv,
        ]
    )
    report, files = run_materialized_dataset_audit(
        dataset_root=args.dataset,
        plan_directory=plan_directory,
        output_directory=output,
        repo_root=config["_repo_root"],
        command=command,
    )
    print(json.dumps({
        "status": report["summary"]["status"],
        "integrity_verified": report["summary"]["integrity_verified"],
        "split_counts": report["split_counts"],
        "class_names": report["class_names"],
        "leakage": report["leakage"],
        "test_directory_exists": report["test_directory_exists"],
        "training_occurred": False,
        "weights_downloaded": False,
        "final_metrics_calculated": False,
        "output_directory": str(files["report"].parent),
    }, indent=2))
    if args.strict and not report["summary"]["integrity_verified"]:
        return 2
    return 0


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "dataset-audit":
        return _dataset_audit_main(arguments[1:])
    if arguments and arguments[0] == "defect-policy-audit":
        return _defect_policy_audit_main(arguments[1:])
    if arguments and arguments[0] == "environment-check":
        return _environment_check_main(arguments[1:])
    if arguments and arguments[0] == "taxonomy-validate":
        return _taxonomy_validate_main(arguments[1:])
    if arguments and arguments[0] == "duplicate-review":
        return _duplicate_review_main(arguments[1:])
    if arguments and arguments[0] == "grouped-split-plan":
        return _grouped_split_plan_main(arguments[1:])
    if arguments and arguments[0] == "grouped-split-materialize":
        return _grouped_split_materialize_main(arguments[1:])
    if arguments and arguments[0] == "model-provenance":
        return _model_provenance_main(arguments[1:])
    if arguments and arguments[0] == "defect-evaluation":
        return _defect_evaluation_main(arguments[1:])
    if arguments and arguments[0] == "metric-approval":
        return _metric_approval_main(arguments[1:])
    if arguments and arguments[0] == "phase3b-gate":
        return _phase3b_gate_main(arguments[1:])
    if arguments and arguments[0] == "external-test-readiness":
        return _external_test_readiness_main(arguments[1:])
    if arguments and arguments[0] == "candidate-dataset-audit":
        return _candidate_dataset_audit_main(arguments[1:])
    if arguments and arguments[0] == "development-split-plan":
        return _development_split_plan_main(arguments[1:])
    if arguments and arguments[0] == "development-split-materialize":
        return _development_split_materialize_main(arguments[1:])
    if arguments and arguments[0] == "gate2-approvals-validate":
        return _gate2_approvals_validate_main(arguments[1:])
    if arguments and arguments[0] == "materialized-dataset-audit":
        return _materialized_dataset_audit_main(arguments[1:])
    return _legacy_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
