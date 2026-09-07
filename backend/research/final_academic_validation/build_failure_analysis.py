"""Build post-hoc failure analysis and threats-to-validity docs.

The generated documents are supplemental academic interpretation based only on
frozen evidence artifacts. They do not alter the underlying research results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "final_research_evidence" / "final_academic_validation"
FAILURE_ANALYSIS_MD = OUTPUT_DIR / "failure_analysis.md"
THREATS_TO_VALIDITY_MD = OUTPUT_DIR / "threats_to_validity.md"

RECON_SUMMARY_JSON = (
    ROOT
    / "final_research_evidence"
    / "reconstruction_multi"
    / "batch_validation_summary.json"
)
DEFECT_RESULTS_JSON = (
    ROOT
    / "final_research_evidence"
    / "defect_detection"
    / "final_test"
    / "independent_final_test_results"
    / "independent_final_test_results.json"
)
FACET_METRICS_JSON = (
    ROOT / "backend" / "research" / "facet_ml" / "facet_orientation_metrics.json"
)
FACET_MANIFEST_JSON = (
    ROOT / "backend" / "research" / "facet_ml" / "facet_orientation_manifest.json"
)
OPTIMIZER_SUMMARY_JSON = (
    ROOT
    / "final_research_evidence"
    / "optimizer_validation"
    / "system_yield_summary.json"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def metric(value: Any, digits: int = 6) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def load_evidence() -> dict[str, Any]:
    return {
        "reconstruction": read_json(RECON_SUMMARY_JSON),
        "defect": read_json(DEFECT_RESULTS_JSON),
        "facet_metrics": read_json(FACET_METRICS_JSON),
        "facet_manifest": read_json(FACET_MANIFEST_JSON),
        "optimizer": read_json(OPTIMIZER_SUMMARY_JSON),
    }


def failure_analysis_text(evidence: dict[str, Any]) -> str:
    reconstruction = evidence["reconstruction"]
    defect = evidence["defect"]
    facet_metrics = evidence["facet_metrics"]
    facet_manifest = evidence["facet_manifest"]
    optimizer = evidence["optimizer"]
    recon_metrics = reconstruction["principal_oriented"]
    defect_aggregate = defect["aggregate"]
    facet_test = facet_metrics["splits"]["test"]
    optimizer_target = optimizer["proposal_target"]["status"]

    lines = [
        "# Post-Hoc Failure Analysis",
        "",
        "This document is a factual post-hoc analysis of frozen research evidence. "
        "It does not change thresholds, labels, specimen membership, model weights, "
        "optimizer outputs, or proposal-target status.",
        "",
        "## Reconstruction",
        "",
        f"The frozen TRUE_DENSE reconstruction aggregate contains "
        f"{reconstruction['dense_aggregate_specimen_count']} physical specimens "
        f"and {reconstruction['dense_aggregate_dimension_count']} dimensions. "
        f"The principal/oriented MAE was {metric(recon_metrics['mae_mm'], 12)} mm, "
        f"RMSE was {metric(recon_metrics['rmse_mm'], 12)} mm, median absolute "
        f"error was {metric(recon_metrics['median_error_mm'], 12)} mm, and maximum "
        f"absolute error was {metric(recon_metrics['max_error_mm'], 12)} mm. The "
        "approximately 0.1 mm target was not achieved.",
        "",
        "- Reflective and translucent quartz remains difficult for multi-view "
        "photogrammetry because stable visible surface texture and internal/edge "
        "appearance can vary across views.",
        "- The frozen evidence separates TRUE_DENSE outputs from SPARSE_FALLBACK "
        "outputs. QZ-09 and QZ-30 were preserved as sparse/degraded evidence and "
        "excluded from the TRUE_DENSE aggregate.",
        "- Capture-quality evidence shows weak or failed capture sets for many "
        "specimens, including low sharpness, fallback extraction, and weak "
        "cross-video overlap in some cases. These are plausible contributors, "
        "not sole proven causes.",
        "- QZ-08 is the major dimensional outlier in the dense aggregate, with a "
        "large principal/oriented maximum error. It keeps RMSE high even though "
        "principal orientation improves MAE relative to production AABB overall.",
        "- Scale depends on the production mass/density path. Physical dimensions "
        "were used only for validation, so scale error, volume error, and shape "
        "proportion error cannot be fully separated from these aggregate metrics.",
        "- Principal orientation reduced the average absolute dimensional error by "
        f"{metric(reconstruction['aabb_to_principal_mae_reduction_mm'], 6)} mm, "
        "but RMSE remained outlier-sensitive.",
        "",
        "## Defect Detection",
        "",
        f"The frozen independent final test contains {defect['image_count']} "
        f"images from {defect['specimen_group_count']} physical specimen groups. "
        f"The aggregate precision was {metric(defect_aggregate['precision'], 12)}, "
        f"recall was {metric(defect_aggregate['recall'], 12)}, F1 was "
        f"{metric(defect_aggregate['f1'], 12)}, and IoU was "
        f"{metric(defect_aggregate['iou'], 12)}. The F1 >= 0.90 target was not "
        "achieved.",
        "",
        "- The independent result shows a development-to-independent "
        "generalization collapse rather than a reporting or integrity failure.",
        "- Recall is extremely low on the held-out final set; most annotated "
        "visible-defect pixels were missed.",
        "- Only Gem_23 produced true-positive binary overlap at the specimen level.",
        "- The inclusion class had zero true-positive pixels in frozen per-class "
        "metrics.",
        "- The dataset is extremely imbalanced toward background pixels. High "
        "specificity is therefore not evidence of useful defect detection.",
        "- No final-test threshold retuning is scientifically allowed; labels, "
        "membership, model hash, and inference settings remain frozen.",
        "- Plausible contributors include limited independent specimen diversity, "
        "annotation difficulty, class ambiguity, model capacity, and domain shift. "
        "The frozen result itself remains valid final-test evidence.",
        "",
        "## Facet ML",
        "",
        "The facet-orientation component performs strongly on held-out simulated "
        "scenario groups. On the frozen test split, model MAE was "
        f"{metric(facet_test['model']['mae'], 3)}, RMSE was "
        f"{metric(facet_test['model']['rmse'], 3)}, Spearman was "
        f"{metric(facet_test['model']['spearman'], 3)}, mean group Spearman was "
        f"{metric(facet_test['model']['mean_group_spearman'], 3)}, top-orientation "
        f"agreement was {metric(facet_test['model']['top_orientation_agreement'], 3)}, "
        f"and mean angular error was "
        f"{metric(facet_test['model']['mean_angular_error_degrees'], 2)} degrees.",
        "",
        "- These metrics validate ranking behavior against a deterministic "
        "simulation-derived inclusion-visibility surrogate.",
        "- No independent expert facet labels are available in the frozen evidence.",
        "- The result does not validate optical/refraction physics, real-world "
        "market beauty, or cutter preference.",
        "- The manifest records only one confirmed independent reconstructed real "
        "specimen for real specimen-grouped ML assessment, so real-world transfer "
        "remains unproven.",
        f"- The simulation dataset contains "
        f"{facet_manifest['dataset']['row_count']} rows across "
        f"{facet_manifest['dataset']['scenario_group_count']} scenario groups, "
        "which supports simulation-bound claims but not expert-grounded claims.",
        "",
        "## Optimizer",
        "",
        f"The frozen Objective-3 system-side status is: {optimizer_target}.",
        "",
        "- Geometric packing and manufacturing-complete cutting plans are different "
        "claims. Manufacturing-complete rows require exact cut-tree verification.",
        "- QZ-05's 36.6 percent Multi-Gem result is preserved as diagnostic because "
        "the exact manufacturing sequence was incomplete in the frozen evidence.",
        "- QZ-01 includes verified multi-gem rows and a diagnostic fourth-gem probe, "
        "but diagnostic evidence is not a saved production option and does not "
        "prove global optimality.",
        "- Search time-budget exhaustion means no global-optimum claim is supported.",
        "- Internal computational baselines are not traditional cutter baselines.",
        "- Post-validation manufacturing improvements are separate engineering "
        "evidence and must not be reinterpreted as original frozen results.",
        "",
    ]
    return "\n".join(lines)


def threat_row(threat: str, effect: str, mitigation: str, remaining: str) -> str:
    return f"| {threat} | {effect} | {mitigation} | {remaining} |"


def threats_to_validity_text() -> str:
    sections = {
        "Internal Validity": [
            (
                "Physical measurement error",
                "Dimensional error estimates may include caliper/protocol noise.",
                "Physical L/W/H were used only for validation, not tuning.",
                "Measurement uncertainty was not independently quantified.",
            ),
            (
                "Mass/density scaling assumptions",
                "Scale, volume, and shape-proportion error can be entangled.",
                "The claim boundary states weight enters through production scaling only.",
                "The current evidence cannot fully decompose scale versus geometry error.",
            ),
            (
                "Manual defect-label subjectivity",
                "Pixel-level labels may vary between annotators.",
                "Labels were frozen before final inference and hash-recorded.",
                "No multi-annotator reliability estimate is available.",
            ),
            (
                "Post-validation system modifications",
                "Later engineering changes could be mistaken for original results.",
                "Evidence separates frozen research results from post-validation improvements.",
                "Readers still need explicit thesis wording to avoid temporal confusion.",
            ),
        ],
        "Construct Validity": [
            (
                "Simulation-derived facet target",
                "High simulation ranking may not equal optical beauty or expert preference.",
                "The manifest names the target as an inclusion-visibility surrogate.",
                "No expert or market-quality facet target is validated.",
            ),
            (
                "Defect metric construction",
                "High specificity can appear strong under extreme background imbalance.",
                "Primary reporting uses F1, recall, precision, and IoU, not specificity alone.",
                "Pixel-level union still simplifies class-specific severity and usability.",
            ),
            (
                "Optimizer baseline definition",
                "Internal computational baselines could be mistaken for traditional cutting.",
                "Optimizer reports mark traditional_cutting_comparison=NO.",
                "Objective-3 expert comparison remains pending.",
            ),
            (
                "Diagnostic versus verified optimizer state",
                "Diagnostic high-yield rows may overstate manufacturable performance.",
                "Verification scope is stored per row.",
                "Some promising rows remain diagnostic rather than production-ready.",
            ),
        ],
        "External Validity": [
            (
                "Small TRUE_DENSE specimen count",
                "Reconstruction conclusions may not generalize across quartz forms.",
                "Specimen-level bootstrap keeps dimensions grouped by physical stone.",
                "Only four TRUE_DENSE validation specimens are available.",
            ),
            (
                "Limited independent defect specimen diversity",
                "Defect detection performance may vary under other stones and defects.",
                "The final test uses specimen-separated groups.",
                "The final set has nine groups and remains small.",
            ),
            (
                "Capture-condition sensitivity",
                "Lighting, sharpness, reflection, and overlap may affect reconstruction success.",
                "A pre-COLMAP capture-quality audit records PASS/BORDERLINE/FAIL evidence.",
                "Capture controls do not retroactively improve frozen results.",
            ),
            (
                "Lack of expert traditional baseline",
                "Optimizer waste-reduction claims cannot be generalized to cutter practice yet.",
                "Form A/Form B tooling preserves independent versus post-system evidence.",
                "Real expert Form A responses are still required.",
            ),
        ],
        "Conclusion/Statistical Validity": [
            (
                "Grouped dependence",
                "Treating dimensions/images as independent would understate uncertainty.",
                "Bootstrap resamples physical specimens or specimen groups only.",
                "Intervals are descriptive because group counts are small.",
            ),
            (
                "Search time limits",
                "Optimizer results do not establish global optima.",
                "Search termination and verification status are recorded.",
                "Better future searches are post-validation evidence, not original results.",
            ),
            (
                "mAP50 uncertainty unavailable",
                "Native segmentation uncertainty cannot be estimated from stored AP samples.",
                "The frozen mAP50 point estimate is retained.",
                "No mAP50 CI is reported.",
            ),
            (
                "Multiple images from same physical specimens",
                "Image-level bootstrap would inflate apparent evidence size.",
                "Defect uncertainty uses specimen-group-level resampling.",
                "Only nine independent physical groups support the final defect interval.",
            ),
        ],
    }
    lines = [
        "# Threats To Validity",
        "",
        "This section records supported threats, mitigations already used, and "
        "remaining limitations. It does not reinterpret frozen results.",
        "",
    ]
    header = "| Threat | Possible effect | Mitigation already used | Remaining limitation |"
    separator = "| --- | --- | --- | --- |"
    for section, rows in sections.items():
        lines.extend([f"## {section}", "", header, separator])
        lines.extend(threat_row(*row) for row in rows)
        lines.append("")
    return "\n".join(lines)


def build_failure_analysis(output_dir: Path = OUTPUT_DIR) -> dict[str, str]:
    global OUTPUT_DIR, FAILURE_ANALYSIS_MD, THREATS_TO_VALIDITY_MD
    OUTPUT_DIR = output_dir
    FAILURE_ANALYSIS_MD = OUTPUT_DIR / "failure_analysis.md"
    THREATS_TO_VALIDITY_MD = OUTPUT_DIR / "threats_to_validity.md"

    evidence = load_evidence()
    write_text(FAILURE_ANALYSIS_MD, failure_analysis_text(evidence))
    write_text(THREATS_TO_VALIDITY_MD, threats_to_validity_text())
    return {
        "failure_analysis_md": FAILURE_ANALYSIS_MD.as_posix(),
        "threats_to_validity_md": THREATS_TO_VALIDITY_MD.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    outputs = build_failure_analysis(args.output_dir)
    print(json.dumps({"status": "SAFE", "outputs": outputs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
