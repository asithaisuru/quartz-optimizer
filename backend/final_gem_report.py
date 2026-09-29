"""Final Gemstone Planning Report: a PDF built from ONE defect-aware run.

Every number comes from that run's saved result.json. Stone Preservation,
Preform Recovery, the legacy analysis_report.json and other runs are never
read here, so the report cannot mix computational models.

Display formatting mirrors the dashboard (FinalGemstonePanel) so the PDF and
the screen show identical values for the same run.
"""
import math
from datetime import datetime

from report_generator import (
    COLORS, CONTENT_WIDTH, PDFReport, _callout, _ensure_room, _key_value_table,
    _metric_cards, _paragraph, _safe_text, _section_title, _styled_table,
)

REPORT_VERSION = "final_gem_report_v1"
TITLE = "FINAL GEMSTONE PLANNING REPORT"
EXCLUSION_LABEL = "Confirmed Defect Exclusion - Final Gem Optimizer"
EXCLUSION_NOTE = "Model-derived exclusion estimate used by this final-gem calculation."
SAFETY_STATEMENT = ("Accepted gemstone placements do not intersect confirmed defect safety "
                    "regions under the implemented geometric model.")
LIMITATIONS = (
    "Computational, model-derived result: weights and exclusions are estimated from the 3D "
    "reconstruction and the calibrated rough weight, not measured on cut stones.",
    "Bounded search: the optimizer searches within time and candidate limits and does not "
    "prove that no higher-yield plan exists.",
    "Not measured polished recovery: final faceted yield is a planning estimate, not the "
    "weight of polished gemstones.",
    "Workshop validation remains required: the cut sequence is operator guidance only, not "
    "CNC or G-code, and must be checked before every cut.",
)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_ct(value):
    number = _number(value)
    return "-" if number is None else f"{number:.2f} ct"


def format_percent(value):
    number = _number(value)
    return "-" if number is None else f"{number:.1f}%"


def format_runtime(seconds):
    number = _number(seconds)
    if number is None:
        return "-"
    if number < 60:
        return f"{number:.{1 if number < 10 else 0}f} s"
    minutes = math.floor(number / 60)
    return f"{minutes} min {math.floor(number - minutes * 60 + 0.5)} s"


def format_status(value):
    if value in (None, ""):
        return "-"
    text = str(value).replace("_", " ")
    return text[:1].upper() + text[1:]


def _mm(value):
    number = _number(value)
    return "-" if number is None else f"{number:g} mm"


def final_gem_summary(result):
    """Dashboard values for one run (same formulas as finalGemstoneSummary)."""
    rough = _number(result.get("rough_weight_ct"))
    excluded = _number(result.get("confirmed_defect_excluded_ct"))
    total = _number(result.get("total_gem_weight_ct"))
    clean = rough - excluded if rough is not None and excluded is not None and rough - excluded > 0 else None
    return {
        "rough_weight_ct": rough,
        "confirmed_defect_count": int(_number(result.get("confirmed_defect_count")) or 0),
        "exclusion_ct": excluded,
        "clean_rough_weight_ct": clean,
        "total_gem_weight_ct": total,
        "clean_yield_percent": total / clean * 100 if clean is not None and total is not None else None,
        "rough_yield_percent": _number(result.get("faceted_yield_percent")),
        "gem_count": int(_number(result.get("gem_count")) or 0),
        "manufacturing_status": result.get("manufacturing_status"),
        "runtime_seconds": _number(result.get("runtime_seconds")),
    }


def dashboard_rows(summary):
    """The dashboard's result card, label for label."""
    return [
        ("Rough Weight", format_ct(summary["rough_weight_ct"])),
        ("Confirmed Defects", f"{summary['confirmed_defect_count']} confirmed"),
        (EXCLUSION_LABEL, format_ct(summary["exclusion_ct"])),
        ("Clean Rough Weight", format_ct(summary["clean_rough_weight_ct"])),
        ("Final Gemstone Weight", format_ct(summary["total_gem_weight_ct"])),
        ("Final Faceted Yield", format_percent(summary["clean_yield_percent"])),
        ("Gem Count", str(summary["gem_count"])),
        ("Manufacturing Status", format_status(summary["manufacturing_status"])),
        ("Runtime", format_runtime(summary["runtime_seconds"])),
    ]


def _gem_axis(orientation):
    """Direction of the gem's own Z (table) axis in the rough frame."""
    try:
        rows = [[float(value) for value in row[:3]] for row in orientation[:3]]
        axis = [rows[0][2], rows[1][2], rows[2][2]]
    except (TypeError, ValueError, IndexError):
        return "-"
    if not all(math.isfinite(value) for value in axis):
        return "-"
    return "[" + ", ".join(f"{0.0 if abs(value) < 5e-4 else value:.3f}" for value in axis) + "]"


def _dimensions(value):
    if not isinstance(value, (list, tuple)) or not value:
        return "-"
    parts = [_number(item) for item in value]
    if any(part is None for part in parts):
        return "-"
    return " x ".join(f"{part:g}" for part in parts) + " mm"


def _plane(step):
    orientation = (step.get("plane") or {}).get("orientation") or {}
    azimuth, elevation = _number(orientation.get("azimuth_deg")), _number(orientation.get("elevation_deg"))
    if azimuth is None or elevation is None:
        return "-"
    return f"Normal az {azimuth:g} deg, el {elevation:g} deg"


def _keep_together(pdf, rows, intro=False):
    """Start a section on a new page unless its title and table fit here."""
    _ensure_room(pdf, 21 + (5 if intro else 0) + rows * 7.2)


class FinalGemReport(PDFReport):
    def __init__(self, job_id, run_id, generated_at):
        super().__init__(job_id, generated_at)
        self.run_id = _safe_text(run_id)
        self.set_title("Quartz Final Gemstone Planning Report")
        self.set_subject("Final defect-aware gemstone placement and cut sequence")

    def header(self):
        if self.page_no() == 1:
            self.set_fill_color(*COLORS["accent"])
            self.rect(0, 0, self.w, 7, style="F")
            self.set_y(14)
            self.set_text_color(*COLORS["accent"])
            self.set_font("Helvetica", "B", 8)
            self.cell(0, 5, "QUARTZ GEMSTONE CUTTING DECISION SUPPORT")
            self.ln(7)
            self.set_text_color(*COLORS["accent_dark"])
            self.set_font("Helvetica", "B", 18)
            self.cell(0, 9, TITLE)
            self.ln(10)
            self.set_text_color(*COLORS["muted"])
            self.set_font("Helvetica", "", 9)
            self.cell(0, 5, "Current defect-aware final-gem optimization run")
            self.ln(8)
            self.set_draw_color(*COLORS["line"])
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(3)
            for label, value in (("JOB ID", self.job_id), ("RUN ID", self.run_id),
                                 ("GENERATED", self.generated_at.strftime("%Y-%m-%d %H:%M:%S"))):
                self.set_font("Helvetica", "B", 7)
                self.set_text_color(*COLORS["muted"])
                self.cell(22, 4.5, label)
                self.set_font("Helvetica", "", 8)
                self.set_text_color(*COLORS["body"])
                self.cell(0, 4.5, value)
                self.ln(4.5)
            self.set_y(62)
            return
        self.set_y(8)
        self.set_text_color(*COLORS["accent_dark"])
        self.set_font("Helvetica", "B", 8)
        self.cell(120, 5, "Final Gemstone Planning Report")
        self.set_text_color(*COLORS["muted"])
        self.set_font("Helvetica", "", 7)
        self.cell(0, 5, f"Run {self.run_id[:12]}", align="R")
        self.ln(7)
        self.set_draw_color(*COLORS["line"])
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_y(20)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(*COLORS["line"])
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*COLORS["muted"])
        self.cell(120, 5, f"Final gemstone report | job {self.job_id[:12]} | run {self.run_id[:12]}")
        self.cell(0, 5, f"Page {self.page_no()} of {{nb}}", align="R")


def create_final_gem_pdf(result, output_path, *, job_id, run_id, result_sha256=None, generated_at=None):
    """Write the report for one completed defect-aware run to output_path."""
    summary = final_gem_summary(result)
    rows = dict(dashboard_rows(summary))
    pdf = FinalGemReport(job_id, run_id, generated_at or datetime.now())
    pdf.add_page()

    _metric_cards(pdf, [
        ("Final gemstone weight", rows["Final Gemstone Weight"], f"{summary['gem_count']} gems"),
        ("Final faceted yield", rows["Final Faceted Yield"], "Of clean rough weight"),
        ("Clean rough weight", rows["Clean Rough Weight"], "Rough - confirmed exclusion"),
        ("Manufacturing", rows["Manufacturing Status"], "Cut sequence status"),
    ])
    _section_title(pdf, "Result Summary",
                   "Values exactly as shown on the dashboard for this run.")
    _key_value_table(pdf, dashboard_rows(summary))

    _keep_together(pdf, 3)
    _section_title(pdf, "1. Rough Stone")
    _key_value_table(pdf, [
        ("Rough weight", rows["Rough Weight"]),
        ("Clean rough weight", rows["Clean Rough Weight"]),
        ("Clean rough weight basis", "Rough weight - confirmed defect exclusion"),
    ])

    _keep_together(pdf, 5)
    _section_title(pdf, "2. Defect Review")
    _key_value_table(pdf, [
        ("Confirmed defect count", str(summary["confirmed_defect_count"])),
        (EXCLUSION_LABEL, rows[EXCLUSION_LABEL]),
        ("Exclusion basis", EXCLUSION_NOTE),
        ("Provisional AI candidates", f"{int(_number(result.get('provisional_candidate_count')) or 0)} "
                                      "(not excluded; only confirmed defects constrain placement)"),
    ])

    _keep_together(pdf, 7)
    _section_title(pdf, "3. Final Gemstone Result")
    _key_value_table(pdf, [
        ("Gem count", str(summary["gem_count"])),
        ("Total final gemstone weight", rows["Final Gemstone Weight"]),
        ("Final faceted yield", f"{rows['Final Faceted Yield']} of clean rough weight"),
        ("Yield vs full rough weight", format_percent(summary["rough_yield_percent"])),
        ("Search status", format_status(result.get("search_state"))),
        ("Placement strategy", _safe_text(result.get("selected_strategy"), "-")),
        ("Runtime", rows["Runtime"]),
    ])

    gems = result.get("gems") or result.get("gem_details") or []
    _keep_together(pdf, min(len(gems), 6), intro=True)
    _section_title(pdf, "4. Gemstone Breakdown",
                   "Orientation is the gem's own Z (table) axis in the centered rough frame.")
    _styled_table(
        pdf,
        ["Gem ID", "Shape", "Weight (ct)", "Dimensions", "Orientation", "Confirmed defect"],
        [20, 30, 22, 40, 38, CONTENT_WIDTH - 150],
        [[
            _safe_text(gem.get("id") or f"gem_{gem.get('index', index + 1)}"),
            _safe_text(gem.get("shape"), "-"),
            f"{_number(gem.get('weight_ct')):.2f}" if _number(gem.get("weight_ct")) is not None else "-",
            _dimensions(gem.get("dimensions_mm")),
            _gem_axis(gem.get("orientation")),
            "None" if gem.get("confirmed_defect_intersection") is False else "Not verified",
        ] for index, gem in enumerate(gems)],
        aligns=["L", "L", "R", "L", "L", "L"],
    )

    plan = result.get("manufacturing_plan") or {}
    settings = {**(plan.get("settings") or {}), **(result.get("settings") or {})}
    sequence = result.get("cut_sequence") or plan.get("sequence") or []
    verified = (plan.get("diagnostics") or {}).get("exact_sequence_verified")
    _keep_together(pdf, 7)
    _section_title(pdf, "5. Manufacturing Verification")
    _key_value_table(pdf, [
        ("Manufacturing status", rows["Manufacturing Status"]),
        ("Blade kerf", _mm(settings.get("blade_kerf_mm"))),
        ("Maximum required depth", _mm(plan.get("maximum_required_depth_mm"))),
        ("Saw depth limit", _mm(settings.get("max_cut_depth_mm"))),
        ("Number of verified cuts", str(len(sequence))),
        ("Exact sequence verification", "Passed" if verified is True else "Not reported" if verified is None else "Failed"),
        ("Output type", "Operator guidance only - not CNC or G-code"),
    ])

    _keep_together(pdf, min(len(sequence), 6) + 1, intro=True)
    _section_title(pdf, "6. Cut Sequence",
                   "Straight full-through saw cuts in order. Plane direction is given as the cut normal's "
                   "azimuth and elevation in the rough frame.")
    if sequence:
        _styled_table(
            pdf,
            ["Cut", "Piece", "Produces", "Depth", "Kerf", "Plane / direction"],
            [12, 30, 44, 22, 18, CONTENT_WIDTH - 126],
            [[
                str(step.get("step", index + 1)),
                _safe_text(step.get("parent_piece_id"), "-"),
                ", ".join(step.get("result_piece_ids") or []) or "-",
                _mm(step.get("required_depth_mm")),
                _mm((step.get("kerf_slab") or {}).get("thickness_mm")),
                _plane(step),
            ] for index, step in enumerate(sequence)],
            aligns=["R", "L", "L", "R", "R", "L"],
        )
    else:
        _paragraph(pdf, "No saw separation is required for this plan.")

    if gems and all(gem.get("confirmed_defect_intersection") is False for gem in gems):
        _callout(pdf, "Defect Safety Statement", SAFETY_STATEMENT)
    else:
        _callout(pdf, "Defect Safety Statement",
                 "Defect-safety verification is not recorded for every gem in this run.", kind="warning")

    _section_title(pdf, "Limitations")
    for item in (*LIMITATIONS, *(result.get("limitations") or [])):
        _ensure_room(pdf, 10)
        _paragraph(pdf, f"- {item}", font_size=8.5)

    _keep_together(pdf, 5)
    _section_title(pdf, "Source")
    _key_value_table(pdf, [
        ("Result source", "Defect-aware final-gem optimization run (result.json of this run only)"),
        ("Optimizer mode", _safe_text(result.get("mode"), "-")),
        ("Run ID", _safe_text(run_id)),
        ("Result SHA-256", _safe_text(result_sha256, "-")),
        ("Defect review fingerprint", _safe_text(result.get("defect_review_sha256"), "-")),
    ])
    pdf.output(str(output_path))
    return output_path
