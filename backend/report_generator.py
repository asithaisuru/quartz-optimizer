import os
import tempfile
from datetime import datetime

import matplotlib
import numpy as np
import trimesh
from fpdf import FPDF
from scipy.spatial import ConvexHull

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_IMG_DIR = os.path.join(BASE_DIR, "temp_plots")
os.makedirs(TEMP_IMG_DIR, exist_ok=True)

CANONICAL_PDF_FILENAME = "report.pdf"
LEGACY_PDF_FILENAMES = (
    "analysis_report.pdf",
    "quartz_analysis_report.pdf",
)

PAGE_MARGIN = 14
CONTENT_WIDTH = 210 - (PAGE_MARGIN * 2)

COLORS = {
    "accent": (30, 91, 110),
    "accent_dark": (34, 53, 61),
    "accent_soft": (232, 241, 243),
    "body": (45, 52, 55),
    "muted": (98, 108, 112),
    "line": (207, 216, 219),
    "panel": (246, 248, 249),
    "table_alt": (249, 250, 250),
    "warning": (151, 101, 24),
    "warning_fill": (252, 247, 235),
    "success": (43, 112, 81),
    "success_fill": (237, 247, 242),
    "white": (255, 255, 255),
}


def is_valid_pdf(path):
    """Return true only for a readable, non-empty PDF file."""
    if not path or not os.path.isfile(path) or not os.access(path, os.R_OK):
        return False
    try:
        if os.path.getsize(path) < 5:
            return False
        with open(path, "rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def find_existing_pdf(job_folder):
    """Locate only known report filenames inside a job directory."""
    for filename in (CANONICAL_PDF_FILENAME, *LEGACY_PDF_FILENAMES):
        candidate = os.path.join(job_folder, filename)
        if is_valid_pdf(candidate):
            return candidate
    return None


def _safe_text(value, fallback="Unavailable"):
    if value is None or value == "":
        value = fallback
    text = str(value)
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u00d7": "x",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _has_value(value):
    return value is not None and value != ""


def _waste_reference_note(waste):
    boundary = "Not a validated traditional-cutting comparison."
    if not isinstance(waste, dict):
        return boundary
    note = waste.get("note")
    if not _has_value(note):
        if _has_value(waste.get("traditional_waste_baseline_percent")):
            return boundary
        return None
    text = _safe_text(note)
    if boundary.casefold() in text.casefold():
        return text
    return f"{text} {boundary}"


def _format_number(value, digits=1, suffix=""):
    if not _has_value(value):
        return "Unavailable"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _safe_text(value)
    if not np.isfinite(number):
        return "Unavailable"
    rendered = f"{number:,.{digits}f}"
    if digits:
        rendered = rendered.rstrip("0").rstrip(".")
    return f"{rendered}{suffix}"


def _format_integer(value):
    if not _has_value(value):
        return "Unavailable"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return _safe_text(value)


def _format_vector(value, digits=2, suffix=""):
    if not isinstance(value, (list, tuple)) or not value:
        return "Unavailable"
    return " x ".join(_format_number(item, digits) for item in value) + suffix


def _format_normal(value):
    if not isinstance(value, (list, tuple)) or not value:
        return "Unavailable"
    return "[" + ", ".join(_format_number(item, 3) for item in value) + "]"


def _humanize(value):
    return _safe_text(str(value).replace("_", " ").strip().title())


class PDFReport(FPDF):
    def __init__(self, job_id, generated_at):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.job_id = _safe_text(job_id)
        self.generated_at = generated_at
        self.set_margins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
        self.set_auto_page_break(auto=True, margin=19)
        self.set_title("Quartz Gemstone Cutting Optimization Report")
        self.set_subject("Computer vision and gemstone cutting analysis")
        self.set_creator("Quartz Gemstone Cutting Optimization System")
        self.set_compression(False)
        self.alias_nb_pages()

    def header(self):
        if self.page_no() == 1:
            self.set_fill_color(*COLORS["accent"])
            self.rect(0, 0, self.w, 7, style="F")
            self.set_y(14)
            self.set_text_color(*COLORS["accent"])
            self.set_font("Helvetica", "B", 8)
            self.cell(0, 5, "TECHNICAL ANALYSIS REPORT")
            self.ln(7)
            self.set_text_color(*COLORS["accent_dark"])
            self.set_font("Helvetica", "B", 18)
            self.cell(
                0,
                9,
                "Quartz Gemstone Cutting Optimization Report",
            )
            self.ln(10)
            self.set_text_color(*COLORS["muted"])
            self.set_font("Helvetica", "", 9)
            self.cell(
                0,
                5,
                "Computer vision, 3D reconstruction and blade-aware cut planning",
            )
            self.ln(8)
            self.set_draw_color(*COLORS["line"])
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)
            self.set_font("Helvetica", "B", 7)
            self.set_text_color(*COLORS["muted"])
            self.cell(18, 5, "JOB ID")
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*COLORS["body"])
            self.cell(97, 5, self.job_id)
            self.set_font("Helvetica", "B", 7)
            self.set_text_color(*COLORS["muted"])
            self.cell(22, 5, "GENERATED")
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*COLORS["body"])
            self.cell(
                0,
                5,
                self.generated_at.strftime("%Y-%m-%d %H:%M"),
                align="R",
            )
            self.set_y(52)
            return

        self.set_y(8)
        self.set_text_color(*COLORS["accent_dark"])
        self.set_font("Helvetica", "B", 8)
        self.cell(120, 5, "Quartz Gemstone Cutting Optimization Report")
        self.set_text_color(*COLORS["muted"])
        self.set_font("Helvetica", "", 7)
        self.cell(0, 5, f"Job {self.job_id[:12]}", align="R")
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
        self.cell(
            120,
            5,
            f"Technical report | {self.job_id[:12]}",
        )
        self.cell(0, 5, f"Page {self.page_no()} of {{nb}}", align="R")


def generate_blueprint(rough_path, cut_path, output_filename, view="top"):
    rough = trimesh.load(rough_path)
    cut = trimesh.load(cut_path)

    plt.figure(figsize=(4, 4), facecolor="white")
    plt.axis("equal")
    plt.axis("off")

    if view == "top":
        rough_points = rough.vertices[:, [0, 1]]
        cut_points = cut.vertices[:, [0, 1]]
        title = "Top View (Crown)"
    else:
        rough_points = rough.vertices[:, [0, 2]]
        cut_points = cut.vertices[:, [0, 2]]
        title = "Side View (Profile)"

    try:
        rough_hull = ConvexHull(rough_points)
        rough_x = np.append(
            rough_points[rough_hull.vertices, 0],
            rough_points[rough_hull.vertices[0], 0],
        )
        rough_y = np.append(
            rough_points[rough_hull.vertices, 1],
            rough_points[rough_hull.vertices[0], 1],
        )
        plt.plot(rough_x, rough_y, color="#35444a", linewidth=2)
    except Exception:
        pass

    try:
        cut_hull = ConvexHull(cut_points)
        cut_x = np.append(
            cut_points[cut_hull.vertices, 0],
            cut_points[cut_hull.vertices[0], 0],
        )
        cut_y = np.append(
            cut_points[cut_hull.vertices, 1],
            cut_points[cut_hull.vertices[0], 1],
        )
        plt.plot(cut_x, cut_y, color="#1e5b6e", linewidth=1.7, linestyle="--")
        plt.fill(cut_x, cut_y, color="#1e5b6e", alpha=0.12)
    except Exception:
        pass

    plt.title(title, color="#22353d", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    save_path = os.path.join(TEMP_IMG_DIR, output_filename)
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    return save_path


def _ensure_room(pdf, height):
    if pdf.get_y() + height > pdf.page_break_trigger:
        pdf.add_page()


def _split_long_token(pdf, token, max_width):
    chunks = []
    current = ""
    for character in token:
        candidate = current + character
        if current and pdf.get_string_width(candidate) > max_width:
            chunks.append(current)
            current = character
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [""]


def _wrap_text(pdf, value, max_width):
    text = _safe_text(value)
    lines = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            if pdf.get_string_width(word) > max_width:
                token_parts = _split_long_token(pdf, word, max_width)
            else:
                token_parts = [word]
            for part in token_parts:
                candidate = f"{current} {part}".strip()
                if current and pdf.get_string_width(candidate) > max_width:
                    lines.append(current)
                    current = part
                elif not current and pdf.get_string_width(part) > max_width:
                    lines.append(part)
                    current = ""
                else:
                    current = candidate
        if current:
            lines.append(current)
    return lines or [""]


def _draw_lines(pdf, x, y, width, lines, line_height, align="L"):
    for index, line in enumerate(lines):
        pdf.set_xy(x + 2, y + 1.7 + (index * line_height))
        pdf.cell(width - 4, line_height, line, align=align)


def _section_title(pdf, title, intro=None):
    intro_height = 0
    if intro:
        pdf.set_font("Helvetica", "", 8.5)
        intro_lines = _wrap_text(pdf, intro, CONTENT_WIDTH)
        intro_height = len(intro_lines) * 4.4 + 2
    _ensure_room(pdf, 13 + intro_height)
    if pdf.get_y() > 54:
        pdf.ln(4)
    y = pdf.get_y()
    pdf.set_fill_color(*COLORS["accent_soft"])
    pdf.rect(pdf.l_margin, y, CONTENT_WIDTH, 9, style="F")
    pdf.set_fill_color(*COLORS["accent"])
    pdf.rect(pdf.l_margin, y, 2, 9, style="F")
    pdf.set_xy(pdf.l_margin + 5, y + 1)
    pdf.set_text_color(*COLORS["accent_dark"])
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(CONTENT_WIDTH - 7, 7, _safe_text(title))
    pdf.set_y(y + 11)
    if intro:
        _paragraph(pdf, intro, font_size=8.5, color=COLORS["muted"])
        pdf.ln(1)


def _paragraph(pdf, text, font_size=9, color=None, line_height=4.6):
    color = color or COLORS["body"]
    pdf.set_text_color(*color)
    pdf.set_font("Helvetica", "", font_size)
    lines = _wrap_text(pdf, text, CONTENT_WIDTH)
    _ensure_room(pdf, len(lines) * line_height)
    for line in lines:
        pdf.cell(0, line_height, line)
        pdf.ln(line_height)


def _metric_cards(pdf, cards):
    gap = 4
    card_width = (CONTENT_WIDTH - (gap * 3)) / 4
    card_height = 25
    _ensure_room(pdf, card_height + 2)
    y = pdf.get_y()
    for index, (label, value, detail) in enumerate(cards):
        x = pdf.l_margin + (index * (card_width + gap))
        pdf.set_fill_color(*COLORS["panel"])
        pdf.set_draw_color(*COLORS["line"])
        pdf.rect(x, y, card_width, card_height, style="DF")
        pdf.set_fill_color(*COLORS["accent"])
        pdf.rect(x, y, card_width, 2, style="F")
        rendered = _safe_text(value)
        value_size = 14
        pdf.set_font("Helvetica", "B", value_size)
        while value_size > 9 and pdf.get_string_width(rendered) > card_width - 6:
            value_size -= 1
            pdf.set_font("Helvetica", "B", value_size)
        pdf.set_text_color(*COLORS["accent_dark"])
        pdf.set_xy(x + 3, y + 5)
        pdf.cell(card_width - 6, 7, rendered)
        pdf.set_text_color(*COLORS["muted"])
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_xy(x + 3, y + 13)
        pdf.cell(card_width - 6, 4, _safe_text(label).upper())
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_xy(x + 3, y + 18)
        detail_text = _safe_text(detail, "")
        detail_lines = _wrap_text(pdf, detail_text, card_width - 6)
        pdf.cell(card_width - 6, 3.5, detail_lines[0] if detail_lines else "")
    pdf.set_y(y + card_height + 3)


def _progress_bar(pdf, label, value, detail=None):
    _ensure_room(pdf, 15)
    y = pdf.get_y()
    pdf.set_text_color(*COLORS["body"])
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(80, 5, _safe_text(label))
    rendered = _format_number(value, 1, "%")
    if detail:
        rendered = f"{rendered} | {_safe_text(detail)}"
    pdf.set_text_color(*COLORS["muted"])
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 5, rendered, align="R")
    pdf.ln(6)
    bar_y = pdf.get_y()
    pdf.set_fill_color(229, 234, 236)
    pdf.rect(pdf.l_margin, bar_y, CONTENT_WIDTH, 4, style="F")
    try:
        normalized = max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        normalized = 0.0
    if normalized:
        pdf.set_fill_color(*COLORS["accent"])
        pdf.rect(
            pdf.l_margin,
            bar_y,
            CONTENT_WIDTH * normalized / 100.0,
            4,
            style="F",
        )
    pdf.set_y(bar_y + 7)


def _styled_table(
    pdf,
    headers,
    widths,
    rows,
    aligns=None,
    font_size=8,
):
    aligns = aligns or ["L"] * len(headers)
    line_height = 4

    def draw_header():
        pdf.set_font("Helvetica", "B", 7.5)
        header_lines = [
            _wrap_text(pdf, header, width - 4)
            for header, width in zip(headers, widths)
        ]
        height = max(8, max(len(lines) for lines in header_lines) * line_height + 3)
        _ensure_room(pdf, height + 7)
        x = pdf.l_margin
        y = pdf.get_y()
        pdf.set_fill_color(*COLORS["accent_dark"])
        pdf.set_draw_color(*COLORS["accent_dark"])
        pdf.set_text_color(*COLORS["white"])
        for width, lines, align in zip(widths, header_lines, aligns):
            pdf.rect(x, y, width, height, style="DF")
            _draw_lines(pdf, x, y, width, lines, line_height, align)
            x += width
        pdf.set_y(y + height)

    draw_header()
    for row_index, row in enumerate(rows):
        pdf.set_font("Helvetica", "", font_size)
        rendered = [_safe_text(value) for value in row]
        cell_lines = [
            _wrap_text(pdf, value, width - 4)
            for value, width in zip(rendered, widths)
        ]
        height = max(7, max(len(lines) for lines in cell_lines) * line_height + 3)
        if pdf.get_y() + height > pdf.page_break_trigger:
            pdf.add_page()
            draw_header()
        x = pdf.l_margin
        y = pdf.get_y()
        fill = COLORS["table_alt"] if row_index % 2 else COLORS["white"]
        pdf.set_fill_color(*fill)
        pdf.set_draw_color(*COLORS["line"])
        pdf.set_text_color(*COLORS["body"])
        for width, lines, align in zip(widths, cell_lines, aligns):
            pdf.rect(x, y, width, height, style="DF")
            _draw_lines(pdf, x, y, width, lines, line_height, align)
            x += width
        pdf.set_y(y + height)
    pdf.ln(2)


def _key_value_table(pdf, rows):
    _styled_table(
        pdf,
        ["Field", "Recorded value"],
        [55, CONTENT_WIDTH - 55],
        rows,
        aligns=["L", "L"],
        font_size=8.5,
    )


def _callout(pdf, title, text, kind="info"):
    if not _has_value(text):
        return
    accent = COLORS["warning"] if kind == "warning" else COLORS["accent"]
    fill = COLORS["warning_fill"] if kind == "warning" else COLORS["accent_soft"]
    pdf.set_font("Helvetica", "", 8)
    lines = _wrap_text(pdf, text, CONTENT_WIDTH - 11)
    height = max(14, 8 + (len(lines) * 4))
    _ensure_room(pdf, height + 2)
    x = pdf.l_margin
    y = pdf.get_y()
    pdf.set_fill_color(*fill)
    pdf.set_draw_color(*COLORS["line"])
    pdf.rect(x, y, CONTENT_WIDTH, height, style="DF")
    pdf.set_fill_color(*accent)
    pdf.rect(x, y, 2, height, style="F")
    pdf.set_xy(x + 6, y + 2)
    pdf.set_text_color(*accent)
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(CONTENT_WIDTH - 9, 4, _safe_text(title))
    pdf.set_text_color(*COLORS["body"])
    pdf.set_font("Helvetica", "", 8)
    _draw_lines(pdf, x + 4, y + 6, CONTENT_WIDTH - 7, lines, 4)
    pdf.set_y(y + height + 3)


def _write_executive_summary(pdf, stats):
    gem_details = stats.get("gem_details")
    manufacturing = stats.get("manufacturing_plan") or {}
    if isinstance(gem_details, list):
        gem_count = len(gem_details)
    else:
        gem_count = manufacturing.get("gem_count")
    _section_title(
        pdf,
        "Executive Summary",
        "Key values from the selected optimization plan. All measurements are "
        "reported from the existing analysis output.",
    )
    _metric_cards(
        pdf,
        [
            (
                "Rough weight",
                _format_number(stats.get("raw_carats"), 2, " ct"),
                "Calibrated source weight",
            ),
            (
                "Estimated cut",
                _format_number(stats.get("estimated_cut_carats"), 2, " ct"),
                "Selected plan estimate",
            ),
            (
                "Predicted yield",
                _format_number(stats.get("yield_percent"), 1, "%"),
                "Cut-to-rough proportion",
            ),
            (
                "Recommended gems",
                _format_integer(gem_count),
                _safe_text(stats.get("recommended_shape"), "Plan unavailable"),
            ),
        ],
    )
    utilization = stats.get("space_utilization") or {}
    _progress_bar(
        pdf,
        "Rough-space utilization",
        utilization.get("occupied_percent"),
        "voxel occupancy estimate",
    )


def _write_analysis_overview(pdf, stats):
    dims = stats.get("rough_dimensions_mm") or stats.get("dimensions_mm")
    light = stats.get("light_analysis") or {}
    light_text = "Unavailable"
    if _has_value(light.get("score")) or _has_value(light.get("grade")):
        light_text = (
            f"{_format_number(light.get('score'), 1)}"
            f" ({_safe_text(light.get('grade'))})"
        )
    _section_title(pdf, "Source and Analysis Overview")
    _key_value_table(
        pdf,
        [
            ("Source volume", _format_number(stats.get("volume_cm3"), 3, " cm3")),
            ("Rough dimensions", _format_vector(dims, 2, " mm")),
            ("Cut mode", _humanize(stats.get("cut_mode"))),
            ("Preferred shape", _safe_text(stats.get("preferred_shape"))),
            ("Recommended plan", _safe_text(stats.get("recommended_shape"))),
            ("Light-performance result", light_text),
        ],
    )


def _write_defect_summary(pdf, stats):
    summary = stats.get("defect_summary") or {}
    detection = stats.get("defect_detection") or {}
    _section_title(
        pdf,
        "Defect Detection Summary",
        "Detector candidates, policy decisions and mapped no-cut evidence from "
        "the current job.",
    )
    if not summary and not detection:
        _callout(
            pdf,
            "Data unavailable",
            "No structured defect-detection summary was supplied for this report.",
            kind="warning",
        )
        return

    _styled_table(
        pdf,
        ["Measure", "Count", "Measure", "Count"],
        [53, 20, 64, 20],
        [
            (
                "Images scanned",
                _format_integer(summary.get("images_scanned")),
                "Raw candidates",
                _format_integer(detection.get("raw_prediction_count")),
            ),
            (
                "Visualization candidates",
                _format_integer(detection.get("visualization_count")),
                "Mapping masks",
                _format_integer(detection.get("mapping_count")),
            ),
            (
                "Mapped 3D points",
                _format_integer(summary.get("point_count")),
                "No-cut points",
                _format_integer(summary.get("no_cut_point_count")),
            ),
            (
                "No-cut masks",
                _format_integer(detection.get("no_cut_count")),
                "Rejected candidates",
                _format_integer(detection.get("rejected_count")),
            ),
        ],
        aligns=["L", "R", "L", "R"],
    )
    _key_value_table(
        pdf,
        [
            ("Detection policy", _safe_text(detection.get("policy"))),
            ("Taxonomy version", _safe_text(detection.get("taxonomy_version"))),
            ("Claim status", _safe_text(detection.get("claim_status"))),
            ("Evidence source", _safe_text(summary.get("source"))),
            ("Evidence status", _safe_text(summary.get("evidence_status"))),
            (
                "Mapping method",
                _safe_text(
                    summary.get("mapping_method")
                    or detection.get("mapping_method")
                ),
            ),
        ],
    )
    _callout(
        pdf,
        "Interpretation boundary",
        summary.get("claim_boundary") or detection.get("reason"),
        kind="warning",
    )
    for warning in detection.get("warnings") or []:
        _callout(pdf, "Detection warning", warning, kind="warning")


def _write_optimization_summary(pdf, stats):
    utilization = stats.get("space_utilization") or {}
    waste = stats.get("waste_reduction") or {}
    diagnostics = stats.get("optimizer_diagnostics") or {}
    facet = stats.get("facet_recommendation") or {}
    blade = (
        diagnostics.get("blade_clearance")
        or diagnostics.get("blade_gap")
        or {}
    )
    pocket = diagnostics.get("pocket_fill") or {}
    recovery = (
        diagnostics.get("preserve_fill")
        or diagnostics.get("repacked_search")
        or {}
    )
    remaining = diagnostics.get("remaining_free_space") or {}
    free = diagnostics.get("free_space_components") or {}
    clearance = diagnostics.get("clearance_model") or {}
    settings = diagnostics.get("optimizer_settings") or {}
    rough_clearance = diagnostics.get("rough_clearance") or {}
    strategy_options = [
        option for option in (stats.get("options") or [])
        if option.get("type") in {"Preserve + Fill", "Repacked Cuttable Plan"}
    ]

    _section_title(
        pdf,
        "Optimization and Yield",
        "Selected-plan yield, occupancy, blade clearance and heuristic facet "
        "orientation diagnostics.",
    )
    _progress_bar(pdf, "Predicted yield", stats.get("yield_percent"))
    _progress_bar(
        pdf,
        "Occupied usable volume",
        utilization.get("occupied_percent"),
    )
    if strategy_options:
        strategy_rows = []
        for option in strategy_options:
            comparison = (
                (option.get("optimizer_diagnostics") or {}).get(
                    "baseline_comparison"
                ) or {}
            )
            delta = comparison.get("yield_delta_percentage_points")
            strategy_rows.append((
                _safe_text(option.get("type")),
                _format_integer(option.get("gem_count")),
                _format_number(option.get("weight"), 2, " ct"),
                _format_number(option.get("yield"), 1, "%"),
                _format_number(
                    (option.get("space_utilization") or {}).get(
                        "occupied_percent"
                    ),
                    1,
                    "%",
                ),
                _format_number(delta, 2, " points"),
            ))
        _styled_table(
            pdf,
            [
                "Verified strategy",
                "Gems",
                "Weight",
                "Yield",
                "Utilization",
                "Yield vs internal baseline",
            ],
            [48, 15, 27, 23, 27, 42],
            strategy_rows,
            aligns=["L", "R", "R", "R", "R", "R"],
        )
    _styled_table(
        pdf,
        ["Optimization measure", "Recorded value", "Optimization measure", "Recorded value"],
        [51, 40, 51, 40],
        [
            (
                "Projected waste",
                _format_number(waste.get("projected_waste_percent"), 1, "%"),
                "Internal reference waste",
                _format_number(
                    waste.get("traditional_waste_baseline_percent"),
                    1,
                    "%",
                ),
            ),
            (
                "Difference vs reference",
                _format_number(
                    waste.get("reduction_vs_baseline_percent"),
                    1,
                    " points",
                ),
                "Occupied volume",
                _format_number(
                    utilization.get("occupied_volume_mesh_units"),
                    6,
                    " mesh units",
                ),
            ),
            (
                "Blade kerf",
                _format_number(
                    clearance.get("blade_kerf_mm", settings.get("blade_kerf_mm")),
                    3,
                    " mm",
                ),
                "Preform allowance",
                _format_number(
                    clearance.get(
                        "preform_margin_mm",
                        settings.get("preform_margin_mm"),
                    ),
                    3,
                    " mm",
                ),
            ),
            (
                "Protected corridor",
                _format_number(
                    clearance.get(
                        "protected_corridor_mm",
                        settings.get("protected_corridor_mm"),
                    ),
                    3,
                    " mm",
                ),
                "Voxelized corridor",
                _format_number(
                    clearance.get("voxelized_protected_corridor_mm"),
                    3,
                    " mm",
                ),
            ),
            (
                "Actual exported gap",
                _format_number(
                    clearance.get(
                        "actual_exported_gap_mm",
                        blade.get("actual_min_gap_mm"),
                    ),
                    3,
                    " mm",
                ),
                "Meets protected corridor",
                "Yes" if clearance.get(
                    "meets_exported_gap",
                    blade.get("meets_target"),
                ) is True else "No",
            ),
            (
                "Rough-surface inset",
                _format_number(
                    rough_clearance.get("target_clearance_mm"),
                    3,
                    " mm",
                ),
                "Pocket gems added",
                _format_integer(recovery.get("added_to_best", pocket.get("added"))),
            ),
            (
                "Free components",
                _format_integer(
                    remaining.get("component_count", free.get("count"))
                ),
                "Candidate placements",
                _format_integer(diagnostics.get("candidate_count")),
            ),
            (
                "Optimizer version",
                _safe_text(diagnostics.get("optimizer_version")),
                "Runtime",
                _format_number(diagnostics.get("runtime_seconds"), 2, " s"),
            ),
            (
                "Facet score",
                _format_number(facet.get("score"), 1),
                "Facet method",
                _humanize(facet.get("method")),
            ),
            (
                "Recommended table normal",
                _format_normal(facet.get("normal")),
                "Visible-defect estimate",
                _format_integer(facet.get("visible_defect_estimate")),
            ),
        ],
        aligns=["L", "R", "L", "R"],
    )
    _callout(
        pdf,
        "Internal waste-reference context",
        _waste_reference_note(waste),
        kind="info",
    )
    _callout(
        pdf,
        "Remaining free-space result",
        (
            ((remaining.get("components") or [{}])[0]).get("rejection_reason")
            or pocket.get("unused_space_reason")
        ),
        kind="warning" if recovery.get("timed_out", pocket.get("timed_out")) else "info",
    )
    components = remaining.get("components") or []
    if components:
        rows = []
        for index, component in enumerate(components[:6], start=1):
            rows.append((
                str(index),
                _format_normal(component.get("centre_mesh_units")),
                _format_vector(
                    component.get("bbox_extent_mesh_units"),
                    3,
                ),
                _format_number(
                    component.get("max_inscribed_diameter_mesh_units"),
                    4,
                ),
                _format_number(
                    component.get("estimated_inscribed_sphere_carat"),
                    2,
                    " ct",
                ),
                _safe_text(component.get("rejection_reason")),
            ))
        _styled_table(
            pdf,
            ["#", "Centre", "Dimensions", "Inscribed diameter", "Max carat", "Rejection status"],
            [8, 37, 34, 30, 22, 51],
            rows,
            aligns=["C", "L", "L", "R", "R", "L"],
        )
    _callout(
        pdf,
        "Facet recommendation basis",
        facet.get("reason"),
        kind="info",
    )


def _write_blueprints(pdf, job_folder, job_id, stats):
    rough_path = os.path.join(job_folder, "dense", "visual_aligned_stone.ply")
    cut_path = os.path.join(job_folder, "dense", "best_cut.ply")
    generated = []
    if not (os.path.isfile(rough_path) and os.path.isfile(cut_path)):
        _section_title(
            pdf,
            "Optimization Blueprints",
            "Orthographic fit views generated from the aligned rough mesh and "
            "selected cut-plan mesh when both files are available.",
        )
        _callout(
            pdf,
            "Blueprint unavailable",
            "The aligned rough mesh or selected cut-plan mesh was not supplied "
            "with this report input.",
            kind="warning",
        )
        return generated

    try:
        top_path = generate_blueprint(
            rough_path,
            cut_path,
            f"{job_id}_top.png",
            "top",
        )
        side_path = generate_blueprint(
            rough_path,
            cut_path,
            f"{job_id}_side.png",
            "side",
        )
        generated.extend([top_path, side_path])
    except Exception:
        _section_title(
            pdf,
            "Optimization Blueprints",
            "Orthographic fit views generated from the aligned rough mesh and "
            "selected cut-plan mesh when both files are available.",
        )
        _callout(
            pdf,
            "Blueprint unavailable",
            "The available mesh files could not be converted into report views.",
            kind="warning",
        )
        return generated

    _ensure_room(pdf, 116)
    _section_title(
        pdf,
        "Optimization Blueprints",
        "Orthographic fit views generated from the aligned rough mesh and selected "
        "cut-plan mesh when both files are available.",
    )
    y = pdf.get_y() + 2
    panel_width = (CONTENT_WIDTH - 6) / 2
    pdf.set_fill_color(*COLORS["panel"])
    pdf.set_draw_color(*COLORS["line"])
    pdf.rect(pdf.l_margin, y, panel_width, 78, style="DF")
    pdf.rect(pdf.l_margin + panel_width + 6, y, panel_width, 78, style="DF")
    pdf.image(top_path, x=pdf.l_margin + 3, y=y + 3, w=panel_width - 6, h=70)
    pdf.image(
        side_path,
        x=pdf.l_margin + panel_width + 9,
        y=y + 3,
        w=panel_width - 6,
        h=70,
    )
    pdf.set_y(y + 81)
    dims = stats.get("rough_dimensions_mm") or stats.get("dimensions_mm")
    _paragraph(
        pdf,
        f"Visual fit reference. Rough dimensions: {_format_vector(dims, 2, ' mm')}.",
        font_size=7.5,
        color=COLORS["muted"],
        line_height=4,
    )
    return generated


def _write_gem_details(pdf, gem_details):
    _section_title(
        pdf,
        "Gem Details",
        "Each row represents one exported gem in the selected cut plan. Weight "
        "and yield values are proportional estimates from the calibrated rough volume.",
    )
    if not isinstance(gem_details, list) or not gem_details:
        _callout(
            pdf,
            "Gem details unavailable",
            "No individual gem records were supplied for this report.",
            kind="warning",
        )
        return

    rows = []
    placements = []
    for gem in gem_details:
        dims = gem.get("dimensions_mm")
        rows.append(
            (
                _format_integer(gem.get("index")),
                _safe_text(gem.get("shape")),
                _format_number(gem.get("weight_ct"), 2),
                _format_number(gem.get("yield_percent"), 2, "%"),
                _format_number(gem.get("plan_share_percent"), 1, "%"),
                _format_number(gem.get("scale"), 4),
                _format_vector(dims, 2),
            )
        )
        placements.append(
            (
                _format_integer(gem.get("index")),
                _format_vector(gem.get("center_mm"), 2, " mm"),
                _format_number(
                    gem.get("volume_mesh_units"),
                    6,
                    " mesh units",
                ),
                _format_number(
                    gem.get("surface_clearance_mesh_units"),
                    6,
                    " mesh units",
                ),
                _safe_text(gem.get("file")),
            )
        )

    _styled_table(
        pdf,
        ["#", "Shape", "Weight ct", "Yield", "Plan share", "Scale", "Dimensions L x W x H mm"],
        [9, 36, 21, 20, 22, 19, 55],
        rows,
        aligns=["R", "L", "R", "R", "R", "R", "R"],
        font_size=7.5,
    )
    _paragraph(
        pdf,
        "Placement and export references",
        font_size=8.5,
        color=COLORS["accent_dark"],
        line_height=5,
    )
    _styled_table(
        pdf,
        ["Gem", "Center coordinates", "Volume", "Surface clearance", "Export file"],
        [13, 49, 35, 39, 46],
        placements,
        aligns=["R", "R", "R", "R", "L"],
        font_size=7,
    )


def _write_manufacturing_plan(pdf, manufacturing):
    if not manufacturing:
        return
    _section_title(
        pdf,
        "Cuttable Rough-Separation Plan",
        "Geometry-verified straight, full-through saw guidance for the selected "
        "strategy. This is not machine-ready CNC output.",
    )
    ready = manufacturing.get("machine_ready")
    ready_text = (
        "Yes" if ready is True else "No" if ready is False else "Unavailable"
    )
    _key_value_table(
        pdf,
        [
            ("Plan version", _safe_text(manufacturing.get("version"))),
            ("Verification status", _humanize(manufacturing.get("status"))),
            ("Gem count", _format_integer(manufacturing.get("gem_count"))),
            (
                "Blade kerf",
                _format_number(
                    (manufacturing.get("settings") or {}).get("blade_kerf_mm"),
                    3,
                    " mm",
                ),
            ),
            (
                "Preform allowance",
                _format_number(
                    (manufacturing.get("settings") or {}).get("preform_margin_mm"),
                    3,
                    " mm",
                ),
            ),
            (
                "Maximum usable saw depth",
                _format_number(
                    (manufacturing.get("settings") or {}).get("max_cut_depth_mm"),
                    2,
                    " mm",
                ),
            ),
            ("Verified cuts", _format_integer(len(manufacturing.get("sequence") or []))),
            (
                "Maximum required depth",
                _format_number(manufacturing.get("maximum_required_depth_mm"), 2, " mm"),
            ),
            (
                "Minimum envelope clearance",
                _format_number(
                    manufacturing.get("minimum_envelope_clearance_mm"), 3, " mm"
                ),
            ),
            (
                "Cuttable yield",
                _format_number(manufacturing.get("cuttable_yield_percent"), 1, "%"),
            ),
            (
                "Difference from geometric plan",
                _format_number(
                    manufacturing.get("yield_difference_percentage_points"), 1, " pts"
                ),
            ),
            ("Machine ready", ready_text),
        ],
    )
    _callout(
        pdf,
        "Machine-readiness boundary",
        manufacturing.get("machine_ready_reason"),
        kind="warning",
    )
    sequence = manufacturing.get("sequence") or []
    if sequence:
        rows = []
        for step in sequence:
            rows.append(
                (
                    _format_integer(step.get("step")),
                    _safe_text(step.get("parent_piece_id")),
                    _humanize(step.get("operation")),
                    _format_number(
                        step.get("required_depth_mm"),
                        2,
                        " mm",
                    ),
                    _format_number(
                        step.get("minimum_envelope_clearance_mm"),
                        3,
                        " mm",
                    ),
                )
            )
        _styled_table(
            pdf,
            ["Step", "Parent piece", "Operation", "Saw depth", "Envelope clearance"],
            [14, 43, 57, 34, 34],
            rows,
            aligns=["R", "L", "L", "R", "R"],
            font_size=7.5,
        )

        for step in sequence:
            _write_cut_step_page(pdf, manufacturing, step)


def _cut_contour_projection(contour):
    points = np.asarray(contour or [], dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 3:
        return None
    centered = points - points.mean(axis=0)
    try:
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    return centered @ axes[:2].T


def _draw_cut_step_diagram(pdf, step, y=None):
    x = PAGE_MARGIN
    y = max(float(y if y is not None else pdf.get_y() + 4), 64.0)
    width, height = CONTENT_WIDTH, min(78.0, 230.0 - y)
    pdf.set_draw_color(*COLORS["line"])
    pdf.set_fill_color(*COLORS["panel"])
    pdf.rect(x, y, width, height, style="DF")
    projected = _cut_contour_projection(step.get("section_contour"))
    if projected is None:
        pdf.set_xy(x, y + height / 2)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*COLORS["muted"])
        pdf.cell(width, 5, "Section contour unavailable", align="C")
        return y + height

    bounds_min = projected.min(axis=0)
    bounds_max = projected.max(axis=0)
    extent = np.maximum(bounds_max - bounds_min, 1e-9)
    scale = min((width - 22) / extent[0], (height - 22) / extent[1])
    centered = (projected - (bounds_min + bounds_max) * 0.5) * scale
    screen = np.column_stack([
        x + width * 0.5 + centered[:, 0],
        y + height * 0.5 - centered[:, 1],
    ])
    pdf.set_draw_color(*COLORS["accent"])
    pdf.set_line_width(0.8)
    for index in range(len(screen)):
        start = screen[index]
        end = screen[(index + 1) % len(screen)]
        pdf.line(float(start[0]), float(start[1]), float(end[0]), float(end[1]))

    center_x, center_y = x + width * 0.5, y + height * 0.5
    pdf.set_draw_color(*COLORS["warning"])
    pdf.set_line_width(0.45)
    pdf.line(x + 8, center_y, x + width - 8, center_y)
    pdf.set_fill_color(*COLORS["warning"])
    pdf.line(center_x, center_y, center_x + 24, center_y - 13)
    pdf.line(center_x + 24, center_y - 13, center_x + 19, center_y - 12)
    pdf.line(center_x + 24, center_y - 13, center_x + 22, center_y - 8)
    pdf.set_xy(x + 4, y + 3)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*COLORS["accent_dark"])
    pdf.cell(width - 8, 4, "Orthographic cut-section contour and feed direction")
    return y + height


def _write_cut_step_page(pdf, manufacturing, step):
    pdf.add_page()
    _section_title(
        pdf,
        f"Saw Cut {step.get('step')}: {_safe_text(step.get('parent_piece_id'))}",
        "Printable orientation sheet for one straight, full-through rough "
        "separation operation.",
    )
    plane = step.get("plane") or {}
    orientation = plane.get("orientation") or {}
    feed_orientation = step.get("feed_orientation") or {}
    results = step.get("result_piece_ids") or []
    _key_value_table(
        pdf,
        [
            ("Plane origin", _format_normal(plane.get("origin"))),
            ("Plane normal", _format_normal(plane.get("normal"))),
            (
                "Plane orientation",
                f"Az {_format_number(orientation.get('azimuth_deg'), 2)} deg / "
                f"El {_format_number(orientation.get('elevation_deg'), 2)} deg",
            ),
            ("Feed direction", _format_normal(step.get("feed_direction"))),
            (
                "Feed orientation",
                f"Az {_format_number(feed_orientation.get('azimuth_deg'), 2)} deg / "
                f"El {_format_number(feed_orientation.get('elevation_deg'), 2)} deg",
            ),
            (
                "Kerf / preform",
                f"{_format_number((step.get('kerf_slab') or {}).get('thickness_mm'), 3)} mm / "
                f"{_format_number((manufacturing.get('settings') or {}).get('preform_margin_mm'), 3)} mm",
            ),
            ("Required saw depth", _format_number(step.get("required_depth_mm"), 2, " mm")),
            ("Resulting pieces", ", ".join(_safe_text(item) for item in results)),
            (
                "Negative-side gems",
                ", ".join(step.get("negative_side_gems") or []),
            ),
            (
                "Positive-side gems",
                ", ".join(step.get("positive_side_gems") or []),
            ),
        ],
    )
    diagram_bottom = _draw_cut_step_diagram(pdf, step, pdf.get_y() + 4)
    pdf.set_y(diagram_bottom + 5)
    _paragraph(
        pdf,
        "Operator checks",
        font_size=9,
        color=COLORS["accent_dark"],
        line_height=5,
    )
    for check in step.get("operator_checks") or []:
        _paragraph(pdf, f"- {_safe_text(check)}", font_size=8, line_height=4.5)
    _callout(
        pdf,
        "Safety and validation boundary",
        "Confirm stable support, clamping, blade condition, visible defects, and "
        "clearance before cutting. This plan has not been physically validated "
        "and is not CNC or G-code output.",
        kind="warning",
    )


def _write_notes_and_limitations(pdf, stats):
    notes = []
    waste = stats.get("waste_reduction") or {}
    detection = stats.get("defect_detection") or {}
    summary = stats.get("defect_summary") or {}
    diagnostics = stats.get("optimizer_diagnostics") or {}
    manufacturing = stats.get("manufacturing_plan") or {}
    pocket = diagnostics.get("pocket_fill") or {}

    for title, value in (
        ("Internal waste reference", _waste_reference_note(waste)),
        ("Defect evidence", summary.get("claim_boundary")),
        ("Detection policy", detection.get("reason")),
        ("Remaining free space", pocket.get("unused_space_reason")),
        ("Manufacturing output", manufacturing.get("machine_ready_reason")),
    ):
        if _has_value(value):
            notes.append((title, _safe_text(value)))
    for warning in detection.get("warnings") or []:
        notes.append(("Detection warning", _safe_text(warning)))

    deduplicated = []
    seen = set()
    for title, value in notes:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            deduplicated.append((title, value))

    _section_title(
        pdf,
        "Notes, Warnings and Limitations",
        "Interpretation boundaries supplied by the analysis are repeated here "
        "for technical review.",
    )
    if not deduplicated:
        _callout(
            pdf,
            "No additional notes",
            "No optional warnings or limitation statements were supplied.",
            kind="info",
        )
        return
    for title, value in deduplicated:
        _callout(pdf, title, value, kind="warning")


def create_pdf(job_folder, job_id, stats):
    job_folder = os.path.abspath(job_folder)
    os.makedirs(job_folder, exist_ok=True)
    stats = stats if isinstance(stats, dict) else {}
    generated_at = datetime.now()
    pdf = PDFReport(job_id, generated_at)
    pdf.add_page()

    blueprint_paths = []
    _write_executive_summary(pdf, stats)
    _write_analysis_overview(pdf, stats)
    _write_defect_summary(pdf, stats)
    _write_optimization_summary(pdf, stats)
    blueprint_paths = _write_blueprints(pdf, job_folder, job_id, stats)
    _write_gem_details(pdf, stats.get("gem_details"))
    _write_manufacturing_plan(pdf, stats.get("manufacturing_plan") or {})
    _write_notes_and_limitations(pdf, stats)

    output_path = os.path.join(job_folder, CANONICAL_PDF_FILENAME)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".report-",
            suffix=".pdf",
            dir=job_folder,
            delete=False,
        ) as handle:
            temporary_path = handle.name
        pdf.output(temporary_path)
        if not is_valid_pdf(temporary_path):
            raise ValueError("Generated report does not have a valid PDF signature.")
        os.replace(temporary_path, output_path)
        temporary_path = None
        return output_path
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
        for path in blueprint_paths:
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass
