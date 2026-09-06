"""Tiny blind polygon annotator for the frozen defect final-test subset.

The tool writes only YOLO segmentation label text files at the exact paths from
the blind annotation package. It does not import or run any model code.
"""

from __future__ import annotations

import csv
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Any

from PIL import Image, ImageTk

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.common import sha256_file
from research.defect_validation.independent_final_test import (
    CLASS_NAMES,
    EXPECTED_IMAGE_COUNT,
    EXPECTED_MEMBERSHIP_SHA256,
    default_package_path,
    repo_root,
    validate_frozen_manifest,
)

CANVAS_WIDTH = 1100
CANVAS_HEIGHT = 720
COLORS = {0: "#ff3b30", 1: "#007aff"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _load_manifest_rows(root: Path) -> list[dict[str, str]]:
    package = default_package_path(root)
    manifest_path = package / "IMMUTABLE_TEST_IMAGES_MANIFEST.csv"
    expected_path = package / "EXPECTED_LABEL_OUTPUTS.csv"
    rows = _read_csv(manifest_path)
    expected = {row["image_id"]: row for row in _read_csv(expected_path)}
    if len(rows) != EXPECTED_IMAGE_COUNT:
        raise ValueError(f"Expected 45 frozen images, found {len(rows)}.")
    for row in rows:
        required = expected.get(row["image_id"], {})
        if required.get("required_output_label_path") != row.get("expected_label_path"):
            raise ValueError(f"Expected label path mismatch for {row['image_id']}.")
    return rows


def _parse_label(path: Path) -> list[dict[str, Any]]:
    polygons: list[dict[str, Any]] = []
    if not path.is_file():
        return polygons
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        class_id = int(float(tokens[0]))
        coords = [float(value) for value in tokens[1:]]
        points = [(coords[index], coords[index + 1]) for index in range(0, len(coords), 2)]
        polygons.append({"class_id": class_id, "points": points})
    return polygons


class BlindPolygonAnnotator:
    def __init__(self, root: Path) -> None:
        self.repo = root
        frozen = validate_frozen_manifest(root)
        if (
            frozen["status"] != "pass"
            or frozen["membership_sha256"] != EXPECTED_MEMBERSHIP_SHA256
        ):
            raise ValueError("Frozen final-test manifest failed validation.")

        self.rows = _load_manifest_rows(root)
        self.index = 0
        self.polygons: list[dict[str, Any]] = []
        self.current_points: list[tuple[float, float]] = []
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.image_size = (0, 0)
        self.photo: ImageTk.PhotoImage | None = None

        self.window = tk.Tk()
        self.window.title("Blind Defect Annotation - Frozen Independent Test")
        self.class_var = tk.IntVar(value=0)

        self.header = tk.Label(self.window, text="", anchor="w", font=("Segoe UI", 11, "bold"))
        self.header.pack(fill="x", padx=8, pady=(8, 2))
        self.status = tk.Label(self.window, text="", anchor="w")
        self.status.pack(fill="x", padx=8, pady=(0, 6))

        self.canvas = tk.Canvas(
            self.window,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            background="#222222",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)

        controls = tk.Frame(self.window)
        controls.pack(fill="x", padx=8, pady=8)
        tk.Radiobutton(
            controls,
            text="0 fracture",
            variable=self.class_var,
            value=0,
        ).pack(side="left")
        tk.Radiobutton(
            controls,
            text="1 inclusion",
            variable=self.class_var,
            value=1,
        ).pack(side="left", padx=(8, 16))
        tk.Button(controls, text="Finish Polygon", command=self.finish_polygon).pack(side="left")
        tk.Button(controls, text="Undo Point/Polygon", command=self.undo).pack(side="left", padx=4)
        tk.Button(controls, text="No Visible Defect", command=self.mark_negative).pack(side="left", padx=4)
        tk.Button(controls, text="Save", command=self.save_current).pack(side="left", padx=(16, 4))
        tk.Button(controls, text="Previous", command=self.previous_image).pack(side="left", padx=4)
        tk.Button(controls, text="Next", command=self.next_image).pack(side="left", padx=4)

        self.help = tk.Label(
            self.window,
            text=(
                "Left click adds polygon points. Right click or Enter finishes. "
                "0/1 selects class. Ctrl+S saves. N/P saves and navigates."
            ),
            anchor="w",
        )
        self.help.pack(fill="x", padx=8, pady=(0, 8))

        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", lambda _event: self.finish_polygon())
        self.window.bind("<Return>", lambda _event: self.finish_polygon())
        self.window.bind("<Control-s>", lambda _event: self.save_current())
        self.window.bind("n", lambda _event: self.next_image())
        self.window.bind("p", lambda _event: self.previous_image())
        self.window.bind("z", lambda _event: self.undo())
        self.window.bind("0", lambda _event: self.class_var.set(0))
        self.window.bind("1", lambda _event: self.class_var.set(1))

        self.load_image(0)

    def row(self) -> dict[str, str]:
        return self.rows[self.index]

    def image_path(self) -> Path:
        return self.repo / self.row()["workspace_image_path"]

    def label_path(self) -> Path:
        return self.repo / self.row()["expected_label_path"]

    def load_image(self, index: int) -> None:
        self.index = max(0, min(len(self.rows) - 1, index))
        row = self.row()
        path = self.image_path()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != row["workspace_sha256"]:
            raise ValueError(f"Frozen image hash mismatch: {path}")

        image = Image.open(path).convert("RGB")
        self.image_size = image.size
        scale = min(CANVAS_WIDTH / image.width, CANVAS_HEIGHT / image.height)
        self.display_scale = scale
        display_size = (int(image.width * scale), int(image.height * scale))
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        display = image.resize(display_size, resample)
        self.photo = ImageTk.PhotoImage(display)
        self.display_offset = (
            (CANVAS_WIDTH - display_size[0]) // 2,
            (CANVAS_HEIGHT - display_size[1]) // 2,
        )
        self.polygons = _parse_label(self.label_path())
        self.current_points = []
        self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("all")
        x0, y0 = self.display_offset
        self.canvas.create_image(x0, y0, anchor="nw", image=self.photo)
        for polygon in self.polygons:
            self.draw_polygon(polygon, closed=True)
        if self.current_points:
            color = COLORS[self.class_var.get()]
            points = [self.to_canvas(point) for point in self.current_points]
            for point in points:
                self.canvas.create_oval(point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3, fill=color)
            if len(points) > 1:
                self.canvas.create_line(*[coord for point in points for coord in point], fill=color, width=2)
        label_exists = "saved" if self.label_path().is_file() else "not saved"
        self.header.config(
            text=(
                f"{self.index + 1}/{len(self.rows)}  "
                f"{self.row()['specimen_id']}  {self.row()['image_filename']}"
            )
        )
        self.status.config(
            text=(
                f"Label: {self.row()['expected_label_path']} ({label_exists}); "
                f"polygons: {len(self.polygons)}"
            )
        )

    def to_canvas(self, point: tuple[float, float]) -> tuple[float, float]:
        width, height = self.image_size
        x0, y0 = self.display_offset
        return (
            x0 + point[0] * width * self.display_scale,
            y0 + point[1] * height * self.display_scale,
        )

    def from_canvas(self, x: float, y: float) -> tuple[float, float] | None:
        width, height = self.image_size
        x0, y0 = self.display_offset
        image_x = (x - x0) / self.display_scale
        image_y = (y - y0) / self.display_scale
        if image_x < 0 or image_y < 0 or image_x > width or image_y > height:
            return None
        return (
            max(0.0, min(1.0, image_x / width)),
            max(0.0, min(1.0, image_y / height)),
        )

    def draw_polygon(self, polygon: dict[str, Any], closed: bool) -> None:
        color = COLORS.get(int(polygon["class_id"]), "#ffcc00")
        points = [self.to_canvas(point) for point in polygon["points"]]
        if len(points) >= 3 and closed:
            self.canvas.create_polygon(
                *[coord for point in points for coord in point],
                outline=color,
                fill="",
                width=2,
            )
        if len(points) >= 2:
            self.canvas.create_line(
                *[coord for point in points for coord in point],
                fill=color,
                width=2,
            )
        for point in points:
            self.canvas.create_oval(point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3, fill=color)

    def add_point(self, event: tk.Event) -> None:
        point = self.from_canvas(float(event.x), float(event.y))
        if point is None:
            return
        self.current_points.append(point)
        self.redraw()

    def finish_polygon(self) -> None:
        if len(self.current_points) < 3:
            if self.current_points:
                messagebox.showinfo("Polygon needs points", "A polygon needs at least three points.")
            return
        self.polygons.append(
            {"class_id": self.class_var.get(), "points": list(self.current_points)}
        )
        self.current_points = []
        self.redraw()

    def undo(self) -> None:
        if self.current_points:
            self.current_points.pop()
        elif self.polygons:
            self.polygons.pop()
        self.redraw()

    def mark_negative(self) -> None:
        if not messagebox.askyesno(
            "No visible defect",
            "Save this image as manually inspected with no visible defect?",
        ):
            return
        self.polygons = []
        self.current_points = []
        self.save_current()

    def save_current(self) -> bool:
        if self.current_points:
            if not messagebox.askyesno(
                "Unfinished polygon",
                "Discard the unfinished polygon points and save completed polygons?",
            ):
                return False
            self.current_points = []
        label = self.label_path()
        label.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for polygon in self.polygons:
            coords = []
            for x, y in polygon["points"]:
                coords.extend([f"{x:.6f}", f"{y:.6f}"])
            lines.append(f"{int(polygon['class_id'])} {' '.join(coords)}")
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.redraw()
        return True

    def next_image(self) -> None:
        if not self.save_current():
            return
        if self.index < len(self.rows) - 1:
            self.load_image(self.index + 1)

    def previous_image(self) -> None:
        if not self.save_current():
            return
        if self.index > 0:
            self.load_image(self.index - 1)

    def run(self) -> None:
        self.window.mainloop()


def main() -> int:
    app = BlindPolygonAnnotator(repo_root())
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
