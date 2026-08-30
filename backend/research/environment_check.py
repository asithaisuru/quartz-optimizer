"""Read-only runtime and dependency readiness inspection."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, Callable

from .common import atomic_write_json, utc_timestamp
from .readiness_common import (
    atomic_write_csv,
    atomic_write_text,
    load_json_config,
    resolve_path,
    write_manifest,
)

DEPENDENCIES = (
    ("numpy", "NumPy", "numpy", ("numpy",)),
    ("pillow", "Pillow", "PIL", ("Pillow",)),
    ("pyyaml", "PyYAML", "yaml", ("PyYAML",)),
    ("opencv", "OpenCV", "cv2", ("opencv-python", "opencv-python-headless")),
    ("torch", "Torch", "torch", ("torch",)),
    ("ultralytics", "Ultralytics", "ultralytics", ("ultralytics",)),
    ("trimesh", "Trimesh", "trimesh", ("trimesh",)),
    ("scipy", "SciPy", "scipy", ("scipy",)),
    ("scikit_learn", "Scikit-learn", "sklearn", ("scikit-learn",)),
    ("matplotlib", "Matplotlib", "matplotlib", ("matplotlib",)),
)
ALLOWED_STATUSES = {
    "available",
    "missing",
    "incompatible",
    "unavailable",
    "not_checked",
}


def load_environment_config(
    path: str | Path,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    return load_json_config(path, output_override=output_override)


def _safe_executable() -> str:
    value = str(Path(sys.executable).resolve())
    try:
        home = str(Path.home().resolve())
        if value.casefold().startswith(home.casefold()):
            value = "<user_home>" + value[len(home):]
    except OSError:
        pass
    return value


def _version(distributions: tuple[str, ...]) -> str | None:
    for distribution in distributions:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _dependency_row(
    key: str,
    display_name: str,
    import_name: str,
    distributions: tuple[str, ...],
    spec_finder: Callable[[str], Any] = importlib.util.find_spec,
    version_reader: Callable[[tuple[str, ...]], str | None] = _version,
) -> dict[str, Any]:
    try:
        available = spec_finder(import_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        available = False
    if not available:
        return {
            "key": key,
            "name": display_name,
            "import_name": import_name,
            "status": "missing",
            "version": None,
            "reason": "Import module was not found.",
        }
    try:
        version = version_reader(distributions)
    except Exception as exc:
        return {
            "key": key,
            "name": display_name,
            "import_name": import_name,
            "status": "unavailable",
            "version": None,
            "reason": f"Version could not be read: {exc}",
        }
    return {
        "key": key,
        "name": display_name,
        "import_name": import_name,
        "status": "available",
        "version": version,
        "reason": None if version else "Installed version metadata is unavailable.",
    }


def audit_environment(
    config: dict[str, Any],
    *,
    spec_finder: Callable[[str], Any] = importlib.util.find_spec,
    version_reader: Callable[[tuple[str, ...]], str | None] = _version,
    executable_finder: Callable[[str], str | None] = shutil.which,
) -> dict[str, Any]:
    started = utc_timestamp()
    configured = config.get("dependencies")
    allowed = set(configured) if isinstance(configured, list) else None
    rows = [
        _dependency_row(
            key,
            display,
            import_name,
            distributions,
            spec_finder,
            version_reader,
        )
        for key, display, import_name, distributions in DEPENDENCIES
        if allowed is None or key in allowed
    ]

    colmap_name = str(config.get("colmap_executable", "colmap"))
    colmap_path = executable_finder(colmap_name)
    rows.append(
        {
            "key": "colmap",
            "name": "COLMAP executable",
            "import_name": None,
            "status": "available" if colmap_path else "missing",
            "version": None,
            "reason": None if colmap_path else "Executable was not found on PATH.",
            "executable": Path(colmap_path).name if colmap_path else None,
        }
    )

    python_minimum = tuple(
        int(part)
        for part in str(config.get("minimum_python_version", "3.10")).split(".")
    )
    python_status = (
        "available"
        if sys.version_info[: len(python_minimum)] >= python_minimum
        else "incompatible"
    )
    python_row = {
        "key": "python",
        "name": "Python",
        "status": python_status,
        "version": platform.python_version(),
        "reason": (
            None
            if python_status == "available"
            else f"Python {config.get('minimum_python_version')} or newer is required."
        ),
    }
    rows.insert(0, python_row)

    cuda = {
        "status": "unavailable",
        "available": None,
        "gpu_name": None,
        "reason": "Torch is not available, so CUDA was not checked.",
    }
    torch_row = next((row for row in rows if row["key"] == "torch"), None)
    if torch_row and torch_row["status"] == "available":
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            cuda = {
                "status": "available",
                "available": cuda_available,
                "gpu_name": (
                    str(torch.cuda.get_device_name(0))
                    if cuda_available
                    else None
                ),
                "reason": None,
            }
        except Exception as exc:
            cuda = {
                "status": "unavailable",
                "available": None,
                "gpu_name": None,
                "reason": f"CUDA inspection failed: {exc}",
            }

    missing = [row["key"] for row in rows if row["status"] == "missing"]
    incompatible = [
        row["key"] for row in rows if row["status"] == "incompatible"
    ]
    return {
        "schema_version": "1.0",
        "audit_type": "environment_readiness",
        "started_utc": started,
        "ended_utc": utc_timestamp(),
        "system": {
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "architecture": platform.machine() or f"{struct.calcsize('P') * 8}-bit",
            "python_executable": _safe_executable(),
            "cpu_identifier": platform.processor() or platform.machine() or "unavailable",
            "cuda": cuda,
        },
        "dependencies": rows,
        "summary": {
            "status": "available" if not missing and not incompatible else "incomplete",
            "available_count": sum(row["status"] == "available" for row in rows),
            "missing_count": len(missing),
            "incompatible_count": len(incompatible),
            "missing": missing,
            "incompatible": incompatible,
            "final_model_runtime_ready": not missing and not incompatible,
        },
        "environment_variables_recorded": False,
        "claim_boundary": (
            "Dependency availability is not model performance. Missing software "
            "is reported as missing, never as a zero-valued metric."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Environment Readiness Check",
        "",
        f"- Status: {summary['status']}",
        f"- Available: {summary['available_count']}",
        f"- Missing: {summary['missing_count']}",
        f"- Incompatible: {summary['incompatible_count']}",
        f"- Final model runtime ready: {str(summary['final_model_runtime_ready']).lower()}",
        "- Environment variables recorded: false",
        "",
        "| Dependency | Status | Version |",
        "|---|---|---|",
    ]
    for row in report["dependencies"]:
        lines.append(
            f"| {row['name']} | {row['status']} | {row.get('version') or 'unavailable'} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def run_environment_check(
    config: dict[str, Any],
    output_override: str | Path | None = None,
    command: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    report = audit_environment(config)
    repo_root = Path(config["_repo_root"])
    output = resolve_path(
        output_override or config.get("output_directory"),
        repo_root,
    )
    if output is None:
        raise ValueError("An output directory is required.")
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "report": output / "environment_check.json",
        "summary": output / "environment_summary.csv",
        "markdown": output / "audit_summary.md",
        "manifest": output / "manifest.json",
    }
    atomic_write_json(files["report"], report)
    atomic_write_csv(
        files["summary"],
        report["dependencies"],
        ("key", "name", "status", "version", "reason", "executable"),
    )
    atomic_write_text(files["markdown"], _markdown(report))
    write_manifest(
        path=files["manifest"],
        repo_root=repo_root,
        tool_name="quartz_environment_readiness_check",
        command=command,
        tool_sources=(
            "backend/research/environment_check.py",
            "backend/research/readiness_common.py",
            "backend/research_benchmark.py",
        ),
        inputs=(("configuration", config.get("_config_path")),),
        outputs=(
            ("report", files["report"]),
            ("summary", files["summary"]),
            ("markdown", files["markdown"]),
        ),
        extra={"environment_variables_recorded": False},
    )
    return report, files
