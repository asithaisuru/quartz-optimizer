"""Probe an already isolated ML environment without loading model weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PACKAGES = (
    ("numpy", "numpy"),
    ("Pillow", "PIL"),
    ("PyYAML", "yaml"),
    ("opencv-python", "cv2"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("ultralytics", "ultralytics"),
    ("trimesh", "trimesh"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_environment(
    *,
    location_category: str,
    install_commands: list[str],
) -> tuple[dict[str, Any], str]:
    rows = []
    for distribution, module_name in PACKAGES:
        version = _version(distribution)
        try:
            importlib.import_module(module_name)
            status = "available"
            reason = None
        except Exception as exc:
            status = "unavailable"
            reason = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "distribution": distribution,
                "module": module_name,
                "version": version,
                "status": status,
                "reason": reason,
            }
        )

    smoke = {}
    cuda = {
        "status": "unavailable",
        "available": None,
        "gpu_name": None,
        "torch_cuda_version": None,
    }
    try:
        import torch

        result = (torch.tensor([1.0, 2.0]) * 2).tolist()
        smoke["small_tensor_operation"] = {
            "status": "pass" if result == [2.0, 4.0] else "fail",
            "result": result,
        }
        available = bool(torch.cuda.is_available())
        cuda = {
            "status": "available",
            "available": available,
            "gpu_name": torch.cuda.get_device_name(0) if available else None,
            "torch_cuda_version": torch.version.cuda,
        }
    except Exception as exc:
        smoke["small_tensor_operation"] = {
            "status": "fail",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    try:
        from ultralytics import YOLO  # noqa: F401

        smoke["ultralytics_yolo_api_import"] = {
            "status": "pass",
            "model_instantiated": False,
            "weights_downloaded": False,
        }
    except Exception as exc:
        smoke["ultralytics_yolo_api_import"] = {
            "status": "fail",
            "reason": f"{type(exc).__name__}: {exc}",
            "model_instantiated": False,
            "weights_downloaded": False,
        }
    try:
        import matplotlib

        matplotlib.use("Agg")
        smoke["matplotlib_non_interactive"] = {
            "status": "pass",
            "backend": matplotlib.get_backend(),
        }
    except Exception as exc:
        smoke["matplotlib_non_interactive"] = {
            "status": "fail",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        check=False,
    )
    freeze_text = "\n".join(
        sorted(
            line.strip()
            for line in freeze.stdout.splitlines()
            if line.strip()
        )
    ) + "\n"
    conflicts = [
        line.strip()
        for line in (pip_check.stdout + pip_check.stderr).splitlines()
        if line.strip() and line.strip() != "No broken requirements found."
    ]
    imports_ready = all(row["status"] == "available" for row in rows)
    smoke_ready = all(item.get("status") == "pass" for item in smoke.values())
    ready = bool(
        location_category == "external_isolated"
        and imports_ready
        and smoke_ready
        and pip_check.returncode == 0
    )
    environment_hash = hashlib.sha256(
        (
            platform.python_version()
            + "\0"
            + freeze_text
            + "\0"
            + json.dumps(smoke, sort_keys=True)
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "1.0",
        "audit_type": "isolated_ml_environment",
        "environment_location_category": location_category,
        "actual_environment_path": str(Path(sys.prefix).resolve()),
        "installation_commands": install_commands,
        "system": {
            "operating_system": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "architecture": platform.machine(),
            "cuda": cuda,
        },
        "dependencies": rows,
        "smoke_tests": smoke,
        "pip_check": {
            "status": "pass" if pip_check.returncode == 0 else "fail",
            "return_code": pip_check.returncode,
            "conflicts": conflicts,
        },
        "summary": {
            "status": "pass" if ready else "blocked",
            "final_model_runtime_ready": ready,
            "available_dependency_count": sum(
                row["status"] == "available" for row in rows
            ),
            "dependency_count": len(rows),
            "environment_sha256": environment_hash,
        },
        "environment_variables_recorded": False,
        "model_weights_downloaded": False,
        "training_occurred": False,
    }, freeze_text


def run_probe(
    output: Path,
    location_category: str,
    install_commands: list[str],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    report, freeze_text = probe_environment(
        location_category=location_category,
        install_commands=install_commands,
    )
    report_path = output / "ml_environment.json"
    freeze_path = output / "package_freeze.txt"
    summary_path = output / "smoke_test_summary.csv"
    markdown_path = output / "audit_summary.md"
    manifest_path = output / "manifest.json"
    _write_json(report_path, report)
    _write_text(freeze_path, freeze_text)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("check", "status", "detail"),
        )
        writer.writeheader()
        for row in report["dependencies"]:
            writer.writerow(
                {
                    "check": row["distribution"],
                    "status": row["status"],
                    "detail": row["version"] or row["reason"],
                }
            )
        for name, value in report["smoke_tests"].items():
            writer.writerow(
                {
                    "check": name,
                    "status": value["status"],
                    "detail": json.dumps(value, sort_keys=True),
                }
            )
        writer.writerow(
            {
                "check": "pip_check",
                "status": report["pip_check"]["status"],
                "detail": json.dumps(report["pip_check"]["conflicts"]),
            }
        )
    _write_text(
        markdown_path,
        "\n".join(
            [
                "# Isolated ML Environment",
                "",
                f"- Status: {report['summary']['status']}",
                f"- Python: {report['system']['python_version']}",
                f"- Dependencies available: {report['summary']['available_dependency_count']}/{report['summary']['dependency_count']}",
                f"- CUDA available: {str(report['system']['cuda']['available']).lower()}",
                f"- GPU: {report['system']['cuda']['gpu_name'] or 'unavailable'}",
                f"- pip check: {report['pip_check']['status']}",
                "- Model weights downloaded: false",
                "- Training occurred: false",
                "- Environment variables recorded: false",
                "",
            ]
        ),
    )
    source = Path(__file__).resolve()
    outputs = [report_path, freeze_path, summary_path, markdown_path]
    _write_json(
        manifest_path,
        {
            "schema_version": "1.0",
            "tool": "quartz_isolated_ml_environment_probe",
            "python_version": platform.python_version(),
            "environment_location_category": location_category,
            "tool_source": {
                "path": source.name,
                "sha256": _hash(source),
            },
            "outputs": [
                {
                    "path": path.name,
                    "sha256": _hash(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in outputs
            ],
            "environment_variables_recorded": False,
            "model_weights_downloaded": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--location-category",
        choices=("external_isolated",),
        required=True,
    )
    parser.add_argument("--install-command", action="append", default=[])
    args = parser.parse_args()
    run_probe(
        Path(args.output).resolve(),
        args.location_category,
        args.install_command,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
