"""Shared helpers for Phase 3A readiness evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from . import TOOL_VERSION
from .common import (
    atomic_write_json,
    git_evidence,
    normalize_path,
    sha256_file,
    utc_timestamp,
)


def find_repo_root(start: str | Path) -> Path:
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for path in (candidate, *candidate.parents):
        if (path / ".git").exists():
            return path
    return Path.cwd().resolve()


def resolve_path(value: str | Path | None, repo_root: str | Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve()


def load_json_config(
    path: str | Path,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a JSON object: {config_path}")
    config["_config_path"] = str(config_path)
    config["_repo_root"] = str(find_repo_root(config_path))
    if output_override is not None:
        config["output_directory"] = str(output_override)
    return config


def atomic_write_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    columns: Iterable[str],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(columns)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: (
                            json.dumps(value, sort_keys=True)
                            if isinstance(value, (dict, list))
                            else value
                        )
                        for key, value in row.items()
                    }
                )
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_text(path: str | Path, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate_file_hash(
    paths: Iterable[str | Path],
    repo_root: str | Path | None = None,
) -> str:
    root = Path(repo_root).resolve() if repo_root is not None else None
    digest = hashlib.sha256()
    resolved = sorted(
        (Path(path).resolve() for path in paths),
        key=lambda path: normalize_path(path, root),
    )
    for path in resolved:
        relative = normalize_path(path, root)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def write_manifest(
    *,
    path: str | Path,
    repo_root: str | Path,
    tool_name: str,
    command: str,
    tool_sources: Iterable[str | Path],
    inputs: Iterable[tuple[str, str | Path | None]],
    outputs: Iterable[tuple[str, str | Path]],
    extra: dict[str, Any] | None = None,
) -> None:
    root = Path(repo_root).resolve()
    source_rows = []
    for source in tool_sources:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = root / source_path
        if source_path.exists():
            source_rows.append(
                {
                    "path": normalize_path(source_path, root),
                    "sha256": sha256_file(source_path),
                }
            )
    input_rows = []
    for name, input_path in inputs:
        resolved = resolve_path(input_path, root)
        if resolved is None:
            input_rows.append(
                {"name": name, "status": "unavailable", "reason": "not supplied"}
            )
        elif resolved.exists() and resolved.is_file():
            input_rows.append(
                {
                    "name": name,
                    "path": normalize_path(resolved, root),
                    "sha256": sha256_file(resolved),
                }
            )
        else:
            input_rows.append(
                {
                    "name": name,
                    "path": normalize_path(resolved, root),
                    "status": "unavailable",
                    "reason": "file not found",
                }
            )
    output_rows = []
    for name, output_path in outputs:
        candidate = Path(output_path)
        if candidate.exists() and candidate.is_file():
            output_rows.append(
                {
                    "name": name,
                    "path": normalize_path(candidate, root),
                    "sha256": sha256_file(candidate),
                    "size_bytes": candidate.stat().st_size,
                }
            )
    value = {
        "schema_version": "1.0",
        "tool": tool_name,
        "tool_version": TOOL_VERSION,
        "created_utc": utc_timestamp(),
        "command": command,
        "git": git_evidence(root),
        "tool_sources": source_rows,
        "inputs": input_rows,
        "outputs": output_rows,
    }
    if extra:
        value.update(extra)
    atomic_write_json(path, value)
