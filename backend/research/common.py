"""Shared utilities for reproducible Quartz research evidence."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

AuditStatus = Literal["pass", "warning", "fail", "unavailable"]

STATUS_PASS: AuditStatus = "pass"
STATUS_WARNING: AuditStatus = "warning"
STATUS_FAIL: AuditStatus = "fail"
STATUS_UNAVAILABLE: AuditStatus = "unavailable"


def utc_timestamp(value: datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp."""

    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA-256 using bounded memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of a byte string."""

    return hashlib.sha256(value).hexdigest()


def normalize_path(path: str | Path, base: str | Path | None = None) -> str:
    """Serialize a path with forward slashes, relative to ``base`` if possible."""

    candidate = Path(path)
    if base is not None:
        try:
            candidate = candidate.resolve().relative_to(Path(base).resolve())
        except (OSError, ValueError):
            candidate = candidate.resolve()
    return candidate.as_posix()


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Write JSON atomically without leaving a partial evidence file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
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
            temporary_path = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def deterministic_random(seed: int) -> random.Random:
    """Return an isolated deterministic random generator."""

    return random.Random(int(seed))


def audit_status(
    *,
    failed: bool = False,
    warning: bool = False,
    unavailable: bool = False,
) -> AuditStatus:
    """Map explicit evidence state to one of the allowed audit statuses."""

    if failed:
        return STATUS_FAIL
    if unavailable:
        return STATUS_UNAVAILABLE
    if warning:
        return STATUS_WARNING
    return STATUS_PASS


def _run_git(repo_root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        capture_output=True,
        check=False,
        timeout=15,
    )


def git_evidence(repo_root: str | Path) -> dict[str, Any]:
    """Record a bounded, secret-free description of the current Git state."""

    root = Path(repo_root).resolve()
    try:
        head = _run_git(root, "rev-parse", "HEAD")
        status = _run_git(root, "status", "--porcelain", "--untracked-files=all")
        diff = _run_git(root, "diff", "--binary", "--no-ext-diff")
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": f"Git state could not be recorded: {exc}",
        }

    if head.returncode != 0 or status.returncode != 0 or diff.returncode != 0:
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": "One or more Git inspection commands failed.",
        }

    status_text = status.stdout.decode("utf-8", errors="replace")
    return {
        "status": STATUS_PASS,
        "commit": head.stdout.decode("ascii", errors="replace").strip(),
        "dirty": bool(status_text.strip()),
        "status_sha256": sha256_bytes(status.stdout),
        "diff_sha256": sha256_bytes(diff.stdout),
    }
