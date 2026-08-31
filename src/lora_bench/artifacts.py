"""Small, CPU-only helpers for reproducible artifact bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(root: str | Path, run_mode: str) -> dict:
    """Hash every staged payload and write a self-excluding manifest.

    ``checksums.json`` cannot contain its own stable digest, so the manifest
    names that sole exclusion explicitly. Every other regular file below
    ``root`` is included using a portable POSIX relative path.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise ValueError(f"Artifact root must be an existing directory: {root_path}")
    if run_mode not in {"smoke", "full"}:
        raise ValueError(f"run_mode must be 'smoke' or 'full', got {run_mode!r}")

    manifest_path = root_path / "checksums.json"
    payload_files = sorted(
        path for path in root_path.rglob("*") if path.is_file() and path != manifest_path
    )
    manifest = {
        "run_mode": run_mode,
        "algorithm": "sha256",
        "self_excluded": "checksums.json",
        "files": {
            path.relative_to(root_path).as_posix(): sha256_file(path) for path in payload_files
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    expected = {
        path.relative_to(root_path).as_posix()
        for path in root_path.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(manifest["files"]) != expected:
        raise AssertionError("Checksum manifest does not cover every staged payload")
    return manifest
