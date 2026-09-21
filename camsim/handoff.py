"""Move a trained checkpoint to a mounted Drive directory and verify the copy.

This module only handles files. It does not import PyTorch or run inference.
"""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_checkpoint(source, destination, cfg, git_commit: str) -> dict:
    """Copy source to destination and write a checked checkpoint.json manifest."""
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size == 0:
        raise ValueError(f"empty checkpoint: {source}")

    destination.mkdir(parents=True, exist_ok=True)
    copied = destination / source.name
    original_hash = sha256_file(source)
    shutil.copy2(source, copied)
    if sha256_file(copied) != original_hash:
        raise OSError(f"checkpoint copy failed SHA-256 check: {copied}")

    run_config = asdict(cfg)
    run_config.pop("path", None)
    manifest = {
        "file": source.name,
        "sha256": original_hash,
        "bytes": source.stat().st_size,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "config": run_config,
    }
    (destination / "checkpoint.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
