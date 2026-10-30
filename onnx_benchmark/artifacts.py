"""Small shared helpers for durable experiment files and provenance."""

import hashlib
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    package_dir = Path(__file__).parent
    return {path.name: sha256_file(path) for path in sorted(package_dir.glob("*.py"))}

