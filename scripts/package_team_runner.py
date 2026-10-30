"""Create a portable team handoff or a teammate's evidence return ZIP."""

import argparse
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def collect_files(root: Path, results_only: bool) -> list[Path]:
    """Use an explicit allowlist so virtual environments and temporary work stay out."""
    workbook = root / "ONNX_Experiment_Tracker.xlsx"
    if not workbook.is_file():
        raise ValueError("Missing ONNX_Experiment_Tracker.xlsx")
    paths = {workbook}
    for folder in ("output/experiment", "output/runs", "logs"):
        directory = root / folder
        if directory.exists():
            paths.update(item for item in directory.rglob("*") if item.is_file() and not item.name.endswith(".lock"))
    if not results_only:
        for name in (
            "README.md", "requirements.txt", "requirements-dev.txt", "requirements-lock.txt",
            "requirements-tracker.txt", "requirements-docs.txt", "pyproject.toml", "Setup_Environment.cmd",
            "Run_Experiments.cmd", "Collect_Results.cmd", "templates/ONNX_Experiment_Tracker.xlsx",
        ):
            file = root / name
            if not file.is_file():
                raise ValueError(f"Missing handoff file: {name}")
            paths.add(file)
        for folder, patterns in (
            ("onnx_benchmark", ("*.py",)), ("experiment_tools", ("*.py",)),
            ("tests", ("*.py",)), ("scripts", ("*.py", "*.ps1")), ("docs", ("*.md",)),
        ):
            for pattern in patterns:
                paths.update((root / folder).glob(pattern))
    return sorted(paths)


def package(root: Path, destination: Path, results_only: bool) -> int:
    paths = collect_files(root, results_only)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"Refusing to replace an existing archive: {destination}")
    manifest: list[str] = []
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for item in paths:
            relative = item.relative_to(root).as_posix()
            content = item.read_bytes()
            archive.writestr(relative, content)
            manifest.append(f"{hashlib.sha256(content).hexdigest()}  {relative}")
        archive.writestr("MANIFEST.sha256", "\n".join(manifest) + "\n")
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC validation failed")
        for line in archive.read("MANIFEST.sha256").decode("utf-8").splitlines():
            expected, name = line.split("  ", 1)
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"ZIP content hash mismatch: {name}")
    return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-only", action="store_true", help="Collect tracker, complete run folders and logs for the group leader")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    name = f"COMP4330_Results_{platform.node()}_{stamp}.zip" if args.results_only else f"COMP4330_Auto_Experiment_Package_{stamp}.zip"
    destination = args.output or ROOT / "output" / name
    count = package(ROOT, destination.resolve(), args.results_only)
    print(f"Packaged {count} files with SHA-256 manifest: {destination}")


if __name__ == "__main__":
    main()
