"""Package source, coursework drafts and measured evidence without the virtualenv."""

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    experiment = ROOT / "output" / "experiment"
    status = json.loads((experiment / "status.json").read_text(encoding="utf-8"))
    verified = json.loads((experiment / "verification" / "verification.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete" or verified.get("passed") is not True:
        raise ValueError("Complete and verify the experiment before packaging")
    paths = [ROOT / name for name in (
        "README.md", ".gitignore", "pyproject.toml", "requirements.txt",
        "requirements-dev.txt", "requirements-lock.txt", "requirements-docs.txt",
    )]
    for folder, pattern in (("onnx_benchmark", "*.py"), ("tests", "*.py"), ("scripts", "*.py"), ("docs", "*.md"), ("output/pdf", "*.pdf"), ("output/docx", "*.docx")):
        paths.extend(sorted((ROOT / folder).glob(pattern)))
    paths.extend(sorted(path for path in experiment.rglob("*") if path.is_file()))
    archive = ROOT / "output" / "comp4330_project_bundle.zip"
    manifest: list[str] = []
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            bundle.writestr(relative, data)
            manifest.append(f"{hashlib.sha256(data).hexdigest()}  {relative}")
        bundle.writestr("MANIFEST.sha256", "\n".join(manifest) + "\n")
    with ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
    print(f"Packaged {len(paths)} files: {archive}")


if __name__ == "__main__":
    main()

