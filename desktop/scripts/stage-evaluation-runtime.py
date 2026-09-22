"""Stage only evaluation code, frozen participant materials and a minimal Python prefix."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
CODE = (
    "core/__init__.py", "core/web/__init__.py", "core/web/evaluation_app.py",
    "core/web/discovery_study.py", "core/web/discovery_scoring.py",
    "core/web/discovery_localization.py", "core/web/evaluation_export.py",
    "neurooracle/src/__init__.py", "neurooracle/src/user_study.py",
)
ASSETS = (
    "evaluation-home.html", "evaluation-home.js", "evaluation-home.css", "study.html",
    "discovery-study.html", "discovery-study.css", "discovery-study.js", "discovery-study-i18n.js",
    "study-workspace.css", "study-workspace.js", "workspace-tokens.css", "evaluation-export.js",
)
BANK_FILES = (
    "case1_tcp_expert_study_v3.json", "case1_tcp_expert_study_v3_reference_notes_v4.json",
    "case1_tcp_expert_pair_assignments_v4.json",
)


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise ValueError(f"Different frozen staging file exists: {destination}; use a fresh target")
    else:
        shutil.copy2(source, destination)


def stage_backend(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    protected = (source / "desktop/runtime", source / "core", source / "neurooracle")
    if target == source or source.is_relative_to(target) or any(target == path or target.is_relative_to(path) for path in protected):
        raise ValueError("Use a separate evaluation staging directory")
    if target.exists() and any(target.iterdir()):
        raise ValueError("Use an empty evaluation backend directory")
    names = list(CODE) + [f"core/web/static/{name}" for name in ASSETS]
    names += [f"neurooracle/data/user_study/{name}" for name in BANK_FILES]
    for name in names:
        copy_file(source / name, target / name)
    spec = importlib.util.spec_from_file_location("stage_discovery", source / "desktop/scripts/stage-discovery-study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.stage(source, target)
    expected = set(names) | {f"core/web/study_materials/{name}" for name in (
        module.TARGET_NAME, module.CATALOG_NAME, module.ASSIGNMENTS_NAME, module.REFERENCE_NOTES_NAME, module.SIGNIFICANCE_NAME)}
    actual = {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("Unexpected files in evaluation backend")
    return {name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in sorted(expected)}


def stage_python(target):
    if sys.platform != "win32":
        raise ValueError("Automatic minimal-prefix preparation currently supports Windows only")
    base = Path(sys.base_prefix).resolve()
    target = target.resolve()
    if target == base or target.is_relative_to(base) or base.is_relative_to(target):
        raise ValueError("Python staging target must be outside the source prefix")
    for source in base.iterdir():
        if source.is_file() and (source.suffix.lower() in {".dll", ".exe"} or source.name.startswith("LICENSE")):
            copy_file(source, target / source.name)
    for directory in ("Lib", "DLLs"):
        for current, directories, files in os.walk(base / directory):
            directories[:] = [name for name in directories if name not in {"site-packages", "__pycache__", "test", "tests", "idlelib", "tkinter", "ensurepip"}]
            for name in files:
                source = Path(current) / name
                if source.suffix not in {".pyc", ".pyo"}:
                    copy_file(source, target / source.relative_to(base))
    for name in ("sqlite3.dll", "libssl-3-x64.dll", "libcrypto-3-x64.dll", "libbz2.dll",
                 "liblzma.dll", "zlib.dll", "ffi.dll", "ffi-7.dll", "ffi-8.dll"):
        source = base / "Library/bin" / name
        if source.is_file():
            copy_file(source, target / "DLLs" / name)
    pending, installed = ["fastapi", "uvicorn"], {}
    while pending:
        name = pending.pop()
        distribution = importlib.metadata.distribution(name)
        normalized = distribution.metadata["Name"].lower().replace("_", "-")
        if normalized in installed:
            continue
        installed[normalized] = distribution.version
        for relative in distribution.files or []:
            if ".." in relative.parts or "__pycache__" in relative.parts or relative.suffix == ".pyc":
                continue
            source = Path(distribution.locate_file(relative))
            if source.is_file():
                copy_file(source, target / "Lib/site-packages" / relative)
        for dependency in distribution.requires or []:
            requirement = Requirement(dependency)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return installed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--target", type=Path, default=ROOT / "desktop/runtime-evaluation")
    parser.add_argument("--backend-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    target = args.target.resolve()
    if args.resume:
        previous = json.loads((target / "evaluation-manifest.json").read_text(encoding="utf-8"))
        if previous.get("mode") != "human-evaluation-only":
            raise ValueError("Not an evaluation runtime")
        with tempfile.TemporaryDirectory() as directory:
            files = stage_backend(args.source, Path(directory) / "backend")
        actual = {path.relative_to(target / "backend").as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (target / "backend").rglob("*") if path.is_file()}
        if files != previous["backend_files"] or actual != files:
            raise ValueError("Source or staged backend changed; use a fresh runtime target")
    else:
        if target.exists() and any(target.iterdir()):
            raise ValueError("Use a fresh target or --resume for the same verified materials")
        files = stage_backend(args.source, target / "backend")
    packages = {} if args.backend_only else stage_python(target / "python")
    manifest = {"mode": "human-evaluation-only", "backend_files": files, "python_packages": packages,
                "platform": sys.platform, "python_version": sys.version, "backend_only": args.backend_only}
    (target / "evaluation-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"target": str(target), "backend_files": len(files), "python_packages": packages}, indent=2))


if __name__ == "__main__":
    main()
