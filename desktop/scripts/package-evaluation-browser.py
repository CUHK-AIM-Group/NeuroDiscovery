"""Package the verified standalone evaluation runtime without Electron."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

DESKTOP = Path(__file__).resolve().parents[1]
NAME = "NeuroDiscovery-Human-Evaluation-1.0.0-Browser-win-x64"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_backend(runtime):
    manifest = json.loads((runtime / "evaluation-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("mode") != "human-evaluation-only" or manifest.get("backend_only"):
        raise ValueError("A complete evaluation-only runtime is required")
    backend = runtime / "backend"
    actual = {path.relative_to(backend).as_posix(): digest(path)
              for path in backend.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    if actual != manifest["backend_files"]:
        raise ValueError("Frozen evaluation backend does not match its manifest")
    if not (runtime / "python/python.exe").is_file():
        raise ValueError("Bundled Windows Python is missing")
    return manifest


def build(runtime, output):
    runtime, output = Path(runtime).resolve(), Path(output).resolve()
    if runtime == output or runtime.is_relative_to(output) or output.is_relative_to(runtime):
        raise ValueError("Output must be separate from the source runtime")
    if output.exists():
        raise ValueError("Output already exists; choose a new directory, never overwrite an existing export")
    verify_backend(runtime)
    package = output / NAME
    package.mkdir(parents=True)
    for name in ("backend", "python"):
        shutil.copytree(runtime / name, package / "runtime" / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    shutil.copy2(runtime / "evaluation-manifest.json", package / "runtime/evaluation-manifest.json")
    verify_backend(package / "runtime")
    for source, target in (("evaluation-browser.py", "evaluation-browser.py"),
                           ("evaluation-browser.cmd", "Start Evaluation.cmd"),
                           ("EVALUATION_BROWSER_README.md", "README.md")):
        shutil.copy2(DESKTOP / source, package / target)
    files = {path.relative_to(package).as_posix(): digest(path)
             for path in sorted(package.rglob("*")) if path.is_file()}
    (package / "PACKAGE_MANIFEST.json").write_text(json.dumps({"mode": "human-evaluation-system-browser",
        "files": files}, indent=2) + "\n", encoding="utf-8")
    archive = output / f"{NAME}.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
    receipt = {"archive": str(archive), "bytes": archive.stat().st_size,
               "sha256": digest(archive), "files": len(files), "zip_crc_verified": True}
    (output / "BUILD_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=DESKTOP / "runtime-evaluation")
    parser.add_argument("--output", type=Path, default=DESKTOP / "dist-evaluation-browser")
    args = parser.parse_args()
    print(json.dumps(build(args.runtime, args.output), indent=2))


if __name__ == "__main__":
    main()
