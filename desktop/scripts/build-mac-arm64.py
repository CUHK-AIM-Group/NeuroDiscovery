"""Native Apple Silicon build of three isolated desktop distributions."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

DESKTOP = Path(__file__).resolve().parents[1]
PRODUCTS = {"full": ("NeuroDiscovery", "org.neurodiscovery.desktop"),
            "evaluation": ("NeuroDiscovery Human Evaluation", "org.neurodiscovery.human-evaluation"),
            "demo": ("NeuroDiscovery Demo", "org.neurodiscovery.demo")}
SHELL_FILES = ["main.js", "llm-settings.js", "llm-credentials.js", "model-library.js",
               "settings-restart.js", "preload.js", "package.json"]


def builder_config(variant, runtime, output):
    name, app_id = PRODUCTS[variant]
    evaluation = variant == "evaluation"
    metadata = {"main": "evaluation-main.js" if evaluation else "main.js"}
    if variant == "demo":
        metadata["distribution"] = "demo"
    else:
        metadata["distribution"] = variant
    return {"extends": None, "appId": app_id, "productName": name,
        "directories": {"output": str(output)},
        "artifactName": name.replace(" ", "-") + "-${version}-mac-${arch}.${ext}",
        "files": ["evaluation-main.js", "evaluation-preload.js", "package.json"] if evaluation else SHELL_FILES,
        "extraMetadata": metadata,
        "extraResources": [{"from": str(runtime), "to": "evaluation-runtime" if evaluation else "runtime",
                            "filter": ["**/*", "!**/__pycache__/**", "!**/*.pyc", "!**/*.pyo"]}],
        "mac": {"target": [{"target": "dmg", "arch": ["arm64"]}, {"target": "zip", "arch": ["arm64"]}],
                "category": "public.app-category.education" if evaluation else "public.app-category.developer-tools",
                "identity": None}}


def run(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def load_stager():
    spec = importlib.util.spec_from_file_location("mac_stager", DESKTOP / "scripts/stage-mac-variants.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_guard():
    if sys.version_info < (3, 12):
        raise RuntimeError("Use Python 3.12 or newer for the build driver")
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("Run with native ARM64 Python on an Apple Silicon Mac, not Windows or Rosetta")
    if os.environ.get("CONDA_SUBDIR", "osx-arm64") != "osx-arm64":
        raise RuntimeError("CONDA_SUBDIR must be osx-arm64")


def safe_extract(archive, target):
    target = Path(target).resolve()
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            destination = (target / member.name).resolve()
            if not destination.is_relative_to(target) or member.isdev() or member.isfifo():
                raise ValueError("Unsafe Python archive entry")
            if member.issym() or member.islnk():
                linked = ((destination.parent if member.issym() else target) / member.linkname).resolve()
                if not linked.is_relative_to(target):
                    raise ValueError("Python archive link escapes runtime")
        bundle.extractall(target, filter="data")


def build(variant, output_root, conda, prepare_only=False):
    native_guard()
    output = Path(output_root).resolve() / variant
    root = DESKTOP.parent.resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}; specify a fresh --output-root")
    if root.is_relative_to(output) or any(output.is_relative_to(root / name)
            for name in ("core", "skills", "neurooracle", "models", "materials", "desktop/runtime",
                         "desktop/runtime-evaluation", "desktop/runtime-demo", "desktop/scripts")):
        raise ValueError("Output overlaps protected source/runtime")
    builder = DESKTOP / "node_modules/.bin/electron-builder"
    if not builder.is_file():
        raise FileNotFoundError("Run npm ci in desktop first")
    if subprocess.check_output(["node", "-p", "process.arch"], text=True).strip() != "arm64":
        raise RuntimeError("Node must be native arm64, not Rosetta")
    output.mkdir(parents=True)
    runtime = output / "runtime"
    files = load_stager().stage(root, runtime / "backend", variant)
    env = dict(os.environ, CONDA_SUBDIR="osx-arm64", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               CSC_IDENTITY_AUTO_DISCOVERY="false")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory(prefix="neurodiscovery-mac-build-") as temporary:
        prefix = Path(temporary) / "python"
        run([conda, "create", "-y", "-p", prefix, "--copy", "python=3.11", "pip"], env=env)
        python = prefix / "bin/python"
        run([python, "-m", "pip", "install", "conda-pack==0.8.1"], env=env)
        requirements = DESKTOP / ("evaluation-runtime-requirements.txt" if variant == "evaluation" else "runtime-requirements.txt")
        run([python, "-m", "pip", "install", "-r", requirements], env=env)
        run([python, "-m", "pip", "check"], env=env)
        run([python, "-c", "import platform; assert platform.machine() == 'arm64'; import ssl,sqlite3,fastapi,uvicorn"], env=env)
        installed = subprocess.check_output([str(python), "-m", "pip", "freeze"], env=env, text=True)
        (output / "PYTHON_PACKAGES.txt").write_text(installed, encoding="utf-8")
        archive = Path(temporary) / "python.tar.gz"
        run([python, "-m", "conda_pack", "-p", prefix, "-o", archive], env=env)
        with archive.open("rb") as stream:
            python_archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        safe_extract(archive, runtime / "python")
        relocated = Path(temporary) / "relocated with spaces"
        safe_extract(archive, relocated)
        relocated_python = relocated / "bin/python"
        run([relocated_python, relocated / "bin/conda-unpack"], env=env)
        run([relocated_python, DESKTOP / "scripts/verify-mac-backend.py", "--backend", runtime / "backend",
             "--variant", variant], env=env)
    manifest = {"mode": "human-evaluation-only" if variant == "evaluation" else variant,
        "platform": "darwin-arm64", "strategy": "conda-prefix-relocatable", "backend_only": False,
        "backend_files": files, "python_archive_sha256": python_archive_sha256,
        "python_packages_sha256": hashlib.sha256(installed.encode()).hexdigest()}
    for name in (["evaluation-manifest.json", "runtime-manifest.json"] if variant == "evaluation" else ["runtime-manifest.json"]):
        (runtime / name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    config = output / "electron-builder.json"
    config.write_text(json.dumps(builder_config(variant, runtime, output / "artifacts"), indent=2) + "\n", encoding="utf-8")
    if not prepare_only:
        run([builder, "--config", config, "--mac", "dmg", "zip", "--arm64"], cwd=DESKTOP, env=env)
        artifacts = []
        for artifact in sorted((output / "artifacts").iterdir()):
            if artifact.suffix in {".dmg", ".zip"}:
                with artifact.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                artifacts.append({"file": artifact.name, "bytes": artifact.stat().st_size, "sha256": digest})
        if {Path(item["file"]).suffix for item in artifacts} != {".dmg", ".zip"}:
            raise RuntimeError("Both DMG and ZIP must be present")
        (output / "BUILD_RECEIPT.json").write_text(json.dumps({"variant": variant, "platform": "darwin-arm64",
            "artifacts": artifacts, "backend_relocation_check": "passed", "gui_acceptance": "pending",
            "signed": False, "notarized": False}, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {variant}: {output}; macOS GUI acceptance is still required")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=[*PRODUCTS, "all"], required=True)
    parser.add_argument("--output-root", type=Path, default=DESKTOP / "dist-mac-arm64")
    parser.add_argument("--conda", default=os.environ.get("CONDA_EXE") or shutil.which("conda"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    native_guard()
    if not args.conda:
        parser.error("Install native arm64 Miniforge and pass --conda /path/to/bin/conda")
    for variant in PRODUCTS if args.variant == "all" else [args.variant]:
        build(variant, args.output_root, args.conda, args.prepare_only)


if __name__ == "__main__":
    main()
