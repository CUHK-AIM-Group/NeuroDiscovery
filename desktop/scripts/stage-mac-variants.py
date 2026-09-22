"""Stage isolated desktop backends; never copy personal runtime state or datasets."""
import argparse
import fnmatch
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("full", "evaluation", "demo")
EXCLUDED_DIRS = frozenset({".git", ".frozen", ".pytest_cache", ".mypy_cache", "tests", "__pycache__",
    "dist", "build", "node_modules", "data", "checkpoints", "benchmark_results", "experiment_results",
    "papers", "runs", "logs", "output", "materials", "study_materials", ".venv", "venv", "tmp"})
EXCLUDED_FILES = ("test_*.py", "*_test.py", "conftest.py", "*.pyc", "*.pyo", "*.log", ".env*",
    "*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.sqlite*", "*.db*", "*.pem", "*.key",
    "*.csv", "*.jsonl", "*.pkl", "*.npy", "*.npz", "*.zip", "*.tar*", "*.exe", "*.dll",
    "desktop-config.json", "neuroclaw_environment.json")
DEMO_EXCLUDED = ("study.html", "study-workspace.*", "discovery-study*", "discovery_*.py",
    "user_study.py", "evaluation*", "build_*.py")
MODEL_EXCLUDED = ("sweep_atlases.py", "sweep_targets.py", "tune_braingnn.py", "run_benchmark.py")
BANK_FILES = ("case1_tcp_expert_study_v3.json", "case1_tcp_expert_study_v3_reference_notes_v4.json",
    "case1_tcp_expert_pair_assignments_v4.json")


def load_script(source, filename):
    spec = importlib.util.spec_from_file_location(filename.replace("-", "_"), source / "desktop/scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fresh_target(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target == source or source.is_relative_to(target):
        raise ValueError("Target must not contain the source repository")
    for name in ("core", "skills", "neurooracle", "models", "materials", "desktop/scripts", "desktop/runtime",
                 "desktop/runtime-evaluation", "desktop/runtime-demo"):
        protected = source / name
        if target == protected or target.is_relative_to(protected):
            raise ValueError("Target overlaps protected source/runtime")
    if target.exists():
        raise ValueError("Target already exists; use a fresh directory")
    return source, target


def copy_source_tree(source, target, name, variant):
    base = source / name
    if not base.is_dir():
        raise FileNotFoundError(base)
    for current, directories, files in os.walk(base, followlinks=False):
        directory = Path(current)
        directories[:] = [item for item in directories if item not in EXCLUDED_DIRS
            and not (name in {"core", "neurooracle"} and item == "scripts")
            and not (directory / item).is_symlink()]
        patterns = EXCLUDED_FILES + (DEMO_EXCLUDED if variant == "demo" else ())
        if name == "models":
            patterns += MODEL_EXCLUDED
        for filename in files:
            original = directory / filename
            if original.is_symlink() or any(fnmatch.fnmatch(filename, pattern) for pattern in patterns):
                continue
            destination = target / original.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)


def stage(source, target, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown distribution")
    source, target = fresh_target(source, target)
    if variant == "evaluation":
        return load_script(source, "stage-evaluation-runtime.py").stage_backend(source, target)
    target.mkdir(parents=True)
    for name in ("core", "skills", "neurooracle", "models"):
        copy_source_tree(source, target, name, variant)
    for name in ("LICENSE", "README.md", "README_zh.md", "SOUL.md", "pyproject.toml"):
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    policy = Path("neurooracle/data/case_study_reaudit/full_graph_v3/RUBRIC.md")
    (target / policy).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / policy, target / policy)
    helpers = load_script(source, "stage-runtime-helpers.py")
    helpers.stage(source, target, source / "desktop/runtime-helper-files.json")
    if variant == "full":
        load_script(source, "stage-discovery-study.py").stage(source, target)
        for name in BANK_FILES:
            relative = Path("neurooracle/data/user_study") / name
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target / relative)
    else:
        (target / "DEMO_DISTRIBUTION.json").write_text(
            '{"distribution":"demo","human_evaluation":false}\n', encoding="utf-8")
        page = target / "core/web/static/index.html"
        content = page.read_text(encoding="utf-8")
        enabled = "const HUMAN_EVALUATION_ENABLED = true;"
        if content.count(enabled) != 1:
            raise ValueError("Expected exactly one evaluation feature flag")
        page.write_text(content.replace(enabled, "const HUMAN_EVALUATION_ENABLED = false;"), encoding="utf-8")
    environment = {"setup_type": "bundled", "python_path": "bundled", "conda_env": "",
        "llm_backend": {"provider": "openai", "model": "gpt-5.5", "base_url": "https://api.openai.com/v1",
                        "api_key_env": "OPENAI_API_KEY"}, "cuda": {"device": "cpu"},
        "toolchain": {}, "compression_mode": "stub"}
    (target / "neuroclaw_environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    return {path.relative_to(target).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(target.rglob("*")) if path.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    files = stage(args.source, args.target, args.variant)
    print(json.dumps({"variant": args.variant, "backend_files": len(files), "target": str(args.target)}))


if __name__ == "__main__":
    main()
