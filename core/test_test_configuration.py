"""Exercise test selection in a clean miniature checkout, without project data."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def checkout(tmp_path):
    shutil.copyfile(Path(__file__).resolve().parents[1] / "conftest.py", tmp_path / "conftest.py")
    scopes = ("runtime", "models", "graph", "benchmark")
    config = ["[tool.pytest.ini_options]"]
    for scope in scopes:
        folder = tmp_path / scope
        folder.mkdir()
        (folder / f"test_{scope}.py").write_text("def test_example(): pass\n", encoding="utf-8")
        config.append(f'{scope}_testpaths = ["{scope}"]')
    (tmp_path / "pyproject.toml").write_text("\n".join(config), encoding="utf-8")
    return tmp_path


def collect(checkout, *args):
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    return subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", *args],
                          cwd=checkout, env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("layer", ["runtime", "models", "graph", "benchmark"])
def test_layer_collects_only_its_scope(checkout, layer):
    result = collect(checkout, "--layer", layer)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{layer}/test_{layer}.py::test_example" in result.stdout.replace("\\", "/")
    assert "1 test collected" in result.stdout


def test_default_collects_all_four_scopes(checkout):
    result = collect(checkout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 tests collected" in result.stdout


def test_explicit_node_id_is_preserved(checkout):
    result = collect(checkout, "runtime/test_runtime.py::test_example")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 test collected" in result.stdout


def test_missing_scope_fails_instead_of_silently_collecting_other_tests(checkout):
    shutil.rmtree(checkout / "graph")
    result = collect(checkout, "--layer", "graph")
    assert result.returncode == 4
    assert "Missing configured test scope: graph" in result.stderr
