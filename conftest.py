"""Shared local/CI test scopes; explicit pytest paths remain supported."""
import pytest


LAYERS = ("runtime", "models", "graph", "benchmark")


def pytest_addoption(parser):
    parser.addoption("--live-graph", action="store_true", default=False,
                     help="Enable integration tests that load the complete published graph")
    parser.addoption("--layer", choices=(*LAYERS, "all"), default="all",
                     help="Select the test scope defined in pyproject.toml (default: all)")
    for layer in LAYERS:
        parser.addini(f"{layer}_testpaths", f"{layer} test files/directories", type="linelist")


def pytest_configure(config):
    # Do not replace explicit paths or node IDs such as file.py::test_name.
    if config.getoption("file_or_dir"):
        return
    layer = config.getoption("layer")
    layers = LAYERS if layer == "all" else (layer,)
    patterns = [path for name in layers for path in config.getini(f"{name}_testpaths")]
    paths = [path for pattern in patterns for path in sorted(config.rootpath.glob(pattern))]
    missing = [pattern for pattern in patterns if not list(config.rootpath.glob(pattern))]
    if missing:
        raise pytest.UsageError("Missing configured test scope: " + ", ".join(missing))
    config.args[:] = [str(path) for path in paths]


def pytest_report_header(config):
    return f"NeuroDiscovery test layer: {config.getoption('layer')}"
