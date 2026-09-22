import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

DESKTOP = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("evaluation_browser", DESKTOP / "evaluation-browser.py")
packager = load("package_evaluation_browser", DESKTOP / "scripts/package-evaluation-browser.py")


class BrowserEvaluationTests(unittest.TestCase):
    def test_loopback_and_conflict(self):
        with launcher.bind_local(0) as listener:
            host, port = listener.getsockname()
            self.assertEqual(host, "127.0.0.1")
            with self.assertRaises(OSError):
                launcher.bind_local(port)
            with socket.create_connection((host, port), timeout=1):
                pass

    def test_opens_browser_only_when_ready(self):
        class Server:
            started = True
            should_exit = False
        with patch.object(launcher.webbrowser, "open", return_value=True) as opened:
            launcher.open_when_ready(Server(), "http://127.0.0.1:17892")
            opened.assert_called_once_with("http://127.0.0.1:17892")
            opened.reset_mock()
            Server.started, Server.should_exit = False, True
            launcher.open_when_ready(Server(), "http://127.0.0.1:17892")
            opened.assert_not_called()

    def test_backend_hash_and_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "backend").mkdir()
            (runtime / "python").mkdir()
            (runtime / "python/python.exe").write_bytes(b"fixture")
            source = runtime / "backend/fixture.py"
            source.write_text("fixture")
            (runtime / "evaluation-manifest.json").write_text(json.dumps({"mode": "human-evaluation-only",
                "backend_only": False, "backend_files": {"fixture.py": packager.digest(source)}}))
            packager.verify_backend(runtime)
            source.write_text("changed")
            with self.assertRaises(ValueError):
                packager.verify_backend(runtime)
            source.write_text("fixture")
            (runtime / "backend/private.sqlite3").write_bytes(b"private")
            with self.assertRaises(ValueError):
                packager.verify_backend(runtime)

    def test_existing_or_overlapping_output_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            existing = root / "existing"
            existing.mkdir()
            for target in (root, runtime, runtime / "nested", existing):
                with self.assertRaises(ValueError):
                    packager.build(runtime, target)
            self.assertEqual(list(existing.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
