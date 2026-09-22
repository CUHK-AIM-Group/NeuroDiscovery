import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.request

DESKTOP = Path(__file__).resolve().parents[1]
PACKAGE = Path(os.environ.get("EVALUATION_WEBVIEW_PACKAGE", str(
    DESKTOP / "dist-evaluation-webview-r3/NeuroDiscovery-Human-Evaluation-1.0.0-WebView-win-x64")))


class WebViewLifecycleTests(unittest.TestCase):
    def start_backend(self, data, port):
        return subprocess.Popen([str(PACKAGE / "runtime/python/python.exe"), "-I", "-B",
            str(PACKAGE / "evaluation-webview-server.py"), "--data", str(data), "--port", str(port)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW)

    def test_owner_pipe_close_stops_server_and_keeps_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            backend = self.start_backend(directory, port)
            try:
                ready = False
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                for attempt in range(100):
                    if backend.poll() is not None:
                        self.fail(backend.stderr.read().decode(errors="replace"))
                    try:
                        with opener.open(f"http://127.0.0.1:{port}/api/health", timeout=0.5) as response:
                            ready = json.load(response)["mode"] == "human-evaluation-only"
                    except OSError:
                        pass
                    if ready:
                        break
                    time.sleep(0.1)
                self.assertTrue(ready)
                backend.stdin.close()
                self.assertEqual(backend.wait(timeout=8), 0)
                self.assertTrue(list(Path(directory).rglob("*.sqlite3")))
                with socket.socket() as probe:
                    self.assertNotEqual(probe.connect_ex(("127.0.0.1", port)), 0)
            finally:
                if backend.poll() is None:
                    backend.kill()
                    backend.wait(timeout=5)
                backend.stdin.close()
                backend.stderr.close()

    def test_occupied_port_is_not_reused_and_no_records_created(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as occupied:
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            backend = self.start_backend(directory, occupied.getsockname()[1])
            try:
                self.assertNotEqual(backend.wait(timeout=10), 0)
                self.assertEqual(list(Path(directory).iterdir()), [])
                with socket.create_connection(occupied.getsockname(), timeout=1):
                    pass
            finally:
                if backend.poll() is None:
                    backend.kill()
                    backend.wait(timeout=5)
                backend.stdin.close()
                backend.stderr.close()


if __name__ == "__main__":
    unittest.main()
