"""Serve evaluation until the owning WebView host closes its input pipe."""
import argparse
import importlib.util
from pathlib import Path
import sys
import threading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--port", type=int, default=17893)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("browser_launcher", root / "evaluation-browser.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    sys.path.insert(0, str(root / "runtime/backend"))
    import uvicorn
    from core.web.evaluation_app import create_app

    with launcher.bind_local(args.port) as listener:
        server = uvicorn.Server(uvicorn.Config(create_app(args.data), access_log=False, log_level="warning"))

        def monitor_owner():
            sys.stdin.buffer.read(1)
            server.should_exit = True

        threading.Thread(target=monitor_owner, daemon=True).start()
        server.run(sockets=[listener])


if __name__ == "__main__":
    main()
