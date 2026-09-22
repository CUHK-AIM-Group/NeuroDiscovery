"""Run the frozen evaluation backend in the user's system browser."""
import argparse
import os
from pathlib import Path
import socket
import sys
import threading
import time
import webbrowser


def bind_local(port):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(128)
        return listener
    except BaseException:
        listener.close()
        raise


def open_when_ready(server, url):
    deadline = time.monotonic() + 60
    while not server.started and time.monotonic() < deadline:
        if server.should_exit:
            return
        time.sleep(0.1)
    if server.started:
        try:
            if not webbrowser.open(url):
                print(f"Open this address in your browser: {url}", flush=True)
        except Exception as error:
            print(f"Browser could not open ({error}). Open: {url}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=17892)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535")
    if args.data is None:
        if not os.environ.get("APPDATA"):
            parser.error("APPDATA is missing; specify --data")
        args.data = Path(os.environ["APPDATA"]) / "NeuroDiscovery-Human-Evaluation-Browser" / "evaluation-data"
    backend = Path(__file__).resolve().parent / "runtime" / "backend"
    sys.path.insert(0, str(backend))
    import uvicorn
    from core.web.evaluation_app import create_app

    try:
        listener = bind_local(args.port)
    except OSError as error:
        print(f"Cannot reserve local port {args.port}: {error}\nNo existing service was reused or stopped.", flush=True)
        return 1
    with listener:
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        app = create_app(args.data)
        server = uvicorn.Server(uvicorn.Config(app, access_log=False, log_level="warning"))
        print(f"Human Evaluation: {url}\nAnswers: {args.data.resolve()}\nKeep this window open while evaluating.\nSave your work, then press Ctrl+C here to stop. Closing a browser tab does NOT stop this server.", flush=True)
        if not args.no_browser:
            threading.Thread(target=open_when_ready, args=(server, url), daemon=True).start()
        try:
            server.run(sockets=[listener])
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
