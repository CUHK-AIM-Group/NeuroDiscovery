"""Public NeuroDiscovery harness entry point; informational commands are offline."""

import argparse
import json
from pathlib import Path
import webbrowser

from core.harness import manifest


def port_number(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Port must be an integer") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535")
    return port


def workspace_directory(value: str) -> Path:
    workspace = Path(value).expanduser().resolve()
    if not workspace.is_dir():
        raise argparse.ArgumentTypeError("Workspace must be an existing directory")
    return workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neurodiscovery", description="NeuroDiscovery research workspace")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capabilities", help="Print the shared capability manifest; no model calls")
    chat = commands.add_parser("chat", help="Open the existing NeuroRuntime interactive session")
    chat.add_argument("--workspace", type=workspace_directory, default=Path.cwd())
    chat.add_argument("--autoresearch", choices=manifest()["autoresearch_modes"], default="off")
    gui = commands.add_parser("gui", help="Serve the GUI locally; does not start research")
    gui.add_argument("--port", type=port_number, default=7083)
    for name, help_text in (("graph", "Open the existing graph workspace"), ("evaluate", "Open Human Evaluation")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--port", type=port_number, default=7083)
        command.add_argument("--open", action="store_true", help="Also open the URL in your browser")
    research = commands.add_parser("autoresearch", help="Show a scope checklist without running research")
    research.add_argument("mode", choices=[mode for mode in manifest()["autoresearch_modes"] if mode != "off"])
    research.add_argument("--language", choices=["en", "zh"], default="en")
    args = parser.parse_args(argv)
    if args.command == "capabilities":
        print(json.dumps(manifest(), indent=2, ensure_ascii=False))
    elif args.command == "autoresearch":
        from core.autoresearch import parse_help_command, render_help_response

        print(render_help_response(parse_help_command(f"/help {args.mode}", args.language)))
    elif args.command == "chat":
        from core.agent.main import AgentSession

        AgentSession(workspace=args.workspace, autoresearch_mode=args.autoresearch).start()
    elif args.command == "gui":
        from core.web.server import run_server

        print(f"NeuroDiscovery: http://127.0.0.1:{args.port}/harness", flush=True)
        run_server(host="127.0.0.1", port=args.port)
    else:
        capability = "graph" if args.command == "graph" else "evaluation"
        url = f"http://127.0.0.1:{args.port}/harness?capability={capability}"
        print(url)
        print("Requires a running `neurodiscovery gui` server on this port.")
        if args.open:
            webbrowser.open(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
