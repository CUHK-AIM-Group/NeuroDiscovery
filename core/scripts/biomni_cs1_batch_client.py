"""Run one blinded CS1 hypothesis batch through Biomni A1."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--biomni-root", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default="http://localhost:9449/v1")
    parser.add_argument("--reasoning-effort", default="high")
    return parser.parse_args()


def scrub(text: str, secret: str) -> str:
    cleaned = text.replace(secret, "<redacted>")
    if len(secret) >= 4:
        cleaned = cleaned.replace(secret[-4:], "<redacted-suffix>")
    return cleaned


def main() -> None:
    args = parse_args()
    secret = os.environ.get("BIOMNI_API_KEY")
    if not secret:
        raise RuntimeError("BIOMNI_API_KEY is required")
    sys.path.insert(0, str(args.biomni_root.resolve()))
    from biomni.agent import A1

    args.out.mkdir(parents=True, exist_ok=True)
    args.workspace.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt.read_text(encoding="utf-8")
    captured = io.StringIO()
    started = time.time()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        agent = A1(
            path=str(args.workspace),
            llm=args.model,
            source="Custom",
            base_url=args.base_url,
            api_key=secret,
            reasoning_effort=args.reasoning_effort,
            use_tool_retriever=False,
            expected_data_lake_files=[],
        )
        log, final = agent.go(prompt)
    (args.out / "agent_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "final.txt").write_text(str(final), encoding="utf-8")
    (args.out / "client.stdout.log").write_text(scrub(captured.getvalue(), secret), encoding="utf-8")
    (args.out / "client_meta.json").write_text(
        json.dumps(
            {
                "duration_seconds": time.time() - started,
                "model": args.model,
                "base_url": args.base_url,
                "reasoning_effort": args.reasoning_effort,
                "use_tool_retriever": False,
                "expected_data_lake_files": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.out / "final.txt")


if __name__ == "__main__":
    main()

