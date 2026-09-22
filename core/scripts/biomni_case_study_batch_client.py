"""Run one blinded case-study hypothesis batch through Biomni A1."""

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
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default="http://127.0.0.1:18082/v1")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--use-tool-retriever",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def scrub(text: str, secret: str) -> str:
    cleaned = text.replace(secret, "<redacted>")
    if len(secret) >= 4:
        cleaned = cleaned.replace(secret[-4:], "<redacted-suffix>")
    return cleaned


def enforce_no_response_storage() -> None:
    """Force store=false on LangChain's OpenAI Responses payloads."""

    from langchain_openai import ChatOpenAI

    original = ChatOpenAI._get_request_payload
    if getattr(original, "_case_study_no_storage", False):
        return

    def guarded(self, input_, *, stop=None, **kwargs):
        payload = original(self, input_, stop=stop, **kwargs)
        payload["store"] = False
        return payload

    guarded._case_study_no_storage = True
    ChatOpenAI._get_request_payload = guarded


def enforce_blinded_resource_policy(agent):
    """Keep Biomni's native workflow while excluding all external resources."""

    selected_resources = {
        "tools": [],
        "data_lake": [],
        "libraries": [],
        "know_how": [],
    }
    updater = getattr(agent, "update_system_prompt_with_selected_resources", None)
    if not callable(updater):
        raise RuntimeError("Biomni A1 cannot enforce the blinded resource policy")
    updater(selected_resources)
    return selected_resources


def main() -> None:
    args = parse_args()
    secret = os.environ.get("BIOMNI_API_KEY")
    if not secret:
        raise RuntimeError("BIOMNI_API_KEY is required")
    enforce_no_response_storage()
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
            use_tool_retriever=args.use_tool_retriever,
            expected_data_lake_files=[],
        )
        selected_resources = enforce_blinded_resource_policy(agent)
        log, final = agent.go(prompt)
    (args.out / "agent_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.out / "final.txt").write_text(str(final), encoding="utf-8")
    (args.out / "client.stdout.log").write_text(
        scrub(captured.getvalue(), secret), encoding="utf-8"
    )
    (args.out / "client_meta.json").write_text(
        json.dumps(
            {
                "duration_seconds": time.time() - started,
                "model": args.model,
                "base_url": args.base_url,
                "reasoning_effort": args.reasoning_effort,
                "use_tool_retriever": args.use_tool_retriever,
                "external_resources_enabled": False,
                "selected_resource_counts": {
                    key: len(value) for key, value in selected_resources.items()
                },
                "expected_data_lake_files": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.out / "final.txt")


if __name__ == "__main__":
    main()


# Last Updated At: 2026-08-01 10:20 HKT
