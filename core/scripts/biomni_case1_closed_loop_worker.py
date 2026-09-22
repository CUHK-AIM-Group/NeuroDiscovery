"""Persistent Biomni A1 worker for sequential Case Study 1 feedback rounds."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biomni-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--disable-tool-retriever", action="store_true")
    return parser.parse_args()


def _scrub(text: str, secret: str) -> str:
    cleaned = str(text).replace(secret, "<redacted>")
    if len(secret) >= 8:
        cleaned = cleaned.replace(secret[-8:], "<redacted-suffix>")
    return cleaned


def _install_no_storage_guard() -> None:
    from langchain_openai import ChatOpenAI

    original = ChatOpenAI._get_request_payload
    if getattr(original, "_case1_closed_loop_no_storage", False):
        return

    def guarded(self, input_, *, stop=None, **kwargs):
        payload = original(self, input_, stop=stop, **kwargs)
        payload["store"] = False
        return payload

    guarded._case1_closed_loop_no_storage = True
    ChatOpenAI._get_request_payload = guarded


def _write_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def _execution_audit(
    log: list[Any], execution_entries: list[Any]
) -> dict[str, Any]:
    codes: list[str] = []
    for entry in execution_entries:
        if not isinstance(entry, dict):
            continue
        message = str(entry.get("triggering_message") or "")
        match = re.search(r"(?is)<execute>\s*(.*?)</execute>", message)
        if match:
            codes.append(match.group(1))
    observations = [
        str(value)
        for value in log
        if isinstance(value, str) and "<observation>" in value
    ]
    observation_text = "\n".join(observations)
    pubmed_query_calls = sum(code.count("query_pubmed(") for code in codes)
    return {
        "audit_source": "biomni_a1_execution_results",
        "executed_blocks": len(codes),
        "executed_code_sha256": [
            hashlib.sha256(code.encode()).hexdigest() for code in codes
        ],
        "pubmed_query_calls": pubmed_query_calls,
        "observation_count": len(observations),
        "pubmed_result_titles": (
            observation_text.count("Title:") if pubmed_query_calls else 0
        ),
        "execution_errors": observation_text.count("<observation>Error:"),
    }


def main() -> None:
    args = parse_args()
    secret = os.environ.get("BIOMNI_API_KEY", "").strip()
    if not secret:
        raise RuntimeError("BIOMNI_API_KEY is required")
    args.out.mkdir(parents=True, exist_ok=True)
    args.workspace.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.biomni_root.resolve()))
    _install_no_storage_guard()

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        from biomni.agent import A1

        agent = A1(
            path=str(args.workspace),
            llm=args.model,
            source="Custom",
            base_url=args.base_url,
            api_key=secret,
            reasoning_effort=args.reasoning_effort,
            use_tool_retriever=not args.disable_tool_retriever,
            expected_data_lake_files=[],
        )
    (args.out / "worker_init.log").write_text(
        _scrub(captured.getvalue(), secret), encoding="utf-8"
    )
    _write_response(
        {
            "type": "ready",
            "persistent_agent": True,
            "tool_retriever_enabled": not args.disable_tool_retriever,
        }
    )

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(raw_line)
            if request.get("type") == "shutdown":
                _write_response({"type": "shutdown_complete"})
                return
            request_id = str(request["request_id"])
            prompt = str(request["prompt"])
            round_dir = Path(str(request["out_dir"]))
            round_dir.mkdir(parents=True, exist_ok=True)
            captured = io.StringIO()
            started = time.time()
            execution_start = len(getattr(agent, "_execution_results", []))
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(
                captured
            ):
                log, final = agent.go(prompt)
            duration = time.time() - started
            execution_entries = list(
                getattr(agent, "_execution_results", [])[execution_start:]
            )
            (round_dir / "agent_log.json").write_text(
                json.dumps(log, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            (round_dir / "final.txt").write_text(str(final), encoding="utf-8")
            (round_dir / "biomni_execution_audit.json").write_text(
                json.dumps(
                    _execution_audit(log, execution_entries),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (round_dir / "worker.stdout.log").write_text(
                _scrub(captured.getvalue(), secret), encoding="utf-8"
            )
            _write_response(
                {
                    "type": "round_result",
                    "request_id": request_id,
                    "ok": True,
                    "final": str(final),
                    "duration_seconds": duration,
                    "persistent_agent": True,
                }
            )
        except Exception as exc:
            _write_response(
                {
                    "type": "round_result",
                    "request_id": str(
                        request.get("request_id", "unknown")
                        if isinstance(request, dict)
                        else "unknown"
                    ),
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": _scrub(str(exc), secret),
                }
            )


if __name__ == "__main__":
    main()
