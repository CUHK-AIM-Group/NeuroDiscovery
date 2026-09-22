"""Probe DeepSeek V4 Pro through OpenCode Go, Ollama Cloud, and DeepSeek.

Secrets are read from ``~/Downloads/keys.txt`` at runtime.  They are never
written to the report, printed, fingerprinted, or placed in command arguments.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_KEYS = Path.home() / "Downloads" / "keys.txt"
TEST_PROMPT = "Return exactly the ASCII token DS4PRO_OK and nothing else."
USER_AGENT = "NeuroClaw-DeepSeek-V4-Pro-Probe/1.0"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_channel_keys(path: Path) -> dict[str, list[str]]:
    """Parse only the three labelled key sections; ignore unrelated exports."""

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    channels = {"opencode": [], "ollama": [], "deepseek": []}
    current: str | None = None
    for raw in lines:
        value = raw.strip()
        if not value:
            current = None
            continue
        lowered = value.lower()
        if "opencode" in lowered and "api key" in lowered:
            current = "opencode"
            continue
        if "ollama" in lowered and "api key" in lowered:
            current = "ollama"
            continue
        if "deepseek" in lowered and "api key" in lowered:
            current = "deepseek"
            continue
        if current is not None:
            channels[current].append(value.strip('"').strip("'"))

    _require(channels["opencode"], "no OpenCode key found in labelled section")
    _require(channels["ollama"], "no Ollama key found in labelled section")
    _require(len(channels["deepseek"]) == 1, "expected exactly one DeepSeek key")
    return channels


def _scrub(text: str, secrets: Iterable[str]) -> str:
    safe = text
    for secret in secrets:
        if secret:
            safe = safe.replace(secret, "<redacted>")
    safe = re.sub(r"(?i)bearer\s+\S+", "Bearer <redacted>", safe)
    safe = re.sub(r"(?i)\bsk-[A-Za-z0-9._-]+", "<redacted>", safe)
    safe = re.sub(r"\bwrk_[A-Za-z0-9]+\b", "wrk_<redacted>", safe)
    safe = re.sub(
        r"\b[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{12,}(?:\.[A-Za-z0-9_-]{12,})?\b",
        "<redacted>",
        safe,
    )
    return safe[:1000]


def _post_json(
    *,
    channel: str,
    key_label: str,
    url: str,
    api_key: str,
    payload: Mapping[str, Any],
    all_secrets: Sequence[str],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    started = time.perf_counter()
    status: int | None = None
    response_text = ""
    error_kind: str | None = None
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            response_text = response.read(2 * 1024 * 1024).decode("utf-8", "replace")
    except HTTPError as exc:
        status = int(exc.code)
        response_text = exc.read(256 * 1024).decode("utf-8", "replace")
        error_kind = "http_error"
    except URLError as exc:
        error_kind = "network_error"
        response_text = str(exc.reason)
    except TimeoutError as exc:
        error_kind = "timeout"
        response_text = str(exc)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    parsed: dict[str, Any] | None = None
    try:
        candidate = json.loads(response_text)
        if isinstance(candidate, dict):
            parsed = candidate
    except json.JSONDecodeError:
        pass

    result: dict[str, Any] = {
        "channel": channel,
        "key_label": key_label,
        "endpoint": url,
        "http_status": status,
        "elapsed_ms": elapsed_ms,
        "ok": status is not None and 200 <= status < 300,
        "error_kind": error_kind,
    }
    if parsed is not None:
        result["response"] = _summarize_response(channel, parsed, all_secrets)
    else:
        result["response"] = {
            "error_detail": _scrub(response_text, all_secrets),
        }
    return result


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
        return "".join(parts)
    return ""


def _summarize_response(
    channel: str, payload: Mapping[str, Any], secrets: Sequence[str]
) -> dict[str, Any]:
    if channel == "ollama":
        message = payload.get("message") or {}
        content = _content_text(message.get("content")) if isinstance(message, dict) else ""
        return {
            "model": payload.get("model"),
            "content": _scrub(content, secrets),
            "done": payload.get("done"),
            "done_reason": payload.get("done_reason"),
            "prompt_tokens": payload.get("prompt_eval_count"),
            "completion_tokens": payload.get("eval_count"),
            "error": _scrub(str(payload.get("error") or ""), secrets) or None,
        }

    choices = payload.get("choices") or []
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") or {} if isinstance(choice, dict) else {}
    content = _content_text(message.get("content")) if isinstance(message, dict) else ""
    usage = payload.get("usage") or {}
    error = payload.get("error")
    if isinstance(error, dict):
        error_detail = error.get("message") or json.dumps(error, ensure_ascii=False)
    else:
        error_detail = error
    return {
        "model": payload.get("model"),
        "content": _scrub(content, secrets),
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "reasoning_returned": bool(
            isinstance(message, dict) and message.get("reasoning_content")
        ),
        "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
        "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
        "total_tokens": usage.get("total_tokens") if isinstance(usage, dict) else None,
        "error": _scrub(str(error_detail or ""), secrets) or None,
    }


def _openai_payload() -> dict[str, Any]:
    return {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 32,
        "stream": False,
    }


def probe_opencode_go(
    keys: Sequence[str], all_secrets: Sequence[str], timeout_seconds: float
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    endpoint = "https://opencode.ai/zen/go/v1/chat/completions"
    for index, key in enumerate(keys, start=1):
        result = _post_json(
            channel="opencode_go",
            key_label=f"opencode_go_key_{index}",
            url=endpoint,
            api_key=key,
            payload=_openai_payload(),
            all_secrets=all_secrets,
            timeout_seconds=timeout_seconds,
        )
        attempts.append(result)
        if result["ok"]:
            return {
                "channel": "opencode_go",
                "ok": True,
                "selected_attempt": len(attempts) - 1,
                "attempts": attempts,
                "unused_keys": len(keys) - index,
            }
    return {
        "channel": "opencode_go",
        "ok": False,
        "selected_attempt": None,
        "attempts": attempts,
        "unused_keys": 0,
    }


def probe_ollama(
    key: str, all_secrets: Sequence[str], timeout_seconds: float
) -> dict[str, Any]:
    result = _post_json(
        channel="ollama",
        key_label="ollama_key_1",
        url="https://ollama.com/api/chat",
        api_key=key,
        payload={
            "model": "deepseek-v4-pro:cloud",
            "messages": [{"role": "user", "content": TEST_PROMPT}],
            "think": False,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 32},
        },
        all_secrets=all_secrets,
        timeout_seconds=timeout_seconds,
    )
    return {"channel": "ollama", "ok": result["ok"], "attempts": [result]}


def probe_deepseek(
    key: str, all_secrets: Sequence[str], timeout_seconds: float
) -> dict[str, Any]:
    result = _post_json(
        channel="deepseek",
        key_label="deepseek_key_1",
        url="https://api.deepseek.com/chat/completions",
        api_key=key,
        payload=_openai_payload(),
        all_secrets=all_secrets,
        timeout_seconds=timeout_seconds,
    )
    return {"channel": "deepseek", "ok": result["ok"], "attempts": [result]}


def run_probe(
    *,
    keys_path: Path,
    output_path: Path,
    timeout_seconds: float,
    only_channels: Sequence[str] = (),
) -> dict[str, Any]:
    keys = load_channel_keys(keys_path)
    all_secrets = [secret for values in keys.values() for secret in values]
    started_at = datetime.now(timezone.utc).isoformat()
    selected = tuple(only_channels) or ("opencode_go", "ollama", "deepseek")
    jobs = {
        "opencode_go": lambda: probe_opencode_go(
            keys["opencode"], all_secrets, timeout_seconds
        ),
        "ollama": lambda: probe_ollama(
            keys["ollama"][0], all_secrets, timeout_seconds
        ),
        "deepseek": lambda: probe_deepseek(
            keys["deepseek"][0], all_secrets, timeout_seconds
        ),
    }
    _require(not (set(selected) - set(jobs)), "unknown channel selection")
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        futures = [executor.submit(jobs[channel]) for channel in selected]
        channel_results = [future.result() for future in futures]

    report = {
        "schema_version": "neuroclaw.deepseek-v4-pro-channel-probe.v2",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model_requested": {
            "opencode_go": "deepseek-v4-pro",
            "ollama": "deepseek-v4-pro:cloud",
            "deepseek": "deepseek-v4-pro",
        },
        "channels_tested": list(selected),
        "test_prompt": TEST_PROMPT,
        "thinking_mode": "disabled_for_connectivity_and_latency_probe",
        "secret_handling": {
            "keys_file_persisted_in_report": False,
            "key_values_persisted_or_printed": False,
            "loaded_key_counts": {name: len(values) for name, values in keys.items()},
        },
        "channels": channel_results,
        "all_tested_channels_ok": all(
            bool(result["ok"]) for result in channel_results
        ),
        "all_three_channels_tested": set(selected)
        == {"opencode_go", "ollama", "deepseek"},
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=Path, default=DEFAULT_KEYS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--only",
        action="append",
        choices=("opencode_go", "ollama", "deepseek"),
        default=[],
        help="Probe only the selected channel; repeat to select more than one.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_probe(
        keys_path=args.keys.resolve(),
        output_path=args.output,
        timeout_seconds=args.timeout_seconds,
        only_channels=args.only,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
