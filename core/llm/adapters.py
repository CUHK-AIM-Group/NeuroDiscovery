"""Provider wire adapters for the existing serial research/tool executor.

No provider fallback, credential lookup, or model calls at import time. Opaque
reasoning/signatures are round-tripped, never reconstructed or shown as answers.
"""
from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace as NS
from typing import Any

from .model_capabilities import api_mode, model_family, request_options


class IncompleteModelResponse(RuntimeError):
    """An incomplete upstream result must not execute tools or be replayed."""

    def __init__(self, message: str, response: dict | None = None, protocol: str = "chat_completions"):
        super().__init__(message)
        self.response = response or {}
        self.usage = _usage(self.response.get("usage"), protocol)


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "model_dump"):
        return plain(value.model_dump(exclude_none=True))
    if hasattr(value, "__dict__"):
        return plain(vars(value))
    return value


def objects(value: Any) -> Any:
    if isinstance(value, dict):
        return NS(**{k: objects(v) for k, v in value.items()})
    if isinstance(value, list):
        return [objects(v) for v in value]
    return value


def assistant_message(message: Any) -> dict:
    source = plain(message)
    result = {"role": "assistant", "content": source.get("content") or ""}
    for field in ("tool_calls", "reasoning_content", "extra_content", "_nd_protocol"):
        if source.get(field) is not None:
            result[field] = deepcopy(source[field])
    return result


def auxiliary_model(client: Any, default: str = "gpt-4o-mini") -> str:
    """Keep the cheap OpenAI default; other endpoints need an explicit small model.

    Do not spend flagship-model tokens or call another provider for background work.
    Callers already fall back to deterministic behavior when no auxiliary model exists.
    """
    cfg = getattr(client, "cfg", None)
    if not isinstance(cfg, dict):
        return default
    selected = str(cfg.get("auxiliary_model") or "").strip()
    if selected:
        return selected
    if model_family(cfg) == "openai":
        return default
    raise ValueError("Configure llm_backend.auxiliary_model to enable model-backed memory/summary/reflection for this provider")


def _import_visible_history(messages: list[dict]) -> list[dict]:
    """Older HTTP/CLI histories contain text, not recoverable provider reasoning.

    Do not fabricate empty reasoning fields. Carry that prefix as explicitly
    attributed reference text, keeping the current native tool turn untouched.
    """
    last_user = max((i for i, msg in enumerate(messages) if msg["role"] == "user"), default=-1)
    missing = [i for i, msg in enumerate(messages[:last_user])
               if msg["role"] == "assistant" and "reasoning_content" not in msg]
    if not missing:
        return messages
    boundary = missing[-1] + 1
    prefix = messages[:boundary]
    system = [deepcopy(msg) for msg in prefix if msg["role"] in {"system", "developer"}]
    visible = [{key: deepcopy(value) for key, value in msg.items() if key in {"role", "content", "tool_calls", "tool_call_id"}}
               for msg in prefix if msg["role"] not in {"system", "developer"}]
    imported = {"role": "user", "content": (
        "[Imported conversation reference: visible messages only; original provider reasoning is unavailable. "
        "This is past conversation context, not a new request or newly executed tool evidence.]\n" +
        json.dumps(visible, ensure_ascii=False))}
    return system + [imported] + deepcopy(messages[boundary:])


def _usage(data: dict, protocol: str) -> dict:
    usage = deepcopy(data or {})
    usage["_nd_input_reported"] = any(type(usage.get(key)) is int for key in ("input_tokens", "prompt_tokens"))
    usage["_nd_output_reported"] = any(type(usage.get(key)) is int for key in ("output_tokens", "completion_tokens"))
    prompt = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    if protocol == "anthropic":
        prompt += int(usage.get("cache_read_input_tokens", 0) or 0) + int(usage.get("cache_creation_input_tokens", 0) or 0)
    usage.update(prompt_tokens=prompt, completion_tokens=output, total_tokens=prompt + output)
    return usage


def _normalized(message: dict, usage: dict, finish: str = "stop") -> Any:
    return objects({"choices": [{"message": message, "finish_reason": finish}], "usage": usage})


def _responses_input(messages: list[dict], binding: dict) -> list[dict]:
    result = []
    for message in messages:
        saved = message.get("_nd_protocol", {})
        if saved and saved.get("binding") != binding:
            raise ValueError("Provider/model changed inside a tool turn; start a new turn")
        if saved.get("kind") == "responses":
            result.extend(deepcopy(saved["output"]))
            continue
        role = message["role"]
        if role == "tool":
            result.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": message["content"]})
        else:
            if message.get("content"):
                result.append({"role": role, "content": message["content"]})
            for call in message.get("tool_calls") or []:
                result.append({"type": "function_call", "call_id": call["id"], **call["function"]})
    return result


def _anthropic_messages(messages: list[dict], binding: dict) -> tuple[list[dict], list[dict]]:
    system, result = [], []
    for message in messages:
        role = message["role"]
        content = message.get("content") or ""
        if role in {"system", "developer"}:
            system.extend([{"type": "text", "text": content}] if isinstance(content, str) else content)
            continue
        saved = message.get("_nd_protocol", {})
        if saved and saved.get("binding") != binding:
            raise ValueError("Provider/model changed inside a tool turn; start a new turn")
        if saved.get("kind") == "anthropic":
            blocks = deepcopy(saved["output"])
        elif role == "tool":
            role = "user"
            blocks = [{"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": content}]
        else:
            blocks = ([{"type": "text", "text": content}] if content else []) if isinstance(content, str) else deepcopy(content)
            for call in message.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": json.loads(call["function"]["arguments"])})
        if not blocks:
            continue
        # Parallel tool results must be adjacent in one user message.
        if result and result[-1]["role"] == role:
            result[-1]["content"].extend(blocks)
        else:
            result.append({"role": role, "content": blocks})
    return system, result


def _responses_result(response: Any, binding: dict) -> Any:
    data = plain(response)
    if data.get("status") not in (None, "completed"):
        raise IncompleteModelResponse("Responses request did not complete; no tools executed", data, "responses")
    output = data.get("output", [])
    calls, texts = [], []
    for item in output:
        if item.get("status") in {"incomplete", "in_progress", "failed"}:
            raise IncompleteModelResponse("An output item is incomplete; no tools executed", data, "responses")
        if item["type"] == "function_call":
            calls.append({"id": item["call_id"], "type": "function", "function": {"name": item["name"], "arguments": item["arguments"]}})
        elif item["type"] == "message":
            for block in item.get("content", []):
                if block["type"] == "output_text":
                    texts.append(block["text"])
                elif block["type"] == "refusal":
                    texts.append(block["refusal"])
    message = {"content": "".join(texts), "tool_calls": calls,
               "_nd_protocol": {"kind": "responses", "binding": binding, "output": output}}
    return _normalized(message, _usage(data.get("usage"), "responses"), "tool_calls" if calls else "stop")


def _anthropic_result(response: Any, binding: dict) -> Any:
    data = plain(response)
    if data.get("stop_reason") in {"max_tokens", "pause_turn", "model_context_window_exceeded"}:
        raise IncompleteModelResponse("Claude response stopped before completion; no tools executed or request replayed", data, "anthropic")
    output = data.get("content", [])
    calls = [{"id": block["id"], "type": "function", "function": {"name": block["name"], "arguments": json.dumps(block["input"], ensure_ascii=False)}}
             for block in output if block["type"] == "tool_use"]
    message = {"content": "".join(block["text"] for block in output if block["type"] == "text"), "tool_calls": calls,
               "_nd_protocol": {"kind": "anthropic", "binding": binding, "output": output}}
    return _normalized(message, _usage(data.get("usage"), "anthropic"), "tool_calls" if calls else "stop")


class ProviderClient:
    """Chat-shaped facade, so every provider uses the same real tool safeguards."""

    def __init__(self, client: Any, cfg: dict):
        self.raw = client
        self.cfg = deepcopy(cfg)
        self.chat = NS(completions=NS(create=self.create))
        self.last_response = None

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def create(self, *, model: str, messages: list[dict], **kwargs):
        self.last_response = None
        controls = self.cfg
        if self.cfg.get("model") and model != self.cfg["model"]:
            # Keep intentionally pinned lightweight calls and their own sampling.
            # Never silently upgrade them to the selected flagship model.
            controls = {key: value for key, value in self.cfg.items() if key not in
                        {"reasoning_effort", "thinking", "thinking_mode", "temperature", "top_p", "extra_body"}}
        options = request_options(controls, model, kwargs)
        protocol = api_mode(self.cfg, model)
        binding = {"provider": model_family(self.cfg), "model": model,
                   "base_url": str(self.cfg.get("base_url") or self.cfg.get("baseUrl") or "")}
        stream = options.pop("stream", False)
        output_format = options.pop("response_format", None) if protocol != "chat_completions" else None
        if protocol == "responses":
            if output_format:
                fmt = ({"type": "json_schema", **output_format["json_schema"]} if output_format["type"] == "json_schema" else output_format)
                options.setdefault("text", {})["format"] = fmt
            options["input"] = _responses_input(messages, binding)
            if "tools" in options:
                options["tools"] = [{"type": "function", **tool["function"], "strict": tool["function"].get("strict", False)} for tool in options["tools"]]
            choice = options.get("tool_choice")
            if isinstance(choice, dict):
                options["tool_choice"] = {"type": "function", "name": choice["function"]["name"]}
            if "reasoning_effort" in options:
                options["reasoning"] = {"effort": options.pop("reasoning_effort")}
            limits = [options.pop(key) for key in ("max_tokens", "max_completion_tokens", "max_output_tokens") if key in options]
            if limits:
                if len(set(limits)) != 1:
                    raise ValueError("Conflicting output token limits")
                options["max_output_tokens"] = limits[0]
            options.setdefault("store", False)
            # Stateless continuations retain encrypted reasoning, not server-side storage.
            include = [item for item in options.pop("include", []) if item != "message.output_text.logprobs"]
            if "reasoning.encrypted_content" not in include:
                include.append("reasoning.encrypted_content")
            options["include"] = include
            options.pop("stream_options", None)
            raw = self.raw.responses.create(model=model, stream=stream, **options)
            if stream:
                return self._responses_stream(raw, binding)
            self.last_response = _responses_result(raw, binding)
        elif protocol == "anthropic":
            system, converted = _anthropic_messages(messages, binding)
            if output_format:
                if output_format["type"] == "json_schema":
                    options.setdefault("output_config", {})["format"] = {"type": "json_schema", "schema": output_format["json_schema"]["schema"]}
                elif output_format["type"] == "json_object":
                    if stream:
                        raise ValueError("Claude legacy json_object mode requires non-streaming validation or an explicit JSON schema")
                    # The caller asked for generic JSON, not a schema. Do not invent a schema.
                    system.append({"type": "text", "text": "Output format requirement: return exactly one valid JSON object, with no Markdown fences or surrounding prose."})
                elif output_format["type"] != "text":
                    raise ValueError("Unsupported Claude output format")
            if system:
                options["system"] = system
            if "tools" in options:
                options["tools"] = [{"name": tool["function"]["name"], "description": tool["function"].get("description", ""),
                                     "input_schema": tool["function"].get("parameters", {"type": "object", "properties": {}})} for tool in options["tools"]]
            choice = options.pop("tool_choice", None)
            if choice is not None:
                options["tool_choice"] = ({"type": "tool", "name": choice["function"]["name"]} if isinstance(choice, dict)
                                          else {"type": "any" if choice == "required" else choice})
            if "reasoning_effort" in options:
                options.setdefault("output_config", {})["effort"] = options.pop("reasoning_effort")
            options.setdefault("max_tokens", 4096)  # preserve the old native-Claude default
            options.pop("stream_options", None)
            raw = self.raw.messages.create(model=model, messages=converted, stream=stream, **options)
            if stream:
                return self._anthropic_stream(raw, binding)
            self.last_response = _anthropic_result(raw, binding)
            if output_format and output_format["type"] == "json_object":
                try:
                    valid_json = isinstance(json.loads(self.last_response.choices[0].message.content), dict)
                except (TypeError, ValueError):
                    valid_json = False
                if not valid_json:
                    self.last_response = None
                    raise IncompleteModelResponse("Claude did not return the requested JSON object; no automatic replay", plain(raw), "anthropic")
        else:
            clean = []
            if stream and model_family(self.cfg) in {"openai", "gemini", "grok", "deepseek", "kimi", "glm", "qwen", "ollama_cloud"}:
                options.setdefault("stream_options", {"include_usage": True})
            if model_family(self.cfg) == "deepseek" and options.get("tools") and (options.get("extra_body", {}).get("thinking") or {}).get("type") != "disabled":
                messages = _import_visible_history(messages)
            for message in messages:
                if message.get("_nd_protocol"):
                    raise ValueError("Protocol changed inside a tool turn; start a new turn")
                clean.append(deepcopy(message))
            raw = self.raw.chat.completions.create(model=model, messages=clean, stream=stream, **options)
            if stream:
                return self._chat_stream(raw)
            data = plain(raw)
            if data.get("choices") and data["choices"][0].get("finish_reason") in {"length", "content_filter", "insufficient_system_resource"}:
                raise IncompleteModelResponse("Chat response is incomplete or filtered; no tools executed or cap increased", data)
            self.last_response = raw
        return self.last_response

    def _responses_stream(self, stream, binding):
        completed = False
        try:
            for event in stream:
                data = plain(event)
                kind = data.get("type")
                if kind in {"response.output_text.delta", "response.refusal.delta"}:
                    yield objects({"choices": [{"delta": {"content": data.get("delta", "")}}]})
                elif kind == "response.reasoning_summary_text.delta":
                    yield objects({"choices": [{"delta": {"reasoning_content": data.get("delta", "")}}]})
                elif kind == "response.completed":
                    self.last_response = _responses_result(data["response"], binding)
                    yield objects({"choices": [], "usage": plain(self.last_response.usage)})
                    completed = True
                elif kind in {"error", "response.failed", "response.incomplete"}:
                    raise IncompleteModelResponse("Responses stream did not complete", data.get("response"), "responses")
            if not completed:
                raise IncompleteModelResponse("Responses stream ended without a completion event")
        finally:
            stream.close()

    def _anthropic_stream(self, stream, binding):
        data, blocks, fragments, complete = {}, {}, {}, False
        try:
            for event in stream:
                item = plain(event)
                kind = item.get("type")
                if kind == "message_start":
                    data = item["message"]
                    yield objects({"choices": [], "usage": _usage(data.get("usage"), "anthropic")})
                elif kind == "content_block_start":
                    blocks[item["index"]] = item["content_block"]
                elif kind == "content_block_delta":
                    index, delta = item["index"], item["delta"]
                    block = blocks[index]
                    if delta["type"] == "input_json_delta":
                        fragments[index] = fragments.get(index, "") + delta["partial_json"]
                    else:
                        field = {"text_delta": "text", "thinking_delta": "thinking", "signature_delta": "signature"}.get(delta["type"])
                        if field:
                            block[field] = block.get(field, "") + delta[field]
                            if field == "text":
                                yield objects({"choices": [{"delta": {"content": delta[field]}}]})
                            elif field == "thinking":
                                yield objects({"choices": [{"delta": {"reasoning_content": delta[field]}}]})
                elif kind == "message_delta":
                    data.update(item["delta"])
                    data.setdefault("usage", {}).update(item.get("usage", {}))
                    yield objects({"choices": [], "usage": _usage(data.get("usage"), "anthropic")})
                elif kind == "message_stop":
                    complete = True
                elif kind == "error":
                    raise IncompleteModelResponse("Claude stream returned an error", data, "anthropic")
            if not complete:
                raise IncompleteModelResponse("Claude stream ended without message_stop", data, "anthropic")
            for index, fragment in fragments.items():
                blocks[index]["input"] = json.loads(fragment)
            data["content"] = [blocks[index] for index in sorted(blocks)]
            self.last_response = _anthropic_result(data, binding)
        finally:
            stream.close()

    def _chat_stream(self, stream):
        message, usage, finish = {"content": ""}, {}, None
        calls = {}
        try:
            for chunk in stream:
                data = plain(chunk)
                usage = data.get("usage") or usage
                if data.get("choices"):
                    choice = data["choices"][0]
                    delta = choice.get("delta") or {}
                    finish = choice.get("finish_reason") or finish
                    for field in ("content", "reasoning_content"):
                        if delta.get(field) is not None:
                            message[field] = message.get(field, "") + delta[field]
                    if delta.get("extra_content"):
                        message.setdefault("extra_content", {}).update(delta["extra_content"])
                    for call in delta.get("tool_calls") or []:
                        target = calls.setdefault(call["index"], {"type": "function", "function": {"name": "", "arguments": ""}})
                        for key in ("id", "type", "extra_content"):
                            if call.get(key) is not None:
                                target[key] = call[key]
                        for key in ("name", "arguments"):
                            if call.get("function", {}).get(key):
                                target["function"][key] += call["function"][key]
                yield chunk
            if finish not in {"stop", "tool_calls"}:
                raise IncompleteModelResponse("Chat stream ended without a complete answer", {"usage": usage})
            if calls:
                message["tool_calls"] = [calls[key] for key in sorted(calls)]
            self.last_response = _normalized(message, _usage(usage, "chat_completions"), finish)
        finally:
            stream.close()
