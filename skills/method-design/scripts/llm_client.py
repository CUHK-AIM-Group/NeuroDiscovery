"""Thin LLM-call adapter for tournament.py.

Wraps OpenAI / Anthropic SDKs into a common signature:
    async def call(messages: list[{"role", "content"}]) -> str

Reads API keys from env: OPENAI_API_KEY or ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os
from typing import Awaitable, Callable

LLMCall = Callable[[list[dict[str, str]]], Awaitable[str]]


def build_llm_call(backend: str = "openai",
                   model: str = "gpt-4o-mini") -> LLMCall:
    if backend == "openai":
        return _build_openai(model)
    if backend == "anthropic":
        return _build_anthropic(model)
    raise ValueError(f"Unknown backend: {backend}")


def _build_openai(model: str) -> LLMCall:
    from openai import AsyncOpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = AsyncOpenAI(api_key=api_key)

    async def _call(messages: list[dict[str, str]]) -> str:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    return _call


def _build_anthropic(model: str) -> LLMCall:
    from anthropic import AsyncAnthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = AsyncAnthropic(api_key=api_key)

    async def _call(messages: list[dict[str, str]]) -> str:
        system = next((m["content"] for m in messages
                       if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        resp = await client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            messages=user_msgs,
            temperature=0.2,
        )
        return resp.content[0].text if resp.content else ""

    return _call
