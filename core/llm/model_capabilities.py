"""Reviewed model presets, not an assertion of account access or future capabilities.

Keep this independent of experiment configuration. See docs/MODEL_COMPATIBILITY.md.
"""
from __future__ import annotations

from copy import deepcopy
from urllib.parse import urlparse

from .provider_profiles import canonical_provider


PRESETS = {
    "ollama_cloud": ("Ollama Cloud", "https://ollama.com/v1", "OLLAMA_API_KEY", ["deepseek-v4.1-flash"]),
    "openai": ("OpenAI / GPT", "https://api.openai.com/v1", "OPENAI_API_KEY", ["gpt-6-astra", "gpt-5.5"]),
    "anthropic": ("Anthropic / Claude", "https://api.anthropic.com", "ANTHROPIC_API_KEY", ["claude-opus-5", "claude-sonnet-5", "claude-fable-5-1", "claude-haiku-4-5-20251001"]),
    "gemini": ("Google / Gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", ["gemini-3.8-flash"]),
    "grok": ("xAI / Grok", "https://api.x.ai/v1", "XAI_API_KEY", ["grok-4.6"]),
    "deepseek": ("DeepSeek", "https://api.deepseek.com", "DEEPSEEK_API_KEY", ["deepseek-v4-flash", "deepseek-v4-pro"]),
    "kimi": ("Kimi / Moonshot", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY", ["kimi-k2.6"]),
    "glm": ("GLM / Zhipu", "https://open.bigmodel.cn/api/paas/v4", "ZHIPUAI_API_KEY", ["glm-5.1", "glm-5", "glm-4.7"]),
    "qwen": ("Qwen / DashScope", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", ["qwen3.8-max", "qwen3.8-flash", "qwen-plus"]),
}


def model_family(cfg: dict, model: str = "") -> str:
    provider = canonical_provider(cfg.get("provider", "openai"))
    # An endpoint's dialect wins over a hosted model's name (e.g. Qwen on Ollama).
    return {"moonshot": "kimi", "dashscope": "qwen", "zhipu": "glm"}.get(provider, provider)


def api_mode(cfg: dict, model: str = "") -> str:
    mode = cfg.get("api_mode") or "auto"
    if mode not in {"auto", "chat_completions", "responses", "anthropic"}:
        raise ValueError("api_mode must be auto, chat_completions, responses or anthropic")
    if mode != "auto":
        return mode
    if model_family(cfg) == "anthropic":
        return "anthropic"
    host = urlparse(cfg.get("base_url") or cfg.get("baseUrl") or "https://api.openai.com/v1").hostname
    model = model or cfg.get("model", "")
    if model_family(cfg) == "openai" and host in {"api.openai.com", "us.api.openai.com", "eu.api.openai.com"} and model.startswith("gpt-6"):
        return "responses"
    return "chat_completions"


def capabilities(cfg: dict, model: str = "") -> dict:
    model = str(model or cfg.get("model", ""))
    family = model_family(cfg, model)
    mode = api_mode(cfg, model)
    efforts: list[str] = []
    thinking: list[str] = []
    temperature = True
    if family == "openai" and model.startswith("gpt-6"):
        efforts, temperature = ["low", "medium", "high", "xhigh", "max"], False
    elif family == "openai" and model.startswith(("gpt-5", "o1", "o3", "o4")):
        efforts = ["low", "medium", "high"]
    elif mode == "anthropic":
        if model.startswith(("claude-opus-5", "claude-sonnet-5", "claude-fable-5")):
            efforts = ["low", "medium", "high", "xhigh", "max"]
            thinking, temperature = ["adaptive"], False
        elif model.startswith(("claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-4-6")):
            efforts = ["low", "medium", "high", "max"]
            thinking = ["adaptive", "disabled"]
            if model.startswith(("claude-opus-4-7", "claude-opus-4-8")):
                efforts.insert(-1, "xhigh")
                temperature = False
    elif family == "gemini":
        efforts = ["minimal", "low", "medium", "high"]
        if model.startswith("gemini-2.5") and "pro" not in model:
            efforts.insert(0, "none")
    elif family == "grok" and model.startswith("grok-4.6"):
        efforts = ["low", "medium", "high", "xhigh"]
    elif family == "deepseek":
        efforts, thinking = ["low", "high", "max"], ["enabled", "disabled"]
    elif family == "ollama_cloud":
        efforts = ["none", "low", "medium", "high"]
    elif family in {"kimi", "glm", "qwen"}:
        thinking = ["enabled", "disabled"]
        if family == "kimi" and model.startswith(("kimi-k2.5", "kimi-k2.6")):
            temperature = False  # provider fixes this by thinking mode
        if family == "qwen" and model.startswith("qwen3.8"):
            efforts = ["none", "low", "medium", "xhigh"]
    return {"api_mode": mode, "reasoning_efforts": efforts, "thinking_modes": thinking,
            "temperature_configurable": temperature, "custom_model_allowed": True}


def provider_catalog() -> dict:
    providers = []
    for provider, (label, url, key_env, models) in PRESETS.items():
        providers.append({"provider": provider, "label": label, "base_url": url,
                          "api_key_env": key_env, "default_model": models[0], "models": list(models)})
    return {"reviewed_on": "2026-09-14", "live_verified": False, "providers": providers}


def request_options(cfg: dict, model: str, extra: dict) -> dict:
    """Map explicit runtime controls without touching cfg, budgets, or credentials."""
    out = deepcopy(extra)
    caps = capabilities(cfg, model)
    family = model_family(cfg, model)
    body = deepcopy(cfg.get("extra_body") or {})
    body.update(out.pop("extra_body", {}) or {})
    if any(key in body for key in ("model", "messages", "input", "tools")):
        raise ValueError("extra_body cannot override model identity, history, or tools")
    for field in ("reasoning_effort", "temperature", "top_p"):
        value = cfg.get(field)
        if value not in (None, "", "default"):
            out.setdefault(field, value)
    effort = out.get("reasoning_effort")
    if effort in (None, "", "default"):
        out.pop("reasoning_effort", None)
    elif caps["reasoning_efforts"] and effort not in caps["reasoning_efforts"]:
        raise ValueError(f"Unsupported reasoning_effort for {model}: use {', '.join(caps['reasoning_efforts'])}")
    elif not caps["reasoning_efforts"] and family in PRESETS:
        raise ValueError(f"No verified reasoning_effort control for {model}; use provider default")

    # Output controls are ceilings. Preserve the smaller budget of auxiliary calls.
    limits = []
    for source in (out, body):
        for field in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            if field in source:
                value = source.pop(field)
                if isinstance(value, bool) or str(value).strip() != str(int(value)) or int(value) < 1:
                    raise ValueError("Output token limits must be positive integers")
                limits.append(int(value))
    if limits and len(set(limits)) != 1:
        raise ValueError("Conflicting output token limits in the same request")
    cap = cfg.get("max_output_tokens")
    if cap not in (None, ""):
        if isinstance(cap, bool) or str(cap).strip() != str(int(cap)) or int(cap) < 1:
            raise ValueError("max_output_tokens must be a positive integer")
        limits.append(int(cap))
    if limits:
        field = "max_completion_tokens" if caps["api_mode"] == "chat_completions" and family == "openai" and model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")) else "max_tokens"
        out[field] = min(limits)

    if "temperature" in out:
        value = float(out["temperature"])
        if not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
        out["temperature"] = value

    thinking = deepcopy(cfg.get("thinking"))
    choice = cfg.get("thinking_mode")
    if choice not in (None, "", "default"):
        if choice not in caps["thinking_modes"]:
            raise ValueError(f"Unsupported thinking_mode for {model}; use provider default")
        if thinking and thinking.get("type") != choice:
            raise ValueError("Conflicting thinking and thinking_mode settings")
        thinking = {**(thinking or {}), "type": choice}
    # SDK-specific parameters belong in extra_body, never as unknown Python kwargs.
    thinking = out.pop("thinking", thinking)
    if thinking:
        if family == "qwen":
            body["enable_thinking"] = thinking["type"] == "enabled"
        elif caps["api_mode"] == "anthropic":
            out["thinking"] = thinking
        else:
            body["thinking"] = thinking
    if not caps["temperature_configurable"]:
        if cfg.get("temperature") not in (None, "", "default") or cfg.get("top_p") not in (None, "", "default"):
            raise ValueError(f"{model} requires provider-default sampling; leave temperature/top_p blank")
        # Legacy generic probes use temperature=0; do not send it to fixed-sampling models.
        for field in ("temperature", "top_p", "top_logprobs", "logprobs"):
            out.pop(field, None)
            body.pop(field, None)
    if family == "grok" and model.startswith("grok-4"):
        for field in ("presence_penalty", "frequency_penalty", "stop"):
            if field in out or field in body:
                raise ValueError(f"{field} is unsupported by Grok reasoning models")
    if family == "kimi" and model.startswith(("kimi-k2.5", "kimi-k2.6")):
        enabled = (thinking or body.get("thinking") or {}).get("type") != "disabled"
        if enabled and out.get("tool_choice", "auto") not in ("auto", "none"):
            raise ValueError("Kimi thinking mode requires tool_choice auto or none")
    if model.startswith("gpt-6") and family == "openai":
        if caps["api_mode"] == "chat_completions" and out.get("tools"):
            raise ValueError("GPT-6 tool calling requires api_mode=responses; confirm proxy support if using a custom URL")
        if urlparse(cfg.get("base_url", "")).hostname == "eu.api.openai.com" and out.get("service_tier") in {"fast", "priority"}:
            raise ValueError("GPT-6 EU data residency requires Standard processing")
    if body:
        out["extra_body"] = body
    return out


def validate_config(cfg: dict) -> dict:
    """Pure preflight; no key lookup or provider request."""
    model = str(cfg.get("model") or "").strip()
    if not model:
        raise ValueError("A model ID is required")
    request_options(cfg, model, {})
    return {**capabilities(cfg, model), "model": model, "live_verified": False}
