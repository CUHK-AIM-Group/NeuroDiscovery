"""Built-in LLM provider profiles.

These profiles cover providers that expose an OpenAI-compatible Chat
Completions API. NeuroClaw keeps their provider names for UI/benchmark
reporting, while the core request path reuses the OpenAI SDK adapter.
"""

from __future__ import annotations

from typing import Any


OPENAI_COMPATIBLE_PROVIDERS = {
    "openai",
    "deepseek",
    "minimax",
    "kimi",
    "moonshot",
}


OPENAI_COMPATIBLE_PROVIDER_PROFILES: dict[str, dict[str, Any]] = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-v4-flash",
        "models": [
            {"provider": "deepseek", "model": "deepseek-v4-flash", "label": "DeepSeek / deepseek-v4-flash"},
            {"provider": "deepseek", "model": "deepseek-v4-pro", "label": "DeepSeek / deepseek-v4-pro"},
            {"provider": "deepseek", "model": "deepseek-chat", "label": "DeepSeek / deepseek-chat"},
            {"provider": "deepseek", "model": "deepseek-reasoner", "label": "DeepSeek / deepseek-reasoner"},
        ],
    },
    "minimax": {
        "label": "MiniMax",
        "base_url": "https://api.minimaxi.com/v1",
        "api_key_env": "MINIMAX_API_KEY",
        "default_model": "MiniMax-M2.7",
        "models": [
            {"provider": "minimax", "model": "MiniMax-M2.7", "label": "MiniMax / MiniMax-M2.7"},
            {"provider": "minimax", "model": "MiniMax-M2.7-highspeed", "label": "MiniMax / MiniMax-M2.7-highspeed"},
            {"provider": "minimax", "model": "MiniMax-M2.5", "label": "MiniMax / MiniMax-M2.5"},
            {"provider": "minimax", "model": "MiniMax-M2", "label": "MiniMax / MiniMax-M2"},
            {"provider": "minimax", "model": "MiniMax-M1", "label": "MiniMax / MiniMax-M1"},
            {"provider": "minimax", "model": "MiniMax-Text-01", "label": "MiniMax / MiniMax-Text-01"},
        ],
    },
    "kimi": {
        "label": "Kimi",
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "default_model": "kimi-k2.6",
        "models": [
            {"provider": "kimi", "model": "kimi-k2.6", "label": "Kimi / kimi-k2.6"},
            {"provider": "kimi", "model": "moonshot-v1-8k", "label": "Kimi / moonshot-v1-8k"},
            {"provider": "kimi", "model": "moonshot-v1-32k", "label": "Kimi / moonshot-v1-32k"},
            {"provider": "kimi", "model": "moonshot-v1-128k", "label": "Kimi / moonshot-v1-128k"},
        ],
    },
}

OPENAI_COMPATIBLE_PROVIDER_PROFILES["moonshot"] = {
    **OPENAI_COMPATIBLE_PROVIDER_PROFILES["kimi"],
    "label": "Moonshot/Kimi",
    "models": [
        {**item, "provider": "moonshot"}
        for item in OPENAI_COMPATIBLE_PROVIDER_PROFILES["kimi"]["models"]
    ],
}

for _provider, _profile in OPENAI_COMPATIBLE_PROVIDER_PROFILES.items():
    for _model in _profile.get("models", []):
        _model.setdefault("base_url", _profile.get("base_url"))
        _model.setdefault("api_key_env", _profile.get("api_key_env"))
        _model.setdefault("openai_compatible", True)


def canonical_provider(provider: str) -> str:
    """Return the normalized provider key used by the runtime."""
    value = str(provider or "").strip().lower()
    if value in {"moonshotai", "moonshot-ai"}:
        return "moonshot"
    if value in {"kimi-ai", "kimi_k2"}:
        return "kimi"
    return value


def is_openai_compatible_provider(provider: str) -> bool:
    return canonical_provider(provider) in OPENAI_COMPATIBLE_PROVIDERS


def get_openai_compatible_profile(provider: str) -> dict[str, Any] | None:
    return OPENAI_COMPATIBLE_PROVIDER_PROFILES.get(canonical_provider(provider))


def apply_openai_compatible_profile_defaults(llm_cfg: dict[str, Any]) -> None:
    """Fill endpoint/model/key defaults for named OpenAI-compatible providers."""
    provider = canonical_provider(str(llm_cfg.get("provider") or "openai"))
    llm_cfg["provider"] = provider

    profile = get_openai_compatible_profile(provider)
    if profile is None:
        return

    if not llm_cfg.get("base_url") and not llm_cfg.get("baseUrl"):
        llm_cfg["base_url"] = profile.get("base_url")
    if not llm_cfg.get("api_key_env"):
        llm_cfg["api_key_env"] = profile.get("api_key_env")
    if not llm_cfg.get("model"):
        llm_cfg["model"] = profile.get("default_model")
    llm_cfg.setdefault("openai_compatible", True)
    llm_cfg.setdefault("provider_label", profile.get("label"))

    if not llm_cfg.get("available_models"):
        llm_cfg["available_models"] = profile.get("models", [])
