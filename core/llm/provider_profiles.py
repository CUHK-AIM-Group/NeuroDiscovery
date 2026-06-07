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
    "qwen",
    "dashscope",
    "baichuan",
    "zhipu",
    "glm",
    "doubao",
    "ark",
    "openrouter",
    "together",
    "groq",
    "fireworks",
    "ollama",
    "llamacpp",
}

INSTALLER_PROVIDER_CHOICES = [
    ("deepseek", "DeepSeek"),
    ("minimax", "MiniMax"),
    ("kimi", "Kimi / Moonshot"),
    ("qwen", "Qwen / DashScope"),
    ("baichuan", "Baichuan"),
    ("zhipu", "Zhipu GLM"),
    ("doubao", "Doubao / Ark"),
    ("openrouter", "OpenRouter"),
    ("together", "Together AI"),
    ("groq", "Groq"),
    ("fireworks", "Fireworks AI"),
    ("ollama", "Ollama OpenAI-compatible"),
    ("llamacpp", "llama.cpp OpenAI-compatible"),
]


OPENAI_COMPATIBLE_PROVIDER_PROFILES: dict[str, dict[str, Any]] = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-v4-flash",
        "tool_calling": "supported",
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
        "tool_calling": "supported",
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
        "tool_calling": "supported",
        "models": [
            {"provider": "kimi", "model": "kimi-k2.6", "label": "Kimi / kimi-k2.6"},
            {"provider": "kimi", "model": "moonshot-v1-8k", "label": "Kimi / moonshot-v1-8k"},
            {"provider": "kimi", "model": "moonshot-v1-32k", "label": "Kimi / moonshot-v1-32k"},
            {"provider": "kimi", "model": "moonshot-v1-128k", "label": "Kimi / moonshot-v1-128k"},
        ],
    },
    "qwen": {
        "label": "Qwen / DashScope",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "default_model": "qwen-plus",
        "tool_calling": "supported",
        "models": [
            {"provider": "qwen", "model": "qwen-plus", "label": "Qwen / qwen-plus"},
            {"provider": "qwen", "model": "qwen-max", "label": "Qwen / qwen-max"},
            {"provider": "qwen", "model": "qwen-turbo", "label": "Qwen / qwen-turbo"},
            {"provider": "qwen", "model": "qwq-plus", "label": "Qwen / qwq-plus"},
        ],
    },
    "baichuan": {
        "label": "Baichuan",
        "base_url": "https://api.baichuan-ai.com/v1",
        "api_key_env": "BAICHUAN_API_KEY",
        "default_model": "Baichuan4-Turbo",
        "tool_calling": "supported",
        "models": [
            {"provider": "baichuan", "model": "Baichuan4-Turbo", "label": "Baichuan / Baichuan4-Turbo"},
            {"provider": "baichuan", "model": "Baichuan4-Air", "label": "Baichuan / Baichuan4-Air"},
            {"provider": "baichuan", "model": "Baichuan3-Turbo", "label": "Baichuan / Baichuan3-Turbo"},
        ],
    },
    "zhipu": {
        "label": "Zhipu GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "ZHIPUAI_API_KEY",
        "default_model": "glm-4-flash",
        "tool_calling": "supported",
        "models": [
            {"provider": "zhipu", "model": "glm-4-flash", "label": "Zhipu GLM / glm-4-flash"},
            {"provider": "zhipu", "model": "glm-4-plus", "label": "Zhipu GLM / glm-4-plus"},
            {"provider": "zhipu", "model": "glm-4-air", "label": "Zhipu GLM / glm-4-air"},
        ],
    },
    "doubao": {
        "label": "Doubao / Ark",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_env": "ARK_API_KEY",
        "default_model": "doubao-seed-1-6-250615",
        "tool_calling": "supported",
        "models": [
            {"provider": "doubao", "model": "doubao-seed-1-6-250615", "label": "Doubao / doubao-seed-1-6-250615"},
            {"provider": "doubao", "model": "doubao-seed-1-6-thinking-250715", "label": "Doubao / doubao-seed-1-6-thinking-250715"},
            {"provider": "doubao", "model": "doubao-1-5-pro-32k-250115", "label": "Doubao / doubao-1-5-pro-32k-250115"},
        ],
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "openai/gpt-4o-mini",
        "tool_calling": "supported",
        "models": [
            {"provider": "openrouter", "model": "openai/gpt-4o-mini", "label": "OpenRouter / openai/gpt-4o-mini"},
            {"provider": "openrouter", "model": "anthropic/claude-3.5-sonnet", "label": "OpenRouter / anthropic/claude-3.5-sonnet"},
            {"provider": "openrouter", "model": "deepseek/deepseek-chat", "label": "OpenRouter / deepseek/deepseek-chat"},
        ],
    },
    "together": {
        "label": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "tool_calling": "supported",
        "models": [
            {"provider": "together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "label": "Together / Llama-3.3-70B-Instruct-Turbo-Free"},
            {"provider": "together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "label": "Together / Llama-3.3-70B-Instruct-Turbo"},
            {"provider": "together", "model": "deepseek-ai/DeepSeek-V3", "label": "Together / DeepSeek-V3"},
        ],
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "tool_calling": "supported",
        "models": [
            {"provider": "groq", "model": "llama-3.3-70b-versatile", "label": "Groq / llama-3.3-70b-versatile"},
            {"provider": "groq", "model": "llama-3.1-8b-instant", "label": "Groq / llama-3.1-8b-instant"},
            {"provider": "groq", "model": "deepseek-r1-distill-llama-70b", "label": "Groq / deepseek-r1-distill-llama-70b"},
        ],
    },
    "fireworks": {
        "label": "Fireworks AI",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "default_model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "tool_calling": "supported",
        "models": [
            {"provider": "fireworks", "model": "accounts/fireworks/models/llama-v3p1-70b-instruct", "label": "Fireworks / llama-v3p1-70b-instruct"},
            {"provider": "fireworks", "model": "accounts/fireworks/models/llama-v3p1-8b-instruct", "label": "Fireworks / llama-v3p1-8b-instruct"},
            {"provider": "fireworks", "model": "accounts/fireworks/models/deepseek-v3", "label": "Fireworks / deepseek-v3"},
        ],
    },
    "ollama": {
        "label": "Ollama",
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "",
        "default_model": "llama3.1",
        "no_api_key_required": True,
        "tool_calling": "model_dependent",
        "models": [
            {"provider": "ollama", "model": "llama3.1", "label": "Ollama / llama3.1"},
            {"provider": "ollama", "model": "llama3.2", "label": "Ollama / llama3.2"},
            {"provider": "ollama", "model": "qwen2.5", "label": "Ollama / qwen2.5"},
            {"provider": "ollama", "model": "deepseek-r1", "label": "Ollama / deepseek-r1"},
        ],
    },
    "llamacpp": {
        "label": "llama.cpp",
        "base_url": "http://localhost:8080/v1",
        "api_key_env": "",
        "default_model": "local-model",
        "no_api_key_required": True,
        "tool_calling": "model_dependent",
        "models": [
            {"provider": "llamacpp", "model": "local-model", "label": "llama.cpp / local-model"},
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
OPENAI_COMPATIBLE_PROVIDER_PROFILES["dashscope"] = {
    **OPENAI_COMPATIBLE_PROVIDER_PROFILES["qwen"],
    "label": "DashScope / Qwen",
    "models": [
        {**item, "provider": "dashscope"}
        for item in OPENAI_COMPATIBLE_PROVIDER_PROFILES["qwen"]["models"]
    ],
}
OPENAI_COMPATIBLE_PROVIDER_PROFILES["glm"] = {
    **OPENAI_COMPATIBLE_PROVIDER_PROFILES["zhipu"],
    "label": "GLM / Zhipu",
    "models": [
        {**item, "provider": "glm"}
        for item in OPENAI_COMPATIBLE_PROVIDER_PROFILES["zhipu"]["models"]
    ],
}
OPENAI_COMPATIBLE_PROVIDER_PROFILES["ark"] = {
    **OPENAI_COMPATIBLE_PROVIDER_PROFILES["doubao"],
    "label": "Ark / Doubao",
    "models": [
        {**item, "provider": "ark"}
        for item in OPENAI_COMPATIBLE_PROVIDER_PROFILES["doubao"]["models"]
    ],
}

for _provider, _profile in OPENAI_COMPATIBLE_PROVIDER_PROFILES.items():
    for _model in _profile.get("models", []):
        _model.setdefault("base_url", _profile.get("base_url"))
        _model.setdefault("api_key_env", _profile.get("api_key_env"))
        _model.setdefault("openai_compatible", True)
        _model.setdefault("tool_calling", _profile.get("tool_calling", "supported"))
        if _profile.get("no_api_key_required"):
            _model.setdefault("no_api_key_required", True)


def canonical_provider(provider: str) -> str:
    """Return the normalized provider key used by the runtime."""
    value = str(provider or "").strip().lower()
    if value in {"moonshotai", "moonshot-ai"}:
        return "moonshot"
    if value in {"kimi-ai", "kimi_k2"}:
        return "kimi"
    if value in {"aliyun", "aliyun-bailian", "bailian", "tongyi"}:
        return "qwen"
    if value in {"zhipuai", "bigmodel", "bigmodelcn"}:
        return "zhipu"
    if value in {"volcengine", "volcano", "volcark", "byteark"}:
        return "ark"
    if value in {"llama.cpp", "llama_cpp", "llama-cpp"}:
        return "llamacpp"
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
    llm_cfg.setdefault("tool_calling", profile.get("tool_calling", "supported"))
    if profile.get("no_api_key_required"):
        llm_cfg.setdefault("no_api_key_required", True)

    if not llm_cfg.get("available_models"):
        llm_cfg["available_models"] = profile.get("models", [])
