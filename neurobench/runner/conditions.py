"""Condition matrix for the 10,000-run NeuroBench harness campaign.

One condition is ``(model, driver, harness on/off)`` and owns one report model
name. Report model names keep the existing scorer's ``<base>_withskills`` /
``<base>_noskills`` convention so the shipped NeuroClaw scorer can read the
reports unchanged (``core/agent/main.py`` ``_split_skill_condition``).

The two arms differ in capability, not only prompt text: the harness arm exposes
the on-demand skill tools (``search_skills`` / ``read_skill``); neither arm
preloads skill files. Discovery and loading are recorded as actual tool calls.
"""
from __future__ import annotations

import dataclasses


DRIVER_HTTP_CHAT = "http_chat"
DRIVER_CODEX_CLI = "codex_cli"
DRIVER_CLAUDE_CODE = "claude_code"

WITH_SKILLS_SUFFIX = "_withskills"
NO_SKILLS_SUFFIX = "_noskills"


@dataclasses.dataclass(frozen=True)
class ConditionSpec:
    key: str
    group: int
    model: str
    channel: str
    driver: str
    harness_enabled: bool
    report_model: str
    reasoning_effort: str = "high"
    request_overrides: dict | None = None

    @property
    def variant(self) -> str:
        return "harness" if self.harness_enabled else "baseline"


def _report_model(model: str, harness_enabled: bool, *, prefix: str = "") -> str:
    """Scorer-compatible report model name.

    ``/`` is replaced by ``.`` because the scorer slugifies ``/`` to ``_`` and a
    directory separator must never appear inside a report model name. ``prefix``
    (the driver) keeps Group 2's CLI runs from colliding with Group 1's direct-HTTP
    runs for the same model, since the scorer would otherwise merge them.
    """
    base = model.replace("/", ".").replace(":", ".")
    if prefix:
        base = f"{prefix}.{base}"
    return f"{base}{WITH_SKILLS_SUFFIX if harness_enabled else NO_SKILLS_SUFFIX}"


# Group 1: eight models x {harness on, off} on the direct HTTP channel.
# The same eight models the previous channel survey measured.
GROUP1_MODELS: list[tuple[str, str, dict]] = [
    ("openai/openai/gpt-5.6-sol", "https://inference-api.nvidia.com/v1", {"nvidia": True}),
    ("azure/anthropic/claude-opus-4-8", "https://inference-api.nvidia.com/v1", {"nvidia": True}),
    ("xai/xai/grok-4.7", "https://inference-api.nvidia.com/v1", {"nvidia": True}),
    ("gcp/google/gemini-3.1-pro-preview", "https://inference-api.nvidia.com/v1", {"nvidia": True}),
    ("deepseek-v4-pro", "https://api.deepseek.com", {"deepseek": True}),
    ("glm-5.3", "https://ollama.com/v1", {"ollama": True}),
    ("minimax-m3", "https://ollama.com/v1", {"ollama": True}),
    ("kimi-k3", "https://ollama.com/v1", {"ollama": True}),
]

# Group 2: one model per external CLI, harness on and off.
GROUP2_MODELS: list[tuple[str, str, str, dict]] = [
    ("openai/openai/gpt-5.6-sol", DRIVER_CODEX_CLI, "https://inference-api.nvidia.com/v1", {"nvidia": True}),
    ("azure/anthropic/claude-opus-4-8", DRIVER_CLAUDE_CODE, "https://inference-api.nvidia.com/v1", {"nvidia": True}),
]

# Group 1 request shaping, keyed by model. Recorded because each was measured:
#   gpt-5.6-sol rejects ``temperature`` (HTTP 400) so it is omitted.
#   grok-4.7 carries a gateway-injected fixed prefix, reported separately.
#   gemini rejects ``tool_choice=any``; ``auto``/``required`` only.
GROUP1_REQUEST_OVERRIDES: dict[str, dict] = {
    "openai/openai/gpt-5.6-sol": {"omit_temperature": True},
    "deepseek-v4-pro": {"omit_temperature": True,
                        "extra_body": {"thinking": {"type": "enabled"}}},
}


def build_condition_table() -> dict[str, ConditionSpec]:
    table: dict[str, ConditionSpec] = {}
    for model, _base_url, _channel in GROUP1_MODELS:
        overrides = {**GROUP1_REQUEST_OVERRIDES.get(model, {}), "reasoning_effort": "high"}
        for harness_enabled in (True, False):
            arm = "on" if harness_enabled else "off"
            key = f"g1__{model.replace('/', '.')}__{arm}"
            table[key] = ConditionSpec(
                key=key,
                group=1,
                model=model,
                channel=_base_url,
                driver=DRIVER_HTTP_CHAT,
                harness_enabled=harness_enabled,
                report_model=_report_model(model, harness_enabled),
                request_overrides=dict(overrides) or None,
                reasoning_effort="high",
            )
    for model, driver, base_url, _channel in GROUP2_MODELS:
        for harness_enabled in (True, False):
            arm = "on" if harness_enabled else "off"
            key = f"g2__{driver}__{model.replace('/', '.')}__{arm}"
            table[key] = ConditionSpec(
                key=key,
                group=2,
                model=model,
                channel=base_url,
                driver=driver,
                harness_enabled=harness_enabled,
                report_model=_report_model(model, harness_enabled, prefix=driver.replace("_", "")),
                reasoning_effort="high",
            )
    names = [condition.report_model for condition in table.values()]
    if len(names) != len(set(names)):
        raise RuntimeError("report model names are not unique across conditions")
    return table


CONDITIONS: dict[str, ConditionSpec] = build_condition_table()


def condition_matrix() -> dict:
    """A serializable description of the campaign, for the manifest."""
    return {
        "group1": {
            "models": [model for model, _b, _c in GROUP1_MODELS],
            "arms": ["harness_on", "harness_off"],
            "tasks": 500,
            "runs": len(GROUP1_MODELS) * 2 * 500,
        },
        "group2": {
            "drivers": [driver for _m, driver, _b, _c in GROUP2_MODELS],
            "models": [model for model, _d, _b, _c in GROUP2_MODELS],
            "arms": ["harness_on", "harness_off"],
            "tasks": 500,
            "runs": len(GROUP2_MODELS) * 2 * 500,
        },
        "total_conditions": len(CONDITIONS),
        "total_runs": len(GROUP1_MODELS) * 2 * 500 + len(GROUP2_MODELS) * 2 * 500,
    }
