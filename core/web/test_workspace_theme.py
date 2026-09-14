"""Offline presentation contracts. Never import the runtime or call a model."""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import json
import re
import shutil
import subprocess

import pytest

STATIC = Path(__file__).parent / "static"
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "workspace-tokens.css").read_text(encoding="utf-8") + (STATIC / "research-workspace.css").read_text(encoding="utf-8")


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []
        self.feed(HTML)

    def handle_starttag(self, tag, attrs):
        self.items.append((tag, dict(attrs)))


def test_unique_dom_ids():
    counts = Counter(attrs["id"] for _, attrs in Elements().items if "id" in attrs)
    assert not {name: count for name, count in counts.items() if count != 1}


@pytest.mark.parametrize("name", [
    "nav-new-chat", "nav-search", "nav-skills", "nav-neurooracle", "nav-settings",
    "project-list", "chat-list", "checkpoints-section", "msg-input", "upload-btn",
    "send-btn", "model-menu-btn", "generation-parameters-btn",
    "expert-study-frame", "study-results-frame", "neurooracle-frame",
])
def test_research_entry_points_remain(name):
    assert any(attrs.get("id") == name for _, attrs in Elements().items)


def test_starter_prompts_and_decorative_icons():
    elements = Elements().items
    prompts = [attrs["data-starter-prompt"] for tag, attrs in elements if "data-starter-prompt" in attrs]
    assert prompts == [
        "Plan a BIDS validation and quality-control pass for my neuroimaging dataset.",
        "Help me design an fMRI preprocessing and first-level analysis workflow.",
        "Draft a FreeSurfer cortical reconstruction QC checklist with commands.",
        "Use the NeuroOracle Graph to generate and evaluate a neuroscience hypothesis.",
    ]
    icons = [attrs for _, attrs in elements if attrs.get("class") == "starter-icon"]
    assert len(icons) == 4
    assert all(attrs.get("aria-hidden") == "true" for attrs in icons)


def test_stylesheet_is_local_and_loaded_after_base_styles():
    assert HTML.index('href="/static/research-workspace.css"') > HTML.index("</style>")
    assert "@import" not in CSS
    assert "url(http" not in CSS


def test_navigation_and_input_accessibility():
    elements = {attrs["id"]: attrs for _, attrs in Elements().items if "id" in attrs}
    assert elements["generation-parameters-btn"]["aria-controls"] == "generation-parameters-dialog"
    assert elements["sidebar-backdrop"]["type"] == "button"
    assert elements["sidebar-backdrop"]["aria-label"] == "Collapse sidebar"
    assert elements["chat-panel"]["tabindex"] == "-1"
    assert "closeMobileSidebar" in HTML
    assert "sidebarToggleBtn.focus()" in HTML
    assert "prefers-reduced-motion: reduce" in CSS
    assert ":focus-visible" in CSS
    assert "welcomeEl.style.display = 'block'" not in HTML
    assert "replacement.focus({preventScroll: true})" in HTML


def test_modes_and_legacy_storage_are_preserved():
    for mode in ("strict", "novelty_first", "weighted"):
        assert f'name="novelty-mode" value="{mode}"' in HTML
    assert HTML.count("novelty_mode: noveltyMode") == 2
    assert "neuroclaw.web.sessions.v1" in HTML
    assert "window.neuroclawDesktop" in HTML
    assert "AUTO_RESEARCH_MODE_END_TO_END" in HTML
    assert ".input-row #send-btn.stop-mode" in CSS


def test_client_javascript_syntax():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for frontend syntax verification")
    program = """
        const fs = require('node:fs');
        const vm = require('node:vm');
        const html = fs.readFileSync(process.argv[1], 'utf8');
        for (const match of html.matchAll(/<script(?:\\s[^>]*)?>([\\s\\S]*?)<\\/script>/g)) {
            if (match[1].trim()) new vm.Script(match[1]);
        }
        for (const file of process.argv.slice(2)) new vm.Script(fs.readFileSync(file, 'utf8'));
    """
    subprocess.run([node, "-e", program, str(STATIC / "index.html"),
                    str(STATIC / "hypothesis-controls.js"), str(STATIC / "model-library.js")], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("narrow", [False, True])
@pytest.mark.parametrize("desktop_collapsed", [False, True])
@pytest.mark.parametrize("drawer_open", [False, True])
def test_mobile_drawer_does_not_overwrite_desktop_preferences(narrow, desktop_collapsed, drawer_open):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for frontend behavior verification")
    functions = "\n".join(re.search(r"^  function " + name + r"\([^\n]*\)[\s\S]+?^  }", HTML, re.M)[0]
                          for name in ("applySidebarCollapsed", "sidebarWidthBounds"))
    program = """
        const vm = require('node:vm');
        const assert = require('node:assert/strict');
        const data = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
        const attributes = {};
        const toggles = {};
        const context = {
            window: {matchMedia: () => ({matches: data.narrow}), innerWidth: data.narrow ? 390 : 1280},
            mainEl: {clientWidth: data.narrow ? 390 : 1280, classList: {toggle: (name, value) => {toggles[name] = value;}}},
            document: {body: {classList: {toggle: (name, value) => {toggles['body:' + name] = value;}}}},
            state: {sidebarCollapsed: data.desktop_collapsed, mobileSidebarOpen: data.drawer_open},
            sidebarToggleBtn: {setAttribute: (name, value) => {attributes[name] = value;}},
            sidebarResizerEl: {setAttribute: () => {}}, tr: value => value,
            savePrefs: () => {throw new Error('Unexpected preference write');},
            SIDEBAR_WIDTH_MIN: 200, SIDEBAR_WIDTH_MAX: 520,
        };
        vm.createContext(context);
        vm.runInContext(data.functions, context);
        context.applySidebarCollapsed(false);
        const collapsed = data.narrow ? !data.drawer_open : data.desktop_collapsed;
        assert.equal(toggles['sidebar-collapsed'], collapsed);
        assert.equal(toggles['body:sidebar-collapsed'], collapsed);
        assert.equal(attributes['aria-expanded'], collapsed ? 'false' : 'true');
        assert.equal(context.state.sidebarCollapsed, data.desktop_collapsed);
        // A 390px drawer must not clamp the remembered desktop width to 200px.
        assert.equal(context.sidebarWidthBounds().maximum, 520);
    """
    subprocess.run([node, "-e", program], input=json.dumps({
        "functions": functions, "narrow": narrow, "desktop_collapsed": desktop_collapsed,
        "drawer_open": drawer_open,
    }), check=True, capture_output=True, text=True)


def luminance(hex_color):
    if len(hex_color) == 4:
        hex_color = "#" + "".join(c * 2 for c in hex_color[1:])
    channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    channels = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in channels]
    return sum(c * weight for c, weight in zip(channels, (.2126, .7152, .0722)))


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("foreground,background", [
    ("text", "bg"), ("text", "surface"), ("text", "surface-elev"),
    ("text-muted", "bg"), ("text-muted", "surface"), ("text-muted", "surface-elev"),
    ("accent-strong", "accent-wash"),
])
def test_text_token_contrast(theme, foreground, background):
    block = re.search(r'html\[data-theme="' + theme + r'"\]\s*\{([^}]+)\}', CSS)[1]
    palette = dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})\b", block))
    light, dark = sorted([luminance(palette[foreground]), luminance(palette[background])], reverse=True)
    assert (light + .05) / (dark + .05) >= 4.5
