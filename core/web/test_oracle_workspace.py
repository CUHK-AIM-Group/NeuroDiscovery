"""Offline visual-integration contracts; never open a real graph or environment."""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess

import pytest

STATIC = Path(__file__).parent / "static"
EXPLORER = (STATIC / "explore.html").read_text(encoding="utf-8")
HOST = (STATIC / "index.html").read_text(encoding="utf-8")


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []
        self.feed(EXPLORER)

    def handle_starttag(self, tag, attrs):
        self.items.append((tag, dict(attrs)))


def test_explorer_ids_are_unique_and_controls_preserved():
    elements = Elements().items
    counts = Counter(attrs["id"] for _, attrs in elements if "id" in attrs)
    assert not {key: n for key, n in counts.items() if n != 1}
    for key in ("searchInput", "atomSelect", "taskSelect", "strictChainToggle", "depthSelect", "limitInput", "edgeTypeSelect", "fontSelect", "layoutSelect", "backBtn", "sigma-container", "detailPane", "graphUpdateBtn"):
        assert counts[key] == 1
    attrs = next(attrs for _, attrs in elements if attrs.get("id") == "strictChainToggle")
    assert "disabled" in attrs
    assert 'role="tabpanel"' in EXPLORER
    assert 'aria-controls="oracleInspector"' in EXPLORER


def test_host_and_explorer_share_palette_and_dialog_styles():
    for html in (HOST, EXPLORER):
        assert html.index('href="/static/workspace-tokens.css"') > html.index("</style>")
        assert 'src="/static/client-workbench.js"' in html
        assert 'href="/static/client-workbench.css"' in html
    assert 'window.ClientWorkbench.dialog' in EXPLORER
    assert 'setProperty("--ui-font-size"' not in EXPLORER
    assert 'setSetting("labelColor", { color })' in EXPLORER
    assert EXPLORER.count('allowInvalidContainer: true') == 2
    assert 'src="/static/vendor/oracle-layout.js"' in EXPLORER
    assert 'forceatlas2@0.10.1/worker.min.js' not in EXPLORER
    assert (STATIC / "vendor/oracle-layout.LICENSE.txt").is_file()


def test_responsive_panels_and_shared_appearance_are_local():
    css = (STATIC / "oracle-workspace.css").read_text(encoding="utf-8")
    bridge = (STATIC / "oracle-workspace.js").read_text(encoding="utf-8")
    for value in ('max-width: 760px', 'data-oracle-inspector="open"', 'data-oracle-browse="closed"', 'prefers-reduced-motion: reduce', ':focus-visible'):
        assert value in css
    for value in ('fetch(', 'localStorage', 'XMLHttpRequest', '/api/'):
        assert value not in bridge
    assert 'html[data-embedded="true"] #themeBtn' in css
    for bound in ("MIN", "MAX"):
        value = re.search(rf'const TEXT_SCALE_{bound} = ([\d.]+);', HOST)[1]
        assert f'Math.{"max" if bound == "MIN" else "min"}({value.lstrip("0")},' in bridge


def test_embedded_explorer_has_one_toolbar_without_a_second_page_heading():
    css = (STATIC / "oracle-workspace.css").read_text(encoding="utf-8")
    assert 'html[data-embedded="true"] .header { display: none; }' in css
    header = re.search(r'<header class="header">([\s\S]*?)</header>', EXPLORER)[1]
    assert '<h1 class="header-sub">NeuroOracle</h1>' in header  # Standalone identity remains.
    assert 'graphCheckUpdateBtn' not in header
    menu = re.search(r'<details class="oracle-workspace-options"[\s\S]*?</details>', EXPLORER)[0]
    for key in ('graphUpdateStatus', 'graphCheckUpdateBtn', 'graphUpdateBtn', 'themeBtn'):
        assert f'id="{key}"' in menu
    assert 'aria-label="Graph actions"' in menu
    assert 'oracle-workspace-menu .update-status { display: block;' in css


@pytest.mark.parametrize("filename", ["explore.html", "oracle-workspace.js", "claim-evidence.js"])
def test_oracle_javascript_syntax(filename):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for frontend verification")
    code = r"""
        const fs=require('node:fs'),vm=require('node:vm');
        const file=process.argv[1], source=fs.readFileSync(file,'utf8');
        if(file.endsWith('.js')) new vm.Script(source);
        else for (const match of source.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) {
            if(match[1].trim()) new vm.Script(match[1]);
        }
    """
    subprocess.run([node, "-e", code, str(STATIC / filename)], check=True, capture_output=True, text=True, encoding="utf-8")


def test_oracle_interaction_regressions():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for frontend verification")
    result = subprocess.run([node, "--test", str(STATIC / "tests/oracle-workspace.test.cjs")], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
