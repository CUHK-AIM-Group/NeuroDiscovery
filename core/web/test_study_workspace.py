"""Presentation regressions; synthetic sessions never touch the real study DB."""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess

from fastapi.testclient import TestClient
import pytest

from core.web.discovery_study import create_preview_app

STATIC = Path(__file__).with_name("static")


class Elements(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.items = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.items.append((tag, dict(attrs)))


@pytest.mark.parametrize("filename", ["discovery-study.html", "study.html"])
def test_studies_share_client_palette_and_local_presentation_assets(filename):
    page = (STATIC / filename).read_text(encoding="utf-8")
    assert 'href="/static/workspace-tokens.css"' in page
    assert 'href="/static/study-workspace.css"' in page
    assert 'src="/static/study-workspace.js"' in page
    assert page.index("workspace-tokens.css") < page.index("study-workspace.css")
    assert 'data-study-ui=' in page


def test_discovery_controls_and_question_ids_are_not_duplicated():
    page = (STATIC / "discovery-study.html").read_text(encoding="utf-8")
    elements = Elements(page).items
    counts = Counter(attrs["id"] for _, attrs in elements if "id" in attrs)
    assert all(count == 1 for count in counts.values())
    for key in ("auth-form", "setup-form", "resume", "restore", "close-study", "resume-code", "export", "pause", "stage-title", "stage-help", "session-code", "cards-nav", "ratings-form", "save-status", "advance", "confirm-dialog", "export-complete", "export-saved", "save-progress"):
        assert counts[key] == 1
    # The legacy export dialog was replaced by the shared evaluation-export.js
    # module (aligned with the ranking study's export flow).
    assert 'id="export-dialog"' not in page
    assert 'src="/static/evaluation-export.js"' in page
    assert all(attrs.get("name") != "consent" for _, attrs in elements)
    assert '清除浏览器不会删除服务器记录' not in page
    # The storage-details disclosure block and the consent checkbox were both
    # removed per the owner request.
    assert '数据保存与评价边界' not in page
    assert '我自愿参与本次评审' not in page
    assert '<div class="sessionbar">' not in page
    toolbar = page.split('<header class="topbar">', 1)[1].split('</header>', 1)[0]
    for key in ("stage-title", "pause", "close-study", "export"):
        assert f'id="{key}"' in toolbar


def test_isolated_preview_serves_shared_assets_without_exposing_materials(tmp_path):
    with TestClient(create_preview_app(data_dir=tmp_path, record_kind="test")) as client:
        for name in ("workspace-tokens.css", "study-workspace.css", "study-workspace.js"):
            response = client.get(f"/static/{name}")
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
        assert client.get('/static/cs1_discovery_pilot_v4.json').status_code == 404


def test_layout_has_flat_panes_and_accessible_narrow_fallback():
    css = (STATIC / "study-workspace.css").read_text(encoding="utf-8")
    for text in ('data-embedded="true"', '.topbar > .brand { display: none;', 'min-height: 44px', 'grid-template-rows: minmax(0, 1fr)', 'align-items: stretch', 'overflow-y: auto', 'touch-action: pan-y', 'max-width: 820px', '.review-grid > * { width: 100%; min-width: 0;', ':focus-visible', 'prefers-reduced-motion: reduce'):
        assert text in css
    bridge = (STATIC / "study-workspace.js").read_text(encoding="utf-8")
    for prohibited in ("fetch(", "localStorage", "/api/", "innerHTML", "location.reload"):
        assert prohibited not in bridge
    assert "scrollWithWheel" in bridge
    assert "{passive: false, capture: true}" in bridge


def test_discovery_toolbar_title_does_not_use_material_context_styles():
    page = (STATIC / "discovery-study.html").read_text(encoding="utf-8")
    toolbar = page.split('<header class="topbar">', 1)[1].split('</header>', 1)[0]
    assert '<span class="study-toolbar-title">Human Evaluation 1</span>' in toolbar
    assert 'class="study-context"' not in toolbar
    css = (STATIC / "study-workspace.css").read_text(encoding="utf-8")
    assert '.topbar { min-height: 52px; justify-content: flex-start; }' in css
    assert '.study-toolbar-title { display: inline; flex: 0 0 auto;' in css
    assert '#close-study { order: -1; display: inline-flex;' in css
    material_css = (STATIC / "discovery-study.css").read_text(encoding="utf-8")
    assert '.study-context.not-consistent' in material_css


def test_extension_uses_expert_study_identity_and_shared_visual_language():
    study = (STATIC / "study.html").read_text(encoding="utf-8")
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    desktop = (STATIC.parents[2] / "desktop" / "main.js").read_text(encoding="utf-8")
    css = (STATIC / "study-workspace.css").read_text(encoding="utf-8")
    for text in ("Human Evaluation 2",):
        assert text in study
        assert text in index
        assert text in desktop
    assert "Human Evaluation 2（扩展）" in index
    assert "Human Evaluation 2（扩展）" in desktop
    assert "Hypothesis Ranking" not in index
    assert "假设排序" not in desktop
    assert 'html[data-study-ui="legacy"] .setup-card' in css
    assert 'html[data-study-ui="legacy"] .hypothesis-card' in css


def test_study_browser_behavior_contracts():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for frontend verification")
    result = subprocess.run([node, "--test", str(STATIC / "tests/study-workspace.test.cjs"), str(STATIC / "tests/discovery-i18n.test.cjs")], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
