import re
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from core.web.server import (
    WEB_TOOL_OUTPUT_EXPORT_LIMIT,
    _client_surface_prompt,
    _response_language_prompt,
    _summarize_web_tool_events,
    create_app,
)


class AutoResearchHelpEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())

    def test_help_endpoint_handles_all_scopes_without_an_llm_call(self):
        cases = (
            ("/help data process my dataset", "data", "Raw or BIDS dataset path"),
            ("/help model prepared ROI features", "model", "Model-ready data path"),
            ("/help idea neuroscience biomarkers", "idea", "Research area, disease/population"),
            ("/help autoresearch full workflow", "end-to-end", "Overall scientific question"),
        )
        for message, mode, detail in cases:
            with self.subTest(mode=mode):
                response = self.client.post(
                    "/api/chat",
                    json={"message": message, "language": "English"},
                )
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual(body["autoresearch_mode"], mode)
                self.assertIn("Please provide", body["content"])
                self.assertIn("**Scope:**", body["content"])
                self.assertIn(detail, body["content"])
                self.assertGreaterEqual(body["content"].count("- [ ]"), 6)
                self.assertIsNone(re.search(r"[\u3400-\u9fff]", body["content"]))
                self.assertEqual(body["model_used"], "local help")

    def test_plain_help_respects_client_language(self):
        response = self.client.post(
            "/api/chat",
            json={"message": "/help", "language": "Chinese"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("选择 autoresearch", response.json()["content"])

        scoped = self.client.post(
            "/api/chat",
            json={"message": "/help data 只处理数据", "language": "Chinese"},
        )
        self.assertEqual(scoped.status_code, 200)
        self.assertIn("**范围：**", scoped.json()["content"])

    def test_desktop_surface_disables_workspace_environment_precheck(self):
        prompt = _client_surface_prompt("desktop")
        self.assertIn("Never inspect", prompt)
        self.assertIn("neuroclaw_environment.json", prompt)
        self.assertIn("Never run or recommend installer/setup.py", prompt)
        self.assertEqual(_client_surface_prompt("web"), "")

    def test_model_backed_responses_receive_explicit_client_language_policy(self):
        english = _response_language_prompt("English")
        chinese = _response_language_prompt("zh-CN")
        self.assertIn("Respond entirely in English", english)
        self.assertIn("Do not use Chinese UI labels", english)
        self.assertIn("Respond in Simplified Chinese", chinese)
        self.assertEqual(_response_language_prompt(""), "")

        index_html = (Path(__file__).with_name("static") / "index.html").read_text(
            encoding="utf-8"
        )
        server_source = Path(__file__).with_name("server.py").read_text(encoding="utf-8")
        self.assertIn(
            "assistant: assistantText, language: currentUiLanguage()",
            index_html,
        )
        self.assertGreaterEqual(server_source.count("_response_language_prompt("), 4)

    def test_help_uses_explicit_english_client_language_for_chinese_description(self):
        response = self.client.post(
            "/api/chat",
            json={
                "message": "/help data 我只想处理数据。请问需要提供哪些信息。",
                "language": "en",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["autoresearch_mode"], "data")
        self.assertIn("Data processing only", body["content"])
        self.assertIn("Please provide", body["content"])
        self.assertNotIn("请提供以下资料", body["content"])

    def test_autoresearch_defaults_to_off_and_is_not_persisted_across_new_chats(self):
        index_html = (Path(__file__).with_name("static") / "index.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("autoResearchMode: AUTO_RESEARCH_MODE_OFF", index_html)
        self.assertNotIn("autoResearchMode: state.autoResearchMode", index_html)
        self.assertNotIn("AUTO_RESEARCH_MODES.includes(parsed.autoResearchMode)", index_html)
        create_session = index_html[index_html.index("function createSession(") :]
        create_session = create_session[: create_session.index("function ensureSession(")]
        self.assertIn("state.autoResearchMode = AUTO_RESEARCH_MODE_OFF", create_session)
        fresh_session = index_html[index_html.index("function startFreshSession(") :]
        fresh_session = fresh_session[: fresh_session.index("function startFreshHomeSession(")]
        self.assertIn("state.autoResearchMode = AUTO_RESEARCH_MODE_OFF", fresh_session)
        self.assertIn("return createSession('New Chat', targetProjectId", fresh_session)

    def test_english_ui_translation_branches_do_not_contain_chinese(self):
        static_root = Path(__file__).with_name("static")
        index_html = (static_root / "index.html").read_text(encoding="utf-8")
        study_html = (static_root / "study.html").read_text(encoding="utf-8")
        explore_html = (static_root / "explore.html").read_text(encoding="utf-8")
        desktop_main = (static_root.parents[2] / "desktop" / "main.js").read_text(
            encoding="utf-8"
        )
        cjk = re.compile(r"[\u3400-\u9fff]")

        def first_literal_arguments(source: str, function_name: str) -> list[str]:
            pattern = re.compile(
                rf"{re.escape(function_name)}\(\s*(['\"`])([\s\S]*?)\1\s*,"
            )
            return [match.group(2) for match in pattern.finditer(source)]

        ui_english = first_literal_arguments(index_html, "uiText")
        desktop_english = first_literal_arguments(desktop_main, "desktopText")
        self.assertGreater(len(ui_english), 20)
        self.assertGreater(len(desktop_english), 10)
        self.assertEqual([text for text in ui_english if cjk.search(text)], [])
        self.assertEqual([text for text in desktop_english if cjk.search(text)], [])

        study_english = [
            match.group(1)
            for match in re.finditer(
                r"\b\w+:\s*\[\s*'([^']*)'\s*,\s*'([^']*)'\s*\]",
                study_html,
            )
        ]
        self.assertGreater(len(study_english), 50)
        self.assertEqual([text for text in study_english if cjk.search(text)], [])

        explore_english = explore_html.split("const I18N = {", 1)[1].split("  zh: {", 1)[0]
        self.assertIsNone(cjk.search(explore_english))

        translation_keys = re.findall(r"^\s*'([^']+)'\s*:\s*'", index_html, re.MULTILINE)
        self.assertGreater(len(translation_keys), 100)
        self.assertEqual([key for key in translation_keys if cjk.search(key)], [])

    def test_user_study_requires_password_token(self):
        locked = self.client.get("/api/studies/config")
        self.assertEqual(locked.status_code, 401)

        rejected = self.client.post("/api/studies/auth", json={"password": "wrong"})
        self.assertEqual(rejected.status_code, 401)

        accepted = self.client.post("/api/studies/auth", json={"password": "123456"})
        self.assertEqual(accepted.status_code, 200)
        token = accepted.json()["token"]
        unlocked = self.client.get(
            "/api/studies/config",
            headers={"X-NeuroOracle-Study-Token": token},
        )
        self.assertEqual(unlocked.status_code, 200)
        self.assertEqual(
            unlocked.json()["case_study"], "case1_tcp_external_validation"
        )
        self.assertEqual(
            unlocked.json()["session_protocol"],
            {
                "completion_basis": "active_time",
                "required_sessions": 6,
                "active_seconds_per_session": 600,
                "pair_pool_per_session": 20,
                "assignment_policy": "fixed_shared_schedule",
                "shared_random_seed": 0,
                "same_questions_for_all_participants": True,
            },
        )

    def test_study_pages_are_native_menu_actions_not_neurooracle_tabs(self):
        static_root = Path(__file__).with_name("static")
        index_html = (static_root / "index.html").read_text(encoding="utf-8")
        study_html = (static_root / "study.html").read_text(encoding="utf-8")
        desktop_main = (static_root.parents[2] / "desktop" / "main.js").read_text(encoding="utf-8")

        self.assertNotIn('id="nav-expert-study"', index_html)
        self.assertNotIn('id="nav-study-results"', index_html)
        self.assertIn('id="expert-study-page"', index_html)
        self.assertIn('id="study-results-page"', index_html)
        self.assertIn("sendMenuAction('open-expert-study')", desktop_main)
        self.assertIn("sendMenuAction('open-study-results')", desktop_main)
        self.assertIn("normalized === 'open-expert-study'", index_html)
        self.assertIn("normalized === 'open-study-results'", index_html)
        self.assertNotIn("data-neurooracle-subview", index_html)
        self.assertNotIn("neurooracle-subnav", index_html)
        self.assertIn("/study?embedded=1", index_html)
        self.assertIn("/study?view=results&embedded=1", index_html)
        self.assertIn("body.embedded-route .tabs { display: none; }", study_html)
        self.assertIn('<div class="brand"><strong data-i18n="expertStudy">Expert Study</strong></div>', study_html)
        self.assertNotIn("NeuroDiscovery</strong>", study_html)
        self.assertNotIn('class="brand-mark"', study_html)

    def test_embedded_study_closes_to_previous_view_and_reloads_tutorial(self):
        static_root = Path(__file__).with_name("static")
        index_html = (static_root / "index.html").read_text(encoding="utf-8")
        study_html = (static_root / "study.html").read_text(encoding="utf-8")

        self.assertIn('id="close-study-btn"', study_html)
        self.assertIn('data-i18n="closeStudy">Close</span>', study_html)
        self.assertIn("closeStudy:['Close','关闭']", study_html)
        self.assertIn(
            "window.parent.postMessage({type:'neurodiscovery:close-study-workspace'}",
            study_html,
        )
        self.assertIn("const STUDY_VIEW_NAMES = ['expert-study', 'study-results']", index_html)
        self.assertIn("studyReturnView: 'chat'", index_html)
        self.assertIn("state.studyReturnView = previousView", index_html)
        self.assertIn("function resetStudyWorkspace(view)", index_html)
        self.assertIn("frame.src = 'about:blank'", index_html)
        self.assertIn("state[keyName] = ''", index_html)
        self.assertIn("function closeStudyWorkspace()", index_html)
        self.assertIn("function applyStudyWorkspaceMode(active)", index_html)
        self.assertIn(
            "document.body.classList.toggle('study-workspace-mode', studyMode)",
            index_html,
        )
        self.assertIn("applyStudyWorkspaceMode(enteringStudyView)", index_html)
        self.assertIn("sidebarEl.toggleAttribute('inert', studyMode)", index_html)
        self.assertIn(
            "body.study-workspace-mode .sidebar,\n"
            "    body.study-workspace-mode .sidebar-toggle",
            index_html,
        )
        self.assertIn(
            "payload?.type !== 'neurodiscovery:close-study-workspace'", index_html
        )
        self.assertIn("if (trustedSource) closeStudyWorkspace();", index_html)

    def test_desktop_zoom_shortcuts_use_persistent_text_scale_and_sync_study(self):
        static_root = Path(__file__).with_name("static")
        index_html = (static_root / "index.html").read_text(encoding="utf-8")
        study_html = (static_root / "study.html").read_text(encoding="utf-8")
        desktop_main = (static_root.parents[2] / "desktop" / "main.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("'CmdOrCtrl+Plus'", desktop_main)
        self.assertIn("'CmdOrCtrl+='", desktop_main)
        self.assertIn("'CmdOrCtrl+-'", desktop_main)
        self.assertIn("'CmdOrCtrl+0'", desktop_main)
        self.assertIn("sendMenuAction(action)", desktop_main)
        self.assertNotIn("role: 'zoomIn'", desktop_main)
        self.assertNotIn("role: 'zoomOut'", desktop_main)
        self.assertNotIn("role: 'resetZoom'", desktop_main)
        self.assertIn("normalized === 'text-scale-increase'", index_html)
        self.assertIn(
            "applyTextScale(state.textScale + TEXT_SCALE_STEP, true)", index_html
        )
        self.assertIn(
            "applyTextScale(state.textScale - TEXT_SCALE_STEP, true)", index_html
        )
        self.assertIn("applyTextScale(TEXT_SCALE_DEFAULT, true)", index_html)
        self.assertIn("function syncStudyTextScaleFrames()", index_html)
        self.assertIn("textScale=${encodeURIComponent(state.textScale)}", index_html)
        self.assertIn(
            "if(payload?.type==='neuroclaw:text-scale')applyHostTextScale(payload.scale)",
            study_html,
        )
        self.assertIn("app.style.zoom = String(scale)", study_html)
        self.assertIn("app.style.width = '100%'", study_html)
        self.assertIn("app.style.height = '100%'", study_html)
        self.assertIn("app.style.minWidth = `${960 / scale}px`", study_html)
        self.assertNotIn("app.style.width = `${100 / scale}%`", study_html)
        self.assertNotIn("app.style.height = `${100 / scale}%`", study_html)

    def test_dark_theme_syncs_native_chrome_and_embedded_workspaces(self):
        static_root = Path(__file__).with_name("static")
        index_html = (static_root / "index.html").read_text(encoding="utf-8")
        study_html = (static_root / "study.html").read_text(encoding="utf-8")
        explore_html = (static_root / "explore.html").read_text(encoding="utf-8")
        desktop_root = static_root.parents[2] / "desktop"
        desktop_main = (desktop_root / "main.js").read_text(encoding="utf-8")
        desktop_preload = (desktop_root / "preload.js").read_text(encoding="utf-8")

        self.assertIn("nativeTheme.themeSource = theme", desktop_main)
        self.assertIn("ipcMain.handle('neuroclaw:set-theme'", desktop_main)
        self.assertIn(
            "setTheme: (theme) => ipcRenderer.invoke('neuroclaw:set-theme', theme)",
            desktop_preload,
        )
        self.assertIn("window.neuroclawDesktop.setTheme(state.theme)", index_html)
        self.assertIn("{ type: 'neurodiscovery:theme', theme: state.theme }", index_html)
        self.assertIn("theme=${state.theme}", index_html)
        self.assertIn('html[data-theme="dark"]', study_html)
        self.assertIn("if(payload?.type==='neurodiscovery:theme')", study_html)
        self.assertIn('payload?.type !== "neurodiscovery:theme"', explore_html)

    def test_language_switch_rebuilds_native_desktop_menus(self):
        static_root = Path(__file__).with_name("static")
        index_html = (static_root / "index.html").read_text(encoding="utf-8")
        desktop_root = static_root.parents[2] / "desktop"
        desktop_main = (desktop_root / "main.js").read_text(encoding="utf-8")
        desktop_preload = (desktop_root / "preload.js").read_text(encoding="utf-8")

        self.assertIn("ipcMain.handle('neuroclaw:set-language'", desktop_main)
        self.assertIn("setApplicationMenu()", desktop_main)
        self.assertIn(
            "setLanguage: (language) => ipcRenderer.invoke('neuroclaw:set-language', language)",
            desktop_preload,
        )
        self.assertIn("window.neuroclawDesktop.setLanguage", index_html)
        self.assertIn(
            ".setLanguage(state.desktopConfig.language", index_html
        )

    def test_expert_study_scrolls_each_literature_list_without_scrolling_the_page(self):
        study_html = (Path(__file__).with_name("static") / "study.html").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "html, body { height: 100%; margin: 0; overflow: hidden;",
            study_html,
        )
        self.assertIn(
            ".duel-workspace { flex: 1; min-height: 0; display: flex; overflow: hidden;",
            study_html,
        )
        self.assertIn(
            ".duel-shell { width: min(1320px, 100%); height: 100%; min-height: 0;",
            study_html,
        )
        self.assertIn(
            ".hypothesis-card { min-width: 0; min-height: 0; display: flex; flex-direction: column; overflow: hidden;",
            study_html,
        )
        self.assertIn(
            ".literature { flex: 1; min-height: 0; display: flex; flex-direction: column; overflow: hidden;",
            study_html,
        )
        self.assertIn(
            "flex: 1; min-height: 0; display: grid; align-content: start; gap: 9px; overflow-y: auto;",
            study_html,
        )
        self.assertIn(
            "overscroll-behavior: contain; scrollbar-gutter: stable;",
            study_html,
        )
        self.assertNotIn(
            ".duel-workspace { flex: 1; min-height: 0; overflow: auto;",
            study_html,
        )

    def test_expert_study_session_controls_use_a_vertical_status_rail(self):
        study_html = (Path(__file__).with_name("static") / "study.html").read_text(
            encoding="utf-8"
        )

        self.assertIn(".workbench { flex-direction: row; }", study_html)
        self.assertIn(
            "width: var(--study-strip-width, 156px); height: auto; flex: 0 0 var(--study-strip-width, 156px);",
            study_html,
        )
        self.assertIn('class="study-strip-resizer"', study_html)
        self.assertIn("cursor: col-resize !important", study_html)
        self.assertIn(
            '<div class="study-session-meta"><strong id="participant-label"></strong><span id="session-progress-label"></span><span id="condition-label"></span></div>',
            study_html,
        )
        self.assertIn(
            '<div class="study-progress-block"><div class="progress-track"><span id="progress-fill"></span></div><span id="progress-text"></span></div>',
            study_html,
        )
        self.assertIn(
            ".strip-actions { margin-top: auto; display: flex; align-items: stretch; flex-direction: column;",
            study_html,
        )
        self.assertIn(
            ".study-strip .directory-trigger { width: 100%; min-height: 54px; align-items: flex-start; flex-direction: column;",
            study_html,
        )
        self.assertIn(
            ".study-strip .directory-trigger b { min-width: 0; border-left: 0; padding-left: 0; }",
            study_html,
        )
        self.assertIn(
            '<button class="ghost" id="pause-btn" data-i18n="pause">Pause</button>',
            study_html,
        )
        self.assertIn(
            '<button class="ghost" id="end-btn" data-i18n="endSession">End session</button>',
            study_html,
        )
        self.assertNotIn(
            ".study-strip { height: 42px; flex: 0 0 42px;",
            study_html,
        )

    def test_expert_study_uses_six_ten_minute_active_time_sessions(self):
        study_html = (Path(__file__).with_name("static") / "study.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("function sessionTargetMs()", study_html)
        self.assertIn("void completeTimedSession('time_limit')", study_html)
        self.assertIn("completion_reason:reason", study_html)
        self.assertIn("pairwise_elo_v2_timeboxed", study_html)
        self.assertIn(
            "Questions range from straightforward to challenging",
            study_html,
        )
        self.assertIn(
            "Judge every pair using your own clinical and scientific experience",
            study_html,
        )
        self.assertNotIn("generator-score gap", study_html)
        self.assertNotIn("生成器分差", study_html)
        self.assertNotIn("100 selected questions: 30 easy", study_html)

    def test_chinese_expert_study_localizes_structured_hypothesis_explanations(self):
        study_html = (Path(__file__).with_name("static") / "study.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("function localizedHypothesisSummary(item)", study_html)
        self.assertIn("const DISEASE_GLOSSARY =", study_html)
        self.assertIn("const ATLAS_GLOSSARY =", study_html)
        self.assertIn("const REGION_GLOSSARY =", study_html)
        self.assertIn("const FEATURE_GLOSSARY =", study_html)
        self.assertIn("function expandedDiseaseTerm(value)", study_html)
        self.assertIn("function expandedRegionTerm(value)", study_html)
        self.assertIn("function expandedFeatureTerm(value)", study_html)
        self.assertIn(
            "该假设认为：与健康对照相比，${disease}患者在${region}的${feature}${change}。",
            study_html,
        )
        self.assertIn(
            "This hypothesis predicts that, compared with healthy controls, patients with ${disease}",
            study_html,
        )
        self.assertIn("感兴趣区低频振幅分数（ROI fALFF", study_html)
        self.assertIn("Eickhoff–Zilles 脑区图谱", study_html)
        self.assertIn("function plainHypothesisSummary(item)", study_html)
        self.assertNotIn("function hypothesisFactsHtml(item,chain,chainTitle)", study_html)
        self.assertNotIn('class="hypothesis-facts"', study_html)
        self.assertIn("escapeHtml(plainHypothesisSummary(item))", study_html)
        self.assertIn("escapeHtml(localizedHypothesisSummary(item))", study_html)
        self.assertIn("short(localizedHypothesisSummary(item),180)", study_html)

    def test_expert_study_visibly_expands_hypothesis_terminology(self):
        study_html = (Path(__file__).with_name("static") / "study.html").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "terminology:['Plain-language terminology','通俗术语说明']", study_html
        )
        self.assertIn("diseaseTerm:['Condition','疾病/人群']", study_html)
        self.assertIn("function hypothesisTerminology(chain)", study_html)
        self.assertIn("function hypothesisOriginalHtml(chainTitle)", study_html)
        self.assertIn('class="hypothesis-terms"', study_html)
        self.assertIn("Schaefer 七网络功能图谱中", study_html)
        self.assertIn("眶额皮层（Orbitofrontal Cortex, OFC）", study_html)
        self.assertIn("低频振幅分数（ROI fALFF", study_html)
        self.assertIn("roi negative partial-correlation connectivity", study_html)
        terminology_position = study_html.index("${hypothesisTerminology(chain)}")
        original_position = study_html.index("${hypothesisOriginalHtml(chainTitle)}")
        self.assertLess(terminology_position, original_position)

    def test_expert_study_explains_each_reference_relevance_without_overclaiming(self):
        study_html = (Path(__file__).with_name("static") / "study.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("whyRelevant:['Why it is relevant','关联原因']", study_html)
        self.assertIn("const DISEASE_EVIDENCE_ALIASES =", study_html)
        self.assertIn("const DISEASE_RELATED_EVIDENCE_ALIASES =", study_html)
        self.assertIn("const FEATURE_EVIDENCE_ALIASES =", study_html)
        self.assertIn("const EVIDENCE_MEASURE_PATTERNS =", study_html)
        self.assertIn("function detectPaperDirection(paper,chain)", study_html)
        self.assertIn("function localizedPaperFinding(item,paper,analysis)", study_html)
        self.assertIn("function paperRelevanceReason(item,paper)", study_html)
        self.assertIn('class="paper-relevance"', study_html)
        self.assertIn("relevanceReasonZh:String(paper.relevance_reason_zh||'').trim()", study_html)
        self.assertIn("relevanceReasonEn:String(paper.relevance_reason_en||'').trim()", study_html)
        self.assertIn(
            "const curated=String(lang==='zh'?(paper.relevanceReasonZh||''):(paper.relevanceReasonEn||'')).trim()",
            study_html,
        )
        self.assertIn("if(curated)return curated", study_html)
        self.assertIn("function paperEvidenceSummary(item,paper)", study_html)
        self.assertIn("function paperEvidenceHtml(item,paper)", study_html)
        self.assertIn("function paperStudySentence(value)", study_html)
        self.assertIn("function paperSupportScoreHtml(paper,fallbackBottomLine,paperSummary)", study_html)
        self.assertIn(
            "paperSupportScoreHtml(paper,summary.bottomLine,paperStudySentence(summary.studied))",
            study_html,
        )
        self.assertIn("function paperRelationAnalysis(item,paper)", study_html)
        self.assertIn("function paperRelationHtml(analysis)", study_html)
        self.assertIn("function paperQuickMetaHtml(paper)", study_html)
        self.assertIn("function paperStudyDetailsHtml(item,paper)", study_html)
        self.assertIn("relationToHypothesis:['Relationship to the hypothesis','与假设的关系']", study_html)
        self.assertIn("diseaseRelation:['Disease','疾病']", study_html)
        self.assertIn("regionRelation:['Brain region','脑区']", study_html)
        self.assertIn("measureRelation:['Measure','指标']", study_html)
        self.assertIn("directionRelation:['Direction','方向']", study_html)
        self.assertIn('class="paper-evidence-row', study_html)
        self.assertIn('class="paper-support-score ${tone}"', study_html)
        self.assertIn("supportScoreNotProbability:['Evidence-support score, not a replication probability.'", study_html)
        self.assertIn("evidenceFit:['Evidence fit','证据匹配']", study_html)
        self.assertIn("studyCredibilityScore:['Study credibility','研究可信度']", study_html)
        self.assertIn("studyFinding:['What the study found','研究发现']", study_html)

    def test_expert_study_moves_question_guidance_to_sidebar_and_supports_cannot_judge_reasons(self):
        study_html = (Path(__file__).with_name("static") / "study.html").read_text(
            encoding="utf-8"
        )

        sidebar_position = study_html.index('id="study-question-meta"')
        workspace_position = study_html.index('class="duel-workspace"')
        self.assertLess(sidebar_position, workspace_position)
        self.assertIn('id="duel-stage"', study_html)
        self.assertIn('id="duel-distance"', study_html)
        self.assertNotIn('id="duel-difference"', study_html)
        self.assertNotIn("function comparisonDifferenceHtml(left,right)", study_html)
        self.assertNotIn("本题重点比较", study_html)
        self.assertIn('id="undecided-menu"', study_html)
        self.assertIn('data-undecided-reason="evidence_insufficient"', study_html)
        self.assertIn('data-undecided-reason="outside_expertise"', study_html)
        self.assertIn("function undecidedReasonLabel(reason)", study_html)
        self.assertIn("undecided_reason:reason||null", study_html)
        self.assertIn("cannot_judge_reason:result?.undecided_reason||null", study_html)
        self.assertIn("cannot_judge_reasons:cannotJudgeReasons", study_html)

    def test_expert_study_extracts_one_to_four_relevant_source_sentences(self):
        study_html = (Path(__file__).with_name("static") / "study.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("function sourceSentences(paper)", study_html)
        self.assertIn("function relevantSourceSentences(item,paper)", study_html)
        self.assertIn("const selected=[scored[0]]", study_html)
        self.assertIn("if(selected.length>=4)break", study_html)
        self.assertIn("selected.push(row)", study_html)
        self.assertIn(
            "const relevantSentences=relevantSourceSentences(item,paper)", study_html
        )
        self.assertIn("relevantSentences.map(text=>", study_html)
        self.assertNotIn("paper.excerpts.map(text=>", study_html)
        self.assertIn(
            "sourceExcerpt:['Most relevant source text','与假设最相关的原文']",
            study_html,
        )

    def test_expert_study_uses_compact_per_hypothesis_literature_readers(self):
        study_html = (Path(__file__).with_name("static") / "study.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("function paperReaderState(item,paperCount)", study_html)
        self.assertIn("function paperCardHtml(item,paper,index,safeId)", study_html)
        self.assertIn('class="paper-index" role="tablist"', study_html)
        self.assertIn('data-paper-view-toggle', study_html)
        self.assertIn("reader.all?papers.map", study_html)
        self.assertIn("paper-list ${reader.all?'all-papers':'single-paper'}", study_html)
        self.assertIn("function rerenderHypothesisCard(hypothesisId", study_html)
        self.assertIn("event.key==='ArrowLeft'", study_html)
        self.assertIn('class="hypothesis-context"', study_html)
        self.assertIn("originalStatement:['Original structured hypothesis','原始结构化假设']", study_html)

    def test_expert_study_reuses_password_until_the_client_closes(self):
        study_html = (Path(__file__).with_name("static") / "study.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("function studyTokenStorage()", study_html)
        self.assertIn("return window.parent.sessionStorage", study_html)
        self.assertIn("tokenStorage.getItem(STUDY_TOKEN_KEY)", study_html)
        self.assertIn("tokenStorage.setItem(STUDY_TOKEN_KEY,payload.token)", study_html)
        self.assertIn(
            "authToken:savedStudyToken, authenticated:Boolean(savedStudyToken)",
            study_html,
        )
        self.assertIn("if(!state.authToken){showView('auth');return;}", study_html)
        self.assertNotIn(
            "sessionStorage.removeItem('neurodiscoveryStudyToken');\n"
            "    const state",
            study_html,
        )

    def test_new_chat_actions_reuse_and_preserve_the_single_empty_session(self):
        index_html = (Path(__file__).with_name("static") / "index.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("function isCompletelyEmptySessionForProject(", index_html)
        self.assertIn("function findReusableEmptySession(projectId = null)", index_html)
        self.assertIn(
            "const reusable = findReusableEmptySession(targetProjectId)", index_html
        )
        self.assertIn(
            "|| !isCompletelyEmptySessionForProject(session, targetProjectId)",
            index_html,
        )
        start_new_chat = index_html[index_html.index("function startNewChat()") :]
        start_new_chat = start_new_chat[: start_new_chat.index("function setActiveView(")]
        self.assertIn("startFreshHomeSession()", start_new_chat)
        self.assertNotIn("createSession('New Chat')", start_new_chat)
        self.assertGreaterEqual(
            index_html.count(
                "startFreshProjectSession(addProjectChatBtn.dataset.projectNewChatId)"
            ),
            2,
        )
        set_active_view = index_html[index_html.index("function setActiveView(") :]
        set_active_view = set_active_view[
            : set_active_view.index("function renderWorkspaceSearch(")
        ]
        self.assertNotIn("cleanupEmptyActiveHomeSession", set_active_view)
        self.assertNotIn("function cleanupEmptyActiveHomeSession(", index_html)

    def test_first_turn_immediately_titles_and_then_model_refines_the_chat(self):
        static_root = Path(__file__).with_name("static")
        index_html = (static_root / "index.html").read_text(encoding="utf-8")
        server_source = Path(__file__).with_name("server.py").read_text(encoding="utf-8")

        self.assertIn("autoTitlePending: true", index_html)
        self.assertIn("function fallbackSessionTitle(text)", index_html)
        self.assertIn(
            "function beginAutoTitleForFirstUserTurn(session, userText)", index_html
        )
        self.assertIn(
            "beginAutoTitleForFirstUserTurn(session, displayText)", index_html
        )
        self.assertIn(
            "async function updateSessionTitleFromFirstTurn(session)", index_html
        )
        self.assertIn("void updateSessionTitleFromFirstTurn(session)", index_html)
        self.assertIn(
            "if (!session || session.autoTitlePending !== true) return false",
            index_html,
        )
        self.assertIn(
            "modelTitle && !isNewChatTitle(modelTitle)", index_html
        )
        self.assertIn("s.autoTitlePending = false", index_html)
        self.assertIn("fetch('/api/chat/title'", index_html)
        self.assertIn('@app.post("/api/chat/title")', server_source)
        self.assertIn(
            '"Return exactly one short title in plain text, 3-10 words',
            server_source,
        )

    def test_first_desktop_chat_without_api_credentials_opens_llm_settings(self):
        static_root = Path(__file__).with_name("static")
        index_html = (static_root / "index.html").read_text(encoding="utf-8")
        desktop_main = (static_root.parents[2] / "desktop" / "main.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("function describeLlmConnectionStatus(config)", desktop_main)
        self.assertIn("apiKeyConfigured: !apiKeyRequired || Boolean(apiKey || environmentKey)", desktop_main)
        self.assertIn(
            "llmConnectionStatus: describeLlmConnectionStatus(config)", desktop_main
        )
        self.assertIn(
            "llmConnectionStatus: describeLlmConnectionStatus(savedConfig)",
            desktop_main,
        )
        self.assertIn("async function ensureChatLlmConfigured()", index_html)
        self.assertIn("state.settingsSection = 'llm'", index_html)
        self.assertIn("reason: 'chat-configuration-required'", index_html)
        self.assertIn("['llmApiKey', 'llmBaseUrl'].includes(row.key)", index_html)
        self.assertIn('class="settings-required-notice"', index_html)
        self.assertIn(
            "Enter an API key and confirm the API endpoint below, then save the settings.", index_html
        )
        self.assertIn("请在下方填写 API key 并确认 API 接入点，然后保存设置。", index_html)
        send_message = index_html[index_html.index("async function sendMessage()") :]
        send_message = send_message[: send_message.index("function stopCurrentRequest(")]
        self.assertIn(
            "if (!(await ensureChatLlmConfigured())) return;", send_message
        )
        self.assertLess(
            send_message.index("if (!(await ensureChatLlmConfigured())) return;"),
            send_message.index("session.messages.push("),
        )

    def test_settings_select_menu_has_a_complete_unclipped_border(self):
        index_html = (Path(__file__).with_name("static") / "index.html").read_text(
            encoding="utf-8"
        )

        settings_card = index_html[index_html.index(".settings-card {") :]
        settings_card = settings_card[: settings_card.index("    }") + 5]
        self.assertIn("overflow: visible;", settings_card)
        self.assertIn(".settings-row:focus-within", index_html)
        self.assertIn(".settings-select-menu {", index_html)
        self.assertIn(".settings-select-menu[hidden] { display: none; }", index_html)
        self.assertIn("border: 1px solid", index_html[index_html.index(".settings-select-menu {") :])
        self.assertIn('role="combobox"', index_html)
        self.assertIn('role="listbox"', index_html)
        self.assertIn("function openSettingsSelect(wrapper", index_html)
        self.assertIn("function closeSettingsSelect(wrapper", index_html)
        self.assertIn("data-settings-select-option", index_html)
        self.assertIn("option.click();", index_html)
        self.assertNotIn(
            'return `<select class="settings-control" ${common}>${options}</select>`;',
            index_html,
        )

    def test_message_edit_actions_save_resend_and_lock_during_agent_reply(self):
        index_html = (Path(__file__).with_name("static") / "index.html").read_text(
            encoding="utf-8"
        )

        add_user_bubble = index_html[index_html.index("function addUserBubble(") :]
        add_user_bubble = add_user_bubble[: add_user_bubble.index("function toolEventTitle(")]
        self.assertIn("const editAllowed = !isSessionWaiting()", add_user_bubble)
        self.assertIn(
            "const isEditing = editAllowed && state.editingMessageId === message.id",
            add_user_bubble,
        )
        self.assertIn(
            "${editAllowed ? actionButton('edit-user', message.id, 'Edit', 'edit') : ''}",
            add_user_bubble,
        )

        begin_request = index_html[index_html.index("function beginSessionRequest(") :]
        begin_request = begin_request[: begin_request.index("function finishSessionRequest(")]
        self.assertIn(
            "if (sessionId === state.activeSessionId) state.editingMessageId = null",
            begin_request,
        )

        message_actions = index_html[index_html.index("chatPanelEl.addEventListener('click'") :]
        message_actions = message_actions[: message_actions.index("if (action === 'regenerate-user')")]
        self.assertIn("if (isSessionWaiting(session.id))", message_actions)
        self.assertIn(
            "const messageMain = actionBtn.closest('.msg-main')", message_actions
        )
        self.assertIn(
            "messageMain ? messageMain.querySelector('.edit-textarea') : null",
            message_actions,
        )
        self.assertNotIn("actionBtn.closest('.bubble')", message_actions)
        self.assertIn("if (action === 'save-edit')", message_actions)
        self.assertIn("resendAsNewTurn(edited, edited)", message_actions)

    def test_message_editing_preserves_scroll_and_uses_neutral_full_width_bubble(self):
        index_html = (Path(__file__).with_name("static") / "index.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("--user-bg: #eef0f3", index_html)
        self.assertIn("--user-bg: #252c35", index_html)
        self.assertNotIn("--user-bg: #1677ff", index_html)
        self.assertNotIn("--user-bg: #2f81f7", index_html)
        self.assertIn(".msg.user.editing .msg-main", index_html)
        self.assertIn(".msg.user.editing .bubble", index_html)
        self.assertIn("width: 100%;", index_html)
        self.assertIn("background: transparent;", index_html)
        self.assertIn("color: var(--user-text);", index_html)
        self.assertIn("#messages.static-message-render .msg", index_html)

        render_active = index_html[index_html.index("function renderActiveSession(") :]
        render_active = render_active[: render_active.index("function addTypingIndicator(")]
        self.assertIn(
            "const preserveScroll = options.preserveScroll === true", render_active
        )
        self.assertIn(
            "const previousScrollTop = preserveScroll ? messagesEl.scrollTop : 0",
            render_active,
        )
        self.assertIn(
            "messagesEl.classList.toggle('static-message-render', preserveScroll)",
            render_active,
        )
        self.assertIn(
            "if (preserveScroll) messagesEl.scrollTop = previousScrollTop",
            render_active,
        )
        self.assertIn("else scrollToBottom()", render_active)
        self.assertIn("function beginEditingUserMessage(messageId)", render_active)
        self.assertIn("editor.focus({ preventScroll: true })", render_active)

        message_actions = index_html[index_html.index("if (action === 'edit-user')") :]
        message_actions = message_actions[
            : message_actions.index("if (action === 'regenerate-user')")
        ]
        self.assertIn("beginEditingUserMessage(msg.id)", message_actions)
        self.assertGreaterEqual(
            message_actions.count("renderActiveSession({ preserveScroll: true })"),
            3,
        )

    def test_checkpoints_are_scoped_to_the_active_chat_and_can_be_deleted(self):
        static_root = Path(__file__).with_name("static")
        index_html = (static_root / "index.html").read_text(encoding="utf-8")
        server_source = Path(__file__).with_name("server.py").read_text(encoding="utf-8")
        agent_source = (static_root.parents[1] / "agent" / "main.py").read_text(
            encoding="utf-8"
        )

        self.assertGreaterEqual(index_html.count("chat_id: session.id"), 3)
        self.assertIn("function checkpointApiUrl(", index_html)
        self.assertIn("workspace_path: workspacePathForSession(session)", index_html)
        self.assertIn('data-cp-delete="${i}"', index_html)
        self.assertIn("method: 'DELETE'", index_html)
        self.assertIn("No checkpoints for this chat yet", index_html)

        set_active = index_html[index_html.index("function setActiveSession(") :]
        set_active = set_active[: set_active.index("function renameSession(")]
        self.assertIn("loadCheckpoints()", set_active)

        self.assertIn("checkpoint_scope=chat_id or None", server_source)
        self.assertIn('scope_id=normalized_chat_id', server_source)
        self.assertIn('@app.delete("/api/checkpoints/{checkpoint_id}")', server_source)
        self.assertIn("checkpoint_scope: str | None = None", agent_source)

        missing_scope = self.client.get("/api/checkpoints")
        self.assertEqual(missing_scope.status_code, 400)
        self.assertEqual(missing_scope.json()["error"], "Missing chat_id")

    def test_visible_product_and_runtime_names_follow_the_new_brand_hierarchy(self):
        static_root = Path(__file__).with_name("static")
        repo_root = static_root.parents[2]
        index_html = (static_root / "index.html").read_text(encoding="utf-8")
        study_html = (static_root / "study.html").read_text(encoding="utf-8")
        explore_html = (static_root / "explore.html").read_text(encoding="utf-8")
        desktop_main = (repo_root / "desktop" / "main.js").read_text(encoding="utf-8")
        desktop_package = (repo_root / "desktop" / "package.json").read_text(
            encoding="utf-8"
        )
        server_source = Path(__file__).with_name("server.py").read_text(encoding="utf-8")
        soul = (repo_root / "SOUL.md").read_text(encoding="utf-8")

        self.assertIn("<title>NeuroDiscovery</title>", index_html)
        self.assertIn('<div class="brand-name">NeuroDiscovery</div>', index_html)
        self.assertIn("NeuroOracle Graph", index_html)
        self.assertIn('header_sub: "NeuroOracle Graph"', explore_html)
        self.assertIn('header_sub: "NeuroOracle 图谱"', explore_html)
        self.assertIn("NeuroRuntime", index_html)
        self.assertNotIn("NeuroClaw", index_html)
        self.assertIn("const APP_NAME = 'NeuroDiscovery'", desktop_main)
        self.assertIn("const LEGACY_USER_DATA_NAME = 'NeuroClaw'", desktop_main)
        self.assertIn('"productName": "NeuroDiscovery"', desktop_package)
        self.assertIn('FastAPI(title="NeuroDiscovery Web UI"', server_source)
        self.assertIn("You are NeuroRuntime, the execution agent inside NeuroDiscovery", soul)
        self.assertIn("NeuroDiscovery-user-study-", study_html)

    def test_chat_export_writes_separate_conversation_and_agent_worklog_files(self):
        static_root = Path(__file__).with_name("static")
        repo_root = static_root.parents[2]
        index_html = (static_root / "index.html").read_text(encoding="utf-8")
        desktop_main = (repo_root / "desktop" / "main.js").read_text(encoding="utf-8")
        desktop_preload = (repo_root / "desktop" / "preload.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('data-menu-action="export-json"', index_html)
        self.assertIn('data-menu-action="export-markdown"', index_html)
        self.assertIn(
            "async function exportSession(sessionId, format = 'json')", index_html
        )
        self.assertIn("function conversationExportPayload(session)", index_html)
        self.assertIn("function agentWorklogExportPayload(session)", index_html)
        self.assertIn("conversationContent,", index_html)
        self.assertIn("worklogContent,", index_html)
        self.assertIn("`${base}_conversation.json`", index_html)
        self.assertIn("`${base}_agent-worklog.json`", index_html)
        self.assertIn("window.neuroclawDesktop.exportChatSession", index_html)
        self.assertNotIn("prompt(tr('Export format: json or md')", index_html)
        self.assertIn(
            "showSuccess(`${tr('Conversation and agent work log exported to:')} ${exportedDirectory}`)",
            index_html,
        )
        self.assertIn(
            "exportChatSession: (request) => ipcRenderer.invoke('neuroclaw:export-chat-session', request)",
            desktop_preload,
        )
        self.assertIn(
            "ipcMain.handle('neuroclaw:export-chat-session'", desktop_main
        )
        self.assertIn(
            "title: desktopText('Export conversation and agent work log', '导出会话内容和 agent 工作记录')",
            desktop_main,
        )
        self.assertIn("await dialog.showSaveDialog(owner, options)", desktop_main)
        self.assertIn("fs.writeFileSync(conversationPath, conversationOutput, 'utf8')", desktop_main)
        self.assertIn("fs.writeFileSync(worklogPath, worklogOutput, 'utf8')", desktop_main)

    def test_web_tool_event_export_keeps_terminal_output_and_execution_metadata(self):
        oversized = "x" * (WEB_TOOL_OUTPUT_EXPORT_LIMIT + 5)
        events = _summarize_web_tool_events(
            [
                {
                    "tool": "run_shell_command",
                    "command": "python test.py",
                    "executed": True,
                    "success": False,
                    "skills_used": ["example"],
                    "result": {
                        "stdout": oversized,
                        "stderr": "traceback",
                        "returncode": 1,
                        "cwd": "C:/workspace",
                        "shell": "cmd.exe",
                        "platform": "win32",
                        "error_type": "command_failed",
                        "failure_stage": "command_execution",
                        "retryable": True,
                        "recovery_hint": "Inspect stderr.",
                    },
                }
            ]
        )
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(len(event["stdout"]), WEB_TOOL_OUTPUT_EXPORT_LIMIT)
        self.assertEqual(event["stdout_preview"], oversized[:1200])
        self.assertTrue(event["stdout_truncated"])
        self.assertEqual(event["stdout_chars"], len(oversized))
        self.assertEqual(event["stderr"], "traceback")
        self.assertEqual(event["returncode"], 1)
        self.assertEqual(event["cwd"], "C:/workspace")
        self.assertEqual(event["shell"], "cmd.exe")
        self.assertEqual(event["error_type"], "command_failed")
        self.assertTrue(event["retryable"])

    def test_expert_study_has_navigable_answer_aware_question_directory(self):
        study_html = (Path(__file__).with_name("static") / "study.html").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="question-directory-btn"', study_html)
        self.assertIn('id="question-directory-groups"', study_html)
        self.assertIn('id="next-unanswered"', study_html)
        self.assertIn("function renderQuestionDirectory()", study_html)
        self.assertIn("function goToQuestion(index,source='direct')", study_html)
        self.assertIn("function recomputePairwiseRatings()", study_html)
        self.assertIn("previousIndex>=0)state.pairResults[previousIndex]=result", study_html)
        self.assertIn("['easy','medium','hard'].map", study_html)
        self.assertIn("question-index.answered", study_html)
        self.assertIn("question-index.undecided", study_html)
        self.assertIn("question-index.current", study_html)
        self.assertIn("easyPair:['Easy','容易']", study_html)
        self.assertIn("mediumPair:['Medium','中等']", study_html)
        self.assertIn("hardPair:['Hard','困难']", study_html)

    def test_expert_study_has_localized_first_launch_introduction(self):
        study_html = (Path(__file__).with_name("static") / "study.html").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="study-intro" hidden', study_html)
        self.assertIn('data-i18n="introTaskTitle">Your task</', study_html)
        self.assertIn('data-i18n="introHowTitle">How to complete the study</', study_html)
        self.assertIn('data-i18n="introSetupTitle">Check the session setup</', study_html)
        self.assertIn('data-i18n="introDirectoryTitle">Use the question directory</', study_html)
        self.assertIn('data-i18n="introDontShow">Don\'t show again</', study_html)
        self.assertIn('data-i18n="introGotIt">Got it</', study_html)
        self.assertIn("const INTRO_DISMISS_KEY =", study_html)
        self.assertIn("localStorage.setItem(INTRO_DISMISS_KEY,'1')", study_html)
        self.assertIn("if (name === 'setup') showStudyIntroIfNeeded();", study_html)
        self.assertIn("el.inert=true;el.setAttribute('aria-hidden','true')", study_html)
        self.assertIn("el.inert=false;el.removeAttribute('aria-hidden')", study_html)
        self.assertIn("introDontShow:[\"Don't show again\",'不再提示']", study_html)
        self.assertIn("introGotIt:['Got it','我已知晓']", study_html)
        self.assertIn('<div class="intro-tier easy"><strong>6</strong>', study_html)
        self.assertIn('<div class="intro-tier medium"><strong>10</strong>', study_html)
        self.assertIn('<div class="intro-tier hard"><strong>3</strong>', study_html)
        intro_markup = study_html.split(
            '<div class="study-intro-layer" id="study-intro" hidden>', 1
        )[1].split('<section class="view auth active"', 1)[0]
        self.assertIsNone(re.search(r"[\u3400-\u9fff]", intro_markup))

    def test_expert_study_language_can_switch_without_reloading_the_session(self):
        static_root = Path(__file__).with_name("static")
        study_html = (static_root / "study.html").read_text(encoding="utf-8")
        index_html = (static_root / "index.html").read_text(encoding="utf-8")

        self.assertEqual(study_html.count('data-study-language="en"'), 2)
        self.assertEqual(study_html.count('data-study-language="zh"'), 2)
        self.assertIn("let lang =", study_html)
        self.assertIn("function setStudyLanguage(nextLanguage", study_html)
        self.assertIn("applyLocalizedStaticText();", study_html)
        self.assertIn("renderDuel({trackPresentation:false})", study_html)
        self.assertIn("function renderDuel({trackPresentation=true}={})", study_html)
        self.assertIn(
            "window.parent.postMessage({type:'neurodiscovery:set-study-language'",
            study_html,
        )
        self.assertIn(
            "payload?.type !== 'neurodiscovery:set-study-language'", index_html
        )
        self.assertIn(
            "async function setCaseStudyWelcomeLanguage(language, "
            "{ refreshStudyWorkspace = true } = {})",
            index_html,
        )
        self.assertIn(
            "{ refreshStudyWorkspace: false }",
            index_html,
        )
        self.assertIn(
            "applyLanguage(false, { refreshStudyWorkspace });",
            index_html,
        )
        self.assertIn(
            "if (refreshStudyWorkspace && (state.activeView === 'expert-study' "
            "|| state.activeView === 'study-results'))",
            index_html,
        )
        self.assertIn(
            "body.study-workspace-mode .starter-grid",
            index_html,
        )
        self.assertIn(
            "if (state.activeView === 'chat') {\n      renderActiveSession();\n"
            "    } else {\n      chatPanelEl.classList.remove('is-empty');",
            index_html,
        )
        self.assertNotIn("location.reload()", study_html)

    def test_case_study_desktop_has_localized_first_launch_workflow_guide(self):
        index_html = (Path(__file__).with_name("static") / "index.html").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="case-study-welcome" hidden', index_html)
        self.assertIn('id="case-study-step-open-title">Open the study</', index_html)
        self.assertIn('id="case-study-step-name-title">Identify yourself</', index_html)
        self.assertIn('id="case-study-step-answer-title">Answer questions</', index_html)
        self.assertIn('id="case-study-step-save-title">Save progress</', index_html)
        self.assertIn('id="case-study-step-export-title">Finish and export</', index_html)
        self.assertIn('id="case-study-tools-title">Controls available during every session</', index_html)
        self.assertIn('id="case-study-feature-evidence-title">Read evidence efficiently</', index_html)
        self.assertIn('id="case-study-feature-navigation-title">Navigate and revise</', index_html)
        self.assertIn('id="case-study-feature-continuity-title">Continue safely</', index_html)
        self.assertIn('id="case-study-feature-controls-title">Language and shortcuts</', index_html)
        self.assertIn('id="case-study-pause-title">Pause before stepping away</', index_html)
        self.assertIn('id="case-study-language-switch"', index_html)
        self.assertIn('data-case-study-language="English"', index_html)
        self.assertIn('data-case-study-language="Simplified Chinese"', index_html)
        self.assertIn(
            "async function setCaseStudyWelcomeLanguage(language, "
            "{ refreshStudyWorkspace = true } = {})",
            index_html,
        )
        self.assertIn("window.neuroclawDesktop.saveConfig", index_html)
        self.assertIn("const CASE_STUDY_WELCOME_KEY =", index_html)
        self.assertIn("localStorage.setItem(CASE_STUDY_WELCOME_KEY, '1')", index_html)
        self.assertIn("loadDesktopSettings().finally(showCaseStudyWelcomeIfNeeded)", index_html)
        self.assertIn("setActiveView('expert-study', { focus: false })", index_html)
        self.assertIn("中途离开前请先暂停", index_html)
        self.assertIn("Paused time is excluded from active answering time.", index_html)
        self.assertIn("The password is requested once per app launch.", index_html)
        self.assertIn("The 1–10 value rates evidence support, not replication probability", index_html)
        self.assertIn("Left / Right Arrow while a literature number is focused", index_html)
        welcome_markup = index_html.split(
            '<div class="case-study-welcome" id="case-study-welcome" hidden', 1
        )[1].split('<div class="skill-info-overlay"', 1)[0]
        self.assertIsNone(re.search(r"[\u3400-\u9fff]", welcome_markup))

    def test_desktop_settings_can_reset_local_application_data_and_restart(self):
        static_root = Path(__file__).with_name("static")
        index_html = (static_root / "index.html").read_text(encoding="utf-8")
        desktop_root = static_root.parents[2] / "desktop"
        desktop_main = (desktop_root / "main.js").read_text(encoding="utf-8")
        desktop_preload = (desktop_root / "preload.js").read_text(encoding="utf-8")

        self.assertIn(
            'id="settings-reset-zone" aria-labelledby="settings-reset-title" hidden',
            index_html,
        )
        self.assertIn('id="settings-reset-btn"', index_html)
        self.assertIn(
            "settingsResetZone.hidden = active.id !== 'advanced'",
            index_html,
        )
        settings_sections = index_html.split(
            "const SETTINGS_SECTIONS = [", 1
        )[1].split("const STUDY_VIEW_NAMES", 1)[0]
        self.assertEqual(
            re.findall(r"\n\s+id: '([^']+)'", settings_sections),
            ["general", "llm", "usage", "runtime", "advanced"],
        )
        self.assertIn("title: 'Models'", settings_sections)
        self.assertIn("Reset application", index_html)
        self.assertIn("重置应用", index_html)
        self.assertIn("window.neuroclawDesktop.resetApplication()", index_html)
        self.assertIn(
            "resetApplication: () => ipcRenderer.invoke('neuroclaw:reset-application')",
            desktop_preload,
        )
        self.assertIn(
            "ipcMain.handle('neuroclaw:reset-application'", desktop_main
        )
        self.assertIn("session.defaultSession.clearStorageData()", desktop_main)
        self.assertIn("session.defaultSession.clearCache()", desktop_main)
        self.assertIn("path.join(os.homedir(), '.neurodiscovery')", desktop_main)
        self.assertIn("path.join(os.homedir(), '.neuroclaw', 'memory')", desktop_main)
        self.assertIn("userRuntimeRoot()", desktop_main)
        self.assertIn("app.relaunch()", desktop_main)
        self.assertIn(
            "Your project folders, datasets, generated outputs, and exported result files are not deleted.",
            desktop_main,
        )

    def test_expert_study_exports_answers_and_detailed_timing_metadata(self):
        static_root = Path(__file__).with_name("static")
        study_html = (static_root / "study.html").read_text(encoding="utf-8")
        desktop_root = static_root.parents[2] / "desktop"
        desktop_main = (desktop_root / "main.js").read_text(encoding="utf-8")
        desktop_preload = (desktop_root / "preload.js").read_text(encoding="utf-8")

        self.assertIn('data-i18n="participantId">Name / ID</span>', study_html)
        self.assertNotIn('id="seed"', study_html)
        self.assertNotIn("Random seed", study_html)
        self.assertIn("random_seed:0", study_html)
        self.assertIn('id="export-results"', study_html)
        self.assertIn('id="export-results-after-submit"', study_html)
        self.assertIn("function buildStudyExportPayload()", study_html)
        self.assertIn("function toIsoDateTime(value)", study_html)
        self.assertIn("total_study_open_time:formatDurationMs", study_html)
        self.assertIn("active_answering_time:formatDurationMs", study_html)
        self.assertIn("total_study_open_time_ms", study_html)
        self.assertIn("total_decision_time_ms", study_html)
        self.assertIn("total_decision_time:formatDurationMs", study_html)
        self.assertIn("decision_time:formatDurationMs", study_html)
        self.assertIn("modification_count", study_html)
        self.assertIn("Array.isArray(stats.attempts)", study_html)
        self.assertIn("const STUDY_DRAFT_KEY =", study_html)
        self.assertIn("function persistStudyDraft()", study_html)
        self.assertIn("function loadResumableSession()", study_html)
        self.assertIn("session_resumed_after_reopen", study_html)
        self.assertIn("persistStudyDraft();void flushEvents()", study_html)
        self.assertIn('id="resume-session-btn"', study_html)
        self.assertIn('id="restart-session-btn"', study_html)
        self.assertIn("method:'DELETE'", study_html)
        self.assertIn("confirmRestartSession", study_html)
        self.assertIn("restartSession:['Delete progress and restart','删除进度并重新开始']", study_html)
        self.assertIn("neuroclaw:export-user-study-results", desktop_main)
        self.assertIn("total_app_open_time_ms", desktop_main)
        self.assertIn("total_app_open_time: formatDurationMs", desktop_main)
        self.assertIn("exportUserStudyResults", desktop_preload)

    def test_expert_study_uses_binary_generator_score_visibility_condition(self):
        study_html = (Path(__file__).with_name("static") / "study.html").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'data-i18n="condition">Show generator scores</span>', study_html
        )
        self.assertIn(
            '<option value="manual" data-i18n="manual">No</option>', study_html
        )
        self.assertIn(
            '<option value="assisted" data-i18n="assisted">Yes</option>', study_html
        )
        self.assertNotIn('<option value="generator"', study_html)
        self.assertIn(
            "condition:['Show generator scores','显示生成器打分']", study_html
        )
        self.assertIn("manual:['No','否']", study_html)
        self.assertIn("assisted:['Yes','是']", study_html)
        self.assertIn(
            "manualMean:['Scores hidden mean','不显示分数平均用时']",
            study_html,
        )
        self.assertIn(
            "assistedMean:['Scores shown mean','显示分数平均用时']",
            study_html,
        )
        self.assertIn(
            "$('condition-label').textContent=`${t('condition')}: ${t(session.condition)}`",
            study_html,
        )
        self.assertIn(
            "generator_scores_visible:state.session.condition==='assisted'",
            study_html,
        )
        self.assertIn("available=['manual','assisted']", study_html)
        self.assertIn(
            "data.sessions.filter(session=>session.status==='completed'&&['manual','assisted'].includes(session.condition)).length",
            study_html,
        )
        self.assertNotIn("t('generatorTime')", study_html)
        self.assertNotIn("get('generator')", study_html)


if __name__ == "__main__":
    unittest.main()
