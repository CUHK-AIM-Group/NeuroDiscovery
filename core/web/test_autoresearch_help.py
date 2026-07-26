import re
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from core.web.server import _client_surface_prompt, _response_language_prompt, create_app


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
        fresh_session = index_html[index_html.index("function startFreshHomeSession(") :]
        fresh_session = fresh_session[: fresh_session.index("function getActiveSession(")]
        self.assertIn("state.autoResearchMode = AUTO_RESEARCH_MODE_OFF", fresh_session)

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
            headers={"X-NeuroDiscovery-Study-Token": token},
        )
        self.assertEqual(unlocked.status_code, 200)
        self.assertEqual(unlocked.json()["case_study"], "case1_transdiagnostic")

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
        self.assertIn("共 300 道题：容易、中等、困难各 100 道。", study_html)

    def test_expert_study_exports_answers_and_detailed_timing_metadata(self):
        static_root = Path(__file__).with_name("static")
        study_html = (static_root / "study.html").read_text(encoding="utf-8")
        desktop_root = static_root.parents[2] / "desktop"
        desktop_main = (desktop_root / "main.js").read_text(encoding="utf-8")
        desktop_preload = (desktop_root / "preload.js").read_text(encoding="utf-8")

        self.assertIn('data-i18n="participantId">Name / ID</span>', study_html)
        self.assertIn('id="seed" type="number" value="0"', study_html)
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
        self.assertIn("neuroclaw:export-user-study-results", desktop_main)
        self.assertIn("total_app_open_time_ms", desktop_main)
        self.assertIn("total_app_open_time: formatDurationMs", desktop_main)
        self.assertIn("exportUserStudyResults", desktop_preload)


if __name__ == "__main__":
    unittest.main()
