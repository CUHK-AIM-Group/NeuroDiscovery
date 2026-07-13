import unittest

from fastapi.testclient import TestClient

from core.web.server import _client_surface_prompt, create_app


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
                self.assertIn(detail, body["content"])
                self.assertGreaterEqual(body["content"].count("- [ ]"), 6)
                self.assertEqual(body["model_used"], "local help")

    def test_plain_help_respects_client_language(self):
        response = self.client.post(
            "/api/chat",
            json={"message": "/help", "language": "Chinese"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("选择 autoresearch", response.json()["content"])

    def test_desktop_surface_disables_workspace_environment_precheck(self):
        prompt = _client_surface_prompt("desktop")
        self.assertIn("Never inspect", prompt)
        self.assertIn("neuroclaw_environment.json", prompt)
        self.assertIn("Never run or recommend installer/setup.py", prompt)
        self.assertEqual(_client_surface_prompt("web"), "")

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


if __name__ == "__main__":
    unittest.main()
