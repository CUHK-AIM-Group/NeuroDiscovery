import unittest

from core.autoresearch import (
    AUTORESEARCH_MODE_DATA,
    AUTORESEARCH_MODE_END_TO_END,
    AUTORESEARCH_MODE_IDEA,
    AUTORESEARCH_MODE_MODEL,
    build_autoresearch_scope_prompt,
    normalize_autoresearch_mode,
    parse_help_command,
    render_help_response,
)


class AutoResearchHelpTests(unittest.TestCase):
    def test_parse_help_command_supports_four_scopes_in_chinese_and_english(self):
        cases = {
            "/help 只处理数据，做完 QC 就停止": AUTORESEARCH_MODE_DATA,
            "/help model I already have ROI features": AUTORESEARCH_MODE_MODEL,
            "/help 只生成 idea，不运行实验": AUTORESEARCH_MODE_IDEA,
            "/help 端到端 autoresearch": AUTORESEARCH_MODE_END_TO_END,
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                request = parse_help_command(command)
                self.assertIsNotNone(request)
                self.assertEqual(request.mode, expected)

    def test_help_without_description_lists_exactly_four_supported_modes(self):
        request = parse_help_command("/help")
        self.assertIsNotNone(request)
        response = render_help_response(request)
        self.assertEqual(response.count("`/help "), 5)  # four entries plus one example
        for mode in ("data", "model", "idea", "autoresearch"):
            self.assertIn(f"`/help {mode} ", response)

    def test_non_help_message_passes_through(self):
        self.assertIsNone(parse_help_command("Please /help me with my model"))
        self.assertIsNone(parse_help_command("/helper data"))

    def test_explicit_scope_wins_over_negated_component_mentions(self):
        idea = parse_help_command("/help 只生成 idea，不处理数据，也不运行模型")
        model = parse_help_command("/help model use prepared data; do not preprocess data")
        data = parse_help_command("/help 只处理数据，不生成 idea，也不运行模型")
        self.assertEqual(idea.mode, AUTORESEARCH_MODE_IDEA)
        self.assertEqual(model.mode, AUTORESEARCH_MODE_MODEL)
        self.assertEqual(data.mode, AUTORESEARCH_MODE_DATA)

    def test_help_without_description_uses_client_language(self):
        request = parse_help_command("/help", "Chinese")
        self.assertEqual(request.language, "zh")
        self.assertIn("选择 autoresearch", render_help_response(request))

    def test_explicit_client_language_overrides_description_language(self):
        english = parse_help_command(
            "/help data 我只想处理数据。请问需要提供哪些信息。",
            "English",
        )
        self.assertEqual(english.mode, AUTORESEARCH_MODE_DATA)
        self.assertEqual(english.language, "en")
        english_response = render_help_response(english)
        self.assertIn("Data processing only", english_response)
        self.assertIn("Please provide", english_response)
        self.assertGreaterEqual(english_response.count("- [ ]"), 6)

        chinese = parse_help_command(
            "/help data I only want to process data. What information should I provide?",
            "Simplified Chinese",
        )
        self.assertEqual(chinese.language, "zh")
        self.assertIn("请提供以下资料", render_help_response(chinese))

    def test_help_response_contains_scope_boundary_and_checklist(self):
        request = parse_help_command("/help 只编写模型和运行模型")
        self.assertIsNotNone(request)
        response = render_help_response(request)
        self.assertIn("不重新处理原始数据", response)
        self.assertIn("不生成研究 idea", response)
        self.assertGreaterEqual(response.count("- [ ]"), 6)

    def test_legacy_on_mode_maps_to_end_to_end_and_prompt_is_scoped(self):
        self.assertEqual(normalize_autoresearch_mode("on"), AUTORESEARCH_MODE_END_TO_END)
        prompt = build_autoresearch_scope_prompt("model")
        self.assertIn("component scope: model", prompt)
        self.assertIn("do not execute components outside it", prompt)
        self.assertEqual(build_autoresearch_scope_prompt("off"), "")


if __name__ == "__main__":
    unittest.main()
