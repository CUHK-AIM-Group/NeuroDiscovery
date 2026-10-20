"""Budget boundary regressions with actual local execution and a mocked model."""
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch

from neurobench.runner.exec_budget_v3 import Budget, BudgetTools, STANDARD, COMPLEX, base_prompt, PYTHON_HOST
from neurobench.runner.exec_dev.common import read_json, write_json, sha
from neurobench.tests.test_exec_protocol_v2 import ProtocolTests, SOLUTION
from neurobench.runner.exec_brief.oracle import evaluate


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ProtocolTests('test_real_artifacts_finalize_with_host_provenance')
        self.fixture.setUp()
        self.count = 0

    def tearDown(self):
        self.fixture.tearDown()

    def tools(self, arm='A', budget=STANDARD, elapsed=0):
        self.count += 1
        work = self.fixture.root / f'budget_work{self.count}'
        private = self.fixture.root / f'budget_attempt{self.count}'
        private.mkdir()
        shutil.copytree(self.fixture.case_root / 'inputs', work / 'inputs')
        (work / 'context/skills').mkdir(parents=True)
        for name in ('task.md', 'CONTRACT.json'):
            shutil.copy2(self.fixture.case_root / name, work / 'context' / name)
        write_json(work / 'context/skill_catalog.json', [])
        for name in ('output', 'scratch'):
            (work / name).mkdir()
        (work / 'python_host.py').write_text(PYTHON_HOST, encoding='utf-8')
        started = time.monotonic() - elapsed
        return BudgetTools(self.fixture.case, work, private, arm, started + budget.task_seconds,
                           budget=budget, started=started)

    def test_every_arm_can_execute_past_legacy_48_and_finishes_at_24(self):
        for arm in ('A', 'B0', 'B1'):
            with self.subTest(arm=arm):
                tool = self.tools(arm)
                for _ in range(48):
                    self.assertTrue(tool.call('read_file', {'path': 'inputs/subjects.json'})['success'])
                self.assertTrue(tool.call('run_python', {'code': SOLUTION})['success'])
                tool.begin_round(24)
                result = tool.call('finish_result', {'status': 'complete'})
                self.assertTrue(result['public_check_pass'], result)
                self.assertEqual(tool.tool_count, 50)
                self.assertEqual(result['budget']['max_model_rounds'], 24)
                self.assertEqual(evaluate(self.fixture.case, Path(tool.submissions[0]['output']))['verdict'], 'PASS')

    def test_tool_96_is_allowed_and_97_is_refused_without_reset(self):
        tool = self.tools()
        for i in range(96):
            result = tool.call('read_file', {'path': 'inputs/subjects.json'})
            self.assertTrue(result['success'], (i, result))
        refused = tool.call('read_file', {'path': 'inputs/subjects.json'})
        self.assertFalse(refused['success'])
        self.assertEqual(tool.tool_count, 96)
        self.assertEqual(refused['budget']['tool_calls_remaining'], 0)

    def test_nr_dispatch_also_works_after_48(self):
        for arm in ('B0', 'B1'):
            tool = self.tools(arm)
            tool.tool_count = 48
            result = tool.call('search_skills', {'query': 'labels', 'limit': 1})
            self.assertEqual(result.get('error_type'), 'no_skill_match', result)
            self.assertEqual(tool.tool_count, 49)

    def test_actual_elapsed_is_not_based_on_old_120_seconds(self):
        tool = self.tools(elapsed=170)
        self.assertTrue(tool.call('run_python', {'code': SOLUTION})['success'])
        result = tool.call('finish_result', {'status': 'complete'})
        self.assertTrue(result['public_check_pass'])
        self.assertGreaterEqual(tool.submissions[0]['elapsed_seconds'], 170)
        self.assertLess(tool.submissions[0]['elapsed_seconds'], 180)

    def test_deadline_and_round_caps_are_enforced(self):
        tool = self.tools(elapsed=301)
        self.assertFalse(tool.call('run_python', {'code': SOLUTION})['success'])
        self.assertFalse(tool.submit()['success'])
        self.assertFalse((tool.output / 'matched.csv').exists())
        tool = self.tools()
        tool.begin_round(24)
        with self.assertRaises(ValueError):
            tool.begin_round(25)

    def test_complex_profile_caps_and_common_schema(self):
        prefix = None
        for arm in ('A', 'B0', 'B1'):
            tool = self.tools(arm, COMPLEX)
            tool.begin_round(48)
            if prefix is None:
                prefix = tool.schemas()[:6]
            self.assertEqual(tool.schemas()[:6], prefix)
            tool.tool_count = 191
            self.assertTrue(tool.call('read_file', {'path': 'inputs/subjects.json'})['success'])
            self.assertFalse(tool.call('read_file', {'path': 'inputs/subjects.json'})['success'])
        self.assertIn('900 seconds', base_prompt(COMPLEX))
        self.assertIn('48 model rounds and 192 tool calls', base_prompt(COMPLEX))
        with self.assertRaises(ValueError):
            Budget(True, 24, 96, 30)

    def test_driver_finishes_at_round_24_with_real_file_and_oracle(self):
        import neurobench.runner.exec_budget_flash_trial as runner
        root = self.fixture.root / 'trial'
        (root / 'methods/skills').mkdir(parents=True)
        write_json(root / 'methods/skill_catalog.json', [])
        calls = []
        def model(history, schemas, log, deadline):
            calls.append(history[-1]['content'])
            n = len(calls)
            raw = [{'name': 'inspect_outputs', 'arguments': {}, 'call_id': f'c{n}'}]
            if n == 24:
                raw = [{'name': 'run_python', 'arguments': {'code': SOLUTION}, 'call_id': 'produce'},
                       {'name': 'finish_result', 'arguments': {'status': 'complete'}, 'call_id': 'finish'}]
            return {'message': {'role': 'assistant', 'content': '', 'reasoning_content': ''},
                    'text': '', 'calls': raw, 'model': 'deepseek-flash'}, {
                        'usage_complete': True, 'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2}
        slot = {'id': 'deepseek-v4.1-flash__N01__A', 'model': 'deepseek-v4.1-flash',
                'case_id': 'N01', 'arm': 'A', 'family_id': 'labels', 'input_kind': 'synthetic_test'}
        with patch.object(runner, 'PACKET', self.fixture.packet), patch('neurobench.runner.exec_brief.channel.flash_round', model):
            result = runner.attempt(root, slot)
        self.assertEqual(result['native_request_count'], 24)
        self.assertTrue(result['final_pass'], result)
        self.assertEqual(result['submissions'][0]['model_round'], 24)
        self.assertIn('"max_model_rounds": 24', calls[-1])
        self.assertIn('300 seconds', (root / 'attempts' / slot['id'] / 'PROMPT.txt').read_text())


if __name__ == '__main__':
    unittest.main()
