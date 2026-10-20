"""Portable protocol regressions using real files and mocked native inference."""
from pathlib import Path
import json
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

from neurobench.runner.exec_dev.common import Case, file_map, read_json, sha, write_json
from neurobench.runner.exec_brief.oracle import evaluate
from neurobench.runner.exec_protocol_v2 import ProtocolTools, PYTHON_HOST, public_checks


SOLUTION = '''import csv,json
subjects=json.loads((INPUT_DIR/'subjects.json').read_text())
with (INPUT_DIR/'labels.csv').open() as f:
    labels={r['subject_id']:r['label'] for r in csv.DictReader(f)}
with (OUTPUT_DIR/'matched.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['subject_id','label'])
    w.writerows((s,labels[s]) for s in subjects)
'''


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.packet = self.root / 'packet'
        self.case_root = self.packet / 'public/cases/N01'
        inputs = self.case_root / 'inputs'
        inputs.mkdir(parents=True)
        write_json(inputs / 'subjects.json', ['s2', 's1'])
        (inputs / 'labels.csv').write_text('subject_id,label\ns1,21\ns2,32\n', encoding='utf-8')
        self.contract = {'case_id': 'N01', 'title': 'Match participant labels',
                         'input_sha256': file_map(inputs), 'required_outputs': ['matched.csv']}
        write_json(self.case_root / 'CONTRACT.json', self.contract)
        (self.case_root / 'task.md').write_text('Deliver the labels for the supplied participants.', encoding='utf-8')
        write_json(self.packet / 'INDEX.json', {'cases': [{'case_id': 'N01',
            'contract_sha256': sha(self.case_root / 'CONTRACT.json'),
            'task_sha256': sha(self.case_root / 'task.md')}]})
        self.case = Case(self.case_root, self.contract)
        self.count = 0

    def tearDown(self):
        self.temp.cleanup()

    def tools(self, arm='A'):
        self.count += 1
        work = self.root / f'work{self.count}'
        private = self.root / f'attempt{self.count}'
        private.mkdir()
        shutil.copytree(self.case_root / 'inputs', work / 'inputs')
        (work / 'context/skills').mkdir(parents=True)
        for name in ('task.md', 'CONTRACT.json'):
            shutil.copy2(self.case_root / name, work / 'context' / name)
        write_json(work / 'context/skill_catalog.json', [])
        for name in ('output', 'scratch'):
            (work / name).mkdir()
        (work / 'python_host.py').write_text(PYTHON_HOST, encoding='utf-8')
        return ProtocolTools(self.case, work, private, arm, time.monotonic() + 120)

    def produce(self, tools):
        result = tools.call('run_python', {'code': SOLUTION})
        self.assertTrue(result['success'], result)

    def test_missing_manifest_preserves_inventory_and_action(self):
        tools = self.tools()
        self.produce(tools)
        diagnostic = tools.call('inspect_outputs', {})
        self.assertEqual(diagnostic['state'], 'awaiting_finalization')
        self.assertEqual(diagnostic['files']['matched.csv'], sha(tools.output / 'matched.csv'))
        self.assertEqual(diagnostic['next_action'], {'tool': 'finish_result', 'arguments': {'status': 'complete'}})
        self.assertFalse(diagnostic['public_check_pass'])

    def test_same_common_finish_schema_in_all_arms(self):
        common = None
        common_prefix = None
        for arm in ('A', 'B0', 'B1'):
            ordered = self.tools(arm).schemas()
            schemas = {s['name']: s for s in ordered}
            finish = schemas['finish_result']
            self.assertFalse(finish['parameters']['additionalProperties'])
            self.assertNotIn('case_id', finish['parameters']['properties'])
            if common is not None:
                self.assertEqual(finish, common)
                self.assertEqual(ordered[:6], common_prefix)
            common = finish
            common_prefix = ordered[:6]

    def test_real_artifacts_finalize_with_host_provenance(self):
        tools = self.tools()
        self.produce(tools)
        tools.begin_round(12)
        result = tools.call('finish_result', {'status': 'complete', 'summary': 'Delivered matched labels.'})
        self.assertTrue(result['public_check_pass'], result)
        self.assertTrue(tools.terminal)
        state = read_json(tools.output / 'result.json')
        self.assertEqual(state['input_sha256'], self.contract['input_sha256'])
        self.assertEqual(state['artifacts']['matched.csv'], sha(tools.output / 'matched.csv'))
        self.assertEqual(evaluate(self.case, Path(tools.submissions[0]['output']))['verdict'], 'PASS')
        self.assertTrue(list((tools.attempt / 'protocol_finalizations').glob('*.json')))

    def test_caller_cannot_forge_provenance_and_can_retry_legal_action(self):
        tools = self.tools()
        self.produce(tools)
        for key in ('case_id', 'input_sha256', 'artifacts'):
            bad = tools.call('finish_result', {'status': 'complete', key: 'forged'})
            self.assertEqual(bad['error'], 'host_owned_fields')
            self.assertFalse((tools.output / 'result.json').exists())
        fixed = tools.call(bad['next_action']['tool'], bad['next_action']['arguments'])
        self.assertTrue(fixed['public_check_pass'], fixed)

    def test_missing_artifact_is_not_fabricated(self):
        tools = self.tools()
        result = tools.call('finish_result', {'status': 'complete'})
        self.assertEqual(result['error'], 'required_outputs_missing')
        self.assertFalse((tools.output / 'result.json').exists())
        self.assertEqual(tools.submissions, [])

    def test_public_success_does_not_approve_wrong_scientific_values(self):
        tools = self.tools()
        (tools.output / 'matched.csv').write_text('subject_id,label\ns2,999\ns1,999\n')
        result = tools.call('finish_result', {'status': 'complete'})
        self.assertTrue(result['public_check_pass'])
        self.assertEqual(evaluate(self.case, Path(tools.submissions[0]['output']))['verdict'], 'FAIL')

    def test_changed_inputs_cannot_receive_host_bound_completion(self):
        tools = self.tools()
        self.produce(tools)
        (tools.workspace / 'inputs/labels.csv').write_text('subject_id,label\ns1,999\n')
        result = tools.call('finish_result', {'status': 'complete'})
        self.assertEqual(result['error'], 'input_integrity_lost')
        self.assertFalse((tools.output / 'result.json').exists())

    def test_failed_submission_remains_immutable_after_repair(self):
        tools = self.tools()
        self.produce(tools)
        write_json(tools.output / 'result.json', {'case_id': 'wrong', 'status': 'complete',
            'input_sha256': {}, 'artifacts': {}})
        first = tools.call('submit', {})
        self.assertFalse(first['public_check_pass'])
        self.assertFalse(first['terminal'])
        original = Path(tools.submissions[0]['output']).parent
        before = file_map(original)
        final = tools.call('finish_result', {'status': 'complete'})
        self.assertTrue(final['public_check_pass'])
        self.assertEqual(len(tools.submissions), 2)
        self.assertEqual(file_map(original), before)

    def test_deadline_and_tool_caps_prevent_finalization(self):
        tools = self.tools()
        self.produce(tools)
        tools.tool_count = 48
        self.assertEqual(tools.call('finish_result', {'status': 'complete'})['error'], 'tool_or_wall_budget_reached')
        self.assertFalse((tools.output / 'result.json').exists())
        tools = self.tools()
        self.produce(tools)
        tools.deadline = time.monotonic() - 1
        self.assertEqual(tools.call('finish_result', {'status': 'complete'})['error'], 'tool_or_wall_budget_reached')
        self.assertFalse((tools.output / 'result.json').exists())
        with self.assertRaises(ValueError):
            tools.begin_round(13)

    def test_blocked_requires_reason_and_preserves_evidence(self):
        tools = self.tools()
        self.assertEqual(tools.call('finish_result', {'status': 'blocked'})['error'], 'blocked_reason_required')
        result = tools.call('finish_result', {'status': 'blocked', 'reason': 'The requested measurement is unavailable.',
                                             'evidence': ['Observed missing required source.']})
        self.assertTrue(result['public_check_pass'])
        self.assertEqual(read_json(tools.output / 'result.json')['status'], 'blocked')

    def test_python_context_reset_is_explicit_and_files_persist(self):
        tools = self.tools()
        first = tools.call('run_python', {'code': "temporary_value=7\n(OUTPUT_DIR/'persist.txt').write_text('kept')"})
        second = tools.call('run_python', {'code': "print('temporary_value' in globals())\nprint((OUTPUT_DIR/'persist.txt').read_text())"})
        self.assertTrue(first['success'] and second['success'])
        self.assertIn('False', second['stdout'])
        self.assertIn('kept', second['stdout'])
        self.assertEqual(second['python_process'], 'fresh_per_call_files_persist_variables_reset')
        tools.begin_round(10)
        self.assertTrue(tools.call('inspect_outputs', {})['budget']['closeout_recommended'])

    def test_legacy_helper_error_has_registered_recovery_action(self):
        tools = self.tools()
        self.produce(tools)
        bad = tools.call('run_python', {'code': "finish_result(case_id='forged')"})
        self.assertFalse(bad['success'])
        self.assertEqual(bad['error'], 'host_owned_fields')
        self.assertEqual(bad['next_action']['tool'], 'finish_result')
        self.assertTrue(tools.call('finish_result', {'status': 'complete'})['public_check_pass'])

    def test_malformed_manifest_retains_artifact_inventory(self):
        tools = self.tools()
        self.produce(tools)
        (tools.output / 'result.json').write_text('{bad-json')
        diagnostic = public_checks(self.contract, tools.output)
        self.assertEqual(diagnostic['state'], 'invalid_manifest')
        self.assertIn('matched.csv', diagnostic['files'])
        self.assertIn('manifest_invalid', diagnostic['errors'])

    def run_mock_attempt(self, model, arm, last_round=False):
        from neurobench.runner import exec_protocol_v2_trial as driver
        self.count += 1
        trial = self.root / f'trial{self.count}'
        (trial / 'methods/skills').mkdir(parents=True)
        write_json(trial / 'methods/skill_catalog.json', [])
        feedback = []

        def native(history, schemas, log, deadline, **kwargs):
            value = json.loads(history[-1]['content'].split(': ', 1)[1])
            feedback.append(value)
            self.assertIn('finish_result', {s['name'] for s in schemas})
            if last_round and len(feedback) < 12:
                calls = [{'name': 'read_file', 'arguments': {'path': 'context/task.md'}, 'call_id': str(len(feedback))}]
            else:
                calls = [{'name': 'run_python', 'arguments': {'code': SOLUTION}, 'call_id': 'produce'},
                         {'name': 'finish_result', 'arguments': {'status': 'complete'}, 'call_id': 'finish'}]
            result = {'output': [], 'response': {'model': model}, 'message': {'role': 'assistant', 'content': ''},
                      'model': 'deepseek-flash', 'text': '', 'calls': calls}
            return result, {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2, 'usage_complete': True}

        with patch.object(driver, 'PACKET', self.packet), \
             patch('neurobench.runner.exec_brief_gpt61_channel.responses_round', side_effect=native), \
             patch('neurobench.runner.exec_brief.channel.flash_round', side_effect=native):
            result = driver.attempt(trial, {'id': model + '__N01__' + arm, 'model': model,
                'case_id': 'N01', 'arm': arm, 'family_id': 'synthetic', 'input_kind': 'normal'})
        self.assertTrue(result['final_pass'], result)
        self.assertEqual(result['status'], 'SUBMITTED')
        self.assertEqual(result['native_request_count'], 12 if last_round else 1)
        self.assertEqual(len(result['submissions']), 1)
        self.assertEqual(feedback[-1]['model_rounds_remaining_after_current'], 0 if last_round else 11)
        if last_round:
            self.assertTrue(feedback[-1]['closeout_recommended'])

    def test_both_channels_and_all_arms_share_real_delivery_path(self):
        for model in ('gpt-6.1-sol', 'deepseek-v4.1-flash'):
            for arm in ('A', 'B0', 'B1'):
                with self.subTest(model=model, arm=arm):
                    self.run_mock_attempt(model, arm)

    def test_finish_in_last_round_needs_no_extra_model_call(self):
        self.run_mock_attempt('deepseek-v4.1-flash', 'A', last_round=True)


if __name__ == '__main__':
    unittest.main()
