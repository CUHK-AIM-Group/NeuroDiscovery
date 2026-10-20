"""Regression checks for protocol failures observed in the original trial."""
from pathlib import Path
import json
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

from neurobench.runner.exec_dev.common import Case, file_map, load_case, read_json, sha, write_json
from neurobench.runner.exec_brief.tools import BriefTools, PYTHON_HOST
from neurobench.runner.exec_brief.channel import flash_round

REPO = Path(__file__).resolve().parents[2]
PACKET = REPO / 'tmp/benchmark_exec_brief_dev_20261019_r1/packet'


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def tools(self):
        work, private = self.root / 'work', self.root / 'attempt'
        for name in ('inputs', 'context', 'output', 'scratch'):
            (work / name).mkdir(parents=True)
        private.mkdir()
        contract = {'case_id': 'N00', 'title': 'Synthetic protocol fixture', 'input_sha256': {}, 'required_outputs': []}
        write_json(work / 'context/CONTRACT.json', contract)
        (work / 'python_host.py').write_text(PYTHON_HOST, encoding='utf-8')
        return BriefTools(Case(work, contract), work, private, 'A', time.monotonic() + 20)

    def test_explicit_utf8_preserves_stdout_and_stderr_bytes(self):
        tools = self.tools()
        r = tools.call('run_python', {'code': "import sys\nprint('中文 α ≥')\nprint('标准差 σ',file=sys.stderr)"})
        self.assertTrue(r['success'], r)
        self.assertIn('中文 α ≥', r['stdout'])
        self.assertIn('标准差 σ', r['stderr'])
        self.assertEqual((tools.attempt / 'python_receipts/001.stdout.bin').read_bytes().decode('utf-8'), r['stdout'])

    def test_manifest_provenance_fields_cannot_be_overridden(self):
        tools = self.tools()
        r = tools.call('run_python', {'code': "finish_result(input_sha256='forged')"})
        self.assertFalse(r['success'])
        self.assertIn('reserved_result_manifest_field', r['stderr'])
        self.assertFalse((tools.output / 'result.json').exists())

    @unittest.skipUnless((PACKET / 'INDEX.json').is_file(), 'Requires the retained local brief packet')
    def test_public_packet_has_no_fault_annotations_and_nominal_block_outputs(self):
        for row in read_json(PACKET / 'INDEX.json')['cases']:
            case = load_case(PACKET, row['case_id'], sha(PACKET / 'INDEX.json'))
            text = json.dumps(case.contract) + (case.root / 'task.md').read_text(encoding='utf-8')
            for marker in ('input_kind', 'recoverable_fault', 'must_block', 'injected_fault', 'mode=repair'):
                self.assertNotIn(marker, text)
        case = load_case(PACKET, 'N08')
        self.assertNotIn('injected_fault', read_json(case.input('preprocessing.json')))
        self.assertTrue(load_case(PACKET, 'N12').contract['required_outputs'])

    def test_flash_preserves_high_and_reasoning_across_tool_turn(self):
        events = [{'model': 'deepseek-flash', 'choices': [{'index': 0, 'delta': {'reasoning_content': 'reason', 'tool_calls': [{'index': 0, 'id': 'call1', 'function': {'name': 'x', 'arguments': '{"a":1}'}}]}, 'finish_reason': 'tool_calls'}]}, {'model': 'deepseek-flash', 'choices': [], 'usage': {'prompt_tokens': 2, 'completion_tokens': 3}}]
        with patch('neurobench.runner.exec_brief.channel._http', return_value=iter(events)) as http, patch('neurobench.runner.exec_brief.channel.credential', return_value='synthetic'):
            result, usage = flash_round([], [], None, time.monotonic() + 1)
        self.assertEqual(http.call_args.args[0]['model'], 'deepseek-flash')
        self.assertEqual(http.call_args.args[0]['reasoning_effort'], 'high')
        self.assertEqual(result['message']['reasoning_content'], 'reason')
        self.assertEqual(result['calls'][0]['arguments'], {'a': 1})
        self.assertTrue(usage['usage_complete'])

    def test_actual_artifacts_finalize_without_extra_model_round(self):
        if not (PACKET / 'INDEX.json').is_file():
            self.skipTest('Requires the retained local brief packet')
        from neurobench.runner.exec_brief_trial import attempt
        root = self.root / 'trial'
        (root / 'methods/skills').mkdir(parents=True)
        write_json(root / 'methods/skill_catalog.json', [])
        code = '''import csv,json
ids=json.loads((INPUT_DIR/'subjects.json').read_text())
with (INPUT_DIR/'labels.csv').open(encoding='utf-8-sig') as f:
 labels={r['subject_id']:r['label'] for r in csv.DictReader(f)}
with (OUTPUT_DIR/'matched.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['subject_id','label']);w.writerows((s,labels[s]) for s in ids)
finish_result()
'''
        result = {'output': [], 'response': {'model': 'gpt-5.6-sol'}, 'calls': [{'name': 'run_python', 'arguments': {'code': code}, 'call_id': 'synthetic1'}], 'text': ''}
        usage = {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2, 'usage_complete': True}
        with patch('neurobench.runner.high_channels.responses_round', return_value=(result, usage)):
            out = attempt(root, {'id': 'synthetic__N01__A', 'model': 'gpt-5.6-sol', 'case_id': 'N01', 'arm': 'A'})
        self.assertTrue(out['final_pass'], out)
        self.assertEqual(out['native_request_count'], 1)
        self.assertEqual(len(out['submissions']), 1)


if __name__ == '__main__':
    unittest.main()
