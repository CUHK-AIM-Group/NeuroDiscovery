"""Driver boundary and receipt tests, without model requests or reference answers."""
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest

from neurobench.runner.exec_trial import (INDEX_SHA, METHODS, PACKET, PYTHON_HOST, TrialTools,
                              public_checks)
from neurobench.runner.exec_dev.common import load_case, read_json


@unittest.skipUnless((PACKET / 'INDEX.json').is_file() and METHODS.is_dir(),
                     'Requires the retained local execution packet and method bundle')
class ExecutionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def tools(self, case_id='E12', arm='A'):
        case = load_case(PACKET, case_id, INDEX_SHA)
        work, attempt = self.root / 'work', self.root / 'attempt'
        shutil.copytree(case.root / 'inputs', work / 'inputs')
        (work / 'context').mkdir()
        shutil.copy2(case.root / 'CONTRACT.json', work / 'context/CONTRACT.json')
        (work / 'output').mkdir(); (work / 'scratch').mkdir(); attempt.mkdir()
        (work / 'python_host.py').write_text(PYTHON_HOST, encoding='utf-8')
        if arm != 'A':
            shutil.copytree(METHODS / 'skills', work / 'context/skills')
        return TrialTools(case, work, attempt, arm, time.monotonic() + 120)

    def test_execute_and_freeze_real_blocked_delivery(self):
        tools = self.tools()
        r = tools.call('run_python', {'code': "finish_result('blocked',reason_code='MISSING_REQUIRED_INPUT',missing_inputs=['timeseries.npy'])"})
        self.assertTrue(r['success'], r)
        self.assertEqual(read_json(tools.output / 'result.json')['status'], 'blocked')
        first = tools.call('submit', {})
        self.assertTrue(first['public_check_pass'])
        self.assertTrue(tools.terminal)
        self.assertEqual(len(tools.submissions), 1)
        self.assertFalse(tools.call('submit', {})['success'])

    def test_child_denies_external_read_without_echoing_payload(self):
        secret = self.root / 'private.txt'
        secret.write_text('NEVER_VISIBLE_PRIVATE_PAYLOAD', encoding='utf-8')
        tools = self.tools()
        r = tools.call('run_python', {'code': 'print(Path(' + repr(str(secret)) + ').read_text())'})
        self.assertFalse(r['success'])
        self.assertIn('cooperative_workspace_file_boundary', r['stderr'])
        self.assertNotIn('NEVER_VISIBLE_PRIVATE_PAYLOAD', json.dumps(r))

    def test_inputs_cannot_be_overwritten_and_network_disabled(self):
        tools = self.tools()
        r = tools.call('run_python', {'code': "(INPUT_DIR/'new.npy').write_bytes(b'bad')"})
        self.assertFalse(r['success'])
        self.assertFalse((tools.workspace / 'inputs/new.npy').exists())
        r = tools.call('run_python', {'code': 'import socket; socket.socket()'})
        self.assertFalse(r['success'])
        self.assertIn('network_disabled', r['stderr'])

    def test_public_checks_do_not_check_hidden_numerical_correctness(self):
        tools = self.tools('E05')
        r = tools.call('run_python', {'code': "np.save(OUTPUT_DIR/'fc.npy',np.zeros((2,2))); finish_result()"})
        self.assertTrue(r['success'], r)
        self.assertTrue(public_checks(tools.case.contract, tools.output)['public_check_pass'])

    def test_public_repair_preserves_first_submission(self):
        tools = self.tools()
        self.assertFalse(tools.call('submit', {})['public_check_pass'])
        frozen = Path(tools.submissions[0]['output'])
        self.assertFalse((frozen / 'result.json').exists())
        tools.call('run_python', {'code': "finish_result('blocked',reason_code='MISSING_REQUIRED_INPUT',missing_inputs=['timeseries.npy'])"})
        self.assertTrue(tools.call('submit', {})['public_check_pass'])
        self.assertFalse((frozen / 'result.json').exists())
        self.assertEqual(len(tools.submissions), 2)

    def test_actual_qc_host_receipt_and_current_candidate_tools(self):
        tools = self.tools('E09', 'B1')
        names = {s['name'] for s in tools.schemas()}
        self.assertTrue({'compile_task', 'search_skills', 'read_skill', 'read_skill_section'} <= names)
        r = tools.call('compile_task', {})
        self.assertTrue(r['success'], r)
        self.assertFalse(r['numerical_expectations_included'])
        r = tools.call('run_qc_tool', {'mode': 'normal'})
        self.assertTrue(r['success'], r)
        receipt = read_json(tools.receipts / '001.json')
        self.assertIn('QC_COMPLETE', receipt['stdout'])
        self.assertEqual(receipt['output_sha256'], r['output_sha256'])


if __name__ == '__main__':
    unittest.main()
