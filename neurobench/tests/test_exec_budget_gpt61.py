"""GPT Responses selection and last-round delivery with the frozen budget tools."""
from pathlib import Path
import unittest
from unittest.mock import patch

from neurobench.runner.exec_dev.common import write_json
from neurobench.tests.test_exec_protocol_v2 import ProtocolTests, SOLUTION


class GptBudgetTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ProtocolTests('test_real_artifacts_finalize_with_host_provenance')
        self.fixture.setUp()
        self.root = self.fixture.root / 'trial'
        (self.root / 'methods/skills').mkdir(parents=True)
        write_json(self.root / 'methods/skill_catalog.json', [])
        self.slot = {'id':'gpt-6.1-sol__N01__A','model':'gpt-6.1-sol','case_id':'N01',
                     'arm':'A','family_id':'labels','input_kind':'synthetic_test'}

    def tearDown(self):
        self.fixture.tearDown()

    def test_final_round_24_uses_gpt_route_and_real_delivery(self):
        import neurobench.runner.exec_budget_gpt61_trial as runner
        from neurobench.runner.exec_brief_gpt61_channel import GPT61_ROUTE
        calls=[]
        def native(history, schemas, log, deadline, *, route):
            self.assertEqual(route,GPT61_ROUTE)
            self.assertEqual(schemas[5]['name'],'finish_result')
            calls.append(history[-1]['content'])
            raw=[{'name':'inspect_outputs','arguments':{},'call_id':str(len(calls))}]
            if len(calls)==24:
                raw=[{'name':'run_python','arguments':{'code':SOLUTION},'call_id':'produce'},
                     {'name':'finish_result','arguments':{'status':'complete'},'call_id':'finish'}]
            return {'output':[],'response':{'model':'gpt-6.1-sol','reasoning':{'effort':'high'}},
                    'calls':raw,'text':''}, {'usage_complete':True,'input_tokens':1,'output_tokens':1,'total_tokens':2}
        with patch.object(runner,'PACKET',self.fixture.packet), \
             patch('neurobench.runner.exec_brief_gpt61_channel.responses_round',native), \
             patch('neurobench.runner.exec_brief.channel.flash_round',side_effect=AssertionError('Flash must not run')):
            result=runner.attempt(self.root,self.slot)
        self.assertTrue(result['final_pass'],result)
        self.assertEqual(result['native_request_count'],24)
        self.assertEqual(result['submissions'][0]['model_round'],24)
        self.assertGreaterEqual(result['submissions'][0]['elapsed_seconds'],0)
        self.assertIn('"max_model_rounds": 24',calls[-1])

    def test_native_failure_is_retained_without_terminal_retry(self):
        import neurobench.runner.exec_budget_gpt61_trial as runner
        with patch.object(runner,'PACKET',self.fixture.packet), \
             patch('neurobench.runner.exec_brief_gpt61_channel.responses_round',side_effect=RuntimeError('native_response_failed:server_error')) as native:
            result=runner.attempt(self.root,self.slot)
        self.assertEqual(native.call_count,1)
        self.assertEqual(result['status'],'UNKNOWN')
        self.assertFalse(result['final_pass'])
        self.assertEqual(result['submissions'],[])
        self.assertTrue((self.root/'attempts'/self.slot['id']/'unfinished_output').is_dir())


if __name__=='__main__':
    unittest.main()
