from copy import deepcopy
from pathlib import Path
import sys
import unittest
import json
import tempfile
import hashlib
import sqlite3
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import paper_batch_model as batch
from paper_batch_model import validate_candidate
from paper_batch_inputs import compact_source,encode_input


class ModelExtractionSafety(unittest.TestCase):
    def test_manual_mode_blocks_send_before_credentials_or_ledger_access(self):
        with patch.object(batch, 'read', return_value={'execution': {'external_models': False}}), \
                patch.object(batch, 'output_dir') as output, \
                patch.object(batch, 'db_connect') as database, \
                patch.object(batch, 'key_for_provider') as credentials:
            result = batch.send_one('existing-request')
            self.assertEqual(result['status'], 'NOT_DISPATCHED_EXTERNAL_MODELS_DISABLED')
            output.assert_not_called()
            database.assert_not_called()
            credentials.assert_not_called()

    def test_manual_mode_blocks_runner_before_lock_creation(self):
        with patch.object(batch, 'read', return_value={'execution': {'external_models': False}}), \
                patch.object(batch, 'output_dir') as output:
            with self.assertRaisesRegex(RuntimeError, 'disabled'):
                batch.run(['existing-request'])
            output.assert_not_called()

    def example(self):
        text='Patients had lower volume (p < 0.01).'
        source={'job':{'work_key':'PMID:1','job_id':'PAPER:1'},'source_snapshot':{'abstract':text}}
        o={'statement':'Lower volume.','quotes':[text],'role':'primary_result',
           'proposition':{'subject':'patients','relation':'group_difference','object':'hippocampus','measurement':'MRI volume','scope':{},'direction':'lower'},
           'evidence_relation':'supports','scope_reason':'Same measurement.','conditions':{},'result':{},
           'statistics':[{'raw':'p < 0.01','kind':'p_value','operator':'<','value':.01}], 'limitations':[]}
        return {'paper_id':'PMID:1','study':{'samples':[],'quotes':[]},'observations':[o],'unresolved':[],'coverage_note':'abstract'},source

    def test_correct_literal_candidate(self):
        v,s=self.example();self.assertEqual(validate_candidate(v,s),[])
    def test_cross_paper_identity_rejected(self):
        v,s=self.example();v['paper_id']='PMID:2';self.assertIn('wrong_paper_id',validate_candidate(v,s))
    def test_altered_quote_rejected(self):
        v,s=self.example();v['observations'][0]['quotes']=['patients had lower volume (p < 0.01).']
        self.assertIn('observation_0_quote_not_unique_literal',validate_candidate(v,s))
    def test_review_background_cannot_be_own_experimental_support(self):
        v,s=self.example();v['observations'][0]['role']='background'
        self.assertIn('observation_0_nonresult_support',validate_candidate(v,s))
    def test_wrong_number_even_with_true_quote_rejected(self):
        v,s=self.example();v['observations'][0]['statistics'][0]['value']=.1
        self.assertIn('observation_0_p_value_or_operator_changed',validate_candidate(v,s))
    def test_wrong_operator_rejected(self):
        v,s=self.example();v['observations'][0]['statistics'][0]['operator']='='
        self.assertIn('observation_0_p_value_or_operator_changed',validate_candidate(v,s))
    def test_repeated_ambiguous_source_span_rejected(self):
        v,s=self.example();s['source_snapshot']['abstract']+=' '+s['source_snapshot']['abstract']
        self.assertIn('observation_0_quote_not_unique_literal',validate_candidate(v,s))
    def test_null_proposition_preserves_unresolved_evidence(self):
        v,s=self.example();v['observations'][0]['proposition']=None;v['observations'][0]['evidence_relation']='unresolved'
        self.assertEqual(validate_candidate(v,s),[])

    def test_malformed_array_is_held_instead_of_crashing_worker(self):
        v,s=self.example();v['observations'][0]['quotes']='not an array'
        self.assertIn('observation_0_quotes_not_array',validate_candidate(v,s))
    def test_changed_sample_count_is_held(self):
        v,s=self.example();s['source_snapshot']['abstract']+=' There were 20 patients.'
        v['study']['samples']=[{'raw':'20 patients','n':200}]
        self.assertIn('sample_number_not_source_bound',validate_candidate(v,s))
    def test_credential_never_sent_to_an_unbound_host(self):
        with self.assertRaises(ValueError):
            batch.key_for_provider({'provider':'https://unbound.invalid/v1','credential_source':'repo_aimgroup_key'})


class DurableDispatchSafety(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.directory=Path(self.temp.name)
        (self.directory/'AUTHORIZATION.json').write_text(json.dumps({'validation_budget_usd':5,'validation_max_requests':60}))
        self.transport={'id':'test','provider':'https://api.openlux.ai/v1'}
        config = self.directory / 'CONFIG.json'
        config.write_text(json.dumps({'execution': {'external_models': True}}))
        self.config = patch.object(batch, 'CONFIG', config)
        self.config.start()
        self.addCleanup(self.config.stop)
        self.output=patch.object(batch,'output_dir',return_value=self.directory);self.output.start()
    def tearDown(self):self.output.stop();self.temp.cleanup()
    def add(self,rid,status='PREPARED',reserve=1):
        c=batch.db_connect()
        c.execute('insert into requests(request_id,status,reserved_usd,transport_json) values(?,?,?,?)',
                  (rid,status,reserve,json.dumps(self.transport)));c.commit();c.close()
    def test_completed_and_inflight_are_never_resent(self):
        self.add('done','CANDIDATE_READY');self.add('unknown','IN_FLIGHT')
        with patch.object(batch,'key_for_provider',side_effect=AssertionError('No network credentials should be loaded')):
            self.assertEqual(batch.send_one('done')['status'],'NOT_RESENT')
            self.assertEqual(batch.send_one('unknown')['status'],'NOT_RESENT')
    def test_unknown_failure_cost_stays_reserved(self):
        self.add('failed','HELD',4.8);self.add('next',reserve=.3)
        with patch.object(batch,'key_for_provider',side_effect=AssertionError('Budget must block before sending')):
            self.assertEqual(batch.send_one('next')['status'],'BUDGET_NOT_DISPATCHED')
    def test_authentication_failure_blocks_that_transport(self):
        self.add('next');c=batch.db_connect()
        c.execute('insert into run_control values(?,?)',('credential_failure:test','{}'));c.commit();c.close()
        with patch.object(batch,'key_for_provider',side_effect=AssertionError('Circuit must block before sending')):
            self.assertEqual(batch.send_one('next')['status'],'NOT_DISPATCHED_CREDENTIAL_FAILURE')


class SourceInputBoundary(unittest.TestCase):
    def test_dispatch_reuses_compiled_source_without_reopening_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary);d=out/'BATCH_MODEL';d.mkdir()
            q=sqlite3.connect(out/'PAPERS.sqlite')
            q.executescript('create table publications(pub_id text,job_id text); create table paper_jobs(job_id text,status text,selected_source_sha text);')
            q.execute('insert into paper_jobs values(?,?,?)',('paper1','ready_for_source_review','source1'));q.commit();q.close()
            value=compact_source('DOI:example','Source title',{'abstract':'A source-only abstract.'})
            sha,blob,_=encode_input(value);database=d/'INPUTS.sqlite';c=sqlite3.connect(database)
            c.execute('create table inputs(job_id text,source_sha256 text,input_sha256 text,input_zlib blob,status text)')
            c.execute('insert into inputs values(?,?,?,?,?)',('paper1','source1',sha,blob,'READY'));c.commit();c.close()
            stat=database.stat()
            (d/'INPUTS_READY.json').write_text(json.dumps({'database':{'path':str(database),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}}))
            (d/'AUTHORIZATION.json').write_text(json.dumps({'prompt_sha256':hashlib.sha256(batch.SYSTEM.encode()).hexdigest(),'max_output_tokens':6144}))
            transport={'id':'bound-test','provider':'https://api.openlux.ai/v1','candidate_models':['test-model'],'request_options':{'max_completion_tokens':6144}}
            with patch.object(batch,'output_dir',return_value=d),patch.object(batch,'settings',return_value=({},out,None,None)),patch.object(batch,'rates_for',return_value=(.000001,2)),patch.object(batch,'packet',side_effect=AssertionError('Archive must not reopen')):
                first=batch.prepare([],['test-model'],transport,job_ids=['paper1'])
                second=batch.prepare([],['test-model'],transport,job_ids=['paper1'])
                self.assertEqual(first,second)
                c=batch.db_connect();self.assertEqual(c.execute('select count(*) from requests').fetchone()[0],1)
                row=c.execute('select body_json,source_sha256,status from requests').fetchone();c.close()
                self.assertEqual(json.loads(json.loads(row[0])['messages'][1]['content']),value)
                self.assertEqual(row[1:],('source1','PREPARED'))

    def test_source_metadata_retained_legacy_answers_excluded(self):
        raw={'abstract_text':'Exact source text.','claim':'legacy answer','root_review':'old judgment',
             'publication_types':['Journal Article'],'paper':{'title':'Source title','publication_types':['Review'],
             'comments_corrections':[{'type':'ErratumIn','pmid':'2'}]}}
        value=compact_source('PMID:1','Fallback',raw)
        self.assertEqual(value['abstract'],'Exact source text.')
        self.assertEqual(value['publication_types'],['Journal Article','Review'])
        self.assertEqual(value['nested_publication_notices']['comments_corrections'][0]['pmid'],'2')
        self.assertNotIn('legacy answer',json.dumps(value));self.assertNotIn('old judgment',json.dumps(value))
    def test_source_bound_encoding_is_stable(self):
        value=compact_source('PMID:1','Title',{'abstract':'An abstract.'})
        self.assertEqual(encode_input(value),encode_input(dict(reversed(list(value.items())))))


if __name__=='__main__':unittest.main()
