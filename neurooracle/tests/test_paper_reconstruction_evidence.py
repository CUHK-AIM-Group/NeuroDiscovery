from copy import deepcopy
from pathlib import Path
import sqlite3
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from paper_evidence_store import CONDITIONS, SCHEMA, anchor, claim_id, metrics, validate, validate_claims


class EvidenceBoundaries(unittest.TestCase):
    def fixtures(self):
        text='Patients had smaller volumes (p < 0.01).'
        claim={'statement':'Patients have smaller volume than controls.',
               'semantic_key':{'subject':'patients','relation_kind':'group_difference',
                               'measurement':'MRI hippocampal volume','necessary_qualifiers':{'comparator':'controls'}},
               'scope_rule':'Same measurement and comparator.'}
        source={'job':{'job_id':'PAPER:1'},'source_sha256':'sha','old_claims_included':False,
                'fixed_layer_write_permission':False,'source_snapshot':{'abstract':text}}
        record={'job_id':'PAPER:1','source_sha256':'sha','review':{'reader':'root','coverage':'available_abstract'},
                'study':{'cohort_overlap_status':'unknown'},'observations':[{
                  'observation_id':'O:1','role':'primary_result','statement':'Volumes were smaller.',
                  'anchors':[anchor(text,text)],'conditions':dict.fromkeys(CONDITIONS),
                  'missing_fields':{'effect_estimate':'not_reported'},
                  'result':{'direction':'lower','significance':'reported_significant','effect_estimate':None,'confidence_interval':None},
                  'statistics':[{'type':'p_value','raw':'p < 0.01','operator':'<','value':.01}],
                  'evidence':[{'claim_key':'a','relation':'supports','rationale':'Literal own result.','scope_match':'matched'}]}]}
        return record,source,{'a':claim}

    def test_anchor_and_numeric_operator_remain_literal(self):
        p,s,c=self.fixtures();validate(p,s,c)
        p['observations'][0]['anchors'][0]['start']=1
        with self.assertRaisesRegex(ValueError,'Source quote'):validate(p,s,c)

    def test_snapshot_substitution_is_rejected(self):
        p,s,c=self.fixtures();p['source_sha256']='another'
        with self.assertRaisesRegex(ValueError,'snapshot'):validate(p,s,c)

    def test_owning_publication_mismatch_is_rejected(self):
        p,s,c=self.fixtures();s['publications']=[{'ids_json':'{"pmid":"123"}'}]
        s['source_snapshot']['paper']={'pmid':'456'}
        with self.assertRaisesRegex(ValueError,'owning identifier'):validate(p,s,c)

    def test_unanchored_statistic_rejected(self):
        p,s,c=self.fixtures();p['observations'][0]['statistics'][0]['raw']='p = 0.01'
        with self.assertRaisesRegex(ValueError,'Statistic'):validate(p,s,c)

    def test_background_cannot_be_experimental_support(self):
        p,s,c=self.fixtures();p['observations'][0]['role']='background'
        with self.assertRaisesRegex(ValueError,'Non-result'):validate(p,s,c)

    def test_null_does_not_become_zero_estimate(self):
        p,s,c=self.fixtures();o=p['observations'][0]
        o['result']['significance']='not_significant';o['evidence'][0]['relation']='opposes'
        validate(p,s,c);self.assertIsNone(o['result']['effect_estimate'])

    def test_claim_identity_depends_on_measurement_not_paper_or_wording(self):
        _,_,c=self.fixtures();a=c['a'];b=deepcopy(a);b['statement']='Paraphrase of the same proposition.'
        self.assertEqual(claim_id(a),claim_id(b))
        b['semantic_key']['measurement']='postmortem Heschl thickness'
        self.assertNotEqual(claim_id(a),claim_id(b))
        a['semantic_key']['pmid']='123'
        with self.assertRaises(ValueError):validate_claims(c)

    def test_same_work_multiple_observations_and_synthesis_do_not_inflate(self):
        c=sqlite3.connect(':memory:');c.executescript(SCHEMA)
        c.execute("insert into claims values('C','{}','Shared proposition','{}')")
        rows=[('J1','W1','declared_identifiers'),('J2','W1','declared_identifiers'),
              ('J3','W3','declared_identifiers'),('J4','W4','version_identity_review')]
        for job,work,identity in rows:c.execute('insert into papers values(?,?,?,?,?,?,?)',(job,work,identity,'sha','b','{}','{}'))
        for oid,job,role,rel in [('o1','J1','primary_result','supports'),('o2','J1','primary_result','supports'),
                ('o3','J2','primary_result','supports'),('o4','J3','synthesis_result','supports'),
                ('o5','J3','primary_result','partial'),('o6','J4','primary_result','supports')]:
            c.execute('insert into observations values(?,?,?,?,?)',(oid,job,role,'result','{}'))
            c.execute('insert into evidence values(?,?,?,?,?)',(oid,'C',rel,'reason','scope'))
        self.assertEqual(metrics(c)['multipaper_claims_with_primary_support'],0)
        c.execute("insert into observations values('o7','J3','primary_result','result','{}')")
        c.execute("insert into evidence values('o7','C','supports','reason','scope')")
        self.assertEqual(metrics(c)['shared_claims'][0]['primary_supporting_work_records'],2)
        self.assertEqual(metrics(c)['independent_cohort_replications_verified'],0)
        c.close()


if __name__=='__main__':unittest.main()
