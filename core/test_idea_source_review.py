"""Source boundaries and eligibility are real; all model responses are synthetic."""
from copy import deepcopy
import hashlib
import json
import sqlite3
import zlib

import pytest

from core.idea_source_review import build_related_evidence, load_source_packets, validate_assessment
from core.novelty_gate import gate_candidates, NoveltyGateError
from core.test_idea_hypotheses import synthetic_candidate, synthetic_batch, synthetic_sources
from core.agent.test_novelty_gate import reviewer


def packet_db(tmp_path):
    (tmp_path/'BATCH_MODEL').mkdir()
    papers = sqlite3.connect(tmp_path/'PAPERS.sqlite')
    papers.executescript('CREATE TABLE publications(pub_id,job_id); CREATE TABLE paper_jobs(job_id,work_key); '
                        'CREATE TABLE potential_versions(job_id,abstract_key);')
    inputs = sqlite3.connect(tmp_path/'BATCH_MODEL/INPUTS.sqlite')
    inputs.execute('CREATE TABLE inputs(job_id,work_key,fold,source_sha256,input_sha256,input_zlib)')
    for i, fold in ((1, 'corpus'), (2, 'heldout'), (3, 'corpus')):
        source, job = 'PMID:'+str(i), 'j'+str(i)
        raw = json.dumps(dict(title='Synthetic source', abstract='Synthetic literal source '+str(i))).encode()
        papers.execute('INSERT INTO publications VALUES(?,?)', (source,job))
        papers.execute('INSERT INTO paper_jobs VALUES(?,?)', (job,source))
        papers.execute('INSERT INTO potential_versions VALUES(?,?)', (job,'protected-version' if i>1 else 'solo'))
        inputs.execute('INSERT INTO inputs VALUES(?,?,?,?,?,?)', (job,source,fold,'synthetic-source-hash',hashlib.sha256(raw).hexdigest(),zlib.compress(raw)))
    papers.commit(); inputs.commit(); papers.close(); inputs.close()


def test_source_loader_reads_corpus_but_never_protected_or_its_known_version(tmp_path, monkeypatch):
    packet_db(tmp_path)
    before = {p:p.read_bytes() for p in tmp_path.rglob('*.sqlite')}
    decompress, seen = zlib.decompress, []
    def checked(raw):
        value=decompress(raw); seen.append(value)
        assert b'Synthetic literal source 1' in value
        return value
    monkeypatch.setattr(zlib, 'decompress', checked)
    got=load_source_packets(['PMID:1','PMID:2','PMID:3','PMID:fake'],base=tmp_path)
    assert got['PMID:1']['status']=='available' and len(seen)==1
    assert got['PMID:2']['status']==got['PMID:3']['status']=='protected_not_read'
    assert got['PMID:fake']['status']=='unavailable'
    assert before=={p:p.read_bytes() for p in before}


def test_source_loader_rejects_wrong_hash_without_rewriting_input(tmp_path):
    packet_db(tmp_path)
    with sqlite3.connect(tmp_path/'BATCH_MODEL/INPUTS.sqlite') as db:
        db.execute("UPDATE inputs SET input_sha256='wrong' WHERE job_id='j1'")
    with pytest.raises(ValueError,match='fingerprint'):
        load_source_packets(['PMID:1'],base=tmp_path)


def test_legacy_medrxiv_identifier_resolves_existing_doi_without_changing_id(tmp_path):
    packet_db(tmp_path)
    sid='PMID:MEDRXIV:10.1101/2025.01.08.24319440'
    with sqlite3.connect(tmp_path/'PAPERS.sqlite') as db:
        db.execute('INSERT INTO publications VALUES(?,?)',('DOI:10.1101/2025.01.08.24319440','j1'))
    result=load_source_packets([sid],base=tmp_path)[sid]
    assert result['status']=='available' and result['source_id']==sid
    assert result['resolved_source_aliases']==['DOI:10.1101/2025.01.08.24319440']


@pytest.mark.parametrize('ambiguous',[False,True])
def test_resolved_alias_never_reads_protected_or_ambiguous_payload(tmp_path,monkeypatch,ambiguous):
    packet_db(tmp_path)
    sid='PMID:MEDRXIV:10.1101/2025.01.08.24319440'
    with sqlite3.connect(tmp_path/'PAPERS.sqlite') as db:
        db.execute('INSERT INTO publications VALUES(?,?)',('DOI:10.1101/2025.01.08.24319440','j2'))
        if ambiguous:db.execute('INSERT INTO publications VALUES(?,?)',(sid,'j1'))
    monkeypatch.setattr(zlib,'decompress',lambda _:pytest.fail('Protected/ambiguous payload must not be read'))
    assert load_source_packets([sid],base=tmp_path)[sid]['status']==('unavailable' if ambiguous else 'protected_not_read')


def run(candidate, call, **kwargs):
    return gate_candidates([candidate],'balanced',call,require_chain=True,resolver=synthetic_batch,**kwargs)


@pytest.mark.parametrize('decision',['reject','hold'])
def test_one_source_objection_cannot_be_outvoted_or_offset_by_high_scores(decision):
    c=synthetic_candidate()
    def call(system,user,label):
        r=json.loads(reviewer({'H1':'substantive_extension'},score=1)(system,user,label))
        if label=='H1 expert:scientific_delta':
            r['source_assessment'].update(decision=decision,bridge='incompatible' if decision=='reject' else 'unknown')
        return json.dumps(r)
    got=run(c,call,peer_response=True)
    assert got['model_calls']==7 and not got['selected_ids']
    assert got['candidates'][0]['chain_gate_verdict']=='revise'
    assert got['candidates'][0]['root_review_required']
    assert got['candidates'][0]['adjudication']['verdict']=='pass'


@pytest.mark.parametrize('bad',['foreign_quote','missing_source','bad_retain','missing_assessment'])
def test_invalid_source_claim_fails_closed(bad):
    def call(system,user,label):
        r=json.loads(reviewer({'H1':'substantive_extension'})(system,user,label))
        a=r['source_assessment']
        if bad=='foreign_quote':a['anchors'][0]['quote']='Invented phrase absent from every source.'
        if bad=='missing_source':a['anchors']=a['anchors'][:1]
        if bad=='bad_retain':a['bridge']='unknown'
        if bad=='missing_assessment':r.pop('source_assessment')
        return json.dumps(r)
    with pytest.raises(NoveltyGateError):run(synthetic_candidate(),call)


def test_refinement_is_seen_by_all_reviews_and_counts_in_budget():
    c=synthetic_candidate();original=deepcopy(c)
    seen=[]
    def call(system,user,label):
        seen.append(label)
        r=json.loads(reviewer({'H1':'substantive_extension'})(system,user,label))
        if 'source_refinement' in label:r['hypothesis']='Synthetic refined, conditional hypothesis.'
        else:assert json.loads(user)['hypothesis']=='Synthetic refined, conditional hypothesis.'
        return json.dumps(r)
    empty=run(c,lambda *x:pytest.fail('8 calls required'),peer_response=True,refine_before_review=True,max_model_calls=7)
    assert empty['model_calls']==0
    got=run(c,call,peer_response=True,refine_before_review=True,max_model_calls=8)
    assert got['model_calls']==len(seen)==8 and got['selected_ids']==['H1']
    row=got['candidates'][0]
    assert row['chain']==c['chain'] and row['draft_before_refinement']['hypothesis']==c['hypothesis']
    assert c==original


def test_refiner_cannot_edit_graph_and_candidate_cannot_forge_sources():
    c=synthetic_candidate();c['source_packets']={'PMID:1':{'text':'Caller injection'}}
    def call(system,user,label):
        assert 'Caller injection' not in user
        r=json.loads(reviewer({'H1':'substantive_extension'})(system,user,label))
        if 'source_refinement' in label:r['chain']={'node_ids':['fake']}
        return json.dumps(r)
    with pytest.raises(NoveltyGateError,match='cannot change chain'):
        run(c,call,refine_before_review=True)


def test_refinement_format_repair_preserves_its_own_contract():
    attempts=[]
    def call(system,user,label):
        r=json.loads(reviewer({'H1':'substantive_extension'})(system,user,label))
        if 'source_refinement' in label:
            attempts.append(system)
            if len(attempts)==1:
                r['source_assessment']['anchors'][0]['quote']='Not a literal source quote.'
            else:
                assert 'Source anchor is not literal' in system
                assert 'including reference_ids and scientific_delta' not in system
                assert 'Not a literal source quote.' in json.loads(user)['previous_invalid_response_excerpt']
                assert 'Not a literal source quote.' not in system
        return json.dumps(r)
    got=run(synthetic_candidate(),call,peer_response=True,refine_before_review=True,max_model_calls=9)
    assert len(attempts)==2 and got['model_calls']==9 and got['selected_ids']==['H1']


def test_missing_abstracts_can_complete_as_hold_with_empty_anchors():
    def call(system,user,label):
        r=json.loads(reviewer({'H1':'uncertain'})(system,user,label))
        r['source_assessment']=dict(bridge='unknown',support='unknown',scope='unknown',
            relation='unknown',decision='hold',reason='No source passage supplied.',anchors=[])
        return json.dumps(r)
    got=run(synthetic_candidate(),call,source_resolver=lambda ids:{s:dict(status='unavailable') for s in ids},
        peer_response=True,refine_before_review=True,max_model_calls=8)
    assert got['reviewed_candidates']==1 and got['model_calls']==8
    assert got['candidates'][0]['chain_gate_verdict']=='revise' and not got['selected_ids']


def test_real_resolver_alias_can_be_cited_without_changing_chain_identity():
    c=synthetic_candidate()
    def source_resolver(ids):
        return {s:dict(status='available',text='Synthetic bound source for '+s,
            resolved_source_aliases=['DOI:10.123/'+s.split(':')[-1]]) for s in ids}
    def call(system,user,label):
        r=json.loads(reviewer({'H1':'substantive_extension'})(system,user,label))
        packet=json.loads(user)
        for anchor in r['source_assessment']['anchors']:
            anchor['source_id']=packet['source_packets'][anchor['source_id']]['resolved_source_aliases'][0]
        if 'reference_ids' in r:r['reference_ids']=['DOI:10.123/1','DOI:10.123/2']
        return json.dumps(r)
    got=run(c,call,source_resolver=source_resolver,peer_response=True,refine_before_review=True)
    assert got['selected_ids']==['H1'] and got['candidates'][0]['source_ids']==c['source_ids']


def test_scope_difference_reaches_review_as_proposed_transfer_not_batch_abort():
    def resolver(queries):
        b=synthetic_batch(queries)
        b['results'][1]['papers'][0]['observations'][0]['original_claim']['population']='children'
        return b
    def call(system,user,label):
        assert json.loads(user)['chain_context']['scope_differences']
        r=json.loads(reviewer({'H1':'substantive_extension'})(system,user,label))
        r['source_assessment']['scope']='transfer_required'
        return json.dumps(r)
    got=gate_candidates([synthetic_candidate()],'balanced',call,require_chain=True,resolver=resolver)
    assert got['selected_ids']==['H1'] and got['candidates'][0]['chain_gate_verdict']=='pass'


def test_adaptive_missing_sources_spends_no_calls_and_is_not_rejection():
    def resolver(queries):
        batch=synthetic_batch(queries)
        for doc in batch['results']:
            for paper in doc['papers']:
                for obs in paper['observations']:obs['source_review']=None
        return batch
    got=gate_candidates([synthetic_candidate()],'balanced',lambda *a:pytest.fail('No source, no model call'),
        require_chain=True,resolver=resolver,source_resolver=lambda ids:{},adaptive_review=True,max_model_calls=0)
    assert got['reviewed_candidates']==got['model_calls']==0
    assert got['screened_candidates'][0]['source_decision']=='hold'
    assert not got['selected_ids'] and not got['incomplete_candidates']


def test_adaptive_agreement_needs_five_calls_not_eight_and_hides_refiner_vote():
    def call(system,user,label):
        if 'expert:' in label:assert 'source_refinement' not in json.loads(user)
        assert 'response:' not in label
        return reviewer({'H1':'substantive_extension'})(system,user,label)
    got=run(synthetic_candidate(),call,adaptive_review=True,peer_response=True,
        refine_before_review=True,max_model_calls=5)
    assert got['model_calls']==5 and got['selected_ids']==['H1']
    assert not got['candidates'][0].get('peer_responses')


def test_adaptive_median_does_not_turn_a_source_dispute_into_a_pass():
    def call(system,user,label):
        r=json.loads(reviewer({'H1':'substantive_extension'},score=.8)(system,user,label))
        if 'scientific_delta' in label:
            r['critic_score']=.1
            r['source_assessment'].update(decision='reject',bridge='incompatible')
        return json.dumps(r)
    got=run(synthetic_candidate(),call,adaptive_review=True,peer_response=True,
        refine_before_review=True,max_model_calls=8)
    row=got['candidates'][0]
    assert got['model_calls']==8 and len(row['peer_responses'])==3
    assert row['science']['critic_score']==.8 and row['root_review_required']
    assert row['chain_gate_verdict']=='revise' and not got['selected_ids']


@pytest.mark.parametrize('grounded',[True,False])
def test_unanimous_rejection_requires_actual_bad_premise_not_uncertainty(grounded):
    def call(system,user,label):
        r=json.loads(reviewer({'H1':'uncertain'})(system,user,label))
        r['source_assessment'].update(decision='reject',bridge='incompatible' if grounded else 'unknown')
        return json.dumps(r)
    got=run(synthetic_candidate(),call,adaptive_review=True,peer_response=True)
    row=got['candidates'][0]
    assert row['chain_gate_verdict']==('fail' if grounded else 'revise')
    assert not got['selected_ids']


def test_adaptive_disagreement_budget_preserves_partial_reviews():
    def call(system,user,label):
        r=json.loads(reviewer({'H1':'substantive_extension'})(system,user,label))
        if 'scientific_delta' in label:r['source_assessment'].update(decision='hold',bridge='unknown')
        return json.dumps(r)
    got=run(synthetic_candidate(),call,adaptive_review=True,peer_response=True,
        refine_before_review=True,max_model_calls=5)
    assert got['model_calls']==5 and not got['candidates']
    assert len(got['incomplete_candidates'][0]['initial_reviews'])==3
    assert got['incomplete_candidates'][0]['source_refinement']


def test_unanimous_source_reject_skips_irrelevant_novelty_debate():
    def call(system,user,label):
        assert 'response:' not in label
        r=json.loads(reviewer({'H1':'uncertain'})(system,user,label))
        if 'prior_art' in label:r['classification']='same_scientific_conclusion'
        r['source_assessment'].update(decision='reject',bridge='incompatible')
        return json.dumps(r)
    got=run(synthetic_candidate(),call,adaptive_review=True,peer_response=True,
        refine_before_review=True,max_model_calls=5)
    assert got['model_calls']==5 and got['candidates'][0]['chain_gate_verdict']=='fail'


@pytest.mark.parametrize('source_decision,adjudication_verdict,expected_science', [
    ('retain','revise','revise'), ('reject','pass','fail'),
])
def test_unresolved_adjudication_reaches_root_without_changing_eligibility(
        source_decision,adjudication_verdict,expected_science):
    def call(system,user,label):
        r=json.loads(reviewer({'H1':'substantive_extension'},score=1)(system,user,label))
        if source_decision=='reject':
            r['source_assessment'].update(decision='reject',bridge='incompatible')
        if 'adjudication' in label:r['verdict']=adjudication_verdict
        return json.dumps(r)
    got=run(synthetic_candidate(),call,adaptive_review=True,peer_response=True,
        refine_before_review=True,max_model_calls=5)
    row=got['candidates'][0]
    assert row['science']['verdict']==expected_science
    assert row['root_review_required'] and got['root_review_required_ids']==['H1']
    assert not got['selected_ids'] and row['science']['critic_score']==1


def test_related_evidence_excludes_chain_sources_and_requires_verbatim_term(monkeypatch):
    import core.topic_evidence as topic_evidence

    def fake_topic(layer, topic, **kwargs):
        return {'terms': ['mci', 'hippocampus'], 'studies': [
            dict(work_key='PMID:CHAIN', title='chain paper', matched_terms=['mci', 'hippocampus'],
                 bibliography={'pmid': '999'}, observations=[dict(source_anchor='chain text')]),
            dict(work_key='PMID:REL', title='related paper', matched_terms=['mci', 'hippocampus'],
                 bibliography={'pmid': '111'}, observations=[dict(source_anchor='related abstract text')]),
            dict(work_key='PMID:PARTIAL', title='partial', matched_terms=['mci'],
                 bibliography={'pmid': '222'}, observations=[dict(source_anchor='partial text')]),
            dict(work_key='PMID:NOANCHOR', title='context only', matched_terms=['mci', 'hippocampus'],
                 bibliography={'pmid': '333'}, observations=[dict(source_anchor=None)]),
        ]}

    monkeypatch.setattr(topic_evidence, 'topic_evidence', fake_topic)
    payload = {'source_ids': ['PMID:999'], 'source_packets': {'PMID:999': {'status': 'available',
               'resolved_source_aliases': ['DOI:10.1/x']}}}
    got = build_related_evidence('mci hippocampus', payload, layer=object())
    assert [s['work_key'] for s in got['studies']] == ['PMID:REL']
    assert got['status'] == 'provided' and got['studies'][0]['source_reviewed'] is True


def test_related_evidence_empty_is_not_novelty_evidence(monkeypatch):
    import core.topic_evidence as topic_evidence
    monkeypatch.setattr(topic_evidence, 'topic_evidence',
                        lambda layer, topic, **kwargs: {'terms': ['mci'], 'studies': []})
    got = build_related_evidence('mci', {'source_ids': [], 'source_packets': {}}, layer=object())
    assert got['status'] == 'none_found' and got['studies'] == []


def test_gate_threads_related_evidence_into_review_payload():
    seen = {}
    def related(payload):
        seen['source_ids'] = payload['source_ids']
        return {'status': 'provided', 'studies': [{'work_key': 'PMID:OTHER', 'text': 'other abstract'}]}
    def call(system, user, label):
        record = json.loads(reviewer({'H1': 'substantive_extension'})(system, user, label))
        if label.endswith('expert:prior_art'):
            assert 'related_evidence' in json.loads(user)
        return json.dumps(record)
    got = run(synthetic_candidate(), call, related_resolver=related,
              adaptive_review=True, peer_response=True, refine_before_review=True, max_model_calls=5)
    assert seen['source_ids'] == got['candidates'][0]['source_ids']
    assert got['candidates'][0]['related_evidence']['studies'][0]['work_key'] == 'PMID:OTHER'


def test_prompts_pin_the_three_exposed_error_classes():
    from core.idea_source_review import ASSESSMENT_INSTRUCTION, REFINEMENT_SYSTEM
    from core.novelty_gate import ADJUDICATION_SYSTEM, REVIEW_SYSTEM
    # 1. An unproven increment is the proposal, not a failed premise.
    for text in (ASSESSMENT_INSTRUCTION, REVIEW_SYSTEM, ADJUDICATION_SYSTEM):
        assert 'increment' in text and ('not a' in text)
    assert 'untested increment' in ASSESSMENT_INSTRUCTION
    # 2. Condition fidelity: never transfer one element's restriction to another.
    for text in (ASSESSMENT_INSTRUCTION, REVIEW_SYSTEM, ADJUDICATION_SYSTEM, REFINEMENT_SYSTEM):
        assert 'allele' in text
    assert 'never transfer' in REVIEW_SYSTEM or 'never transfer' in ASSESSMENT_INSTRUCTION
    # 3. Hypothesis/prediction alignment on the comparison baseline.
    assert 'comparison baseline' in REVIEW_SYSTEM and 'comparison baseline' in ADJUDICATION_SYSTEM
    assert 'match the hypothesis' in REFINEMENT_SYSTEM
    # Related-evidence block is described to the reviewers.
    assert 'related_studies' in REVIEW_SYSTEM
