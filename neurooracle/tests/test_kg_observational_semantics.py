from copy import deepcopy
from xml.sax.saxutils import escape
import pytest
from neurooracle.src import kg_observational_semantics as repair
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.tests.test_kg_literal_endpoint_repair import edges


def public_fixture():
    articles=[]
    for spec in repair.SPECS.values():
        doi=f'<ArticleId IdType="doi">{spec["doi"]}</ArticleId>' if spec['doi'] else ''
        text=escape(' '.join(spec['fragments']))
        articles.append(f'<PubmedArticle><MedlineCitation><PMID>{spec["pmid"]}</PMID><Article><ArticleTitle>Own article {spec["pmid"]}</ArticleTitle><Abstract><AbstractText>{text}</AbstractText></Abstract></Article></MedlineCitation><PubmedData><ArticleIdList>{doi}</ArticleIdList></PubmedData></PubmedArticle>')
    return '<PubmedArticleSet>'+''.join(articles)+'</PubmedArticleSet>'


def fixture(cid=repair.REBOX):
    spec=repair.SPECS[cid]
    md=dict(id=cid,subject_id='CUI:subject',subject_name='subject',object_id='CUI:object',object_name='outcome',predicate=spec['predicate'],negated=False,
        raw_text='Original quotation with design context.',claim='Original extraction sentence.',
        evidence=dict(direction='',study_type='',methodology='',p_value=None,sample_size=None,effect_size=None,legacy_value='Original quotation.'),
        source_paper=dict(pmid=spec['pmid'],doi=spec['doi']),metadata=dict(original_predicate=spec['original_predicate']),
        scope_reaudit=dict(decision='finalized',claim_evidence_sha256='historical',decision_basis='Historical review, not renewed.'),conditions={})
    row=dict(id=cid,preferred_name=f"subject {spec['predicate']} outcome",metadata=md)
    return row,{cid:dict(claim_id=cid,claim_sha256=digest(row))},repair.source_proof(public_fixture())


@pytest.mark.parametrize('cid',sorted(repair.SPECS))
def test_three_field_contexts_are_finite_and_invertible(cid):
    row,reviews,proof=fixture(cid);before=deepcopy(row);event,current=repair.reviewed_claim(row,[],proof,reviews)
    assert repair.reverse_claim(current,event)==before and repair.change_claim(row,event)==current and row==before
    for key in ('source_paper','scope_reaudit','raw_text','claim','conditions','metadata','subject_id','object_id'):
        assert current['metadata'][key]==before['metadata'][key]
    assert current['metadata']['predicate']=='is_associated_with'
    for key in ('p_value','sample_size','effect_size','legacy_value'):assert current['metadata']['evidence'][key]==before['metadata']['evidence'][key]


def test_design_limits_are_not_inferred_as_causal_or_between_arm_effect():
    assert 'do not establish a between-arm' in repair.SPECS[repair.REBOX]['methodology']
    assert repair.SPECS[repair.CONTINUATION]['study_type']=='nonrandomized unblinded continuation phase'
    assert repair.SPECS[repair.CONTINUATION]['direction']=='negative'
    assert 'abstract does not report' in repair.SPECS[repair.INPATIENT]['methodology']


@pytest.mark.parametrize('kind',['raw','audit','stats','source','endpoint','negated','legacy','new_node','extra_field','existing_design'])
def test_no_unapproved_changes_or_overwrite_of_existing_design(kind):
    row,reviews,proof=fixture();event,current=repair.reviewed_claim(row,[],proof,reviews)
    if kind=='new_node':action=lambda:repair.literal_node('new node')
    elif kind=='extra_field':
        event['field_changes'].append(dict(path=['metadata','raw_text'],old=row['metadata']['raw_text'],new='new'))
        action=lambda:repair.change_claim(row,event)
    elif kind=='existing_design':
        row['metadata']['evidence']['study_type']='existing context';reviews[row['id']]['claim_sha256']=digest(row)
        action=lambda:repair.reviewed_claim(row,[],proof,reviews)
    else:
        if kind=='raw':current['metadata']['raw_text']='new'
        elif kind=='audit':current['metadata']['scope_reaudit']['decision']='renewed'
        elif kind=='stats':current['metadata']['evidence']['sample_size']=74
        elif kind=='source':current['metadata']['source_paper']['pmid']='different'
        elif kind=='endpoint':current['metadata']['subject_id']='new'
        elif kind=='negated':current['metadata']['negated']=True
        else:current['metadata']['claim']='new extraction'
        action=lambda:repair.reverse_claim(current,event)
    with pytest.raises(ValueError):action()


@pytest.mark.parametrize('kind',['pmid','doi','scope','comparison'])
def test_own_public_abstract_and_qualification_are_required(kind):
    xml=public_fixture()
    if kind=='pmid':xml=xml.replace('15131518','15131519')
    elif kind=='doi':xml=xml.replace(repair.SPECS[repair.REBOX]['doi'],'10.1/wrong')
    elif kind=='scope':xml=xml.replace('not randomized or blinded','randomized and blinded')
    else:xml=xml.replace('compared with baseline','compared with placebo')
    with pytest.raises(ValueError):repair.source_proof(xml)


@pytest.mark.parametrize('kind',['valid','negation','history','extra','missing','duplicate'])
def test_scientific_edge_history_kept_and_closed(kind):
    row,reviews,proof=fixture();_,current=repair.reviewed_claim(row,[],proof,reviews);owned=edges(row)
    owned[-1][1]['metadata'].update(negated=False,original_predicate='improves')
    if kind=='negation':owned[-1][1]['metadata']['negated']=True
    elif kind=='history':owned[-1][1]['metadata']['original_predicate']='causes'
    elif kind=='extra':owned[-1][1]['metadata']['additional']='value'
    elif kind=='missing':owned.pop()
    elif kind=='duplicate':owned.append((4,deepcopy(owned[-1][1])))
    if kind=='valid':
        evs=repair.reviewed_edges(row['id'],row,current,owned);assert len(evs)==1
        e=evs[0];old=dict(owned)[e['ordinal']];new=repair.apply_edge(old,e)
        assert old['metadata']==new['metadata'] and repair.apply_edge(new,e,reverse=True)==old
    else:
        with pytest.raises(ValueError):repair.reviewed_edges(row['id'],row,current,owned)


def issue_fixture():
    events=[];issues=[]
    for cid in repair.SPECS:
        row,reviews,proof=fixture(cid);e,_=repair.reviewed_claim(row,[],proof,reviews);events.append(e)
        issues.append(dict(claim_id=cid,current_node_sha256=e['claim_sha256']))
    issues.extend(dict(claim_id='CLM:held'+str(i),current_node_sha256=str(i)) for i in range(14))
    return issues,dict(events=events)


def test_exact_three_issue_records_closed_not_fourteen_unknowns():
    issues,p=issue_fixture();before=deepcopy(issues);out=repair.remaining_issues(issues,p)
    assert out==issues[3:] and len(out)==14 and issues==before


@pytest.mark.parametrize('kind',['missing','extra','duplicate','hash','event'])
def test_remaining_issues_require_exact_current_source(kind):
    issues,p=issue_fixture()
    if kind=='missing':issues.pop()
    elif kind=='extra':issues.append(dict(claim_id='extra',current_node_sha256='x'))
    elif kind=='duplicate':issues[-1]=issues[0]
    elif kind=='hash':issues[0]['current_node_sha256']='wrong'
    else:p['events'].pop()
    with pytest.raises(ValueError):repair.remaining_issues(issues,p)
