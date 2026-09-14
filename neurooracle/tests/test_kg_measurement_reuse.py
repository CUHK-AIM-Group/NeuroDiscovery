from copy import deepcopy
import pytest

from neurooracle.src import kg_measurement_reuse as repair
from neurooracle.src.kg_literal_endpoint_repair import literal_node


QUOTE = ('In Human Connectome Project Young Adult participants, PE severity on the Achenbach thought problems scale '
         'was predicted by increased language network (LAN) and dorsal attention network (DAN) areas and decreased '
         'cingulo-opercular network area (r < 0.12).')


def sources():
    pub = ('<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>39829963</PMID><Article><Abstract>'
           '<AbstractText>' + QUOTE.replace('<', '&lt;') + '</AbstractText></Abstract></Article></MedlineCitation>'
           '<PubmedData><ArticleIdList><ArticleId IdType="doi">10.1016/j.bpsgos.2024.100386</ArticleId>'
           '</ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>')
    pmc = '''<article><front><article-meta><article-id pub-id-type="pmid">39829963</article-id>
      <article-id pub-id-type="doi">10.1016/j.bpsgos.2024.100386</article-id>
      <article-id pub-id-type="pmcid">PMC11740805</article-id></article-meta></front><body>
      <sec><title>Imaging Procedure</title><p>Connectom 3T with resting-state BOLD.</p></sec>
      <sec><title>Functional Network Template Matching</title><p>frontoparietal, dorsal attention, language, cingulo-opercular</p></sec>
      <sec><title>Surface Area and Topography of Functional Networks in HCP-YA</title>
      <p>relative cortical surface against total cortical area: FPN DAN LAN COP</p></sec>
      <sec><title>Psychotic-Like Experiences and Functional Network Size and Topography in HCP-YA</title>
      <p>stepwise regression</p></sec>
      <table-wrap id="tbl2"><table><thead><tr><th>Network</th><th>ASR Depression/Anxiety</th><th>ASR Thought Problems</th></tr></thead>
      <tbody><tr><td>Cingulo-Opercular</td></tr><tr><td>r</td><td>-.041</td><td>−0.088</td></tr>
      <tr><td>p</td><td>.21</td><td>.007<xref ref-type="table-fn">b</xref></td></tr></tbody></table></table-wrap>
      </body></article>'''
    return pub.encode(), pmc.encode()


def proof():
    return repair.public_network_evidence(*sources())


def claim(cid=repair.COP):
    return dict(id=cid, metadata=dict(id=cid, subject_id='CUI:C1414531',
        subject_name=repair.NETWORKS[cid], object_id='CUI:outcome', object_name='psychotic experience severity',
        object_type='OUTCOME', predicate='correlates_with' if cid == repair.COP else 'predicts', negated=False,
        raw_text=QUOTE, source_paper=dict(pmid='39829963', doi='10.1016/j.bpsgos.2024.100386', year=2024),
        conditions=dict(cohort='HCP-YA', age='young adults'), evidence=dict(effect_size=0.12,
            effect_metric='r', p_value=None, sample_size=1003,
            methodology='resting-state fMRI; individual-specific template-matching; logistic regression; Achenbach Self-Report Scale'),
        metadata=dict(subject_id='CUI:C1414531', subject_type='network', raw_stats=dict(effect_size_raw='0.12', sample_size_raw='1003'))))


def changes(row):
    reason, basis = repair.gate(row, 'subject', proof())
    assert reason is None
    node = literal_node(row['metadata']['subject_name'])
    return [dict(side='subject', old_id=row['metadata']['subject_id'], target_id=node['id'],
                 name=node['preferred_name'], basis=basis)]


def test_public_table_uses_thought_problem_column_and_excludes_footnote():
    p = proof()
    assert p['cop_pearson_r'] == -0.088 and p['cop_nominal_p'] == 0.007
    assert p['abstract'] == QUOTE and len(p['section_sha256']) == 4
    assert p['pmcid'] == 'PMC11740805'


@pytest.mark.parametrize('which,old,new', [
    (0,'<PMID>39829963','<PMID>11111111'),
    (0,'10.1016/j.bpsgos.2024.100386','10.1/wrong'),
    (1,'pub-id-type="pmid">39829963','pub-id-type="pmid">11111111'),
    (1,'PMC11740805','PMC11111111'),
    (1,'10.1016/j.bpsgos.2024.100386','10.1/wrong'),
    (1,'Imaging Procedure','Different Procedure'),
    (1,'Connectom 3T','different scanner'),
    (1,'resting-state BOLD','task scan'),
    (1,'dorsal attention','different network'),
    (1,'total cortical area','regional volume'),
    (1,'stepwise regression','logistic regression'),
    (1,'id="tbl2"','id="tbl3"'),
    (1,'ASR Thought Problems','Different Outcome'),
    (1,'−0.088','0.088'),
    (1,'.007<xref','.07<xref'),
])
def test_source_scope_and_table_changes_rejected(which, old, new):
    data = list(sources()); data[which] = data[which].decode().replace(old, new).encode()
    with pytest.raises(ValueError): repair.public_network_evidence(*data)


@pytest.mark.parametrize('typ', [None, '', 'network'])
def test_network_exception_accepts_nested_source_type(typ):
    row = claim()
    if typ is not None: row['metadata']['subject_type'] = typ
    assert repair.gate(row, 'subject', proof()) == (None, 'owning_imaging_methods_and_complete_network_measurement')


@pytest.mark.parametrize('problem', ['case', 'name', 'source', 'doi', 'quote', 'method', 'inner_type', 'outer_type', 'inner_id', 'other_claim', 'other_side'])
def test_network_scope_never_inferred_from_another_claim(problem):
    row = claim(); md = row['metadata']; side = 'subject'
    if problem == 'case': md['subject_name'] = md['subject_name'].title()
    if problem == 'name': md['subject_name'] = 'cingulo-opercular network'
    if problem == 'source': md['source_paper']['pmid'] = '11111111'
    if problem == 'doi': md['source_paper']['doi'] = '10.1/wrong'
    if problem == 'quote': md['raw_text'] = 'Same network.'
    if problem == 'method': md['evidence']['methodology'] = 'RNA sequencing'
    if problem == 'inner_type': md['metadata']['subject_type'] = 'BIOMARKER'
    if problem == 'outer_type': md['subject_type'] = 'GENE_TARGET'
    if problem == 'inner_id': md['metadata']['subject_id'] = 'wrong'
    if problem == 'other_claim': row['id'] = 'CLM:other'
    if problem == 'other_side': side = 'object'
    assert repair.gate(row, side, proof())[0] is not None


@pytest.mark.parametrize('name', ['bilateral insular gray-matter loss rate', 'bilateral medial prefrontal cortical thinning',
    'dorsolateral prefrontal cortex and ventral diencephalon volumes', 'bilateral temporal lobe hypoperfusion',
    'left inferior temporal gray-matter concentration'])
def test_explicit_measurement_forms_do_not_shorten_names(name):
    row = claim(); row['id'] = 'CLM:explicit'; md = row['metadata']
    md['subject_name'] = name; md['metadata']['subject_type'] = 'IMAGING_MARKER'
    assert repair.gate(row, 'subject', proof())[0] is None
    event, out = repair.reviewed_claim(row, changes(row), proof())
    assert out['metadata']['subject_name'] == name and event['science_changes'] == []


@pytest.mark.parametrize('name,typ', [('brain volumes', 'BIOMARKER'), ('COMT genotype cortical thinning', 'IMAGING_MARKER'),
    ('plasma protein gray-matter concentration', 'IMAGING_MARKER'), ('brain changes', 'IMAGING_MARKER')])
def test_new_measurement_forms_keep_type_and_molecular_guards(name, typ):
    row = claim(); row['id'] = 'CLM:other'; row['metadata']['subject_name'] = name
    row['metadata']['metadata']['subject_type'] = typ
    assert repair.gate(row, 'subject', proof())[0] is not None


@pytest.mark.parametrize('cid', sorted(repair.BOUNDS))
def test_source_proven_numeric_repair_is_invertible_and_preserves_context(cid):
    row = claim(cid); before = deepcopy(row)
    event, out = repair.reviewed_claim(row, changes(row), proof())
    assert row == before and repair.reverse_claim(out, event) == before
    for field in ('raw_text', 'source_paper', 'conditions', 'negated', 'predicate', 'subject_name', 'object_name'):
        assert out['metadata'][field] == before['metadata'][field]
    ev = out['metadata']['evidence']
    if cid == repair.COP:
        assert ev['effect_size'] == -0.088 and ev['p_value'] == .007
        assert 'nominal p value' in ev['methodology']
    else:
        assert ev['effect_size'] is None and ev['p_value'] is None
        assert out['metadata']['metadata']['raw_stats']['effect_size_raw'] == 'r < 0.12'
        assert 'stepwise regression' in ev['methodology'] and 'Pearson' not in ev['methodology']
    assert ev['sample_size'] == 1003 and ev['effect_metric'] == 'r'


@pytest.mark.parametrize('field,value', [('object_name', 'different outcome'), ('raw_text', 'r = .12'),
    ('predicate', 'causes')])
def test_numeric_source_shape_must_match(field, value):
    row = claim(); row['metadata'][field] = value
    with pytest.raises(ValueError): repair.science_changes(row, proof())


@pytest.mark.parametrize('field,value', [('effect_size', .13), ('effect_metric', 'OR'), ('p_value', .05),
    ('methodology', 'resting-state fMRI only')])
def test_numeric_evidence_shape_must_match(field, value):
    row = claim(); row['metadata']['evidence'][field] = value
    with pytest.raises(ValueError): repair.science_changes(row, proof())


def test_unapproved_scientific_path_and_stale_output_are_rejected():
    row = claim(); event, out = repair.reviewed_claim(row, changes(row), proof())
    bad = deepcopy(event); bad['science_changes'][0]['path'] = ['negated']
    with pytest.raises(ValueError): repair.change_claim(row, bad)
    bad = deepcopy(out); bad['metadata']['conditions']['age'] = 'children'
    with pytest.raises(ValueError): repair.reverse_claim(bad, event)

