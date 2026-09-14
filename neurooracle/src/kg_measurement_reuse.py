"""Complete measurement identities and narrowly source-proven numeric repairs."""
from copy import deepcopy
import hashlib
import re
from xml.etree import ElementTree as ET

from .kg_literal_endpoint_repair import endpoint_gate as prior_gate, GENES
from .kg_bulk_identity import change_claim as identity_change
from .kg_identity_pilot import digest,nonidentity_claim
from .relation_evidence import name_key

NETWORKS={
    'CLM:84b8e4b85bfb':'language network cortical surface area',
    'CLM:650051f0b777':'dorsal attention network cortical surface area',
    'CLM:951f2bb246a3':'cingulo-opercular network cortical surface area',
    'CLM:1ad2235bf261':'dorsal attention network cortical surface area',
    'CLM:3c2917bbe241':'language network cortical surface area',
    'CLM:1b4579c82247':'frontoparietal network cortical surface area',
}
BOUNDS={'CLM:84b8e4b85bfb','CLM:650051f0b777','CLM:951f2bb246a3'}
COP='CLM:951f2bb246a3'
FORMS=re.compile(r'\b(?:cortical thinning|volumes|(?:gray|grey)[- ]matter (?:loss|reduction|deficits|density|concentration)|hypoperfusion)\b',re.I)
ALLOWED_PATHS={('evidence','effect_size'),('evidence','p_value'),('evidence','methodology'),('metadata','raw_stats','effect_size_raw')}


def text(element): return ' '.join(''.join(element.itertext()).split())


def public_network_evidence(pubmed_bytes,pmc_bytes):
    pub=ET.fromstring(pubmed_bytes); entries=pub.findall('./PubmedArticle')
    if len(entries)!=1 or entries[0].findtext('./MedlineCitation/PMID')!='39829963': raise ValueError('wrong PubMed owner')
    pub=entries[0]
    if pub.findtext("./PubmedData/ArticleIdList/ArticleId[@IdType='doi']")!='10.1016/j.bpsgos.2024.100386': raise ValueError('wrong PubMed DOI')
    root=ET.fromstring(pmc_bytes); entries=[root] if root.tag=='article' else root.findall('./article')
    if len(entries)!=1: raise ValueError('multiple PMC owners')
    article=entries[0]
    ids={e.get('pub-id-type'):e.text for e in article.findall('./front/article-meta/article-id')}
    if ids.get('pmid')!='39829963' or ids.get('doi')!='10.1016/j.bpsgos.2024.100386' or ids.get('pmcid')!='PMC11740805': raise ValueError('PMC/PubMed identities differ')
    sections={text(s.find('title')):s for s in article.findall('./body//sec') if s.find('title') is not None}
    wanted=('Imaging Procedure','Functional Network Template Matching','Surface Area and Topography of Functional Networks in HCP-YA',
        'Psychotic-Like Experiences and Functional Network Size and Topography in HCP-YA')
    if not set(wanted)<=set(sections): raise ValueError('missing owning methods/measurement definition')
    imaging=text(sections[wanted[0]]); matching=text(sections[wanted[1]]); definition=text(sections[wanted[2]])
    if not all(s in imaging for s in ('Connectom 3T','resting-state BOLD')): raise ValueError('imaging source does not reproduce')
    if not all(s in matching.lower() for s in ('frontoparietal','dorsal attention','language','cingulo-opercular')): raise ValueError('network definition incomplete')
    if not all(s in definition for s in ('relative cortical surface','total cortical area','FPN','DAN','LAN','COP')): raise ValueError('surface-area scope differs')
    if 'stepwise regression' not in text(sections[wanted[3]]): raise ValueError('regression analysis not confirmed')
    tables=article.findall("./body//table-wrap[@id='tbl2']")
    if len(tables)!=1: raise ValueError('missing unique owning Table 2')
    table=tables[0]
    headers=[text(e) for e in table.findall('./table/thead/tr/th')]
    if headers!=['Network','ASR Depression/Anxiety','ASR Thought Problems']: raise ValueError('wrong symptom/column')
    group=None; estimates={}
    for row in table.findall('./table/tbody/tr'):
        cells=row.findall('td')
        if len(cells)==1: group=text(cells[0]);continue
        if group=='Cingulo-Opercular':
            metric=text(cells[0]); value=cells[2].text or ''
            if metric not in {'r','p'} or metric in estimates: raise ValueError('ambiguous COP estimate')
            estimates[metric]=float(value.strip().replace('−','-'))
    if estimates!={'r':-0.088,'p':0.007}: raise ValueError('reviewed COP values changed')
    abstracts=' '.join(text(e) for e in pub.findall('./MedlineCitation/Article/Abstract/AbstractText'))
    return dict(pmid='39829963',doi=ids['doi'],pmcid=ids['pmcid'],
        pubmed_sha256=hashlib.sha256(pubmed_bytes).hexdigest(),pmc_sha256=hashlib.sha256(pmc_bytes).hexdigest(),
        section_sha256={s:hashlib.sha256(ET.tostring(sections[s],encoding='utf-8')).hexdigest() for s in wanted},
        table_sha256=hashlib.sha256(ET.tostring(table,encoding='utf-8')).hexdigest(),
        cop_pearson_r=estimates['r'],cop_nominal_p=estimates['p'],abstract=abstracts,
        measurement_scope='individual-specific functional-network cortical surface area relative to total cortex',
        license='Mamah et al. 2024, CC BY 4.0; DOI 10.1016/j.bpsgos.2024.100386')


def gate(record,side,proof):
    md=record['metadata'];reason=prior_gate(md,side)
    if reason is None:return None,'prior_explicit_measurement'
    if reason=='not_unambiguous_measurement_surface' and FORMS.search(str(md.get(side+'_name') or '')):
        return None,'explicit_measurement_form_not_name_equivalence'
    if record['id'] not in NETWORKS or side!='subject':return reason,None
    if reason!='not_explicit_imaging_type':return reason,None
    if name_key(md.get('subject_name'))!=NETWORKS[record['id']]:return 'network_full_name_changed',None
    if md.get('subject_type') not in (None,'','network') or (md.get('metadata') or {}).get('subject_type')!='network':return 'network_type_shape_changed',None
    paper=md.get('source_paper') or {}
    if paper.get('pmid')!=proof['pmid'] or str(paper.get('doi')).lower()!=proof['doi']:return 'network_source_identity_changed',None
    quote=' '.join(str(md.get('raw_text') or '').split())
    if len(quote)<70 or quote not in proof['abstract']:return 'not_complete_owning_abstract_quote',None
    if 'resting-state fMRI' not in (md.get('evidence') or {}).get('methodology',''):return 'current_method_scope_differs',None
    return None,'owning_imaging_methods_and_complete_network_measurement'


def field(md,path):
    for key in path:md=md[key]
    return md


def set_field(md,path,value):
    for key in path[:-1]:md=md[key]
    md[path[-1]]=deepcopy(value)


def science_changes(record,proof):
    if record['id'] not in BOUNDS:return []
    md=record['metadata'];ev=md.get('evidence') or {};raw=(md.get('metadata') or {}).get('raw_stats') or {}
    if md.get('object_name')!='psychotic experience severity' or '(r < 0.12)' not in md.get('raw_text',''):
        raise ValueError('bound belongs to another endpoint/source')
    if ev.get('effect_size')!=0.12 or ev.get('effect_metric')!='r' or raw.get('effect_size_raw')!='0.12' or ev.get('p_value') is not None:
        raise ValueError('reviewed numeric source shape changed')
    changes=[]
    def add(path,value):changes.append(dict(path=list(path),old=field(md,path),new=value))
    if record['id']==COP:
        if md.get('predicate')!='correlates_with':raise ValueError('COP analysis/predicate differs')
        add(('evidence','effect_size'),proof['cop_pearson_r'])
        add(('evidence','p_value'),proof['cop_nominal_p'])
        add(('metadata','raw_stats','effect_size_raw'),'−0.088')
        method='Pearson correlation (Table 2; nominal p value)'
    else:
        if md.get('predicate')!='predicts':raise ValueError('regression analysis/predicate differs')
        # Do not substitute univariate Pearson values for predictive/stepwise
        # regression coefficients. Preserve the abstract bound, not fake r=.12.
        add(('evidence','effect_size'),None)
        add(('metadata','raw_stats','effect_size_raw'),'r < 0.12')
        method='stepwise regression (abstract reports only a shared bound)'
    previous=ev['methodology']
    if previous.count('logistic regression')!=1:raise ValueError('reviewed method shape changed')
    add(('evidence','methodology'),previous.replace('logistic regression',method))
    return changes


def change_claim(record,event):
    out=identity_change(record,event)
    for change in event.get('science_changes',[]):
        path=tuple(change['path'])
        if path not in ALLOWED_PATHS or field(out['metadata'],path)!=change['old']:raise ValueError('unapproved scientific field')
        set_field(out['metadata'],path,change['new'])
    if 'current_node_sha256' in event and digest(out)!=event['current_node_sha256']:raise ValueError('reviewed output hash differs')
    return out


def reviewed_claim(record,changes,proof):
    for change in changes:
        reason,basis=gate(record,change['side'],proof)
        if reason or basis!=change['basis']:raise ValueError(reason or 'source basis differs')
        if name_key(record['metadata'][change['side']+'_name'])!=change['name']:raise ValueError('name changed')
    event=dict(claim_id=record['id'],claim_sha256=digest(record),changes=changes,
        science_changes=science_changes(record,proof),nonidentity_sha256=digest(nonidentity_claim(record)))
    out=change_claim(record,event);event['current_node_sha256']=digest(out)
    return event,out


def reverse_claim(record,event):
    if digest(record)!=event['current_node_sha256']:raise ValueError('current record changed')
    original=deepcopy(record)
    for change in event.get('science_changes',[]):
        path=tuple(change['path'])
        if path not in ALLOWED_PATHS or field(original['metadata'],path)!=change['new']:raise ValueError('wrong scientific output')
        set_field(original['metadata'],path,change['old'])
    for change in event['changes']:
        field_name=change['side']+'_id'
        if original['metadata'][field_name]!=change['target_id']:raise ValueError('wrong current endpoint')
        original['metadata'][field_name]=change['old_id']
        inner=original['metadata'].get('metadata') or {}
        if field_name in inner:
            if inner[field_name]!=change['target_id']:raise ValueError('wrong inner endpoint')
            inner[field_name]=change['old_id']
    if digest(original)!=event['claim_sha256'] or change_claim(original,event)!=record:raise ValueError('inverse approved changes differ')
    return original
