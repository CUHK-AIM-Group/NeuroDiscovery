"""One finite R67 batch: nominal identity, five field-level science repairs.

This does not infer assay equivalence, independent studies, consensus, or a new
approval of historical audits. Raw excerpts and source records are immutable.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
from .kg_identity_pilot import digest
from .kg_scientific_definition_repair import at_path, rewrite, protected_digest
from .kg_historic_literal_repair import literal_node
from .kg_literal_endpoint_repair import edge_owner, reviewed_edges as source_edges
from .kg_gene_boundary_repair import word_interior_hits
from neurooracle.scripts.build_umls_simplification_candidate import compact
from neurooracle.scripts.inspect_kg_claim_deletion import exact_references
from neurooracle.scripts.inspect_kg_relation_semantic_batch import incidence_rows
from neurooracle.scripts.kg_accepted_candidate_lineage import require

VERSION='kg.finite_relation_semantic_batch.v1'
TARGETS={
 'dementia diagnosis':'CLM_CONCEPT:dementia_diagnosis_ca5ee37aa629',
 'hippocampal subfield volume':'CLM_CONCEPT:hippocampal_subfield_volume_f33e02afbddc',
 'hippocampal volume':'CLM_CONCEPT:hippocampal_volume',
 'left hippocampal volume':'CLM_CONCEPT:left_hippocampal_volume',
 'lifetime major depressive disorder':'CLM_CONCEPT:lifetime_major_depressive_disorder_1b4fff5c2c8d',
 'persistent cognitive impairment':'CLM_CONCEPT:persistent_cognitive_impairment',
 'reduced hippocampal volume':'CLM_CONCEPT:reduced_hippocampal_volume',
 'smaller hippocampal volume':'CLM_CONCEPT:smaller_hippocampal_volume_dfd6b5e256c9',
 'structural-functional connectivity coupling':'CLM_CONCEPT:structural_functional_connectivity_coupling_4066b49a6653',
 'treatment-resistant depression':'CLM_CONCEPT:treatment_resistant_depression_be4259f72455',
 'white matter fractional anisotropy':'CLM_CONCEPT:white_matter_fractional_anisotropy',
 'white matter microstructure':'CLM_CONCEPT:white_matter_microstructure_169d2d434859'}
SMA='CLM:61f3088497c0'
GBA='CLM:a0340c2ca3ca'
HIPPO='CLM:30090d82ead711541e448f9bbbe6245e'
EXERCISE='CLM:7e21b1d1073d47ec4bdc154d76487b86'
MDD='CLM:case3_topup_20260730_manual_b0005_054_01'
SCIENCE={SMA,GBA,HIPPO,EXERCISE,MDD}
GBA_NAME="Parkinson's disease dementia-like functional connectivity phenotype"
NEW_NODE=literal_node(GBA_NAME)
SCFC='CLM:140b874ce167ef7b'
SCFC_OLD='CLM_CONCEPT:clinical_risk_genetic_risk_first_episode_and_chronic_schizophrenia_stages'
SCFC_TARGET='CLM_CONCEPT:first_episode_and_chronic_schizophrenia_stages_9f60833d5757'
SCFC_NAME='first-episode and chronic schizophrenia stages'
STALE_HIPPO_EDGES={
 195878:('CLM:b40b93da26c117be','5bc553d89dd461c53ed1523e6498842393467544f2a4d2192f90396ca1a358ae','target_id'),
 229360:('CLM:5d81f0beebf648a9','34121c175dc0fb7a8cab0e6d0d036af175952c86165c4943f63c6813d6b7daab','target_id'),
 462638:('CLM:b40b93da26c117be','d552e12a4107f1b800017dee75d81ed3d42f2d0e3306c0ea9a3f61b64fb6dd2a','source_id'),
 462639:('CLM:5d81f0beebf648a9','9f6c7f62cb523d5476a713c873225b91d507250b8eac6afd3d65c570b7d01c42','source_id')}
RETIRED_EDGE_PAIRS={195878:195876,229360:229358,462638:90314,462639:90315,232555:232558,232560:232561}
SCIENCE_FIELDS={
 SMA:{('metadata','predicate'):'is_associated_with',
      ('metadata','evidence','sample_size'):883,
      ('metadata','evidence','methodology'):'Task-fMRI coordinate meta-analysis; ED<HC right SMA cluster (MNI 4,10,50), Bayesian threshold only; canonical ALE not significant.',
      ('metadata','evidence','replicability'):'Exploratory hypoactivation branch: 17 experiments, 883 subjects; not 35 independent replications of this cluster.',
      ('metadata','evidence','direction'):'Lower right SMA activation in ED at this cluster only; exploratory association, not causation or a region-wide direction.'},
 GBA:{('metadata','object_id'):NEW_NODE['id'],('metadata','object_name'):GBA_NAME,
      ('metadata','metadata','object_type'):'imaging_marker',('metadata','predicate'):'is_associated_with',
      ('metadata','evidence','sample_size'):53,
      ('metadata','evidence','methodology'):'Seed-to-voxel rs-fMRI: 12 GBA E365K/T408M carriers and 41 noncarriers with PD, dementia excluded. Caudate-occipital reductions resemble prior PDD findings; no PDD/DLB diagnostic comparison.',
      ('metadata','evidence','replicability'):'Single cohort; PDD-like interpretation compares prior literature, not independent replication. DLB-like metabolism belongs to PET, not this fMRI relation.',
      ('metadata','evidence','direction'):'Reduced caudate-occipital connectivity in carriers; qualitative PDD-like pattern, not actual dementia or causal risk.'},
 HIPPO:{('metadata','metadata','object_type'):'imaging_marker'},
 EXERCISE:{('metadata','negated'):True,('metadata','evidence','sample_size'):40,
      ('metadata','evidence','study_type'):'cross_sectional',
      ('metadata','evidence','methodology'):'Self-reported high versus low regular exercise, 20 participants per group; manually traced hippocampal volumes. Distinct from continuous cardiorespiratory fitness associations.',
      ('metadata','evidence','direction'):'No detected between-group hippocampal volume difference; not evidence of equivalence or absence of all fitness associations.'},
 MDD:{('metadata','subject_type'):'DISEASE',('metadata','predicate'):'is_associated_with',
      ('metadata','evidence','sample_size'):None,
      ('metadata','evidence','methodology'):'Observational preprint v1: GS BrainAge N=1067, DNAmAge N=684; UKB BrainAge N=12018. Multiple analyses, no single denominator for this compound endpoint.',
      ('metadata','evidence','direction'):'Higher age measures in lifetime MDD for selected clocks/cohort; raw mean differences exclude Horvath DNAmAge and UKB BrainAge. Association, not causal aging.'}}

def canonical_name(name):
    # Finite reviewed orthography only. No general case-folded entity matching.
    for canonical in TARGETS:
        if name in {canonical,canonical[0].upper()+canonical[1:]}:return canonical
    if name=='white-matter fractional anisotropy':return 'white matter fractional anisotropy'
    return None

def check_witness(w):
    require(not w['aliases'] and not w['semantic_types'] and not w['external_ids'],'external identity not approved')
    require(w['definition_sha256']==digest('') and w['spatial_mapping_sha256']==digest(None),'measurement definition not approved')
    require(set(w['metadata_keys'])<={'anchor_role','atom_type','atom_types','curation_scope','staging_source'},'unreviewed concept payload')
    require(w['source_vocab'] in {'replay_anchor_mint','manual_claim_anchor','manual_general_claim_anchor'},'unreviewed origin')

def changes_for(record,plan):
    cid=record['id'];md=record['metadata'];projection=plan['projections'][cid]
    require(digest(record)==projection['claim_sha256'],'fresh claim binding differs')
    changes=[];queued={(r['claim_id'],r['side']):r for r in plan['queued']}
    by_id={nid:g['name'] for g in plan['groups'] for nid in g['candidate_ids']}
    for side in ('subject','object'):
        name=md[side+'_name'];old=md[side+'_id'];canonical=canonical_name(name)
        if not canonical:continue
        if old in by_id:
            require(by_id[old]==canonical,'complete incident scope not in own group')
            check_witness(plan['all_witnesses'][old]);reason='same_nominal_concept'
        elif (cid,side) in queued:
            q=queued[cid,side];w=plan['all_witnesses'][old]
            require(q['current_node_id']==old and q['name']==name and 'T028' in w['semantic_types'],'unreviewed gene queue')
            require(word_interior_hits(name,[w['name'],*w['aliases']]),'gene word-interior error not reproduced')
            reason='gene_endpoint_to_existing_nominal_concept'
        else:continue
        target=TARGETS[canonical];check_witness(plan['all_witnesses'][target])
        if (old,name)!=(target,canonical):changes.append(dict(side=side,name=name,new_name=canonical,old_id=old,target_id=target,reason=reason))
    if cid==GBA:
        require(md['object_id']=='CUI:C1420009' and md['object_name']=="Parkinson's disease dementia and Lewy body dementia phenotypes in GBA variant carriers",'GBA original endpoint differs')
        changes.append(dict(side='object',name=md['object_name'],new_name=GBA_NAME,old_id=md['object_id'],target_id=NEW_NODE['id'],reason='own_source_fmri_scope_not_gene_or_actual_dementia'))
    if cid==SCFC:
        require(md['object_id']==SCFC_OLD and md['object_name']==SCFC_NAME,'SCFC exact literal scope differs')
        changes.append(dict(side='object',name=SCFC_NAME,new_name=SCFC_NAME,old_id=SCFC_OLD,target_id=SCFC_TARGET,reason='complete_mention_matches_existing_owned_edges'))
    return changes

def fields_for(record,changes):
    md=record['metadata'];inner=md.get('metadata') or {};updates={}
    for ch in changes:
        side=ch['side']
        for suffix,old,new in (('_id',ch['old_id'],ch['target_id']),('_name',ch['name'],ch['new_name'])):
            require(md[side+suffix]==old,'original complete endpoint differs')
            updates['metadata',side+suffix]=new
            if side+suffix in inner:
                require(inner[side+suffix]==old,'nested endpoint conflict');updates['metadata','metadata',side+suffix]=new
    updates.update(SCIENCE_FIELDS.get(record['id'],{}))
    for field in ('predicate','negated'):
        if ('metadata',field) in updates and field in inner:
            require(inner[field]==md[field],'nested science conflict');updates['metadata','metadata',field]=updates['metadata',field]
    fields=[(p,at_path(record,p),n) for p,n in updates.items() if at_path(record,p)!=n]
    if fields:
        current=rewrite(record,fields)['metadata']
        old=f"{md['subject_name']} {md['predicate']} {md['object_name']}"
        new=f"{current['subject_name']} {current['predicate']} {current['object_name']}"
        if old!=new:
            require(record['preferred_name']==old,'derived label is not a complete relation label');fields.append((('preferred_name',),old,new))
    return fields

def make_event(record,plan):
    changes=changes_for(record,plan);fields=fields_for(record,changes)
    if not fields:return None,record
    current=rewrite(record,fields);md=current['metadata']
    require(md['subject_id']!=md['object_id'],'new endpoint collapse')
    event=dict(claim_id=record['id'],claim_sha256=digest(record),current_node_sha256=digest(current),changes=changes,
        field_changes=[dict(path=list(p),old=o,new=n) for p,o,n in fields],protected_fields_sha256=protected_digest(record,fields),
        original_audit_sha256=digest(record['metadata'].get('scope_reaudit')),scientific_fields_reviewed=record['id'] in SCIENCE,
        old_audit_not_revalidated=True,raw_text_and_source_preserved=True)
    require(protected_digest(current,fields)==event['protected_fields_sha256'],'protected fields changed')
    return event,current

def reviewed_claim(record,event,plan):
    got,current=make_event(record,plan)
    require(got=={k:v for k,v in event.items() if k not in {'old_relation_id','new_relation_id'}},'finite reviewed event differs')
    return current

def reverse_claim(record,event):
    require(digest(record)==event['current_node_sha256'],'current claim differs')
    fields=[(tuple(r['path']),r['old'],r['new']) for r in event['field_changes']]
    original=rewrite(record,fields,reverse=True)
    require(digest(original)==event['claim_sha256'] and protected_digest(record,fields)==event['protected_fields_sha256'],'inverse/protected fields differ')
    return original

def reviewed_edge_bundle(cid,original,current,edge_rows):
    checked=deepcopy(original);normalized=[]
    if cid==SCFC:
        require(original['metadata']['object_id']==SCFC_OLD and current['metadata']['object_id']==SCFC_TARGET,'finite SCFC consistency scope differs')
        checked['metadata']['object_id']=SCFC_TARGET
    for ordinal,row in edge_rows:
        value=deepcopy(row)
        if ordinal in STALE_HIPPO_EDGES:
            owner,sha,field=STALE_HIPPO_EDGES[ordinal]
            require(cid==owner and digest(row)==sha and row[field]=='CLM_CONCEPT:hippocampal_volume_ea94a989a01c'
                and original['metadata']['subject_id']=='CLM_CONCEPT:7fc77cbc15bc49bf'
                and original['metadata']['subject_name']=='hippocampal volume','unreviewed stale literal edge')
            value[field]=original['metadata']['subject_id']
        if cid==SCFC and value['target_id']==SCFC_OLD:value['target_id']=SCFC_TARGET
        normalized.append((ordinal,value))
    source_edges(cid,checked,checked,[(o,r) for o,r in normalized if o not in RETIRED_EDGE_PAIRS])
    before=checked['metadata'];after=current['metadata'];events=[];current_rows=[];norm=dict(normalized);deleted=[]
    for ordinal,row in edge_rows:
        updates={}
        if row['relation_type']=='about':
            side=next(s for s in ('subject','object') if before[s+'_id']==norm[ordinal]['target_id']);updates['target_id',]=after[side+'_id']
        else:
            require((row.get('metadata') or {}).get('negated',before['negated'])==before['negated'],'science edge negation mismatch')
            for field,value in [('source_id',after['subject_id']),('target_id',after['object_id']),('relation_type',after['predicate'])]:updates[field,]=value
            if before['negated']!=after['negated']:
                require('negated' in row.get('metadata',{}),'missing explicit edge negation');updates['metadata','negated']=after['negated']
        fields=[(p,at_path(row,p),n) for p,n in updates.items() if at_path(row,p)!=n]
        out=rewrite(row,fields);current_rows.append((ordinal,out))
        if fields and ordinal not in RETIRED_EDGE_PAIRS:
            require(out['source_id']!=out['target_id'],'new edge self loop')
            events.append(dict(ordinal=ordinal,claim_id=cid,edge_sha256=digest(row),current_edge_sha256=digest(out),
                field_changes=[dict(path=list(p),old=o,new=n) for p,o,n in fields],protected_fields_sha256=protected_digest(row,fields)))
    complete=dict(current_rows);source=dict(edge_rows)
    for ordinal,kept in RETIRED_EDGE_PAIRS.items():
        if ordinal not in complete:continue
        require(kept in complete and complete[ordinal]==complete[kept],'duplicate edges differ outside approved identity fields')
        old=source[ordinal]
        require(edge_owner(old)==cid and old['relation_type']==source[kept]['relation_type'],'duplicate ownership differs')
        deleted.append(dict(ordinal=ordinal,kept_ordinal=kept,claim_id=cid,edge_sha256=digest(old),
            kept_original_edge_sha256=digest(source[kept]),current_kept_edge_sha256=digest(complete[kept]),
            original_endpoints={k:old[k] for k in ('source_id','target_id')},
            nonendpoint_sha256=digest({k:v for k,v in old.items() if k not in {'source_id','target_id'}})))
    source_edges(cid,current,current,[(o,r) for o,r in current_rows if o not in RETIRED_EDGE_PAIRS])
    return events,sorted(deleted,key=lambda e:e['ordinal'])

def reviewed_edges(cid,original,current,edge_rows):return reviewed_edge_bundle(cid,original,current,edge_rows)[0]

def restore_removed_edge(kept_original,proof):
    require(digest(kept_original)==proof['kept_original_edge_sha256'],'duplicate kept source hash differs')
    out=deepcopy(kept_original);out.update(proof['original_endpoints'])
    require(digest(out)==proof['edge_sha256'] and digest({k:v for k,v in out.items() if k not in {'source_id','target_id'}})==proof['nonendpoint_sha256'],'duplicate source evidence differs')
    return out

def source_edge_ordinal(current_ordinal,deleted):
    source=current_ordinal
    for value in sorted(deleted):
        if value<=source:source+=1
    return source

def apply_edge(row,event,reverse=False):
    require(digest(row)==event['current_edge_sha256' if reverse else 'edge_sha256'],'edge hash differs')
    fields=[(tuple(r['path']),r['old'],r['new']) for r in event['field_changes']]
    require(all(p in {('source_id',),('target_id',),('relation_type',),('metadata','negated')} for p,_,_ in fields),'unapproved edge field')
    out=rewrite(row,fields,reverse=reverse)
    require(digest(out)==event['edge_sha256' if reverse else 'current_edge_sha256'] and protected_digest(out,fields)==event['protected_fields_sha256'],'edge evidence changed')
    return out

def ordered_incidences(values):return sorted(values,key=lambda r:(r['node_id'],r['claim_id'],r['side']))

def no_retired_references(kind,key,row,plan):
    removed={r['node_id'] for r in plan['removed_nodes']}
    require(not (kind=='node' and key in removed),'retired node remains')
    text=compact(row)
    if any(nid in text for nid in removed):require(not list(exact_references(row,removed)),'retired exact reference remains')

class BatchTransform:
    def __init__(self,plan):
        self.plan=plan;self.events={e['claim_id']:e for e in plan['events']};self.edges={e['ordinal']:e for e in plan['edge_events']}
        self.removed={r['node_id']:r for r in plan['removed_nodes']};self.originals={};self.refs=defaultdict(list)
        self.deleted_edges={e['ordinal']:e for e in plan['removed_edges']};self.seen_deleted=set()
        require(len(self.events)==len(plan['events']) and len(self.edges)==len(plan['edge_events']),'duplicate event')
        self.digests={k:hashlib.sha256() for k in ('nodes','edges')}
        self.seen_targets=set();self.seen_removed=set();self.seen_edges=set();self.incidences=[]

    def records(self,records):
        inserted=False
        for kind,key,row in records:
            if kind=='edge' and not inserted:
                yield 'node',NEW_NODE['id'],deepcopy(NEW_NODE);inserted=True
            if kind=='node':
                require(key!=NEW_NODE['id'],'new concept already exists')
                if key in self.plan['existing_targets']:
                    require(digest(row)==self.plan['existing_targets'][key],'concept hash differs');self.seen_targets.add(key)
                self.incidences.extend(incidence_rows(key,row,self.plan['target_witnesses']))
                if key in self.removed:
                    require(digest(row)==self.removed[key]['node_sha256'] and not any(self.removed[key]['detail_references'].values()),'unsafe deletion')
                    self.seen_removed.add(key);continue
            if kind=='edge':
                owner=edge_owner(row)
                if owner in self.events:self.refs[owner].append((int(key),row))
                if int(key) in self.deleted_edges:
                    require(digest(row)==self.deleted_edges[int(key)]['edge_sha256'],'removed duplicate changed')
                    self.seen_deleted.add(int(key));continue
            if kind!='metadata':self.digests[kind+'s'].update(compact(row).encode()+b'\n')
            out=row
            if kind=='node' and key in self.events:out=reviewed_claim(row,self.events[key],self.plan);self.originals[key]=row
            elif kind=='edge':
                if int(key) in self.edges:out=apply_edge(row,self.edges[int(key)]);self.seen_edges.add(int(key))
            no_retired_references(kind,key,out,self.plan)
            yield kind,key,out
        require(inserted and set(self.originals)==set(self.events) and self.seen_edges==set(self.edges),'repair closure differs')
        require(self.seen_removed==set(self.removed) and self.seen_targets==set(self.plan['existing_targets']),'node witness closure differs')
        require(ordered_incidences(self.incidences)==self.plan['source_incidences'],'source incidence closure differs')
        reproduced=[];deleted=[]
        for cid,row in self.originals.items():
            changed,removed=reviewed_edge_bundle(cid,row,reviewed_claim(row,self.events[cid],self.plan),self.refs[cid]);reproduced.extend(changed);deleted.extend(removed)
        require(sorted(reproduced,key=lambda e:e['ordinal'])==self.plan['edge_events'],'edge event closure differs')
        require(self.seen_deleted==set(self.deleted_edges) and sorted(deleted,key=lambda e:e['ordinal'])==self.plan['removed_edges'],'duplicate edge retirement proof differs')
