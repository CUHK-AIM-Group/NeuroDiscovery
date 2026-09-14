"""Current-only census beyond the historic FANCE/VCP watch list.

Lexical classification is diagnostic, never an instruction to merge/delete.
Exact symbols also require organism/protein/abbreviation context. No KG writes,
no model calls, no archived record preimages. Native guards bind a same-phase
reread to the already completed full R42 source verification.
"""
from collections import Counter
from pathlib import Path
import os
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from build_umls_simplification_candidate import IncrementalJsonReader,compact,walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.claim_semantics import concept_atom_roles,declared_type_atoms

OUTPUT=j.OUTPUT/'round42_source_scope'


def classify(name,labels):
    """Whole-label / phrase / word-interior distinctions; no identity inference."""
    if not isinstance(name,str) or not name.strip():return 'missing_name',[]
    text=name.strip().casefold()
    labels=sorted({s.strip().casefold() for s in labels if isinstance(s,str) and s.strip()})
    if text in labels:return 'exact_label_context_still_required',[text]
    phrase=[s for s in labels if s in text and re.search(r'(?<!\w)'+re.escape(s)+r'(?!\w)',text)]
    if phrase:return 'symbol_phrase_requires_scope_review',phrase
    interior=[s for s in labels if s in text]
    if interior:return 'word_interior_alias_only',interior
    return 'no_label_alignment',[]


def progress(phase,**values):
    state=dict(status='READ_ONLY_RUNNING',pid=os.getpid(),at=j.utc_now(),phase=phase,**values)
    j.atomic_json(OUTPUT/'GENE_CENSUS_STATE.json',state);print(compact(state),flush=True)


def main():
    require(not (OUTPUT/'ALL_GENE_ENDPOINT_CENSUS.json').exists(),'census exists; inspect current result')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');source=j.read_json(OUTPUT/'SOURCE_INSPECTION.json')
    require(c['status']=='COMPLETED' and c['active_process'] is None and source['graph']==c['current_graph'] and source['source_full_sha_verified'],'source boundary')
    require(j.fingerprint(source['code']['path'])==source['code'],'source verifier changed')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    code=j.fingerprint(Path(__file__));genes={};n=0
    progress('CANONICAL_GENE_NODE_CENSUS')
    with Path(c['current_graph']['path']).open(encoding='utf8') as f:
        for kind,key,row in walk_graph(IncrementalJsonReader(f)):
            if kind=='edge':break
            if kind!='node':continue
            n+=1
            # Scope is canonical gene nodes, not extracted molecular phrases.
            ext=row.get('external_ids') or {}
            if key.startswith('CUI:') and ('T028' in (row.get('semantic_types') or []) or ext.get('HGNC') or ext.get('NCBI_Gene')):
                if 'gene_target' in {a.value for a in concept_atom_roles(row)}:
                    labels=[row['preferred_name'],*(row.get('aliases') or [])]
                    genes[key]=dict(node_id=key,name=row['preferred_name'],node_sha256=digest(row),labels=labels,
                        semantic_types=row.get('semantic_types'),external_ids=ext)
            if n%1000000==0:progress('CANONICAL_GENE_NODE_CENSUS',nodes=n,canonical_genes=len(genes))
    require(n==c['counts']['nodes'],'source node census differs')
    j.guards(c['current_graph'])
    held={(r['claim_id'],r['side']) for r in rows(c['current_gene_holds']['path'])}
    events=[];counts=Counter();by_gene=Counter();by_role=Counter();claims=0
    progress('ALL_CLAIM_GENE_ENDPOINTS',canonical_genes=len(genes))
    with Path(c['current_graph']['path']).open(encoding='utf8') as f:
        for kind,key,row in walk_graph(IncrementalJsonReader(f)):
            if kind=='edge':break
            if kind!='node' or not key.startswith('CLM:'):continue
            claims+=1;md=row['metadata'];inner=md.get('metadata') or {};paper=md.get('source_paper') or {}
            for side in ('subject','object'):
                nid=md.get(side+'_id')
                if nid not in genes:continue
                gene=genes[nid];category,matches=classify(md.get(side+'_name'),gene['labels'])
                roles=sorted(a.value for a in declared_type_atoms(md.get(side+'_type') or inner.get(side+'_type')))
                counts[category]+=1;by_gene[nid]+=1;by_role['|'.join(roles)]+=1
                events.append(dict(claim_id=key,claim_sha256=digest(row),side=side,name=md.get(side+'_name'),
                    current_node_id=nid,gene_name=gene['name'],category=category,matched_labels=matches,
                    declared_roles=roles,source_pmid=paper.get('pmid'),source_doi=paper.get('doi'),
                    in_existing_449_queue=(key,side) in held,raw_text_sha256=digest(md.get('raw_text')),
                    classification_is_not_a_repair_or_error_verdict=True))
            if claims%250000==0:progress('ALL_CLAIM_GENE_ENDPOINTS',claims=claims,endpoints=len(events))
    require(claims==c['counts']['claims'],'claim census differs')
    require(held<={(r['claim_id'],r['side']) for r in events},'canonical gene scope missed current queue')
    write_rows(OUTPUT/'ALL_GENE_ENDPOINTS.jsonl',events)
    used={r['current_node_id'] for r in events}
    write_rows(OUTPUT/'CURRENT_GENE_NODE_WITNESSES.jsonl',[genes[nid] for nid in sorted(used)])
    suspicious={'word_interior_alias_only','no_label_alignment'}
    outside=[r for r in events if not r['in_existing_449_queue'] and r['category'] in suspicious]
    result=dict(status='CURRENT_ALL_GENE_ENDPOINT_CENSUS_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],
        source_inspection=j.fingerprint(OUTPUT/'SOURCE_INSPECTION.json'),code=code,
        reused_same_phase_full_sha_with_native_guards=True,canonical_gene_nodes=len(genes),
        used_canonical_gene_nodes=len(used),claims=claims,gene_endpoints=len(events),categories=dict(counts),roles=dict(by_role),
        outside_existing_queue_lexical_review_candidates=len(outside),
        outside_existing_queue_claims=len({r['claim_id'] for r in outside}),
        outside_queue_by_gene=dict(Counter(r['gene_name'] for r in outside)),
        artifacts={n:j.fingerprint(OUTPUT/n) for n in ('ALL_GENE_ENDPOINTS.jsonl','CURRENT_GENE_NODE_WITNESSES.jsonl')},
        not_a_confirmed_error_count=True,graph_modified=False,record_preimages_saved=False,
        limits=['Only current canonical CUI gene nodes with T028/HGNC/NCBI_Gene and gene role.',
            'Lexical mismatch does not prove scientific error; exact symbol does not prove correct organism/protein identity.'])
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and j.fingerprint(Path(__file__))==code,'control/code advanced')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    j.atomic_json(OUTPUT/'ALL_GENE_ENDPOINT_CENSUS.json',result)
    j.atomic_json(OUTPUT/'GENE_CENSUS_STATE.json',dict(status='COMPLETED',at=j.utc_now(),pid=os.getpid()))
    print(compact(result),flush=True)


if __name__=='__main__':main()
