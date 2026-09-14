"""Read-only public-source refresh for the 21 current qualified semantic holds."""
from pathlib import Path
import json
import sqlite3
import sys
import requests
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_scoped_structure import source_documents
from neurooracle.src.kg_paper_identity import title_key

OUTPUT=j.OUTPUT/'round45_semantic_sources'


def main():
    require(not (OUTPUT/'PUBLIC_SOURCE_FETCH.json').exists(),'source evidence exists; inspect/reuse')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');census=j.read_json(c['current_paper_census']['path'])
    held=rows(c['current_issues']['path']);require(len(held)==21,'not the selected 21-current-hold set')
    j.guards([c['current_graph'],c['current_detail_store'],census['database'],c['formal_sources']])
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    selected=[]
    for issue in held:
        cid=issue['claim_id'];r=db.execute('SELECT c.node_sha,p.pmid,p.doi,p.payload FROM claims c JOIN papers p ON p.sig=c.paper_sig WHERE c.cid=?',(cid,)).fetchone()
        require(r is not None and r[0]==issue['current_node_sha256'],'current semantic claim/census differs')
        require(r[1].isdigit() and r[1]==str(issue['pmid']),'own explicit current PMID differs')
        paper=json.loads(r[3])
        selected.append(dict(claim_id=cid,claim_sha256=r[0],pmid=r[1],doi=r[2],current_title=paper.get('title'),
            current_paper_sha256=digest(paper),review_category=issue['original_issue_category']))
    db.close();pmids=sorted({r['pmid'] for r in selected});OUTPUT.mkdir(exist_ok=True)
    response=requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',params=dict(db='pubmed',id=','.join(pmids),retmode='xml',tool='NeuroClawKGReview'),timeout=45)
    response.raise_for_status();content=response.content.decode('utf8');docs=source_documents([content])
    require(set(docs)==set(pmids),'requested public PMID set differs')
    for r in selected:
        d=docs[r['pmid']]
        r.update(public_title=d['title'],public_abstract_sha256=digest(d['abstract']),public_dois=d['dois'],
            doi_matches=not r['doi'] or r['doi'].casefold() in {v.casefold() for v in d['dois']},
            title_matches=title_key(r['current_title'])==title_key(d['title']))
    j.atomic_text(OUTPUT/'pubmed_semantic_holds.xml',content)
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c,'source campaign advanced during refresh')
    j.guards([c['current_graph'],c['current_detail_store'],census['database'],c['formal_sources']])
    result=dict(status='PUBLIC_SOURCES_FETCHED_NOT_ADJUDICATED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        current_census=c['current_paper_census'],source_issues=c['current_issues'],selected=selected,code=j.fingerprint(Path(__file__)),
        response=j.fingerprint(OUTPUT/'pubmed_semantic_holds.xml'),url=response.url,public_identifiers_only=True,no_claim_suffix_inference=True,
        graph_modified=False,record_preimages_saved=False,registry_not_automatically_upgraded=True)
    j.atomic_json(OUTPUT/'PUBLIC_SOURCE_FETCH.json',result)
    print('CURRENT_SEMANTIC_PUBLIC_SOURCES_FETCHED',dict(claims=len(selected),papers=len(docs),title_differences=sum(not r['title_matches'] for r in selected),doi_differences=sum(not r['doi_matches'] for r in selected)),flush=True)


if __name__=='__main__':main()
