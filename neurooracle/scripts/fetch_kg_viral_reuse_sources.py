"""Fetch three explicit public PMIDs that already reference the full viral node."""
from pathlib import Path
import sqlite3
import sys
import requests
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from kg_accepted_candidate_lineage import require
from neurooracle.src.kg_scoped_structure import source_documents

OUTPUT=j.OUTPUT/'round44_source_resolution'
OWNERS={
    'CLM:7ee398d42656898c':('80e4bc38e98741f3c54ea18025b82ec6ad2198e77d43f3bffa49fa703af9af6c','12165132','10.1089/08977150260139093'),
    'CLM:8e7e3673f69c7156':('a55b3a08830ec09ab89876a7568e285af69ce70c435745de32586b265204030c','12485887','10.1111/j.1749-6632.2002.tb04659.x'),
    'CLM:584a2b9d1bb1bbf3':('b13a5f0a8f8f81729fef6c324962caceacaff0019d18a38ca1125be873278499','18490064','10.1016/j.bbr.2008.03.038'),
}


def main():
    require(not (OUTPUT/'VIRAL_REUSE_SOURCE_FETCH.json').exists(),'public evidence exists; inspect/reuse')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');require(c['status']=='COMPLETED' and c['active_process'] is None,'writer active')
    census=j.read_json(c['current_paper_census']['path']);j.guards([c['current_graph'],census['database'],c['formal_sources']])
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    for cid,(h,pmid,doi) in OWNERS.items():
        require(db.execute('SELECT c.node_sha,p.pmid,p.doi FROM claims c JOIN papers p ON p.sig=c.paper_sig WHERE c.cid=?',(cid,)).fetchone()==(h,pmid,doi),'current explicit source differs')
    db.close()
    response=requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',params=dict(db='pubmed',id=','.join(sorted(v[1] for v in OWNERS.values())),retmode='xml',tool='NeuroClawKGReview'),timeout=45)
    response.raise_for_status();content=response.content.decode('utf8');docs=source_documents([content])
    require(set(docs)=={v[1] for v in OWNERS.values()},'public PMID set differs')
    for _,pmid,doi in OWNERS.values():
        require(doi.casefold() in {d.casefold() for d in docs[pmid]['dois']},'own public DOI differs')
        require('vaccinia virus complement control protein' in docs[pmid]['title'].casefold(),'full protein title identity differs')
    j.atomic_text(OUTPUT/'viral_reuse_pubmed.xml',content)
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c,'source advanced');j.guards([c['current_graph'],census['database'],c['formal_sources']])
    result=dict(at=j.utc_now(),graph=c['current_graph'],current_acceptance=c['current_acceptance'],census=c['current_paper_census'],
        code=j.fingerprint(Path(__file__)),response=j.fingerprint(OUTPUT/'viral_reuse_pubmed.xml'),url=response.url,
        explicit_current_claim_owners=OWNERS,all_public_titles_and_dois_checked=True,claim_scientific_conclusions_not_revalidated=True,
        no_claim_suffix_inference=True,public_identifiers_only=True,graph_modified=False,record_preimages_saved=False)
    j.atomic_json(OUTPUT/'VIRAL_REUSE_SOURCE_FETCH.json',result);print('VIRAL_REUSE_PUBLIC_SOURCES_VERIFIED',sorted(docs),flush=True)


if __name__=='__main__':main()
