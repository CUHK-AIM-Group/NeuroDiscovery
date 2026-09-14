"""R34 public owning articles and directly linked publication notices only."""
import hashlib
from pathlib import Path
import sys
import time
from xml.etree import ElementTree as ET

import httpx

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from kg_accepted_candidate_lineage import require
from fetch_kg_complete_titles import own_articles, API
from neurooracle.src.kg_paper_identity import normalize_pmid

OUTPUT=journal.OUTPUT/"round34_publication_status"
PREVIOUS=journal.OUTPUT/"round33_complete_title_evidence"
LINK_TYPES=frozenset({"RetractionIn","RetractionOf","CorrectedandRepublishedIn","CorrectedandRepublishedFrom",
    "RetractedandRepublishedIn","RetractedandRepublishedFrom","RepublishedIn","RepublishedFrom","ExpressionOfConcernIn","ExpressionOfConcernFor"})


def status_links(article):
    rows=[]
    for el in article.findall("./MedlineCitation/CommentsCorrectionsList/CommentsCorrections"):
        if el.get("RefType") not in LINK_TYPES: continue
        p=el.find("PMID"); pmid=normalize_pmid(p.text if p is not None else None)
        rows.append(dict(relation=el.get("RefType"),pmid=pmid))
    return sorted(rows,key=lambda r:(r["relation"],r["pmid"]))


def main():
    c=journal.read_json(journal.OUTPUT/"CAMPAIGN.json")
    require(c["active_process"] is None and c["status"]=="COMPLETED","writer active")
    journal.guards([c["current_graph"],c["current_detail_store"],c["formal_sources"]])
    warning=PREVIOUS/"CURRENT_SOURCE_WARNINGS.json"; w=journal.read_json(warning)
    require(w["graph"]==c["current_graph"],"warnings not current")
    roots=sorted({r["pmid"] for r in w["authorities"]},key=int); require(len(roots)==7,"review changed scope")
    OUTPUT.mkdir(exist_ok=True); cache=OUTPUT/"pubmed_xml"; cache.mkdir(exist_ok=True)
    plan=dict(graph=c["current_graph"],warning=journal.fingerprint(warning),root_pmids=roots,
        limit="roots plus one hop of explicit publication-status links, at most 40 IDs, sequential <=1 request/s",endpoint=API)
    if (OUTPUT/"NOTICE_REQUEST.json").exists(): require(journal.read_json(OUTPUT/"NOTICE_REQUEST.json")==plan,"request changed")
    else: journal.atomic_json(OUTPUT/"NOTICE_REQUEST.json",plan)
    available={}; witnesses=[]
    # Reuse verified R33 public XML without duplicating its cache.
    for old in journal.read_json(PREVIOUS/"XML_FETCH.json")["witnesses"]:
        if not set(roots)&set(old["requested_pmids"]):continue
        require(journal.fingerprint(Path(old["response"]["path"]))==old["response"],"old XML changed")
        for pmid,a in own_articles(Path(old["response"]["path"]).read_bytes()).items():
            if pmid in roots: available[pmid]=a
        witnesses.append(dict(**old,reused_for_root_pmids=sorted(set(roots)&set(old["requested_pmids"]))))
    with httpx.Client(timeout=30,follow_redirects=True,headers={"User-Agent":"NeuroClawKGIdentityAudit/1.0"}) as client:
        for phase in ("roots","linked_notices"):
            wanted=set(roots) if phase=="roots" else {r["pmid"] for p in roots for r in status_links(available[p]) if r["pmid"]}
            require(len(wanted|set(roots))<=40,"unexpected publication notice fanout")
            missing=sorted(wanted-set(available),key=int)
            if not missing:continue
            path=cache/(hashlib.sha256(",".join(missing).encode()).hexdigest()[:20]+".xml")
            params=dict(db="pubmed",id=",".join(missing),retmode="xml",tool="NeuroClawKGIdentityAudit")
            if not path.exists():
                for attempt in range(3):
                    try:
                        response=client.get(API,params=params); response.raise_for_status()
                        found=own_articles(response.content); require(set(found)==set(missing),"incomplete notice response")
                        journal.atomic_text(path,response.text);break
                    except (httpx.HTTPError,ValueError,ET.ParseError):
                        if attempt==2:raise
                        time.sleep(2+attempt)
                time.sleep(1.05)
            found=own_articles(path.read_bytes()); require(set(found)==set(missing),"unrequested cached notice")
            available.update(found)
            witnesses.append(dict(requested_pmids=missing,response=journal.fingerprint(path),url=str(httpx.URL(API,params=params))))
            print(phase+": "+str(len(missing))+" XML records",flush=True)
    result=dict(status="FETCHED_NOT_ADOPTED",at=journal.utc_now(),request=journal.fingerprint(OUTPUT/"NOTICE_REQUEST.json"),
        witnesses=witnesses,root_pmids=roots,linked_pmids=sorted(set(available)-set(roots),key=int),
        root_links={p:status_links(available[p]) for p in roots},graph_modified=False,private_claim_text_sent=False)
    journal.guards([c["current_graph"],c["formal_sources"]]); require(journal.read_json(journal.OUTPUT/"CAMPAIGN.json")==c,"KG changed during retrieval")
    journal.atomic_json(OUTPUT/"NOTICE_FETCH.json",result)
    for p in roots:
        print(p,[e.text for e in available[p].findall("./MedlineCitation/Article/PublicationTypeList/PublicationType")],status_links(available[p]),flush=True)
    for p in result["linked_pmids"]:
        print("linked",p,[e.text for e in available[p].findall("./MedlineCitation/Article/PublicationTypeList/PublicationType")],status_links(available[p]),flush=True)


if __name__=="__main__":main()
