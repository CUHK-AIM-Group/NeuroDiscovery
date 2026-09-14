"""R35 full-SHA source inspection; save hashes/verdicts, not KG record preimages."""
from collections import Counter,defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import unicodedata

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from build_umls_simplification_candidate import hashed_reader,walk_graph,compact
from kg_accepted_candidate_lineage import require
from fetch_kg_complete_titles import own_articles
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import identifiers,normalize_pmid,VerifiedPaperIdentities

OUTPUT=journal.OUTPUT/"round35_source_bibliography"
R31=journal.OUTPUT/"round31_paper_identity"
R33=journal.OUTPUT/"round33_complete_title_evidence"


def content_key(value):
    return " ".join(unicodedata.normalize("NFKC",html.unescape(str(value or ""))).casefold().split())


def source_text(article):
    paths=("./MedlineCitation/Article/Abstract/AbstractText","./MedlineCitation/OtherAbstract/AbstractText","./BookDocument/Abstract/AbstractText")
    return " ".join("".join(e.itertext()) for path in paths for e in article.findall(path))


def load_public_xml():
    articles={}; witnesses={}
    manifest=journal.read_json(R33/"XML_FETCH.json")
    for w in manifest["witnesses"]:
        require(journal.fingerprint(Path(w["response"]["path"]))==w["response"],"XML source changed")
        found=own_articles(Path(w["response"]["path"]).read_bytes())
        require(set(found)<=set(w["requested_pmids"]),"unrequested XML")
        for p,a in found.items():
            require(p not in articles,"duplicate XML PMID")
            articles[p]=a;witnesses[p]=dict(response_sha256=w["response"]["sha256"],response_file=Path(w["response"]["path"]).name,url=w["url"])
    return articles,witnesses


def main():
    OUTPUT.mkdir(exist_ok=True)
    require(not (OUTPUT/"SOURCE_INSPECTION.json").exists(),"inspection exists; reuse it")
    c=journal.read_json(journal.OUTPUT/"CAMPAIGN.json");require(c["status"]=="COMPLETED" and c["active_process"] is None,"writer active")
    journal.guards([c["current_graph"],c["current_detail_store"],c["formal_sources"]])
    census=journal.read_json(R31/"CENSUS.json");journal.guards(census["database"])
    registry=VerifiedPaperIdentities(journal.read_json(c["current_paper_identities"]["path"]))
    records=registry.export_payload()["records"];aliases={kind:defaultdict(set) for kind in ("doi","pmcid")}
    for p,row in records.items():
        for kind in aliases:
            for v in row[kind]:aliases[kind][v].add(p)
    targets={};categories=defaultdict(set)
    for row in census["outer_pmid_conflicts"]:
        targets[row["claim_id"]]=row["claim_sha256"];categories[row["claim_id"]].add("outer_pmid")
    for line in Path(c["current_paper_issues"]["path"]).read_text(encoding="utf8").splitlines():
        row=json.loads(line)
        if "complete_title_disagreement" in row["reasons"]:
            require(targets.get(row["claim_id"],row["claim_sha256"])==row["claim_sha256"],"inconsistent current claim hash")
            targets[row["claim_id"]]=row["claim_sha256"];categories[row["claim_id"]].add("title")
    articles,witnesses=load_public_xml(); abstracts={p:content_key(source_text(a)) for p,a in articles.items()}
    counts=Counter(); seen=set(); output=[]; printed=Counter(); schema=Counter()
    code=[journal.fingerprint(Path(__file__))]
    with hashed_reader(Path(c["current_graph"]["path"])) as (reader,sha):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=="node" and key in targets:
                require(digest(row)==targets[key],"selected current claim changed")
                seen.add(key);md=row["metadata"];p=md.get("source_paper") or {};ids=identifiers(p)
                inner=md.get("metadata") or {};alternates={normalize_pmid(h.get("pmid")) for h in (md,inner) if h.get("pmid")}
                candidates=({ids["pmid"]} if ids["pmid"] in records else set()) | (alternates&set(records))
                for field in aliases:candidates.update(aliases[field].get(ids[field],set()))
                quote=content_key(md.get("raw_text"));strong=len(quote)>=40 and len(re.findall(r"\w+",quote))>=8
                matches=[pmid for pmid in sorted(candidates,key=int) if strong and quote in abstracts.get(pmid,"")]
                label="+".join(sorted(categories[key]));counts[label]+=1
                counts[label+":exact_abstract_match"]+=len(matches)==1
                schema.update(md.keys())
                item=dict(claim_id=key,claim_sha256=targets[key],categories=sorted(categories[key]),
                    authority_candidates=sorted(candidates,key=int),exact_abstract_match_pmids=matches,
                    raw_text_sha256=hashlib.sha256(str(md.get("raw_text") or "").encode()).hexdigest(),normalized_quote_sha256=hashlib.sha256(quote.encode()).hexdigest(),
                    raw_quote_characters=len(quote),strong_quote=strong,available_abstract_pmids=sorted(candidates&set(abstracts),key=int),
                    current_identity=registry.resolve(md),matching_witnesses={p:witnesses[p] for p in matches})
                output.append(item)
                sample_label=label+(":match" if matches else ":no_match")
                if printed[sample_label]<3:
                    print(compact(dict(sample=sample_label,id=key,source_paper=p,outer_pmids=sorted(alternates),raw_text=str(md.get("raw_text") or "")[:1500],
                        evidence=md.get("evidence"),metadata_keys=list(inner),matches=matches,
                        abstract_sample={p:source_text(articles[p])[:1600] for p in sorted(candidates) if p in articles})),flush=True)
                    printed[sample_label]+=1
            if counts[kind]%250000==0:
                state=dict(status="READ_ONLY_SCANNING",at=journal.utc_now(),kind=kind,count=counts[kind],selected_seen=len(seen),selected_total=len(targets))
                journal.atomic_json(OUTPUT/"INSPECTION_STATE.json",state);print(compact(state),flush=True)
        require(sha.hexdigest()==c["current_graph"]["sha256"],"full source graph SHA mismatch")
    require(seen==set(targets),"selected records missing")
    require([journal.fingerprint(Path(__file__))]==code,"inspection code changed during scan")
    journal.guards([c["current_graph"],c["current_detail_store"],c["formal_sources"],census["database"]])
    require(journal.read_json(journal.OUTPUT/"CAMPAIGN.json")==c,"current campaign advanced")
    result=dict(status="INSPECTED_NOT_MODIFIED",at=journal.utc_now(),graph=c["current_graph"],counts=dict(counts),selected_claims=len(seen),
        selected_schema_keys=dict(schema),members=output,xml_fetch=journal.fingerprint(R33/"XML_FETCH.json"),
        source_census=journal.fingerprint(R31/"CENSUS.json"),paper_registry=c["current_paper_identities"],source_full_sha_verified=True,
        graph_modified=False,record_preimages_saved=False,code=code)
    journal.atomic_json(OUTPUT/"SOURCE_INSPECTION.json",result)
    journal.atomic_json(OUTPUT/"INSPECTION_STATE.json",dict(status="INSPECTION_COMPLETE",at=result["at"],selected_claims=len(seen),counts=dict(counts)))
    print(compact(dict(status=result["status"],counts=dict(counts),selected_claims=len(seen))),flush=True)


if __name__=="__main__":main()
