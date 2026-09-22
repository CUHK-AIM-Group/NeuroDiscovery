"""Fetch PubMed abstracts used to resolve targeted manual Case-2 source gaps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


SCHEMA = "case2_manual_source_supplements.v1"


def _node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pmid", action="append", required=True)
    args = parser.parse_args()
    pmids = list(dict.fromkeys(str(value).strip() for value in args.pmid if value))
    query = urllib.parse.urlencode(
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    )
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + query
    request = urllib.request.Request(url, headers={"User-Agent": "NeuroClaw/3 source-audit"})
    with urllib.request.urlopen(request, timeout=30) as response:
        root = ET.fromstring(response.read())

    existing: dict[str, dict[str, object]] = {}
    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA:
            raise ValueError("source supplement schema mismatch")
        existing = {
            str(entry["paper_key"]): entry for entry in payload.get("entries") or []
        }

    fetched: set[str] = set()
    retrieved_on = datetime.now(timezone.utc).date().isoformat()
    for article in root.findall("./PubmedArticle"):
        pmid = _node_text(article.find("./MedlineCitation/PMID"))
        if not pmid:
            continue
        abstract_nodes = article.findall("./MedlineCitation/Article/Abstract/AbstractText")
        source_text = " ".join(_node_text(node) for node in abstract_nodes if _node_text(node))
        if not source_text:
            raise ValueError(f"PubMed {pmid} has no abstract text")
        title = _node_text(article.find("./MedlineCitation/Article/ArticleTitle"))
        paper_key = f"pmid:{pmid}"
        existing[paper_key] = {
            "paper_key": paper_key,
            "title": title,
            "source": "NCBI PubMed E-utilities efetch",
            "source_url": (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
                + urllib.parse.urlencode({"db": "pubmed", "id": pmid, "retmode": "xml"})
            ),
            "retrieved_on": retrieved_on,
            "source_text": source_text,
            "source_text_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        }
        fetched.add(pmid)

    missing = sorted(set(pmids) - fetched)
    if missing:
        raise ValueError(f"PubMed did not return requested records: {missing}")
    payload = {
        "schema_version": SCHEMA,
        "entries": [existing[key] for key in sorted(existing)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "fetched": len(fetched),
                "entries": len(existing),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
