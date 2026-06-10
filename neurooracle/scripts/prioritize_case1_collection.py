"""Build a manual curation queue from Case Study 1 collect-only papers.

This is intentionally local and deterministic: it ranks cached abstracts for
manual review and does not extract claims or touch the knowledge graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_DATA_DIR = Path("neurooracle/data/cs_runs/phase2_case1_transdiagnostic_v1")


@dataclass(frozen=True)
class PatternGroup:
    name: str
    weight: int
    patterns: tuple[str, ...]


GROUPS = (
    PatternGroup("meta_or_review", 26, (
        r"\bmeta[-\s]?analysis\b", r"\bsystematic review\b", r"\breview\b",
        r"\bmega[-\s]?analysis\b", r"\bconsortium\b", r"\bmultisite\b",
        r"\blarge[-\s]?scale\b", r"\bpooled\b", r"\bharmoniz",
    )),
    PatternGroup("cross_disorder_or_dimensional", 34, (
        r"\btransdiagnostic\b", r"\bcross[-\s]?disorder\b",
        r"\bcross[-\s]?diagnostic\b", r"\bshared neural\b",
        r"\bshared brain\b", r"\brdoc\b", r"\bresearch domain criteria\b",
        r"\bp[-\s]?factor\b", r"\bgeneral psychopathology\b", r"\bhitop\b",
        r"\binternalizing\b", r"\bexternalizing\b",
    )),
    PatternGroup("large_cohort_or_named_dataset", 24, (
        r"\benigma\b", r"\buk biobank\b", r"\babcd\b",
        r"\bphiladelphia neurodevelopmental cohort\b", r"\bpnc\b",
        r"\bhuman connectome project\b", r"\bhcp\b",
        r"\bucla consortium for neuropsychiatric phenomics\b",
        r"\btransdiagnostic connectome project\b", r"\btcp\b",
    )),
    PatternGroup("structural_mri_marker", 24, (
        r"\bcortical thickness\b", r"\bsurface area\b",
        r"\bgray matter\b", r"\bgrey matter\b", r"\bsubcortical volume\b",
        r"\bhippocamp", r"\bamygdala\b", r"\bbrain volume\b",
        r"\bstructural covariance\b", r"\bmri\b",
    )),
    PatternGroup("functional_connectivity_marker", 24, (
        r"\bfunctional connect", r"\bresting[-\s]?state\b", r"\brs[-\s]?fmri\b",
        r"\bfmri\b", r"\bconnectome\b", r"\bdefault mode\b",
        r"\bsalience network\b", r"\bfrontoparietal\b", r"\bgraph theory\b",
        r"\balff\b", r"\bfalff\b", r"\breho\b", r"\bregional homogeneity\b",
    )),
    PatternGroup("diffusion_marker", 20, (
        r"\bdti\b", r"\bdiffusion\b", r"\bwhite matter\b",
        r"\bfractional anisotropy\b", r"\bmean diffusivity\b",
        r"\btract\b",
    )),
    PatternGroup("brain_age_normative", 22, (
        r"\bbrain[-\s]?age\b", r"\bnormative model", r"\bnormative deviation",
        r"\bdeviation pattern\b",
    )),
    PatternGroup("clinical_or_cognitive_outcome", 22, (
        r"\bcognit", r"\bexecutive function\b", r"\bworking memory\b",
        r"\bsocial cognition\b", r"\bsymptom", r"\bseverity\b",
        r"\bdiagnos", r"\bclinical\b", r"\bfunctioning\b",
        r"\bimpairment\b", r"\bnegative symptoms\b", r"\bpositive symptoms\b",
        r"\bpanss\b", r"\bhamd\b", r"\bmadrs\b", r"\bphq[-\s]?9\b",
        r"\banhedonia\b", r"\bamotivation\b",
    )),
    PatternGroup("psychiatric_disorder", 18, (
        r"\bschizophren", r"\bpsychosis\b", r"\bpsychotic\b", r"\bbipolar\b",
        r"\bdepress", r"\bautis", r"\basd\b", r"\badhd\b",
        r"\battention[-\s]?deficit", r"\bobsessive[-\s]?compulsive\b",
        r"\bocd\b", r"\banxiety\b", r"\bptsd\b",
        r"\bpost[-\s]?traumatic stress\b", r"\bsubstance use\b",
        r"\balcohol use\b", r"\beating disorder\b", r"\banorexia\b",
        r"\bpsychiatr", r"\bmental disorder\b", r"\bmental health\b",
    )),
)

NOISE_PATTERNS = (
    r"\bmouse\b", r"\bmice\b", r"\brat\b", r"\brodent\b",
    r"\brodents\b", r"\bzebrafish\b", r"\banimal model\b",
    r"\bcell line\b", r"\bin vitro\b", r"\bendothelial\b",
    r"\bvascular function\b", r"\bbone\b", r"\btrabecular\b",
    r"\bperivascular adipose\b", r"\bhealthcare students\b",
    r"\bmedical students\b", r"\bprotocol\b", r"\bstudy protocol\b",
    r"\bcommentary\b", r"\beditorial\b", r"\bletter\b",
    r"\brationale and methods\b", r"\bmethods of an interdisciplinary\b",
    r"\bimage acquisition\b", r"\bquality assurance\b",
    r"\bhead motion\b", r"\bscoping review\b",
    r"\bsegmentation\b", r"\bsemi[-\s]?automatic\b",
    r"\bopen resource\b", r"\bdata resource\b",
    r"\bdataset protocol\b",
)

SOURCE_WEIGHT = {
    "pubmed": 12,
    "europepmc": 10,
    "openalex": 8,
    "anysearch": 4,
    "medrxiv": 0,
    "biorxiv": -4,
    "arxiv": -8,
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def load_abstracts(path: Path) -> dict[str, dict]:
    records = {}
    for row in read_jsonl(path):
        pmid = str(row.get("pmid") or "").strip()
        if pmid:
            records[pmid] = row
    return records


def load_manual_pmids(path: Path) -> set[str]:
    pmids: set[str] = set()
    for row in read_jsonl(path):
        pmid = str(row.get("pmid") or "").strip()
        if pmid:
            pmids.add(pmid)
    return pmids


def load_extracted_case1_pmids(path: Path) -> set[str]:
    pmids: set[str] = set()
    for row in read_jsonl(path):
        md = row.get("metadata") or {}
        is_case1 = (
            str(row.get("id") or "").startswith("CLM:CASE1MAN:")
            or row.get("disease") == "manual_case1_curated"
            or md.get("curation_scope") == "case1_transdiagnostic"
        )
        if not is_case1:
            continue
        paper = row.get("source_paper") or {}
        pmid = str(paper.get("pmid") or "").strip()
        if pmid:
            pmids.add(pmid)
    return pmids


def has_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def matches_for_group(text: str, group: PatternGroup) -> list[str]:
    found: list[str] = []
    for pattern in group.patterns:
        if re.search(pattern, text):
            found.append(pattern.replace("\\b", "").replace("\\", ""))
    return found


def score_record(row: dict, abstract_row: dict | None, curated_pmids: set[str]) -> dict:
    title = row.get("title") or ""
    abstract = (abstract_row or {}).get("abstract") or ""
    text = f"{title}\n{abstract}".lower()
    pmid = (row.get("pmid") or "").strip()
    source = (row.get("source") or "").strip().lower()

    score = SOURCE_WEIGHT.get(source, 0)
    reasons: list[str] = []
    tags: list[str] = []
    matched_groups: set[str] = set()

    for group in GROUPS:
        matches = matches_for_group(text, group)
        if matches:
            matched_groups.add(group.name)
            tags.append(group.name)
            score += group.weight
            reasons.append(f"{group.name}:{', '.join(matches[:3])}")

    noise_hits = [p.replace("\\b", "").replace("\\", "") for p in NOISE_PATTERNS if re.search(p, text)]
    if noise_hits:
        score -= 70
        tags.append("possible_noise")
        reasons.append(f"noise:{', '.join(noise_hits[:3])}")

    try:
        year = int(row.get("year") or 0)
    except ValueError:
        year = 0
    if year >= 2024:
        score += 6
        tags.append("recent")
    elif year >= 2020:
        score += 3

    has_imaging = bool(matched_groups & {
        "structural_mri_marker",
        "functional_connectivity_marker",
        "diffusion_marker",
        "brain_age_normative",
    })
    has_psychiatric = bool(matched_groups & {
        "psychiatric_disorder",
        "cross_disorder_or_dimensional",
    })
    has_outcome = "clinical_or_cognitive_outcome" in matched_groups

    if not has_imaging:
        score -= 35
        tags.append("missing_imaging_anchor")
    if not has_psychiatric:
        score -= 28
        tags.append("missing_psychiatric_anchor")
    if not has_outcome and "meta_or_review" not in matched_groups:
        score -= 12
        tags.append("weak_outcome_anchor")

    if pmid in curated_pmids:
        status = "already_curated"
        score -= 500
    elif score >= 95 and has_imaging and has_psychiatric:
        status = "ready_for_manual_curation"
    elif score >= 65 and has_imaging and has_psychiatric:
        status = "secondary_manual_curation"
    else:
        status = "low_priority_or_noise"

    if score >= 135:
        tier = "A"
    elif score >= 105:
        tier = "B"
    elif score >= 75:
        tier = "C"
    else:
        tier = "D"

    return {
        "priority_score": score,
        "priority_tier": tier,
        "queue_status": status,
        "topic_tags": ";".join(dict.fromkeys(tags)),
        "priority_reasons": " | ".join(reasons[:8]),
        "has_abstract": "yes" if bool(abstract) else "no",
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-batches", type=int, default=10)
    args = parser.parse_args()

    data_dir = args.data_dir
    collection_csv = data_dir / "collection_metadata.csv"
    abstract_cache = data_dir / "abstract_cache.jsonl"
    manual_claims = data_dir / "manual_case1_claims.jsonl"
    extracted_claims = data_dir / "extracted_claims.jsonl"

    abstracts = load_abstracts(abstract_cache)
    curated_pmids = load_manual_pmids(manual_claims) | load_extracted_case1_pmids(extracted_claims)

    with collection_csv.open("r", encoding="utf-8", newline="") as f:
        collection_rows = list(csv.DictReader(f))

    ranked_rows: list[dict] = []
    seen_ids: set[str] = set()
    for row in collection_rows:
        pmid = (row.get("pmid") or "").strip()
        if not pmid or pmid in seen_ids:
            continue
        seen_ids.add(pmid)
        abstract_row = abstracts.get(pmid)
        score = score_record(row, abstract_row, curated_pmids)
        abstract = (abstract_row or {}).get("abstract") or ""
        ranked_rows.append({
            **score,
            "pmid": pmid,
            "doi": row.get("doi") or "",
            "year": row.get("year") or "",
            "source": row.get("source") or "",
            "title": row.get("title") or "",
            "journal": row.get("journal") or "",
            "abstract_length": row.get("abstract_length") or str(len(abstract)),
            "collected_at": row.get("collected_at") or "",
            "abstract_preview": re.sub(r"\s+", " ", abstract)[:500],
        })

    ranked_rows.sort(
        key=lambda r: (
            r["queue_status"] not in ("ready_for_manual_curation", "secondary_manual_curation"),
            -int(r["priority_score"]),
            -(int(r["year"]) if str(r["year"]).isdigit() else 0),
            r["pmid"],
        )
    )
    for idx, row in enumerate(ranked_rows, start=1):
        row["rank"] = idx

    fieldnames = [
        "rank", "priority_tier", "priority_score", "queue_status",
        "pmid", "doi", "year", "source", "title", "journal",
        "abstract_length", "has_abstract", "topic_tags", "priority_reasons",
        "collected_at", "abstract_preview",
    ]
    queue_csv = data_dir / "case1_manual_curation_queue.csv"
    write_csv(queue_csv, ranked_rows, fieldnames)

    batches_dir = data_dir / "manual_curation_batches"
    ready_rows = [r for r in ranked_rows if r["queue_status"] in {
        "ready_for_manual_curation",
        "secondary_manual_curation",
    }]
    for batch_idx in range(args.n_batches):
        start = batch_idx * args.batch_size
        batch = ready_rows[start:start + args.batch_size]
        if not batch:
            break
        write_csv(
            batches_dir / f"case1_manual_batch_{batch_idx + 1:03d}.csv",
            batch,
            fieldnames,
        )

    by_status = Counter(r["queue_status"] for r in ranked_rows)
    by_tier = Counter(r["priority_tier"] for r in ranked_rows)
    by_source_ready = Counter(r["source"] for r in ready_rows)
    tag_counts = Counter()
    for row in ranked_rows:
        for tag in (row["topic_tags"] or "").split(";"):
            if tag:
                tag_counts[tag] += 1

    summary_path = data_dir / "CASE1_COLLECTION_PRIORITY_SUMMARY.md"
    top_examples = ready_rows[:20]
    lines = [
        "# Case Study 1 Collection Priority Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- Collection metadata: `{collection_csv}`",
        f"- Abstract cache: `{abstract_cache}`",
        f"- Curated PMID anchors excluded/deprioritized: {len(curated_pmids)}",
        "",
        "## Outputs",
        "",
        f"- Full ranked queue: `{queue_csv}`",
        f"- Batch directory: `{batches_dir}`",
        "",
        "## Counts",
        "",
        f"- Collection rows read: {len(collection_rows)}",
        f"- Unique candidate IDs ranked: {len(ranked_rows)}",
        f"- Ready or secondary manual curation: {len(ready_rows)}",
        "",
        "### By Status",
        "",
    ]
    for key, value in by_status.most_common():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "### By Tier", ""])
    for key in ("A", "B", "C", "D"):
        lines.append(f"- {key}: {by_tier.get(key, 0)}")
    lines.extend(["", "### Ready Queue By Source", ""])
    for key, value in by_source_ready.most_common():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "### Common Tags", ""])
    for key, value in tag_counts.most_common(20):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top 20 Ready Candidates", ""])
    for row in top_examples:
        lines.append(
            f"{row['rank']}. [{row['priority_tier']}/{row['priority_score']}] "
            f"{row['year']} {row['pmid']} {row['title']} "
            f"({row['source']}; {row['topic_tags']})"
        )
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "queue_csv": str(queue_csv),
        "summary": str(summary_path),
        "batches_dir": str(batches_dir),
        "collection_rows": len(collection_rows),
        "ranked_unique_candidates": len(ranked_rows),
        "ready_or_secondary": len(ready_rows),
        "by_status": dict(by_status),
        "by_tier": dict(by_tier),
        "ready_by_source": dict(by_source_ready),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
