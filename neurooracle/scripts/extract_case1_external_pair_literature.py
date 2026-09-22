"""Extract prior literature for selected TCP external-validation hypotheses.

This stage deliberately does not generate final relevance prose.  It creates
an auditable queue containing the selected paper, abstract/evidence excerpt,
and empty bilingual fields for hypothesis-specific manual review.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd


DISEASES = {
    "ADHD": {
        "name_en": "Attention-Deficit/Hyperactivity Disorder",
        "name_zh": "注意缺陷多动障碍",
        "terms": (
            "attention deficit hyperactivity",
            "attention-deficit/hyperactivity",
            "adhd",
        ),
    },
    "bipolar": {
        "name_en": "Bipolar Disorder",
        "name_zh": "双相障碍",
        "terms": ("bipolar disorder", "bipolar", "mania", "manic"),
    },
    "psychosis_SZ_SZA": {
        "name_en": "Schizophrenia-spectrum Psychosis",
        "name_zh": "精神分裂症谱系精神病",
        "terms": (
            "schizophrenia",
            "schizoaffective",
            "non-affective psychosis",
            "psychosis",
            "psychotic",
        ),
    },
}

FEATURES = {
    "roi_falff_proxy": (
        "ROI fALFF",
        "感兴趣区低频振幅分数",
        ("falff", "fractional amplitude of low-frequency fluctuation"),
    ),
    "roi_alff_proxy": (
        "ROI ALFF",
        "感兴趣区低频振幅",
        ("alff", "amplitude of low-frequency fluctuation"),
    ),
    "roi_temporal_mean": (
        "ROI mean signal",
        "感兴趣区平均信号",
        ("mean bold signal", "mean signal", "regional signal"),
    ),
    "roi_temporal_mean_abs": (
        "ROI mean absolute signal",
        "感兴趣区平均绝对信号",
        ("mean absolute signal", "regional signal"),
    ),
    "roi_temporal_std": (
        "ROI temporal standard deviation",
        "感兴趣区时序标准差",
        ("signal variability", "temporal variability", "standard deviation"),
    ),
    "roi_temporal_variance": (
        "ROI temporal variance",
        "感兴趣区时序方差",
        ("signal variability", "temporal variability", "variance"),
    ),
    "corr_mean": (
        "ROI mean whole-brain functional connectivity",
        "感兴趣区平均全脑功能连接",
        ("functional connectivity", "resting-state connectivity"),
    ),
    "corr_mean_abs": (
        "ROI mean absolute whole-brain functional connectivity",
        "感兴趣区平均绝对全脑功能连接",
        ("functional connectivity", "resting-state connectivity"),
    ),
    "corr_positive_mean": (
        "ROI positive functional connectivity",
        "感兴趣区正向功能连接",
        ("positive functional connectivity", "functional connectivity"),
    ),
    "corr_negative_mean": (
        "ROI negative functional connectivity",
        "感兴趣区负向功能连接",
        ("negative functional connectivity", "anticorrelation", "functional connectivity"),
    ),
    "corr_node_degree_abs_top10": (
        "ROI functional-connectivity node degree",
        "感兴趣区功能连接节点度",
        ("node degree", "degree centrality", "functional connectivity"),
    ),
    "partial_mean": (
        "ROI mean partial-correlation connectivity",
        "感兴趣区平均偏相关功能连接",
        ("partial correlation", "functional connectivity"),
    ),
    "partial_mean_abs": (
        "ROI mean absolute partial-correlation connectivity",
        "感兴趣区平均绝对偏相关功能连接",
        ("partial correlation", "functional connectivity"),
    ),
    "partial_positive_mean": (
        "ROI positive partial-correlation connectivity",
        "感兴趣区正向偏相关功能连接",
        ("partial correlation", "positive connectivity"),
    ),
    "partial_negative_mean": (
        "ROI negative partial-correlation connectivity",
        "感兴趣区负向偏相关功能连接",
        ("partial correlation", "negative connectivity", "anticorrelation"),
    ),
}

REGION_FAMILIES = {
    "frontal": ("frontal", "precentral", "orbitofrontal", "ofc", "pfc"),
    "temporal": ("temporal", "fusiform", "parahippocamp", "hippocamp", "amygdala"),
    "parietal": ("parietal", "postcentral", "precuneus", "supramarginal", "angular"),
    "occipital": ("occipital", "calcarine", "lingual", "cuneus", "visual"),
    "cingulate": ("cingulate", "cingulum", "anterior cingulate", "posterior cingulate"),
    "insula": ("insula", "insular"),
    "thalamus": ("thalam",),
    "striatal": ("caudate", "putamen", "pallid", "striat", "accumbens"),
    "cerebellum": ("cerebell", "vermis"),
    "default": ("default mode", "dmn", "default"),
    "attention": ("attention network", "dorsal attention", "ventral attention", "salience"),
    "sensorimotor": ("sensorimotor", "somatomotor", "motor network"),
}

IMAGING_TERMS = (
    "fmri",
    "functional magnetic resonance",
    "functional connectivity",
    "resting-state",
    "resting state",
    "bold",
    "alff",
    "falff",
    "brain network",
    "neuroimaging",
)


def jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(item, dict):
                yield item


def paper_key(paper: dict[str, Any]) -> str:
    doi = re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        str(paper.get("doi") or "").strip().casefold(),
    )
    return str(
        doi
        or paper.get("pmid")
        or paper.get("title")
        or ""
    ).strip().casefold()


def paper_url(paper: dict[str, Any]) -> str:
    doi = str(paper.get("doi") or "").strip()
    pmid = str(paper.get("pmid") or "").strip()
    if doi:
        return f"https://doi.org/{doi}"
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return "https://scholar.google.com/scholar?q=" + quote_plus(
        str(paper.get("title") or "")
    )


def disease_tags(text: str) -> set[str]:
    lower = text.casefold()
    return {
        disease
        for disease, profile in DISEASES.items()
        if any(term in lower for term in profile["terms"])
    }


def load_abstracts(path: Path) -> dict[str, str]:
    abstracts: dict[str, str] = {}
    for item in jsonl(path):
        paper = item.get("paper") if isinstance(item.get("paper"), dict) else {}
        abstract = str(item.get("abstract") or "").strip()
        if not abstract:
            continue
        for key in (
            str(item.get("pmid") or "").strip().casefold(),
            str(paper.get("doi") or "").strip().casefold(),
            str(paper.get("title") or "").strip().casefold(),
        ):
            if key:
                abstracts[key] = abstract
    return abstracts


def normalized_title_tokens(title: str) -> set[str]:
    stop = {"a", "an", "the", "in", "of", "for", "on", "to", "and", "with", "by", "et", "al"}
    normalized = re.sub(r"[^\w\s]", " ", title.casefold())
    return {
        token
        for token in re.sub(r"\s+", " ", normalized).strip().split()
        if token and token not in stop
    }


def first_author_surname(authors: str) -> str:
    first = re.split(r"[;|]", authors.strip(), maxsplit=1)[0]
    if "," in first:
        return first.split(",", maxsplit=1)[0].strip().casefold()
    return first.split(maxsplit=1)[0].strip().casefold() if first else ""


def deduplicate_papers(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by DOI, then PMID, then title Jaccard + first author."""

    deduped: list[dict[str, Any]] = []
    doi_index: dict[str, int] = {}
    pmid_index: dict[str, int] = {}
    author_index: dict[str, list[int]] = defaultdict(list)

    def merge(target: dict[str, Any], incoming: dict[str, Any]) -> None:
        for field in ("title", "authors", "year", "journal", "doi", "pmid", "url", "abstract"):
            if not target.get(field) and incoming.get(field):
                target[field] = incoming[field]
        target["claims"] = list(
            dict.fromkeys([*target.get("claims", []), *incoming.get("claims", [])])
        )
        target["disease_tags"] = sorted(
            set(target.get("disease_tags", []))
            | set(incoming.get("disease_tags", []))
        )
        target["url"] = paper_url(target)
        target["_searchable"] = " ".join(
            [target["title"], target["abstract"], *target["claims"]]
        ).casefold()

    for record in records:
        doi = re.sub(
            r"^https?://(?:dx\.)?doi\.org/",
            "",
            str(record.get("doi") or "").strip().casefold(),
        )
        pmid = str(record.get("pmid") or "").strip()
        match_index = doi_index.get(doi) if doi else None
        if match_index is None and pmid:
            match_index = pmid_index.get(pmid)
        author = first_author_surname(str(record.get("authors") or ""))
        tokens = normalized_title_tokens(str(record.get("title") or ""))
        if match_index is None and author and tokens:
            for candidate_index in author_index.get(author, []):
                candidate_tokens = normalized_title_tokens(
                    str(deduped[candidate_index].get("title") or "")
                )
                union = tokens | candidate_tokens
                jaccard = len(tokens & candidate_tokens) / len(union) if union else 0.0
                if jaccard >= 0.90:
                    match_index = candidate_index
                    break
        if match_index is None:
            match_index = len(deduped)
            deduped.append(record)
            if author:
                author_index[author].append(match_index)
        else:
            merge(deduped[match_index], record)
        if doi:
            doi_index[doi] = match_index
        if pmid:
            pmid_index[pmid] = match_index
    return deduped


def build_paper_index(
    claims_path: Path,
    abstracts_path: Path,
) -> list[dict[str, Any]]:
    abstracts = load_abstracts(abstracts_path)
    records: dict[str, dict[str, Any]] = {}
    for claim in jsonl(claims_path):
        paper = claim.get("source_paper")
        if not isinstance(paper, dict):
            continue
        raw_text = str(claim.get("raw_text") or "").strip()
        searchable = " ".join(
            str(claim.get(key) or "")
            for key in (
                "disease",
                "subject_name",
                "object_name",
                "predicate",
                "raw_text",
            )
        )
        tags = disease_tags(searchable)
        if not tags:
            continue
        key = paper_key(paper)
        if not key:
            continue
        record = records.setdefault(
            key,
            {
                "title": str(paper.get("title") or "").strip(),
                "authors": str(paper.get("authors") or "").strip(),
                "year": paper.get("year"),
                "journal": str(paper.get("journal") or "").strip(),
                "doi": str(paper.get("doi") or "").strip(),
                "pmid": str(paper.get("pmid") or "").strip(),
                "claims": [],
                "disease_tags": set(),
            },
        )
        record["disease_tags"].update(tags)
        if raw_text and raw_text not in record["claims"]:
            record["claims"].append(raw_text)

    output = []
    for key, record in records.items():
        abstract = (
            abstracts.get(str(record["pmid"]).casefold())
            or abstracts.get(str(record["doi"]).casefold())
            or abstracts.get(str(record["title"]).casefold())
            or ""
        )
        record["abstract"] = abstract
        record["url"] = paper_url(record)
        record["_searchable"] = " ".join(
            [record["title"], abstract, *record["claims"]]
        ).casefold()
        record["disease_tags"] = sorted(record["disease_tags"])
        output.append(record)
    return deduplicate_papers(output)


def clean_region(row: pd.Series) -> str:
    value = str(row.get("anatomy_full") or "").strip()
    if not value or value.casefold() == "nan":
        value = str(row.get("roi_name") or "").strip()
    source = str(row.get("source") or "").removesuffix("_multiatlas")
    prefix = source + "_"
    if value.casefold().startswith(prefix.casefold()):
        value = value[len(prefix):]
    return value.replace("_", " ").strip()


def region_families(region: str) -> set[str]:
    lower = region.casefold()
    return {
        family
        for family, terms in REGION_FAMILIES.items()
        if any(term in lower for term in terms)
    }


def selected_ids(selection: pd.DataFrame) -> set[str]:
    ids: set[str] = set()
    for column in ("confirmed_id", "falsified_id", "left_id", "right_id"):
        if column in selection:
            ids.update(
                str(value)
                for value in selection[column].dropna()
                if str(value).strip()
            )
    return ids


def select_papers(
    papers: list[dict[str, Any]],
    *,
    disease: str,
    region: str,
    feature: str,
    limit: int,
) -> list[dict[str, Any]]:
    feature_terms = FEATURES[feature][2]
    families = region_families(region)
    scored: list[tuple[float, int, dict[str, Any], list[str]]] = []
    for paper in papers:
        if disease not in paper["disease_tags"]:
            continue
        text = paper["_searchable"]
        feature_match = any(term.casefold() in text for term in feature_terms)
        matched_families = sorted(
            family
            for family in families
            if any(term in text for term in REGION_FAMILIES[family])
        )
        imaging_match = any(term in text for term in IMAGING_TERMS)
        if not imaging_match or not (feature_match or matched_families):
            continue
        score = 5.0
        score += 3.0 if feature_match else 0.0
        score += min(4.0, 2.0 * len(matched_families))
        score += 0.75 if paper["abstract"] else 0.0
        score += 0.25 if paper["doi"] or paper["pmid"] else 0.0
        scored.append(
            (
                score,
                int(paper.get("year") or 0),
                paper,
                matched_families,
            )
        )
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]["title"]))

    selected = []
    seen_titles: set[str] = set()
    for score, _, paper, matched_families in scored:
        title_key = re.sub(r"\W+", "", paper["title"].casefold())
        if not title_key or title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        excerpt = next(
            (
                claim
                for claim in paper["claims"]
                if any(term.casefold() in claim.casefold() for term in feature_terms)
                or any(
                    term in claim.casefold()
                    for family in matched_families
                    for term in REGION_FAMILIES[family]
                )
            ),
            paper["abstract"][:800]
            if paper["abstract"]
            else (paper["claims"][0] if paper["claims"] else ""),
        )
        selected.append(
            {
                key: paper[key]
                for key in (
                    "title",
                    "authors",
                    "year",
                    "journal",
                    "doi",
                    "pmid",
                    "url",
                    "abstract",
                )
            }
            | {
                "excerpt": excerpt,
                "prior_evidence_match_score": round(score, 3),
                "matched_region_families": matched_families,
                "manual_relevance_status": "pending",
                "relevance_reason_zh": "",
                "relevance_reason_en": "",
                "abstract_verified": bool(paper["abstract"]),
                "abstract_source": (
                    "full_v2_abstract_cache"
                    if paper["abstract"]
                    else "full_v2_extracted_claim"
                ),
            }
        )
        if len(selected) >= limit:
            break
    return selected


def build_hypothesis(row: pd.Series, papers: list[dict[str, Any]]) -> dict[str, Any]:
    disease = str(row["disease"])
    feature = str(row["feature"])
    region = clean_region(row)
    direction = (
        "increased"
        if float(row["tcp_adjusted_residual_d"]) > 0
        else "decreased"
    )
    direction_zh = "升高" if direction == "increased" else "降低"
    disease_en = str(DISEASES[disease]["name_en"])
    disease_zh = str(DISEASES[disease]["name_zh"])
    feature_en, feature_zh, _ = FEATURES[feature]
    literature = select_papers(
        papers,
        disease=disease,
        region=region,
        feature=feature,
        limit=5,
    )
    return {
        "id": str(row["candidate_id"]),
        "title": f"{disease_en} → {region} → {feature_en} ({direction})",
        "title_zh": f"{disease_zh} → {region} → {feature_zh}（{direction_zh}）",
        "summary": (
            f"Relative to healthy controls, patients with {disease_en} are "
            f"hypothesized to show {direction} {feature_en} in {region}."
        ),
        "summary_zh": (
            f"该假设认为：与健康对照相比，{disease_zh}患者在 {region} 的"
            f"{feature_zh}{direction_zh}。"
        ),
        "source_name": disease_en,
        "target_name": f"{region} | {feature_en}",
        "composite_score": float(row["score_neurodiscovery"]),
        "literature": literature,
        "metadata": {
            "case_study": "case1_tcp_external_validation",
            "comparison_task_id": (
                f"{disease}|{row['source']}|{feature}"
            ),
            "candidate_tuple": {
                "disease": disease,
                "disease_name": disease_en,
                "atlas_name": str(row["source"]).removesuffix("_multiatlas"),
                "source": str(row["source"]),
                "roi_index": int(row["roi_index"]),
                "region_name": region,
                "feature_id": feature,
                "feature_name": feature_en,
                "direction": direction,
            },
            "literature_count": len(literature),
        },
    }


def queue_rows(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        candidate = hypothesis["metadata"]["candidate_tuple"]
        for paper_rank, paper in enumerate(hypothesis["literature"], start=1):
            rows.append(
                {
                    "candidate_id": hypothesis["id"],
                    "hypothesis_title": hypothesis["title"],
                    "hypothesis_title_zh": hypothesis["title_zh"],
                    "disease": candidate["disease"],
                    "atlas": candidate["atlas_name"],
                    "roi_index": candidate["roi_index"],
                    "region": candidate["region_name"],
                    "feature": candidate["feature_id"],
                    "direction": candidate["direction"],
                    "paper_rank": paper_rank,
                    "paper_title": paper["title"],
                    "authors": paper["authors"],
                    "year": paper["year"],
                    "journal": paper["journal"],
                    "doi": paper["doi"],
                    "pmid": paper["pmid"],
                    "url": paper["url"],
                    "abstract": paper["abstract"],
                    "evidence_excerpt": paper["excerpt"],
                    "matched_region_families": "|".join(
                        paper["matched_region_families"]
                    ),
                    "abstract_verified": paper["abstract_verified"],
                    "abstract_source": paper["abstract_source"],
                    "manual_relevance_status": "pending",
                    "what_the_paper_studied_zh": "",
                    "specific_support_zh": "",
                    "evidence_strength_zh": "",
                    "boundary_zh": "",
                    "what_the_paper_studied_en": "",
                    "specific_support_en": "",
                    "evidence_strength_en": "",
                    "boundary_en": "",
                }
            )
    return rows


def write_review_batches(
    queue: pd.DataFrame,
    output_dir: Path,
    *,
    hypotheses_per_batch: int,
) -> list[dict[str, Any]]:
    batch_root = output_dir / "relevance_batches"
    batch_root.mkdir(parents=True, exist_ok=True)
    if queue.empty:
        return []
    candidate_ids = list(dict.fromkeys(queue["candidate_id"].astype(str)))
    manifest: list[dict[str, Any]] = []
    for start in range(0, len(candidate_ids), hypotheses_per_batch):
        batch_number = start // hypotheses_per_batch + 1
        batch_ids = candidate_ids[start : start + hypotheses_per_batch]
        batch = queue[queue["candidate_id"].astype(str).isin(batch_ids)].copy()
        path = batch_root / f"batch_{batch_number:03d}.csv"
        batch.to_csv(path, index=False)
        manifest.append(
            {
                "batch": batch_number,
                "path": str(path),
                "hypotheses": len(batch_ids),
                "references": int(len(batch)),
                "status": "pending_manual_relevance_review",
            }
        )
    return manifest


def pair_literature_overlap(
    selection: pd.DataFrame,
    hypotheses: list[dict[str, Any]],
) -> pd.DataFrame:
    reference_ids = {
        hypothesis["id"]: {
            str(paper.get("pmid") or paper.get("doi") or paper.get("title") or "")
            for paper in hypothesis["literature"]
        }
        for hypothesis in hypotheses
    }
    rows = []
    for _, pair in selection.iterrows():
        left_id = str(pair.get("left_id") or "")
        right_id = str(pair.get("right_id") or "")
        left = reference_ids.get(left_id, set())
        right = reference_ids.get(right_id, set())
        union = left | right
        shared = left & right
        rows.append(
            {
                "pair_id": str(pair.get("pair_id") or ""),
                "left_id": left_id,
                "right_id": right_id,
                "shared_references": len(shared),
                "reference_jaccard": len(shared) / len(union) if union else 0.0,
                "passes_overlap_gate": (
                    len(left) == 5
                    and len(right) == 5
                    and (len(shared) / len(union) if union else 0.0) <= 0.50
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--abstracts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hypotheses-per-batch", type=int, default=10)
    args = parser.parse_args()

    selection = pd.read_csv(args.selection, low_memory=False)
    labels = pd.read_csv(args.labels, low_memory=False)
    ids = selected_ids(selection)
    labels = labels[labels["candidate_id"].astype(str).isin(ids)].copy()
    papers = build_paper_index(args.claims, args.abstracts)
    hypotheses = [
        build_hypothesis(row, papers)
        for _, row in labels.sort_values("rank").iterrows()
    ]
    queue = queue_rows(hypotheses)
    summary = {
        "schema_version": "case1-tcp-external-literature-queue-v1",
        "selected_hypotheses": len(hypotheses),
        "hypotheses_with_five_papers": sum(
            len(item["literature"]) == 5 for item in hypotheses
        ),
        "hypotheses_with_fewer_than_five_papers": sum(
            len(item["literature"]) < 5 for item in hypotheses
        ),
        "selected_reference_records": len(queue),
        "unique_papers": len(
            {
                str(row["pmid"] or row["doi"] or row["paper_title"])
                for row in queue
            }
        ),
        "manual_relevance_reasons_completed": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queue_frame = pd.DataFrame(queue)
    queue_frame.to_csv(
        args.output_dir / "case1_tcp_external_literature_relevance_queue.csv",
        index=False,
    )
    batches = write_review_batches(
        queue_frame,
        args.output_dir,
        hypotheses_per_batch=args.hypotheses_per_batch,
    )
    overlap = pair_literature_overlap(selection, hypotheses)
    overlap.to_csv(
        args.output_dir / "case1_tcp_external_pair_literature_overlap_audit.csv",
        index=False,
    )
    summary["manual_review_batches"] = len(batches)
    summary["hypotheses_per_batch"] = args.hypotheses_per_batch
    summary["pairs_passing_literature_overlap_gate"] = int(
        overlap["passes_overlap_gate"].sum()
    ) if not overlap.empty else 0
    summary["pairs_failing_literature_overlap_gate"] = int(
        (~overlap["passes_overlap_gate"]).sum()
    ) if not overlap.empty else 0
    summary["maximum_pair_reference_jaccard"] = (
        float(overlap["reference_jaccard"].max()) if not overlap.empty else 0.0
    )
    payload = {
        **summary,
        "hypotheses": hypotheses,
    }
    (args.output_dir / "case1_tcp_external_hypotheses_with_literature.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "case1_tcp_external_relevance_batch_manifest.json").write_text(
        json.dumps({"batches": batches}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "case1_tcp_external_literature_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
