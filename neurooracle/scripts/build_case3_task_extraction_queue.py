"""Merge Case Study 3 literature collections into extraction queues."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from neurooracle.scripts.collect_case3_task_template_literature import TASK_QUERIES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = (
    ROOT
    / "neurooracle"
    / "data"
    / "cs_runs"
    / "case3_hindcasting"
    / "merged_task_literature_20260616_v1"
)


IMAGING_TERMS = {
    "mri", "fmri", "pet", "spect", "dti", "diffusion", "connectivity",
    "connectome", "cortical", "hippocampal", "amygdala", "subcortical",
    "neuroimaging",
    "brain age", "radiomics", "suvr", "amyloid", "tau", "fdg",
    "functional connectivity", "structural connectivity", "cortical thickness",
    "brain volume", "gray matter", "grey matter", "white matter",
    "fractional anisotropy", "mean diffusivity", "resting state",
}

TASK_SIGNAL_TERMS = {
    "biomarker_discovery": {"biomarker", "diagnosis", "classification", "marker", "predict"},
    "disease_subtyping": {"subtype", "subtyping", "cluster", "stratification", "heterogeneity"},
    "progression_prediction": {"progression", "conversion", "longitudinal", "decline", "follow-up"},
    "imaging_genetics": {"gene", "genetic", "gwas", "polygenic", "apoe", "variant", "transcriptomic"},
    "differential_diagnosis": {"differential", "distinguish", "discriminate", "classification"},
    "drug_response_prediction": {"treatment response", "drug response", "medication", "antidepressant", "antipsychotic"},
    "personalised_treatment": {"personalized", "personalised", "precision", "treatment selection", "stratification"},
    "drug_repurposing": {"repurposing", "drug", "connectivity signature", "pharmacoimaging"},
    "adverse_event_prediction": {"adverse", "toxicity", "side effect", "safety"},
    "neuromodulation_target": {"stimulation", "tms", "dbs", "neuromodulation", "target"},
    "functional_localization": {"task", "activation", "localization", "mapping"},
    "cognitive_decoding": {"decode", "decoding", "cognitive state", "prediction", "classification"},
    "connectome_behavior": {"behavior", "behaviour", "cognition", "individual differences", "symptom"},
    "brain_age": {"brain age", "age gap", "predicted age", "aging", "ageing"},
    "prognosis": {"prognosis", "outcome", "survival", "disability", "longitudinal"},
}

STUDY_DESIGN_BONUS = {
    "meta-analysis": 2.0,
    "systematic review": 1.5,
    "review": 0.6,
    "longitudinal": 1.2,
    "multi-site": 0.8,
    "multisite": 0.8,
    "enigma": 1.2,
    "uk biobank": 1.0,
    "adni": 1.0,
    "abcd": 1.0,
}


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _norm_doi(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    value = value.removeprefix("doi:")
    return value.rstrip("/")


def _norm_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _candidate_key(row: dict[str, str]) -> str:
    doi = _norm_doi(row.get("doi") or "")
    if doi:
        return f"doi:{doi}"
    identifier = (row.get("id") or row.get("pmid") or "").strip()
    if identifier:
        return f"id:{identifier}"
    title = _norm_title(row.get("title") or "")
    year = (row.get("year") or "").strip()
    return f"title:{year}:{title}"


def _load_cache_abstracts(paths: Iterable[Path]) -> dict[str, str]:
    abstracts: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                pmid = str(obj.get("pmid") or "").strip()
                abstract = _norm_text(obj.get("abstract") or "")
                if pmid and abstract:
                    abstracts[pmid] = abstract
    return abstracts


def _iter_rows(path: Path, run_label: str) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = dict(row)
            if "id" not in out or not out.get("id"):
                out["id"] = out.get("pmid") or ""
            out["source_run"] = run_label
            yield out


def _task_name(label: str) -> str:
    label = label or ""
    if label.startswith("task:"):
        return label.split(":", 1)[1]
    if label.startswith("chain:"):
        return label
    return label


def _has_term(text: str, term: str) -> bool:
    parts = re.findall(r"[a-z0-9]+", term.lower())
    if not parts:
        return False
    pattern = r"\b" + r"[\s-]+".join(re.escape(part) for part in parts) + r"\b"
    return re.search(pattern, text) is not None


def _any_term(text: str, terms: Iterable[str]) -> bool:
    return any(_has_term(text, term) for term in terms)


def _score_for_task(task: str, title: str, abstract: str, source_count: int, label_count: int) -> float:
    abstract_head = (abstract or "")[:2500]
    text = f"{title} {abstract_head}".lower()
    title_l = title.lower()
    score = 0.0
    imaging_hit = _any_term(text, IMAGING_TERMS)
    title_imaging_hit = _any_term(title_l, IMAGING_TERMS)
    task_hit = _any_term(text, TASK_SIGNAL_TERMS.get(task, set()))
    if imaging_hit:
        score += 2.0
    if title_imaging_hit:
        score += 1.0
    for term in TASK_SIGNAL_TERMS.get(task, set()):
        if _has_term(text, term):
            score += 1.5
        if _has_term(title_l, term):
            score += 1.0
    for phrase in TASK_QUERIES.get(task, []):
        terms = [t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if len(t) > 3]
        if not terms:
            continue
        overlap = sum(1 for t in terms if t in text)
        score += min(2.0, overlap * 0.25)
    for term, bonus in STUDY_DESIGN_BONUS.items():
        if _has_term(text, term):
            score += bonus
    score += min(2.0, 0.5 * max(0, source_count - 1))
    score += min(1.5, 0.3 * max(0, label_count - 1))
    if len(abstract) < 400:
        score -= 2.0
    if not imaging_hit:
        score = min(score, 2.5)
    elif not task_hit:
        score = min(score, 4.0)
    return round(score, 3)


def build_queue(collections: list[tuple[Path, str]], cache_paths: list[Path], out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    abstracts = _load_cache_abstracts(cache_paths)
    merged: dict[str, dict[str, object]] = {}

    for path, run_label in collections:
        for row in _iter_rows(path, run_label):
            key = _candidate_key(row)
            if key not in merged:
                identifier = row.get("id") or row.get("pmid") or ""
                merged[key] = {
                    "key": key,
                    "id": identifier,
                    "doi": _norm_doi(row.get("doi") or ""),
                    "title": _norm_text(row.get("title") or ""),
                    "authors": _norm_text(row.get("authors") or ""),
                    "year": row.get("year") or "",
                    "journal": _norm_text(row.get("journal") or ""),
                    "labels": set(),
                    "sources": set(),
                    "source_runs": set(),
                    "queries": set(),
                    "abstract_length": 0,
                }
            item = merged[key]
            item["labels"].add(row.get("label") or "")  # type: ignore[union-attr]
            item["sources"].add(row.get("source") or "")  # type: ignore[union-attr]
            item["source_runs"].add(row.get("source_run") or "")  # type: ignore[union-attr]
            query = row.get("query") or ""
            if query:
                item["queries"].add(query)  # type: ignore[union-attr]
            if not item.get("doi"):
                item["doi"] = _norm_doi(row.get("doi") or "")
            if not item.get("id"):
                item["id"] = row.get("id") or row.get("pmid") or ""
            try:
                item["abstract_length"] = max(
                    int(item.get("abstract_length") or 0),
                    int(row.get("abstract_length") or 0),
                )
            except Exception:
                pass

    candidate_rows: list[dict[str, object]] = []
    queue_rows: list[dict[str, object]] = []
    for item in merged.values():
        identifier = str(item.get("id") or "")
        abstract = abstracts.get(identifier, "")
        labels = sorted(x for x in item["labels"] if x)  # type: ignore[index]
        sources = sorted(x for x in item["sources"] if x)  # type: ignore[index]
        source_runs = sorted(x for x in item["source_runs"] if x)  # type: ignore[index]
        queries = sorted(x for x in item["queries"] if x)  # type: ignore[index]
        base = {
            "key": item["key"],
            "id": identifier,
            "doi": item.get("doi") or "",
            "title": item.get("title") or "",
            "year": item.get("year") or "",
            "journal": item.get("journal") or "",
            "labels": "|".join(labels),
            "sources": "|".join(sources),
            "source_runs": "|".join(source_runs),
            "queries": " || ".join(queries[:8]),
            "abstract_length": max(int(item.get("abstract_length") or 0), len(abstract)),
            "has_abstract": bool(abstract),
        }
        candidate_rows.append(base)
        for label in labels:
            task = _task_name(label)
            if task.startswith("chain:") or task not in TASK_SIGNAL_TERMS:
                continue
            score = _score_for_task(
                task,
                str(item.get("title") or ""),
                abstract,
                len(sources),
                len(labels),
            )
            queue_rows.append({
                **base,
                "task": task,
                "priority_score": score,
                "priority_tier": "core" if score >= 6 else "standard" if score >= 3 else "low",
            })

    candidate_rows.sort(key=lambda r: (str(r["year"]), str(r["title"])))
    queue_rows.sort(key=lambda r: (str(r["task"]), -float(r["priority_score"]), str(r["year"]), str(r["title"])))

    candidate_csv = out_dir / "merged_candidates.csv"
    queue_csv = out_dir / "task_priority_extraction_queue.csv"
    _write_csv(candidate_csv, candidate_rows)
    _write_csv(queue_csv, queue_rows)

    by_task = defaultdict(lambda: {"total": 0, "core": 0, "standard": 0, "low": 0})
    for row in queue_rows:
        stats = by_task[str(row["task"])]
        stats["total"] += 1
        stats[str(row["priority_tier"])] += 1

    summary = {
        "collections": [str(path) for path, _ in collections],
        "cache_paths": [str(path) for path in cache_paths],
        "merged_candidates": len(candidate_rows),
        "queue_rows": len(queue_rows),
        "has_abstract": sum(1 for row in candidate_rows if row["has_abstract"]),
        "by_task": dict(sorted(by_task.items())),
        "outputs": {
            "merged_candidates": str(candidate_csv),
            "task_priority_extraction_queue": str(queue_csv),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_collection(value: str) -> tuple[Path, str]:
    if "=" in value:
        label, path = value.split("=", 1)
        return Path(path), label
    path = Path(value)
    return path, path.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", action="append", required=True, help="RUN_LABEL=path/to/collection_metadata.csv")
    parser.add_argument("--cache", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = build_queue(
        collections=[_parse_collection(v) for v in args.collection],
        cache_paths=args.cache,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
