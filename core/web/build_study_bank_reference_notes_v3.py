"""Add paper-level citation counts to the HE2 reference-note sidecar.

The frozen v2 notes retain abstracts, cohort extraction, p-value extraction,
and journal metadata. This version adds an OpenAlex citation count for every
unique bank reference. The UI uses this paper-level metric only when no p value
is directly bound to the selected evidence; unrelated abstract p values are not
shown as substitutes.
"""
import copy
import hashlib
import json
from pathlib import Path

from core.web.build_study_bank_reference_notes_v2 import BANK, BANK_SHA256

ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE = BANK.with_name(BANK.stem + "_reference_notes_v2.json")
SOURCE_SHA256 = "229746366d11c1aff6f6c43988319bf876ea015be7859ae5102297bd263c01b1"
CITATIONS = BANK.with_name(BANK.stem + "_citation_counts_v1.json")
CITATIONS_SHA256 = "3b90bf6835d223ee9a868491c4d9104ba6b69c0eb146f237d5fc32aa859e2567"
TARGET = BANK.with_name(BANK.stem + "_reference_notes_v3.json")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checked(path, expected):
    if _sha256(path) != expected:
        raise ValueError(f"Pinned input changed: {path}")
    return json.loads(Path(path).read_bytes().decode("utf-8-sig"))


def build():
    notes = copy.deepcopy(_checked(SOURCE, SOURCE_SHA256))
    citations = _checked(CITATIONS, CITATIONS_SHA256)
    if notes.get("bank_sha256") != BANK_SHA256:
        raise ValueError("The v2 notes are not bound to the frozen HE2 bank.")
    if citations.get("unique_bank_references") != len(notes["references"]):
        raise ValueError("Citation inventory does not cover the reference inventory.")
    citation_rows = citations.get("citations") or {}
    if set(citation_rows) != set(notes["references"]):
        raise ValueError("Every reference must have exactly one citation-count record.")

    for key, note in notes["references"].items():
        row = citation_rows[key]
        note.setdefault("credibility", {})["citation_count"] = {
            "count": int(row["count"]),
            "source": row["source"],
            "retrieved_at": row["retrieved_at"],
            "source_url": row["source_url"],
        }

    notes.update(
        schema_version="case1-tcp-expert-study-v2-reference-notes-v3",
        created_at="2026-09-21T20:45:00+08:00",
    )
    notes["sources"]["citation_counts"] = str(CITATIONS.relative_to(ROOT))
    notes["disclosure_zh"] = (
        "本地摘要来自冻结语料库与 PubMed 摘要缓存；队列规模与 p 值为摘要文本规则提取，"
        "未经人工核对。界面只显示与所选证据直接绑定的 p 值；没有此类 p 值时，改为显示 "
        "2026-09-21 获取的 OpenAlex 单篇论文引用量。期刊 JIF 属期刊层面指标。参考文献仍为"
        "知识图谱自动匹配候选，本轮未逐条人工复核，也不代表全图谱中已验证的最相近论文。"
    )
    return notes


def main():
    document = build()
    data = (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if TARGET.exists() and TARGET.read_bytes() != data:
        raise ValueError("A different v3 reference-notes file already exists.")
    TARGET.write_bytes(data)
    print(json.dumps({
        "notes": TARGET.name,
        "notes_sha256": hashlib.sha256(data).hexdigest(),
        "references": len(document["references"]),
        "citation_counts": sum(
            "citation_count" in (note.get("credibility") or {})
            for note in document["references"].values()
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
