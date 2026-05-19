"""Phase 2: LLM 声明抽取 + Biomarker 关注扫描

从 PubMed 文献中提取结构化科学声明，解析实体并入图。
支持渐进式升级：先从 abstract 快速抽取，后用全文增强 evidence。

Usage:
    # 主流程
    python -m neurooracle.phase2 --broad --max-workers 8
    python -m neurooracle.phase2 --diseases "Alzheimer's disease" --year-start 2025 --year-end 2025

    # 2.6 biomarker mention scanner
    python -m neurooracle.phase2 biomarker-scan \\
        --graph neurooracle/data/knowledge_graph.json \\
        --claims neurooracle/data/extracted_claims.jsonl \\
        --output neurooracle/data/biomarker_mentions.json \\
        --mode local
"""

import sys

from .src.claim_extractor import ClaimExtractor
from .src.claim_ingestion import ingest_claims
from .src.evidence_enhancer import EvidenceEnhancer, enhance_and_update
from .src.batch_extract import main as batch_extract_main
from .src.biomarker_scan import main as biomarker_scan_main

if __name__ == "__main__":
    # Dispatch: first positional arg "biomarker-scan" routes to that tool.
    if len(sys.argv) > 1 and sys.argv[1] == "biomarker-scan":
        sys.argv.pop(1)  # drop the subcommand token so argparse in biomarker_scan sees clean argv
        biomarker_scan_main()
    else:
        batch_extract_main()
