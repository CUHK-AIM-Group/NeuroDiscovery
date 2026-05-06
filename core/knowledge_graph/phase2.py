"""Phase 2: LLM 声明抽取

从 PubMed 文献中提取结构化科学声明，解析实体并入图。
支持渐进式升级：先从 abstract 快速抽取，后用全文增强 evidence。

Usage:
    python -m core.knowledge_graph.phase2 --broad --max-workers 8
    python -m core.knowledge_graph.phase2 --diseases "Alzheimer's disease" --year-start 2025 --year-end 2025
"""

from .src.claim_extractor import ClaimExtractor
from .src.claim_ingestion import ingest_claims
from .src.evidence_enhancer import EvidenceEnhancer, enhance_and_update
from .src.batch_extract import main as batch_extract_main

if __name__ == "__main__":
    batch_extract_main()
