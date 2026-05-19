"""Phase 1: 结构化数据入库

从 NeuroNames、MeSH、DisGeNET、CognitiveAtlas 构建知识图谱骨架。
UMLS 对齐确保跨来源实体一致性。

Usage:
    python -m neurooracle.phase1
"""

from .src.ingest_pipeline import run_full_ingestion
from .src.umls_integration import align_graph_to_umls

if __name__ == "__main__":
    from .src.ingest_pipeline import main
    main()
